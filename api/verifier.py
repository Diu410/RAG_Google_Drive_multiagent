"""Source-grounded answer verification and deterministic confidence scoring."""
from __future__ import annotations

import json
import logging
import math
import re
from enum import Enum
from typing import Any, Iterable, Literal

from pydantic import BaseModel, Field
from pydantic import ValidationError
from model_profiles import get_profile

from model_runtime import get_llm_provider
from pydantic_models import ConfidenceComponents, VerificationClaim, VerificationResult
from settings import (
    FAST_LLM_NUM_CTX,
    MAX_VERIFICATION_RETRIES,
    RAG_LLM_NUM_PREDICT,
    VERIFIER_CLAIM_SUPPORT_WEIGHT,
    VERIFIER_CONFIDENCE_THRESHOLD,
    VERIFIER_ENABLED,
    VERIFIER_AUTO_STRATEGIES,
    VERIFIER_RETRIEVAL_WEIGHT,
    VERIFIER_SOURCE_AGREEMENT_WEIGHT,
)

logger = logging.getLogger(__name__)


class VerificationStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    CONFLICTING = "CONFLICTING"
    INSUFFICIENT_INFORMATION = "INSUFFICIENT_INFORMATION"


class ClaimAssessment(BaseModel):
    text: str = Field(min_length=1)
    status: VerificationStatus
    source_chunk_ids: list[str] = Field(default_factory=list)
    basis: Literal["EXPLICIT", "INFERENCE", "UNANSWERED"] = "EXPLICIT"
    evidence_quote: str = ""


class LLMVerification(BaseModel):
    status: VerificationStatus
    claims: list[ClaimAssessment] = Field(default_factory=list)


_STATUS_FACTOR = {
    VerificationStatus.SUPPORTED: 1.0,
    VerificationStatus.PARTIALLY_SUPPORTED: 0.80,
    VerificationStatus.INSUFFICIENT_INFORMATION: 0.35,
    VerificationStatus.UNSUPPORTED: 0.20,
    VerificationStatus.CONFLICTING: 0.10,
}


def _clip(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 4)


def _quote_source_ids(quote: str, source_by_id: dict[str, str]) -> list[str]:
    """Match quoted spans exactly while tolerating wrappers and omitted text."""
    spans = [" ".join(part.strip(" \t\r\n\"'«»„“”‘’.,;:…").casefold().split())
             for part in re.split(r"\.{3,}|…+", quote)]
    spans = [part for part in spans if part]
    if not spans or any(len(part) < 12 for part in spans):
        return []
    normalized_sources = {chunk_id: " ".join(content.casefold().split())
                          for chunk_id, content in source_by_id.items()}
    matched_ids: list[str] = []
    for span in spans:
        match = next((chunk_id for chunk_id, content in normalized_sources.items()
                      if span in content), None)
        if match is None:
            return []
        if match not in matched_ids:
            matched_ids.append(match)
    return matched_ids


def _aggregate_status(claims: list[ClaimAssessment], fallback: VerificationStatus) -> VerificationStatus:
    if not claims:
        return VerificationStatus.INSUFFICIENT_INFORMATION
    statuses = {claim.status for claim in claims}
    if VerificationStatus.CONFLICTING in statuses:
        return VerificationStatus.CONFLICTING
    if statuses == {VerificationStatus.SUPPORTED}:
        return VerificationStatus.SUPPORTED
    if VerificationStatus.SUPPORTED in statuses or VerificationStatus.PARTIALLY_SUPPORTED in statuses:
        return VerificationStatus.PARTIALLY_SUPPORTED
    if VerificationStatus.UNSUPPORTED in statuses:
        return VerificationStatus.UNSUPPORTED
    return fallback


def _retrieval_quality(results: list[dict[str, Any]]) -> float:
    """Normalize retrieval by rank and channel agreement; never mix raw score scales."""
    if not results:
        return 0.0
    limit = min(len(results), 5)
    discounts = [1.0 / math.log2(rank + 2) for rank in range(limit)]
    rank_coverage = sum(discounts) / sum(1.0 / math.log2(rank + 2) for rank in range(5))
    channel_agreement = sum(
        1.0 if len(set(results[index].get("found_by") or [])) > 1 else 0.65
        for index in range(limit)
    ) / limit
    return _clip(0.7 * rank_coverage + 0.3 * channel_agreement)


def calculate_confidence(assessment: LLMVerification,
                         results: list[dict[str, Any]]) -> tuple[float, ConfidenceComponents]:
    status = _aggregate_status(assessment.claims, assessment.status)
    if assessment.claims:
        support_values = {
            VerificationStatus.SUPPORTED: 1.0,
            VerificationStatus.PARTIALLY_SUPPORTED: 0.5,
            VerificationStatus.INSUFFICIENT_INFORMATION: 0.0,
            VerificationStatus.UNSUPPORTED: 0.0,
            VerificationStatus.CONFLICTING: 0.0,
        }
        claim_support = sum(support_values[c.status] for c in assessment.claims) / len(assessment.claims)
        agreement = sum(
            (1.0 if len(set(c.source_chunk_ids)) >= 2 else 0.6)
            if c.status == VerificationStatus.SUPPORTED else
            (0.35 if c.status == VerificationStatus.PARTIALLY_SUPPORTED else 0.0)
            for c in assessment.claims
        ) / len(assessment.claims)
    else:
        claim_support = agreement = 0.0
    components = ConfidenceComponents(
        retrieval=_retrieval_quality(results),
        claim_support=_clip(claim_support),
        source_agreement=_clip(agreement),
    )
    weighted = (
        components.retrieval * VERIFIER_RETRIEVAL_WEIGHT
        + components.claim_support * VERIFIER_CLAIM_SUPPORT_WEIGHT
        + components.source_agreement * VERIFIER_SOURCE_AGREEMENT_WEIGHT
    )
    return _clip(weighted * _STATUS_FACTOR[status]), components


def _source_payload(results: Iterable[dict[str, Any]], max_chars: int = 14000) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    used = 0
    for rank, result in enumerate(results, start=1):
        text = str(result.get("text", ""))
        remaining = max_chars - used
        if remaining <= 0:
            break
        text = text[:remaining]
        used += len(text)
        payload.append({
            "rank": rank,
            "document_id": result.get("document_id"),
            "chunk_id": str(result.get("chunk_id", "")),
            "text": text,
            "metadata": result.get("metadata") or {},
            "found_by": result.get("found_by") or [],
        })
    return payload


class Verifier:
    def __init__(self, client: Any | None = None):
        self.client = client or get_llm_provider()

    def bounded_sources(self, prefix, sources, model, output_tokens, think=False):
        """Keep source JSON valid while fitting the actual model's input budget."""
        sources = [dict(item) for item in sources]
        counter = getattr(self.client, 'count_prompt', None)
        if not callable(counter):
            return sources
        profile = get_profile(model)
        while True:
            prompt = prefix + json.dumps(sources, ensure_ascii=False, default=str)
            count = counter([{'role': 'user', 'content': prompt}], profile.alias, think)
            # Custom providers may not expose token counting (including mocks).
            if not isinstance(count, int) or count + output_tokens + 32 <= profile.context_size:
                return sources
            if not sources:
                return sources
            largest = max(range(len(sources)), key=lambda index: len(json.dumps(sources[index], ensure_ascii=False, default=str)))
            item = sources[largest]
            if item.get('metadata'):
                item['metadata'] = {}
            elif len(item.get('text', '')) > 128:
                item['text'] = item['text'][:len(item['text']) // 2]
            else:
                sources.pop(largest)

    def verify(self, question: str, answer: str, results: list[dict[str, Any]],
               strategy: str, model: str, *, think: bool = False) -> VerificationResult:
        sources = _source_payload(results)
        valid_ids = {item["chunk_id"] for item in sources if item["chunk_id"]}
        if not sources:
            return VerificationResult(
                status=VerificationStatus.INSUFFICIENT_INFORMATION.value,
                confidence=0.0,
                components=ConfidenceComponents(retrieval=0.0, claim_support=0.0, source_agreement=0.0),
                warning="Недостаточно источников для проверки ответа.",
            )
        prompt = (
            "Проверь одним проходом все проверяемые фактические утверждения ответа по источникам. "
            "Не оценивай стиль и не раскрывай ход рассуждений. Для каждого утверждения выбери один статус: "
            "SUPPORTED, PARTIALLY_SUPPORTED, UNSUPPORTED, CONFLICTING или INSUFFICIENT_INFORMATION. "
            "Раздели каждый ответ на отдельные существенные утверждения; не объединяй поддержанный факт с домыслом. "
            "Для каждого утверждения укажи basis: EXPLICIT — прямо сказано в источнике; "
            "INFERENCE — осторожная интерпретация, явно обозначенная в самом ответе; "
            "UNANSWERED — честное указание, что точный смысл или факт источником не раскрыт. "
            "Для EXPLICIT и INFERENCE укажи evidence_quote — короткую ДОСЛОВНУЮ цитату из указанного chunk_id; "
            "предпочитай один непрерывный фрагмент без добавленных слов и пересказа. "
            "Ссылка на файл без подходящей цитаты не доказывает утверждение. "
            "Не считай SUPPORTED категоричное определение метафоры, если текст даёт лишь образ. "
            "Осторожная интерпретация может быть SUPPORTED, если она совместима с дословной цитатой "
            "и явно названа возможной трактовкой, а не фактом. "
            "Не подтверждай факты только потому, что они были в предыдущем ответе ассистента. "
            "Указывай только chunk_id из переданного списка; не придумывай ссылки, документы и страницы.\n"
            f"Стратегия: {strategy}\nВопрос: {question}\nОтвет: {answer}\n"
            "Источники JSON: "
        )
        output_tokens = min(RAG_LLM_NUM_PREDICT, 1200)
        sources = self.bounded_sources(prompt, sources, model, output_tokens, think)
        if not sources:
            return VerificationResult(status='INSUFFICIENT_INFORMATION', confidence=0.0,
                                      warning='Источники не помещаются в контекст проверки. Сократите вопрос или ответ.')
        valid_ids = {item['chunk_id'] for item in sources if item['chunk_id']}
        prompt += json.dumps(sources, ensure_ascii=False, default=str)
        response = self.client.structured_generate(
            profile=model,
            messages=[{"role": "user", "content": prompt}],
            schema=LLMVerification.model_json_schema(),
            thinking=think,
            temperature=0, max_tokens=output_tokens,
        )
        assessment = LLMVerification.model_validate_json(response.text)
        source_by_id = {item["chunk_id"]: item["text"] for item in sources}
        for claim in assessment.claims:
            if claim.basis != "UNANSWERED" and (strategy == "WORKSPACE_QA" or not claim.source_chunk_ids):
                # The model often copies a filename rather than the opaque chunk_id.
                # It may also omit optional citation IDs. Recover a real ID
                # only from a literal quote that exists in the supplied sources.
                claim.source_chunk_ids = _quote_source_ids(claim.evidence_quote, source_by_id)
            else:
                claim.source_chunk_ids = list(dict.fromkeys(
                    chunk_id for chunk_id in claim.source_chunk_ids if chunk_id in valid_ids
                ))
                if claim.basis != 'UNANSWERED' and claim.status == VerificationStatus.SUPPORTED:
                    cited_sources = {value: source_by_id[value] for value in claim.source_chunk_ids}
                    quote_ids = set(_quote_source_ids(claim.evidence_quote, cited_sources))
                    quote_ids.update(value for value, text in cited_sources.items()
                                     if _quote_source_ids(claim.evidence_quote, {value: text}))
                    claim.source_chunk_ids = [value for value in claim.source_chunk_ids if value in quote_ids]
            if claim.status == VerificationStatus.SUPPORTED and not claim.source_chunk_ids and claim.basis != "UNANSWERED":
                claim.status = VerificationStatus.UNSUPPORTED
        assessment.status = _aggregate_status(assessment.claims, assessment.status)
        confidence, components = calculate_confidence(assessment, [item for item in results if item.get('chunk_id') in valid_ids])
        claims = [VerificationClaim(text=c.text, status=c.status.value,
                                    source_chunk_ids=c.source_chunk_ids, basis=c.basis,
                                    evidence_quote=c.evidence_quote or None) for c in assessment.claims]
        used_ids = list(dict.fromkeys(chunk_id for claim in claims for chunk_id in claim.source_chunk_ids))
        unsupported = [c.text for c in claims if c.status == VerificationStatus.UNSUPPORTED.value]
        conflicting = [c.text for c in claims if c.status == VerificationStatus.CONFLICTING.value]
        return VerificationResult(
            status=assessment.status.value,
            claims=claims,
            unsupported_claims=unsupported,
            conflicting_claims=conflicting,
            source_chunk_ids=used_ids,
            confidence=confidence,
            components=components,
        )

    def correct(self, question: str, answer: str, results: list[dict[str, Any]],
                verification: VerificationResult, model: str, *, think: bool = False) -> str:
        prompt = (
            "Исправь ответ один раз: убери или исправь неподтверждённые и противоречивые утверждения. "
            "Используй только переданные источники. Если данных недостаточно, явно скажи об этом. "
            "Сохрани подтверждённую часть: процитируй относящийся к вопросу фрагмент, "
            "затем отдельно обозначь возможную интерпретацию и пределы того, что известно. "
            "Сохрани краткость исходного ответа: не превращай исправление в расширенную биографию или обзор. "
            "Не превращай метафору в установленное определение и не добавляй новых деталей. "
            "Верни только исправленный ответ без анализа проверки.\n"
            f"Вопрос: {question}\nПредыдущий ответ: {answer}\n"
            f"Результат проверки: {verification.model_dump_json()}\n"
            "Источники: "
        )
        output_tokens = min(RAG_LLM_NUM_PREDICT, max(256, len(answer) // 2))
        sources = self.bounded_sources(prompt, _source_payload(results), model, output_tokens, think)
        if not sources:
            raise ValueError('Correction sources do not fit the context')
        prompt += json.dumps(sources, ensure_ascii=False, default=str)
        response = self.client.generate(
            profile=model,
            messages=[{"role": "user", "content": prompt}],
            thinking=think,
            temperature=0, max_tokens=output_tokens,
        )
        corrected = response.text.strip()
        if not corrected:
            raise ValueError("Correction returned an empty answer")
        return corrected


def unavailable_result(error=None) -> VerificationResult:
    warning = ('Проверка не завершена: модель вернула некорректный JSON.'
               if isinstance(error, (json.JSONDecodeError, ValidationError))
               else 'Проверка достоверности временно недоступна.')
    return VerificationResult(status="UNAVAILABLE", warning=warning)


def verify_answer(question: str, answer: str, results: list[dict[str, Any]], strategy: str,
                  model: str, verifier: Verifier | None = None) -> tuple[str, VerificationResult | None]:
    """Verify a final answer and perform at most the configured single correction."""
    results = [item for item in results if item.get("chunk_id")]
    if not VERIFIER_ENABLED or strategy == "GENERAL_CHAT" or not results:
        return answer, None
    service = verifier or Verifier()
    try:
        verification = service.verify(question, answer, results, strategy, model, think=False)
    except Exception as exc:
        logger.warning("Answer verification unavailable: %s", exc)
        return answer, unavailable_result(exc)

    corrected_answer = answer
    needs_correction = verification.status in {
        "PARTIALLY_SUPPORTED", "UNSUPPORTED", "CONFLICTING", "INSUFFICIENT_INFORMATION"
    } and (strategy == "WORKSPACE_QA" or (
        verification.confidence is not None
        and verification.confidence < VERIFIER_CONFIDENCE_THRESHOLD
    ))
    for _ in range(MAX_VERIFICATION_RETRIES if needs_correction else 0):
        try:
            candidate = service.correct(question, corrected_answer, results, verification, model, think=False)
            final_verification = service.verify(question, candidate, results, strategy, model, think=False)
            corrected_answer, verification = candidate, final_verification
        except Exception as exc:
            logger.warning("Answer correction unavailable; returning original answer: %s", exc)
            return answer, verification
        break
    if verification.confidence is not None and verification.confidence < VERIFIER_CONFIDENCE_THRESHOLD:
        verification.warning = "Ответ подтверждён источниками не полностью; информации может быть недостаточно."
    return corrected_answer, verification


def verify_knowledge_answer(question: str, answer: str, results: list[dict[str, Any]], strategy: str,
                            model: str) -> tuple[str, VerificationResult | None]:
    """Apply automatic verification only where its extra model pass is useful.

    Explicit verification in the writing workshop continues to call
    ``verify_answer`` directly and is therefore unaffected by this latency
    policy.
    """
    if strategy.upper() not in VERIFIER_AUTO_STRATEGIES:
        return answer, None
    return verify_answer(question, answer, results, strategy, model)
