"""Adaptive prompt and response budgets for the local model context window."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field

from model_profiles import get_profile
import model_runtime

logger = logging.getLogger(__name__)


class AnswerDetail(str, Enum):
    BRIEF = "BRIEF"
    NORMAL = "NORMAL"
    DETAILED = "DETAILED"


class ContextCoverage(str, Enum):
    FOCUSED = "FOCUSED"
    BALANCED = "BALANCED"
    BROAD = "BROAD"


class BudgetDecision(BaseModel):
    detail: AnswerDetail
    coverage: ContextCoverage
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(max_length=200)


@dataclass(frozen=True)
class ResponseBudget:
    detail: AnswerDetail
    coverage: ContextCoverage
    max_output_tokens: int
    max_source_characters: int
    max_history_characters: int
    reason: str


_DETAIL_TOKENS = {
    AnswerDetail.BRIEF: 192,
    AnswerDetail.NORMAL: 768,
    AnswerDetail.DETAILED: 1400,
}
_DETAIL_RE = re.compile(
    r"\b(?:подробн\w*|детальн\w*|разв[её]рнут\w*|глубок\w*|проанализир\w*|"
    r"исчерпывающ\w*|по пунктам)\b", re.IGNORECASE,
)
_BRIEF_RE = re.compile(r"\b(?:кратко|коротко|одним предложением|без подробностей)\b", re.IGNORECASE)
_BROAD_RE = re.compile(
    r"\b(?:все(?:х|ми)?|весь|всю|всё|кажд\w*|целиком|полностью|внутри директор\w*|"
    r"по всем файлам|обзор)\b",
    re.IGNORECASE,
)


class ResponseBudgetPlanner:
    """Lets the LLM choose intent-level detail, then enforces physical model limits."""

    def decide(self, question: str, model: str | None = None) -> BudgetDecision:
        if _BRIEF_RE.search(question):
            return BudgetDecision(detail=AnswerDetail.BRIEF, coverage=ContextCoverage.FOCUSED,
                                  confidence=1, reason="Пользователь явно запросил краткий ответ")
        if _DETAIL_RE.search(question):
            coverage = ContextCoverage.BROAD if _BROAD_RE.search(question) else ContextCoverage.BALANCED
            return BudgetDecision(detail=AnswerDetail.DETAILED, coverage=coverage, confidence=1,
                                  reason="Пользователь явно запросил подробный анализ")
        from source_grounding import requests_source_overview
        if requests_source_overview(question) and len(question.split()) <= 12:
            return BudgetDecision(detail=AnswerDetail.NORMAL,
                                  coverage=ContextCoverage.BROAD if _BROAD_RE.search(question) else ContextCoverage.BALANCED,
                                  confidence=1, reason="Короткий вопрос о предмете: прямое объяснение по источникам")
        from agent_planner import is_single_information_request
        from source_grounding import factual_query
        factual = factual_query(question)
        if (len(factual.split()) <= 30 and is_single_information_request(factual)
                and not re.search(r'\b(?:почему|объясн\w*|расскаж\w*|опиш\w*)\b', factual, re.I)
                and (re.match(r'^\s*как\s+зовут\b', factual, re.I)
                     or re.match(r'^\s*(?:какой|какая|какое|какие|сколько|когда|где)\b', factual, re.I)
                     and re.search(r'\b(?:рост|нагрузк\w*|масс\w*|вес|телефон\w*|код\w*|номер\w*|'
                                   r'дат[ауы]|время|имя|стоимост\w*|цен[ауы])\b', factual, re.I))):
            return BudgetDecision(detail=AnswerDetail.BRIEF, coverage=ContextCoverage.FOCUSED,
                                  confidence=1, reason="Прямой вопрос об имени или параметрах по источнику")
        if len(factual.split()) <= 30 and is_single_information_request(factual):
            return BudgetDecision(detail=AnswerDetail.NORMAL, coverage=ContextCoverage.BALANCED,
                                  confidence=1, reason="Прямой информационный вопрос по источникам")
        try:
            result = model_runtime.get_llm_provider().structured_generate(
                profile=model,
                messages=[{
                    "role": "system",
                    "content": (
                        "Определи необходимую подробность ответа и охват источников. "
                        "BRIEF — простой факт или короткий ответ; NORMAL — объяснение или обычный вопрос; "
                        "DETAILED — анализ, пересказ, сравнение или многочастный результат. "
                        "FOCUSED — достаточно одного точного фрагмента; BALANCED — нужны несколько релевантных "
                        "фрагментов; BROAD — требуется охватить файл целиком или много файлов. "
                        "Не отвечай на вопрос, верни только JSON по схеме."
                    ),
                }, {"role": "user", "content": question}],
                schema=BudgetDecision.model_json_schema(), thinking=False, temperature=0, max_tokens=120,
            )
            return BudgetDecision.model_validate_json(result.text)
        except Exception as exc:
            logger.warning("Response budget planning unavailable; using safe fallback: %s", exc)
            return BudgetDecision(
                detail=AnswerDetail.NORMAL,
                coverage=ContextCoverage.BROAD if _BROAD_RE.search(question) else ContextCoverage.BALANCED,
                confidence=0, reason="Безопасный бюджет по умолчанию",
            )

    def plan(self, question: str, model: str | None = None) -> ResponseBudget:
        decision = self.decide(question, model)
        profile = get_profile(model)
        reserve = 420  # chat template, instructions, question and source labels
        requested_output = _DETAIL_TOKENS[decision.detail]
        max_output = min(requested_output, profile.max_tokens,
                         max(256, profile.context_size - reserve - 512))
        available_input_tokens = max(512, profile.context_size - max_output - reserve)
        history_share = .18 if decision.coverage != ContextCoverage.BROAD else .10
        history_tokens = max(192, int(available_input_tokens * history_share))
        source_tokens = max(320, available_input_tokens - history_tokens)
        return ResponseBudget(
            detail=decision.detail, coverage=decision.coverage,
            max_output_tokens=max_output, max_source_characters=source_tokens * 3,
            max_history_characters=history_tokens * 3, reason=decision.reason,
        )
