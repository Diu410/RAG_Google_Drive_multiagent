"""Rules-first query routing with a structured provider fallback."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from pydantic import BaseModel, Field, ValidationError, field_validator

from settings import FAST_CONTEXT_CHUNKS, ROUTER_LLM_FALLBACK
from model_runtime import get_llm_provider
from llm_contract import LLMProvider

logger = logging.getLogger(__name__)


class QueryStrategy(str, Enum):
    GENERAL_CHAT = "GENERAL_CHAT"
    DOCUMENT_QA = "DOCUMENT_QA"
    FACT_LOOKUP = "FACT_LOOKUP"
    DOCUMENT_COMPARISON = "DOCUMENT_COMPARISON"
    DEEP_ANALYSIS = "DEEP_ANALYSIS"
    GRAPH_QUERY = "GRAPH_QUERY"


class RouteDecision(BaseModel):
    strategy: QueryStrategy
    confidence: float = Field(ge=0.0, le=1.0)
    entities: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("entities")
    @classmethod
    def clean_entities(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(value.strip()[:200] for value in values if value.strip()))


FALLBACK_ROUTE = RouteDecision(strategy=QueryStrategy.DOCUMENT_QA, confidence=0.0, entities=[])


@dataclass(frozen=True)
class RetrievalPlan:
    mode: str = "hybrid"
    result_k: int = FAST_CONTEXT_CHUNKS
    # Exact local terminology and propagated PDF section headings should beat
    # semantically similar but unrelated chunks from other documents.
    lexical_weight: float = 1.25
    query_variants: list[str] = field(default_factory=list)
    graph_interface: bool = False


def retrieval_plan(decision: RouteDecision, requested_mode: str | None = None) -> RetrievalPlan | None:
    if decision.strategy == QueryStrategy.GENERAL_CHAT:
        return None
    mode = requested_mode or "hybrid"
    if decision.strategy == QueryStrategy.FACT_LOOKUP:
        return RetrievalPlan(mode=mode, lexical_weight=2.0 if mode == "hybrid" else 1.0)
    if decision.strategy == QueryStrategy.DOCUMENT_COMPARISON:
        return RetrievalPlan(mode=mode, result_k=max(FAST_CONTEXT_CHUNKS * 2, 4),
                             query_variants=decision.entities)
    if decision.strategy == QueryStrategy.DEEP_ANALYSIS:
        return RetrievalPlan(mode=mode, result_k=FAST_CONTEXT_CHUNKS * 2)
    if decision.strategy == QueryStrategy.GRAPH_QUERY:
        return RetrievalPlan(mode=mode, graph_interface=True)
    return RetrievalPlan(mode=mode)

_GENERAL_RE = re.compile(
    r"^\s*(?:привет(?:ик|ствую)?|здравствуй(?:те)?|добрый\s+(?:день|вечер|утро)|"
    r"спасибо|благодарю|как\s+(?:(?:твои|у\s+тебя)\s+)?дела|hello|hi|hey|thanks)\s*[!.?]*\s*$", re.IGNORECASE,
)
_GRAPH_RE = re.compile(
    r"\b(?:кто|что|как|чем)\s+(?:связан[аоы]?|участву(?:ет|ют)|относится)|"
    r"\b(?:связи|связь|отношения|взаимосвязи|цепочк\w*)\s+между\b",
    re.IGNORECASE,
)
_COMPARISON_RE = re.compile(r"\b(?:сравни(?:ть|вай)?|сопоставь|различия|чем\s+отлича(?:ется|ются))\b", re.IGNORECASE)
_DEEP_RE = re.compile(r"\b(?:проанализируй|глубокий\s+анализ|оцени\s+риски|анализ\s+рисков|выяви\s+риски)\b", re.IGNORECASE)
_FACT_RE = re.compile(
    r"\b(?:найди|укажи|какой|какая|какое|где)\b.{0,45}\b(?:номер|дат[ауеы]?|идентификатор|инн|кпп|"
    r"огрн|артикул|код|сумм[ауеы]?|телефон|email|договор[ауе]?)\b|\b(?:№|id)\s*[\w.-]+",
    re.IGNORECASE,
)
_DOCUMENT_RE = re.compile(
    r"\b(?:что\s+(?:написано|сказано|указано)|согласно\s+документу|в\s+документе|по\s+документам|"
    r"найди\s+(?:в|по)\s+документ|summari[sz]e\s+(?:the\s+)?(?:report|document)|"
    r"where\s+is\s+(?:the\s+)?(?:report|document))\b", re.IGNORECASE,
)
_QUOTED_RE = re.compile(r"[«\"]([^»\"]{1,200})[»\"]")


def _comparison_entities(query: str) -> list[str]:
    quoted = _QUOTED_RE.findall(query)
    if len(quoted) >= 2:
        return quoted[:8]
    tail = re.split(r"\b(?:документ(?:ы|а|ов)?|файл(?:ы|а|ов)?)\b", query, maxsplit=1, flags=re.IGNORECASE)
    candidate = tail[-1]
    parts = re.split(r"\s+(?:и|с|vs\.?|против)\s+|\s*,\s*", candidate, flags=re.IGNORECASE)
    cleaned = [re.sub(r"[?.!]+$", "", part).strip() for part in parts]
    return [part for part in cleaned if part][:8]


def route_by_rules(query: str) -> RouteDecision | None:
    from conversation_control import session_announcement_reply
    if _GENERAL_RE.fullmatch(query) or session_announcement_reply(query):
        return RouteDecision(strategy=QueryStrategy.GENERAL_CHAT, confidence=1.0)
    if _GRAPH_RE.search(query):
        return RouteDecision(strategy=QueryStrategy.GRAPH_QUERY, confidence=0.95, entities=_QUOTED_RE.findall(query))
    if _COMPARISON_RE.search(query):
        return RouteDecision(strategy=QueryStrategy.DOCUMENT_COMPARISON, confidence=0.95,
                             entities=_comparison_entities(query))
    if _DEEP_RE.search(query):
        return RouteDecision(strategy=QueryStrategy.DEEP_ANALYSIS, confidence=0.9,
                             entities=_QUOTED_RE.findall(query))
    if _FACT_RE.search(query):
        return RouteDecision(strategy=QueryStrategy.FACT_LOOKUP, confidence=0.9,
                             entities=_QUOTED_RE.findall(query))
    if _DOCUMENT_RE.search(query):
        return RouteDecision(strategy=QueryStrategy.DOCUMENT_QA, confidence=0.9,
                             entities=_QUOTED_RE.findall(query))
    return None


class QueryRouter:
    def __init__(self, client_factory: Callable[[], LLMProvider] = get_llm_provider,
                 use_llm_fallback: bool = True):
        self.client_factory = client_factory
        self.use_llm_fallback = use_llm_fallback

    def route(self, query: str, model: str) -> RouteDecision:
        deterministic = route_by_rules(query)
        if deterministic is not None:
            return deterministic
        if not self.use_llm_fallback:
            return FALLBACK_ROUTE.model_copy(deep=True)
        try:
            decision = self._route_with_llm(query, model)
            return decision if decision.confidence >= .65 else FALLBACK_ROUTE.model_copy(deep=True)
        except Exception as exc:
            logger.warning("LLM query routing failed; using document QA fallback: %s", exc)
            return FALLBACK_ROUTE.model_copy(deep=True)

    def _route_with_llm(self, query: str, model: str) -> RouteDecision:
        client = self.client_factory()
        response = client.structured_generate(
            profile=model,
            messages=[
                {"role": "system", "content": (
                    "Classify the user's request. Return only the requested JSON. "
                    "Choose GENERAL_CHAT only when no external factual knowledge is needed: social conversation, "
                    "discussion of information already stated by the user, brainstorming, or explicitly creative/fictional writing. "
                    "Choose DOCUMENT_QA for every request that needs factual claims, explanations about the real world, "
                    "or knowledge not supplied in the conversation, as well as requests to use, find, summarize, or answer "
                    "from local documents. A factual question does not need to mention documents explicitly. "
                    "FACT_LOOKUP asks for an exact number, date, code or identifier; DOCUMENT_COMPARISON compares named "
                    "documents or objects; DEEP_ANALYSIS requests synthesis or risk analysis; GRAPH_QUERY asks about "
                    "relationships between people, organizations, projects or other entities. Extract only explicitly "
                    "named entities. Tone and slang do not determine the route; whether unsupported factual knowledge is needed does. "
                    "Confidence describes routing certainty."
                )},
                {"role": "user", "content": query},
            ],
            schema=RouteDecision.model_json_schema(),
            thinking=False,
            temperature=0, max_tokens=180,
        )
        content = response.text
        try:
            return RouteDecision.model_validate_json(content)
        except (ValidationError, json.JSONDecodeError, TypeError) as exc:
            raise ValueError("Router returned invalid structured JSON") from exc


query_router = QueryRouter(use_llm_fallback=ROUTER_LLM_FALLBACK)
