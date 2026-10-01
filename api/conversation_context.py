"""Conversation-aware query resolution shared by every retrieval skill."""
from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import BaseModel, Field

from model_runtime import get_llm_provider


class ResolvedQuery(BaseModel):
    standalone_question: str = Field(min_length=1, max_length=32000)
    depends_on_history: bool


def _message(message) -> tuple[str, str]:
    if isinstance(message, dict):
        return str(message.get("role", "")), str(message.get("content", ""))
    return str(getattr(message, "type", getattr(message, "role", ""))), str(getattr(message, "content", ""))


def normalized_history(history: list | None, limit: int = 6) -> list[dict[str, str]]:
    messages = []
    for raw in (history or [])[-limit:]:
        role, content = _message(raw)
        if not content.strip():
            continue
        messages.append({
            "role": "user" if role.casefold() in {"human", "user"} else "assistant",
            "content": content[-3000:],
        })
    return messages


@dataclass
class ConversationResolver:
    """Turn a conversational follow-up into a retrieval-safe standalone query.

    The original question is never replaced in persisted history.  Resolution
    only supplies a separate query for routing and retrieval.
    """

    def resolve(self, question: str, history: list, model: str) -> ResolvedQuery:
        messages = normalized_history(history)
        if not messages:
            return ResolvedQuery(standalone_question=question, depends_on_history=False)
        payload = json.dumps(messages, ensure_ascii=False)
        response = get_llm_provider().structured_generate(
            profile=model,
            messages=[{
                "role": "system",
                "content": (
                    "Определи, зависит ли новый вопрос от предыдущего диалога. Разреши местоимения, "
                    "пропущенные сущности и ссылки вроде «это», «там», «тот человек». Верни самостоятельный "
                    "вопрос, пригодный для поиска по файлам. Не отвечай на вопрос, не добавляй факты и не "
                    "меняй смысл. Если вопрос самодостаточен или меняет тему, верни его без изменений."
                ),
            }, {
                "role": "user",
                "content": f"История JSON: {payload}\nНовый вопрос: {question}",
            }],
            schema=ResolvedQuery.model_json_schema(),
            thinking=False,
            temperature=0,
            max_tokens=256,
        )
        resolved = ResolvedQuery.model_validate_json(response.text)
        if not resolved.depends_on_history:
            resolved.standalone_question = question
        return resolved
