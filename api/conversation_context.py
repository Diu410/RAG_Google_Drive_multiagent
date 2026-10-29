"""Conversation-aware query resolution shared by every retrieval skill."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from pydantic import BaseModel, Field

from model_runtime import get_llm_provider


class ResolvedQuery(BaseModel):
    standalone_question: str = Field(min_length=1, max_length=32000)
    depends_on_history: bool
    referent: str = Field(default='',max_length=160,
        description='Буквальное имя предмета из истории, заменяющее местоимение; без новых свойств.')


def strip_dialogue_prefix(question):
    question = re.sub(r'^(?:странно|интересно|забавно|непонятно)'
        r'(?:\s+(?:это|вс[её]|конечно|очень|как-то)){0,5}\s*[,!.]\s*(?:а\s+|и\s+)?'
        r'(?=(?:кто|что|чего|чем|кому|кого|кем|как|почему|зачем|откуда|куда|где|когда|како\w*|сколько)\b)',
        '',question.strip(),count=1,flags=re.I)
    return re.sub(r'^(?:(?:отлично|хорошо|понятно|спасибо|ладно|ок(?:ей)?)\s*[,!.]?\s*)?'
                  r'(?:а\s+|и\s+)?', '', question.strip(),count=1,flags=re.I)


def is_narrow_confirmation(question):
    """A short check of one proposition is not a request to repeat its overview."""
    text = strip_dialogue_prefix(question)
    return bool(len(text.split()) <= 12 and text.count('?') == 1
        and re.match(r'^(?:(?:то\s+есть|значит),?|верно\s+ли,?\s+что|правильно\s+(?:я\s+)?понимаю,?\s+что)\s+\S',text,re.I)
        and not re.search(r'\b(?:и|или|также|почему|зачем|как|когда|где|подроб\w*|деталь\w*|анализ\w*|'
            r'объясни\w*|расскажи\w*|сначала|затем|потом|открой\w*|прочитай\w*|найди\w*|'
            r'сохрани\w*|создай\w*|запиши\w*|удали\w*|переименуй\w*|запусти\w*)\b|[;:\n]',text,re.I))


def has_explicit_overview_subject(question):
    """A named overview subject does not need the previous topic substituted."""
    match = re.fullmatch(r'(?:кто\s+так(?:ой|ая|ое)|что\s+(?:за|такое)|'
        r'что\s+(?:(?:ты|вы)\s+)?зна(?:ешь|ете)\s+об?|'
        r'что\s+(?:можешь|можете)\s+рассказать\s+об?|расскажи(?:те)?\s+об?)\s+(.+?)[?!. ]*',
        strip_dialogue_prefix(question),re.I)
    if not match:
        return False
    subject = match[1].strip()
    return bool(subject and not re.search(r'\b(?:он|она|оно|они|его|е[её]|их|ему|ей|им|'
        r'н[её]м|ней|эт\w*|тот|та|те|так\w*|данн\w*|текущ\w*|предыдущ\w*)\b',subject,re.I))


def asks_history_question(question):
    text=command_text_for_history(question)
    if re.search(r'\b(?:по|в|из)\s+(?:документ|файл|источник)\w*',text):return False
    return bool(re.search(r'^(?:напомни(?:те)?\s*,?\s*(?:что\s+я|о\s+ч[её]м\s+мы|как\s+мы)|'
        r'что\s+я\s+(?:просил|говорил|запретил|разрешил)|'
        r'какие\s+(?:ограничения|решения)\s+(?:я|мы)|'
        r'о\s+ч[её]м\s+мы\s+(?:говорили|договаривались))\b',text))


def command_text_for_history(question):
    from command_intent import command_text
    return command_text(question).strip()


def archive_marker_explanation(question, history):
    """Recover from an old leaked service prefix, not from a real file request."""
    previous = next((content for role,content in map(_message,reversed(history or []))
        if role in {'assistant','ai'} and content.strip()), '')
    marker = re.match(r'^\s*\[Архив\s+(\d+)\]',previous,re.I)
    if not marker:
        return None
    text = command_text_for_history(question)
    if (re.search(r'\b(?:что\s+за|что\s+(?:это|значит)|почему|зачем|откуда)\b',text)
            and re.search(r'\b(?:архив|метк\w*)\b',text)
            and not re.search(r'\.[a-zа-я0-9]{1,10}\b',text)
            and (not re.search(r'\d+',text) or marker[1] in re.findall(r'\d+',text))):
        return ('«[Архив '+marker[1]+']» — служебная метка контекста, ошибочно попавшая в мой ответ. '
            'Она не является названием найденного файла. Извините за путаницу.')
    return None


def preserve_question_focus(question, resolved, history):
    """Resolve a referent without turning a relation question into its cause."""
    original=strip_dialogue_prefix(question)
    causal = r'\b(?:почему|зачем|причин\w*|основани\w*)\b'
    drift = not re.search(causal,original,re.I) and re.search(causal,resolved.standalone_question,re.I)
    pronouns=r'\b(?:он|она|оно|они|его|е[её]|их|ему|ей|им|н[её]м|ней)\b'
    if drift or resolved.depends_on_history and re.search(pronouns,original,re.I):
        referent=resolved.referent.strip()
        if not referent:
            # A previous explicit identity question names a topic, not facts
            # from the assistant's potentially rejected answer.
            for message in reversed(normalized_history(history)):
                if message['role']=='user':
                    match=re.fullmatch(r'(?:кто\s+так(?:ой|ая)|что\s+такое|что\s+знаешь\s+о)\s+(.+?)[?!. ]*',
                                       message['content'].strip(),re.I)
                    if match:
                        referent=match[1];break
        if referent and any(referent.casefold() in m['content'].casefold() for m in normalized_history(history)):
            resolved.standalone_question=re.sub(pronouns,referent,original,flags=re.I)
        else:
            resolved.standalone_question=original
    return resolved


def _message(message) -> tuple[str, str]:
    if isinstance(message, dict):
        return str(message.get("role", "")), str(message.get("content", ""))
    return str(getattr(message, "type", getattr(message, "role", ""))), str(getattr(message, "content", ""))


def normalized_history(history: list | None, limit: int = 6) -> list[dict[str, str]]:
    messages = []
    prepared=bool(history) and all(isinstance(m,dict) and m.get('_context_selected') for m in history)
    for raw in (history or []) if prepared else (history or [])[-limit:]:
        role, content = _message(raw)
        if not content.strip():
            continue
        messages.append({
            "role": "user" if role.casefold() in {"human", "user"} else "assistant",
            "content": content if prepared else content[-3000:],
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
        if not messages or has_explicit_overview_subject(question):
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
                    "меняй смысл. Сохрани тип и узость вопроса: «чего/чей/какой» требует предмета или "
                    "принадлежности, а не причин, оснований, обязанностей или биографии. referent — "
                    "только буквальное имя из истории. Предыдущий ответ помогает разрешить ссылку, "
                    "но не является доказательством. Если вопрос самодостаточен или меняет тему, верни его без изменений."
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
        return preserve_question_focus(question,resolved,history)
