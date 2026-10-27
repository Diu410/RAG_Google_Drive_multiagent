"""Rules-first workload scheduling; duration never grants tool permissions."""
from dataclasses import asdict, dataclass
import re

from command_intent import command_text, requests_file_open
from local_tool_intent import (directory_summary_request, permitted_local_operations, repeated_user_request,
                               document_search_query, document_search_selection, previous_answer_export)
from model_profiles import get_profile


@dataclass(frozen=True)
class TaskPolicy:
    kind: str
    seconds: float
    reason: str

    def as_dict(self):return asdict(self)


def classify_task(question, model='BALANCED', workspace=False):
    import settings
    question=repeated_user_request(question)
    text=command_text(question)
    profile=get_profile(model)
    from conversation_control import session_announcement_reply
    if session_announcement_reply(question):
        return TaskPolicy('SHORT',settings.SHORT_TASK_SECONDS,'Обозначение диалога без поиска и генерации')
    if workspace:
        if previous_answer_export(question):
            return TaskPolicy('SHORT',settings.SHORT_TASK_SECONDS,'Экспорт готового ответа без генерации')
        selection = document_search_selection(question)
        if document_search_query(question) is not None or selection and selection['operation'] != 'select':
            return TaskPolicy('SHORT',settings.SHORT_TASK_SECONDS,'Поиск или выбор документа без полного текстового ответа')
        from repository_agent import repository_intent
        if repository_intent(question):
            return TaskPolicy('LONG',settings.LONG_TASK_SECONDS,'Исследование проекта, правки и проверки')
        if directory_summary_request(question):
            return TaskPolicy('LONG',settings.LONG_TASK_SECONDS,'Документ по всему содержимому папки')
        operations=permitted_local_operations(question)
        if 'create_document' in operations and re.search(r'\b(?:с\s+текстом|пуст\w*)\b',text):
            return TaskPolicy('SHORT',settings.SHORT_TASK_SECONDS,'Сохранение готового текста без генерации')
        if 'create_document' in operations and not re.search(r'\b(?:с\s+текстом|пуст\w*)\b',text):
            return TaskPolicy('LONG',settings.LONG_TASK_SECONDS,'Подготовка и сверка нового документа')
        if re.search(r'\b(?:подробн\w*|проанализируй|исследуй)\b',text) and re.search(
                r'\b(?:вс[её]\w*|целиком|полностью|кажд\w*)\b',text):
            return TaskPolicy('LONG',settings.LONG_TASK_SECONDS,'Развёрнутая обработка нескольких источников')
        from workspace_intent import display_file_target, asks_last_opened_file
        if requests_file_open(question) or display_file_target(question) or asks_last_opened_file(question):
            return TaskPolicy('SHORT',settings.SHORT_TASK_SECONDS,'Прямая файловая операция')
    if re.fullmatch(r'\s*(?:привет|здравствуй(?:те)?|спасибо|hello|hi|thanks)[!.? ]*',text,re.I):
        return TaskPolicy('SHORT',settings.SHORT_TASK_SECONDS,'Короткая реплика')
    return TaskPolicy('STANDARD',profile.request_seconds,'Обычный ответ с бюджетом выбранного профиля')


def apply_policy(scope, policy):
    if scope is None:return
    scope.deadline=scope.started_at+policy.seconds
    scope.task_kind=policy.kind


def refine_long_task(scope, model, units, seconds_per_unit=18):
    import settings
    seconds=min(settings.LONG_TASK_SECONDS,max(get_profile(model).request_seconds,units*seconds_per_unit))
    policy=TaskPolicy('LONG',seconds,'Бюджет по фактическому числу частей задачи')
    apply_policy(scope,policy)
    return policy
