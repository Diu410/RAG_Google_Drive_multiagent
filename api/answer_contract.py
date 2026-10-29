"""Per-turn answer authority. Source selection and permission to speculate are separate.

The contract is created from the current user's words, never from a retrieved
document or a previous assistant answer. ContextVars carry it through existing
sync providers without changing the public/legacy agent APIs.
"""
from contextvars import ContextVar
from dataclasses import asdict, dataclass, replace
import re

from command_intent import command_text


current_contract = ContextVar('answer_contract', default=None)


def requests_interpretation(question):
    text = command_text(question)
    # Discussing or forbidding speculation is not permission to speculate.
    if re.search(r'\b(?:не\s+(?:надо\s+|нужно\s+)?(?:гадай|додумывай|предполагай|интерпретируй)|'
                 r'без\s+(?:догадок|предположений|интерпретаций)|только\s+по\s+(?:тексту|источникам|документам))\b', text):
        return False
    patterns = (r'\b(?:как|что)\s+(?:же\s+)?(?:ты|вы)\s+(?:(?:сам|сами|лично)\s+)?(?:думаешь|думаете|считаешь|считаете)\b',
                r'\b(?:по\s+твоему\s+мнению|на\s+твой\s+взгляд|твоя\s+интерпретация|тво[её]\s+мнение)\b',
                r'\b(?:предложи|выскажи|дай)\s+(?:(?:свою|свои|сво[её]|возможные)\s+)?(?:гипотез\w*|интерпретаци\w*|догадк\w*|предположени\w*)\b',
                r'\b(?:предположи|поразмышляй|интерпретируй)\b',
                r'\b(?:what\s+do\s+you\s+think|in\s+your\s+opinion|your\s+interpretation|suggest\s+(?:a\s+)?hypothesis)\b')
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            before = text[max(0, match.start()-65):match.start()]
            if not re.search(r'\b(?:не|not|если|когда|например)\b[^.!?;\n]*$', before):
                return True
    return False


def question_kind(question):
    text = command_text(question)
    if re.search(r'\b(?:чего\s+(?:\w+\s+){0,4}добива\w*|каков\w*\s+цел\w*|каки\w*\s+цел\w*|'
                 r'к\s+чему\s+(?:\w+\s+){0,4}стрем\w*|что\s+.+?хочет\s+добиться)\b', text):
        return 'GOAL'
    if re.search(r'\b(?:почему|зачем|откуда|отчего|причин\w*)\b', text):
        return 'CAUSE'
    if re.search(r'\b(?:кто|кем|чь[еёяи]|как\s+зовут|имя|название)\b', text):
        return 'IDENTITY'
    if re.search(r'\b(?:перескажи|пересказать|о\s+ч[её]м|обзор|пересказ)\b', text):
        return 'OVERVIEW'
    return 'INFORMATION'


def information_request(question):
    """Conversational wrappers/multiple question marks do not change the task."""
    text = command_text(question)
    if re.search(r'\b(?:создай|сохрани|запиши|открой|прочитай|запусти|удали|переименуй|найди|сравни)\b', text):
        return False
    return bool(re.search(r'\b(?:кто|кем|кого|что|чего|чей|чья|каков\w*|како[йея]|какие|'
                          r'почему|зачем|откуда|отчего|куда|сколько|когда|где|как|who|what|why|how)\b', text))


@dataclass(frozen=True)
class AnswerContract:
    question: str
    source: str = 'AUTO'
    focus: str = 'INFORMATION'
    allow_interpretation: bool = False
    workspace_id: str | None = None
    source_scope: dict | None = None
    topic: str = ''
    resolved_question: str = ''
    brief: bool = False

    def as_dict(self):
        return asdict(self)

    def resolved(self, question, referent=''):
        # Authority and focus stay anchored to the literal current turn.
        source = self.source
        if source == 'AUTO' and self.workspace_id and referent and ('?' in self.question or information_request(self.question)):
            source = 'WORKSPACE'
        return replace(self, resolved_question=question, topic=referent or self.topic, source=source)

    def instructions(self):
        focus = {'GOAL': 'Запрошена цель: чего добивается участник, не причины и не история её появления.',
                 'IDENTITY': 'Запрошены имя, природа или принадлежность; не заменяй их целями и биографией.',
                 'CAUSE': 'Запрошена причина или происхождение; соседнее описание и самоназвание не устанавливают историю появления.',
                 'OVERVIEW': 'Запрошена общая картина содержания, а не перечисление отсутствующих подробностей.'}
        return ('\nКонтракт текущего ответа: '+focus.get(self.focus, 'Ответь именно на исходный вопрос.')+
            ' Сведения передавай в области источника: слова, самоописание и намерение автора — не установленная реальность. '
            'Автор, персонаж, адресат и пользователь чата различны, пока источник не устанавливает связь. '
            'В этом фактическом тексте не добавляй собственных гипотез. '+
            ('Пользователь явно запросил мнение; отдельная интерпретация будет подготовлена после сверки оснований. '
             if self.allow_interpretation else 'Собственная интерпретация в этой реплике не запрошена. ')+
            'Недостаток данных обозначь один раз и только по запрошенному предмету. '
            'Производная справка не является независимым подтверждением первичного текста.')

    def accepts_replacement(self, before, after):
        from source_grounding import is_missing_information_claim
        causal = r'\b(?:причин\w*|потому|так\s+как|почему|истори\w*|выполненн\w*\s+путь)\b'
        if self.focus != 'CAUSE' and re.search(causal, command_text(after)) and not re.search(causal, command_text(before)):
            return False
        if is_missing_information_claim(after) and not is_missing_information_claim(before):
            # A generic abstention is not a correction of a participant/action.
            terms = set(re.findall(r'[а-яёa-z]{4,}', command_text(self.question)))
            terms -= {'источник', 'источниках', 'документах', 'файлах', 'сказал', 'понятно', 'тогда'}
            roots = {word[:5] for word in terms}
            if not roots.intersection(word[:5] for word in re.findall(r'[а-яёa-z]{4,}', command_text(after))):
                return False
        return True


def make_contract(question, workspace_id=None, source_scope=None):
    from query_router import route_by_rules, QueryStrategy
    from source_grounding import requests_document_library, overview_subject
    from conversation_context import asks_history_question
    route = route_by_rules(question)
    social = bool(route and route.strategy == QueryStrategy.GENERAL_CHAT)
    opinion = requests_interpretation(question)
    source = ('HISTORY' if asks_history_question(question) else
              'CONVERSATION' if social else
              'LIBRARY' if requests_document_library(question) else
              'WORKSPACE' if workspace_id and (information_request(question) or opinion) else 'AUTO')
    return AnswerContract(question=question, source=source, focus=question_kind(question),
        allow_interpretation=opinion, workspace_id=workspace_id, source_scope=source_scope,
        topic=overview_subject(question), resolved_question=question,
        brief=bool(re.search(r'\b(?:вкратце|(?:кратк|коротк)(?:о|ий|ая|ое|ие|ую|ого|ой|их|ому|им|ими)|'
                             r'briefly|brief)\b',command_text(question))))
