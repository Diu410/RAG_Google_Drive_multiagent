"""Action intent from the user, independent of model confidence and rewrites."""
import re

from command_intent import command_text, negates_comparison, requests_file_open


_DISPLAY = re.compile(
    r'\b(?:прочитай|прочти|read|(?:выведи|покажи)\s+(?:содержим\w*|текст)|show\s+(?:the\s+)?contents?\s+of)'
    r'\s+(?:(?:файл|документ|заметк)\w*\s+|(?:the\s+)?file\s+)?(.+?)\s*[?!]*$', re.I)
_INSIDE = re.compile(
    r'^\s*(?:а\s+)?(?:что\s+(?:внутри|в|написано\s+в|находится\s+в)|'
    r'what\s+(?:is\s+)?(?:inside|in))\s+'
    r'(?:(?:файл|документ|заметк)\w*\s+|(?:the\s+)?file\s+)?(.+?)\s*[?!]*$', re.I)
_DIRECTORY_TARGET = re.compile(r'^(?:папк|директор|workspace|воркспейс|folder|directory)\w*\b', re.I)
_PRONOUN_TARGET = re.compile(r'^(?:эт\w*|текущ\w*|данн\w*|нем|нём|ней|его|е[её]|it|this|that)\b', re.I)


def _literal_target(match, text):
    target = match.group(1).strip().rstrip('?!.').strip(' "\'«»`“”')
    if re.fullmatch(r'md|markdown|txt|pdf|docx|html?|csv|tsv|json',target,re.I):
        label=re.search(r'((?:файл|документ|заметк)\w*)\s+$',text[:match.start(1)],re.I)
        if label:target=label[1]+' '+target
    return target


def references_created_file(text: str) -> bool:
    return bool(re.search(r'\b(?:созданн\w*|сохран[её]нн\w*|подготовленн\w*|сконвертированн\w*)\s+'
        r'(?:(?:тобой|вами)\s+)?(?:файл|документ|заметк|копи)\w*\b|'
        r'\b(?:файл|документ|заметк|копи)\w*\s*,?\s+котор\w*\s+(?:ты|вы)\s+'
        r'(?:создал\w*|сохранил\w*|подготовил\w*)\b',command_text(text),re.I))


def asks_last_opened_file(text: str) -> bool:
    return bool(re.fullmatch(r'(?:а\s+)?(?:(?:ты|вы)\s+(?:что|ч[её])|'
        r'(?:что|ч[её]|какой\s+(?:файл|документ))\s+(?:ты|вы))\s+'
        r'(?:(?:вообще|именно|сейчас|только\s+что)\s+)?открыл[аи]?\s*[?!.]*',command_text(text).strip(),re.I))


def display_file_target(text: str) -> str:
    """Return a literal file description; folder requests and pronouns stay separate."""
    commands = command_text(text)
    match = _DISPLAY.search(text)
    if match and not re.search(r'\b(?:прочитай|прочти|read|выведи|покажи|show)\b', commands):
        match = None  # A quoted instruction is data.
    match = match or _INSIDE.search(text)
    if not match:
        return ''
    target = _literal_target(match,text)
    literal = bool(re.search(r'\.\w{1,10}$',target))
    if (not target or _DIRECTORY_TARGET.search(target) or (not literal and (
            _PRONOUN_TARGET.search(target) or references_created_file(target)))
            or re.match(r'^(?:о|об|про|about)\b', target, re.I)):
        return ''
    return target


def open_file_target(text: str) -> str:
    if not requests_file_open(text):
        return ''
    match = re.search(r'\b(?:открой(?:те)?|открывай(?:те)?|открыть|open)\s+(?:файл\w*\s+)?(.+?)\s*[?!]*$', text, re.I)
    if not match:
        return ''
    target = _literal_target(match,text)
    literal = bool(re.search(r'\.\w{1,10}$',target))
    if _DIRECTORY_TARGET.search(target) or (not literal and (
            _PRONOUN_TARGET.search(target) or references_created_file(target))):
        return ''
    return target


def is_contents_question(text: str) -> bool:
    return bool(_INSIDE.search(text))


def references_active_file(text: str) -> bool:
    return bool(re.search(r'\b(?:эт(?:от|ого|ом|ой|у|а|о)|тот|данн\w*)\s+(?:файл|заметк|документ)\w*\b|'
                          r'\b(?:его|е[её])\s+(?:содержим|текст)\w*\b|'
                          r'\b(?:открой(?:те)?|прочитай(?:те)?|прочти(?:те)?|покажи(?:те)?|open|read|show)\s+'
                          r'(?:его|е[её]|it)\b', command_text(text))) or references_created_file(text)


def requests_active_file_read(text: str) -> bool:
    commands = command_text(text)
    return references_active_file(text) and bool(re.search(
        r'(?<!не )\b(?:прочитай|прочти|покажи|read|show)\b',commands))


def permits_workspace_operation(operation: str, original: str) -> bool:
    """A model may choose how to execute an action, but cannot invent the action."""
    text = command_text(original)
    if operation == 'open_file':
        return requests_file_open(original)
    if operation == 'list_file_dates':
        from workspace_planner import WorkspaceOperation, WorkspacePlanner
        plan = WorkspacePlanner().plan(original)
        return bool(plan and any(step.operation == WorkspaceOperation.LIST_FILE_DATES for step in plan.steps))
    if operation == 'find_latest':
        return bool(re.search(r'\b(?:последн\w*|актуальн\w*|свеж\w*|latest|newest)\b', text))
    if operation == 'compare_versions':
        return bool(re.search(r'\b(?:сравн\w*|сопостав\w*|разниц\w*|изменил\w*|compare|difference)\b', text)
                    and not negates_comparison(original))
    if operation == 'list_directory':
        return bool(re.search(r'\b(?:папк|директор|workspace|воркспейс|files|folders|directory)\w*\b|'
                              r'\b(?:список|перечисли|какие)\b.*\bфайл\w*', text))
    return True
