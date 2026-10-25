"""Source-selection intent and exclusion of instructions addressed to an assistant.

Documents stay unchanged. These helpers only prepare factual retrieval/evidence;
explicit file reads continue to return the original text.
"""
import re
from command_intent import command_text


_LIBRARY_REQUEST = re.compile(
    r'\b(?:импортирован\w*\s+библиотек\w*|библиотек\w*\s+(?:документ\w*|источник\w*)|'
    r'(?:загруженн|импортирован)\w*\s+документ\w*|document\s+library|imported\s+(?:library|documents)|'
    r'document\s+(?:rag|graphrag))\b', re.IGNORECASE,
)
_NEGATED_LIBRARY = re.compile(
    r'\b(?:не|not|without|без)\s+(?:\w+\s+){0,3}$', re.IGNORECASE,
)
_SOURCE_POLICY = re.compile(
    r'\bне\s+(?:следуй|выполняй|исполняй)\s+(?:инструкци|команд)\w*'
    r'(?:\s+из\s+(?:источник|документ|файл)\w*)?\s*[.!]?|'
    r"\b(?:do\s+not|don't)\s+follow\s+(?:instructions|commands)\s+from\s+(?:sources|documents)\s*[.!]?",
    re.IGNORECASE,
)
_OVERRIDE = re.compile(
    r'\b(?:игнорируй|забудь|отмени)\s+(?:все\s+)?(?:предыдущ\w*|системн\w*|пользоват\w*|инструкци\w*)|'
    r'\b(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|prior|system|user|instructions)\b|'
    r'<\|(?:im_start|start_header_id)\|>\s*system|\[INST\]|'
    r'\b(?:system\s+prompt|системн\w*\s+промпт)\s*:',
    re.IGNORECASE,
)


def requests_document_library(question: str) -> bool:
    text = command_text(question)
    return any(not _NEGATED_LIBRARY.search(text[max(0, match.start() - 60):match.start()])
               for match in _LIBRARY_REQUEST.finditer(text))


def factual_query(question: str) -> str:
    return _SOURCE_POLICY.sub(' ', question).strip() or question


def factual_source_text(content: str) -> str:
    """Exclude entire instruction-bearing lines, including quoted false facts."""
    return ''.join(line for line in content.splitlines(keepends=True) if not _OVERRIDE.search(line))


def missing_information_reply(question: str) -> str:
    if re.search(r'\b(?:смысл|означает|значени\w*|meaning|means)\b', question, re.IGNORECASE):
        return 'В найденных документах точный смысл не раскрыт. Недостаточно информации для надёжного объяснения.'
    return 'В найденных документах недостаточно информации для ответа на этот вопрос.'


def requests_source_overview(question: str) -> bool:
    """Distinguish an introduction/overview from a request for one missing fact."""
    text = command_text(factual_query(question))
    return bool(re.search(
        r'\b(?:что\s+(?:(?:ты|вы)\s+)?(?:знаешь|знаете|известно|можешь\s+рассказать|можете\s+рассказать)|'
        r'что\s+(?:такое|за)|расскажи(?:те)?\s+(?:мне\s+)?(?:о|об|про)|кто\s+так(?:ой|ая|ие)|'
        r'(?:сделай|дай|составь)\s+(?:краткий\s+|подробный\s+)?обзор|'
        r'(?:what\s+do\s+you\s+know\s+about|tell\s+me\s+about|who\s+is))\b', text, re.I))


def overview_subject(question: str) -> str:
    """An explicitly named subject; this is not a model inferred entity."""
    match = re.search(
        r'\b(?:кто\s+так(?:ой|ая|ие)|что\s+(?:такое|за)|'
        r'что\s+(?:(?:ты|вы)\s+)?(?:знаешь|знаете|известно)\s+(?:о|об|про)|'
        r'расскажи(?:те)?\s+(?:мне\s+)?(?:о|об|про)|who\s+is|'
        r'what\s+is|tell\s+me\s+about|what\s+do\s+you\s+know\s+about)\s+([^?!.\n]+)',
        factual_query(question), re.I)
    if not match:
        return ''
    topic = re.split(r'\b(?:из\s+(?:файлов|документов)|в\s+(?:файлах|документах)|'
                     r'по\s+(?:файлам|документам)|from\s+(?:files|documents))\b', match[1], flags=re.I)[0]
    return topic.strip(' \t\r\n«»"\'`')


def overview_topic_terms(question: str) -> set[str]:
    """Explicit overview subjects only; unresolved pronouns need contextualization."""
    topic = overview_subject(question)
    stop = {'этот', 'эта', 'это', 'этого', 'такой', 'такая', 'вообще', 'подробно',
            'кратко', 'пожалуйста', 'него', 'ней', 'них', 'this', 'that', 'please', 'about'}
    return {word[:5] for word in re.findall(r'[а-яёa-z]{4,}', topic.casefold().replace('ё', 'е'))
            if word not in stop}


def focused_source_text(question: str, text: str) -> str:
    """Quote topic passages with their paragraph context, not an inferred summary.

    This is a generation view for short introductions. Full sources remain in
    the response and verifier. General advice from other paragraphs must not
    become the definition of a named organization merely by file proximity.
    """
    if re.search(r'\b(?:подроб\w*|полн\w*|деталь\w*|анализ\w*|detailed|comprehensive)\b', question, re.I):
        return text
    terms = overview_topic_terms(question)
    if not terms:
        return text
    paragraphs = re.split(r'\n\s*\n|\n', text)
    selected = []
    for paragraph in paragraphs:
        folded = paragraph.casefold().replace('ё', 'е')
        if any(term in folded for term in terms):
            # A short first-person passage may identify its speaker only at the
            # end. Keep the description with that identity instead of reducing
            # a character introduction to the final title and adjacent sentence.
            named_speaker = any(all(term in designation.casefold().replace('ё', 'е') for term in terms)
                                for designation in explicit_self_designations(paragraph))
            if len(paragraph) <= 900 or (named_speaker and len(paragraph) <= 2400):
                selected.append(paragraph)
                continue
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            indexes = set()
            for i, sentence in enumerate(sentences):
                if any(term in sentence.casefold().replace('ё', 'е') for term in terms):
                    indexes.update(range(max(0, i - 1), min(len(sentences), i + 2)))
            selected.append(' '.join(sentences[i] for i in sorted(indexes)))
    return '\n[…]\n'.join(selected) if selected else ''


def without_suggested_self_examples(text: str) -> str:
    """Authoring view only: suggested affirmations cannot supply a biography.

    The original source remains available to the reviewer. Explicit requests
    to describe examples bypass this view at the caller.
    """
    suggestion=re.compile(r'\b(?:проговари\w*|повторя\w*|самовнушен\w*|представь\w*)\b',re.I)
    person=re.compile(r'\b(?:я|меня|мне|мой|моя|мо[её]|I|my)\b',re.I)
    return '\n'.join(line for line in text.splitlines() if not (suggestion.search(line) and person.search(line)))


def explicit_self_designations(text: str) -> list[str]:
    """Literal positive self-identification outside a nearby example/negation."""
    pattern = re.compile(r'\b(?:это\s+(?:всё\s+)?я|Я)\s*[—-]\s*([^.!?\n]{12,180})', re.I)
    excluded = re.compile(r'\b(?:например|проговари\w*|повторя\w*|представ\w*|допустим|'
        r'for example|imagine|если бы|условн\w*\s+пример|не\s+(?:говорил|сказал|называл|утверждал))\b', re.I)
    text = factual_source_text(text)
    result = []
    for match in pattern.finditer(text):
        if excluded.search(text[max(0, match.start() - 200):match.start()]):
            continue
        designation = re.split(r'\s+и\s+(?:мне|я|мой|моя|у меня)\b', match[1], flags=re.I)[0].strip()
        if not re.match(r'(?:не|not)\b', designation, re.I):
            result.append(designation)
    return result


def documentary_overview(question: str, sources: list[dict]) -> str | None:
    """Recover explicit self-designation, never a fictional biography or doctrine.

    Used only after a failed generative answer. Eligibility comes from a literal
    self-identification outside an example and the named question subject in
    that designation. The resulting prose still undergoes normal verification.
    """
    topic = overview_subject(question)
    terms = overview_topic_terms(question)
    if not terms or len(topic) > 80 or re.search(r'\b(?:и|and)\b', topic, re.I):
        return None
    candidates = []
    for source in sources:
        text = factual_source_text(source['text'])
        for designation in explicit_self_designations(text):
            if not all(term in designation.casefold().replace('ё', 'е') for term in terms):
                continue
            candidates.append((source, designation))
    # Competing self-designations need a real answer, not an arbitrary winner.
    if not candidates or len({d.casefold() for _, d in candidates}) != 1:
        return None
    source, designation = candidates[0]
    named = topic[0].upper() + topic[1:]
    who = bool(re.search(r'\b(?:кто\s+так(?:ой|ая|ие)|who\s+is)\b', question, re.I))
    # A concept inside the title is not the named speaker. Only an exact name
    # in the final name span permits a personal self-description; otherwise
    # report the literal mention without identifying the topic with the voice.
    name_span = ' '.join(re.findall(r'[а-яёa-z]+', designation, re.I)[-2:]).casefold()
    named_speaker = bool(re.search(r'(?<!\w)' + re.escape(topic.casefold()) + r'(?!\w)', name_span))
    if who and named_speaker:
        answer = f'В тексте {named} называет себя «{designation}».'
    else:
        answer = f'{named} упоминается в самоназвании говорящего в тексте — «{designation}».'
        # A further direct description may accompany the designation. Only
        # literal non-first-person topic statements, with no inferred rules.
        for sentence in re.split(r'(?<=[.!?])\s+', factual_source_text(source['text'])):
            sentence = sentence.strip()
            folded = sentence.casefold().replace('ё', 'е')
            head = re.findall(r'[а-яёa-z]{4,}', topic.casefold().replace('ё', 'е'))[0][:5]
            if (12 <= len(sentence) <= 240 and folded.startswith(head)
                    and sentence.casefold() not in designation.casefold()
                    and not re.search(r'\b(?:я|например|меня|мне|мы)\b', sentence, re.I)):
                answer += f' В этом же тексте сказано: «{sentence.removesuffix(".")}».'
                break
    path = source.get('metadata', {}).get('source', source['chunk_id'])
    return answer + f'\n\nИсточники: `{path}`.'


def is_missing_information_claim(text: str) -> bool:
    """Normalize explicit abstention even when the verifier chose the wrong basis."""
    return bool(re.search(
        r'\bне\s+(?:указан|раскрыт|раскрыва|назван|называ|предоставлен|найден|сообщ[её]н)\w*\b|'
        r'\b(?:нет|недостаточно)\s+(?:данных|сведений|информации)\b|'
        r'\b(?:источник|документ|файл|текст)\w*[^.!?\n]{0,100}\b'
        r'не\s+(?:содерж\w*|привед\w*|упомина\w*)\s+(?:данны|данных|сведени|сведений|информаци|упоминани)\w*\b|'
        r'\b(?:информаци|сведени|данны)\w*[^.!?\n]{0,100}\bотсутств\w*\b|'
        r'\bотсутств\w*\s+(?:в\s+)?(?:предоставленн\w*\s+|найденн\w*\s+)?(?:источник|документ|файл|текст)\w*\b|'
        r'\b(?:not\s+(?:stated|specified|provided|disclosed|found)|insufficient\s+information|'
        r'no\s+(?:information|data|evidence)\s+(?:about|on|for))\b', text, re.I))
