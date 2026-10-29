"""Narrow rejection of explicit future-to-present changes, never entailment.

Dictionary morphology is used only for unambiguous Russian finite verbs and
literal noun anchors. Ambiguity and unsupported constructions stay with the
semantic auditor; a matching tense cannot certify an assertion.
"""
from functools import lru_cache
from difflib import SequenceMatcher
import re


@lru_cache(maxsize=1)
def _analyzer():
    from pymorphy3 import MorphAnalyzer
    return MorphAnalyzer(lang='ru')


@lru_cache(maxsize=8192)
def _word(word):
    parses = _analyzer().parse(word.casefold())
    known = [p for p in parses if p.is_known]
    if not known:
        return None, None, None
    poses = {str(p.tag.POS) for p in known}
    lemmas = {p.normal_form for p in known}
    if len(poses) != 1 or len(lemmas) != 1:
        return None, None, None
    pos, lemma = next(iter(poses)), next(iter(lemmas))
    tenses = {str(p.tag.tense) for p in known}
    tense = next(iter(tenses)) if pos == 'VERB' and len(tenses) == 1 else None
    return pos, tense, lemma


def _records(text):
    tokens = re.findall(r'[а-яё]+|[.!?;,\n…]', text.casefold())
    records = []
    for index, word in enumerate(tokens):
        pos, tense, lemma = _word(word)
        if pos != 'VERB' or tense not in {'pres', 'futr'}:
            continue
        objects = set()
        for following in tokens[index+1:index+13]:
            if following in '.!?;,\n…':
                break
            next_pos, _, next_lemma = _word(following)
            if next_pos == 'VERB':
                break
            if next_pos == 'INFN' and tense == 'futr' and lemma == 'быть':
                lemma = next_lemma  # Explicit auxiliary future, not a new fact.
            if next_pos == 'NOUN':
                objects.add(next_lemma)
        if objects:
            records.append({'word':word,'lemma':lemma,'tense':tense,'objects':objects})
    return records


def _same_form_family(left, right):
    if left == right:
        return True
    # Recognise only close regular aspect forms. This is NOT a thesaurus or a
    # proof of equivalence: a shared literal object and no present counterpart
    # are also required before the guard may reject the temporal projection.
    return (left[:3] == right[:3] and abs(len(left)-len(right)) <= 2
            and SequenceMatcher(None,left,right).ratio() >= .85)


def future_as_present(claim, quote):
    """Return the explicit conflicting forms, or None (not 'supported')."""
    sources = _records(quote)
    future = [row for row in sources if row['tense'] == 'futr']
    if not future:
        return None
    for target in _records(claim):
        if target['tense'] != 'pres':
            continue
        matches = [row for row in future if _same_form_family(target['lemma'],row['lemma'])
                   and target['objects'] & row['objects']]
        if not matches:
            continue
        anchors = set().union(*(target['objects'] & row['objects'] for row in matches))
        if any(row['tense']=='pres' and _same_form_family(target['lemma'],row['lemma'])
               and anchors & row['objects'] for row in sources):
            continue
        return {'claim_verb':target['word'],'source_verbs':sorted({row['word'] for row in matches}),
                'noun_anchors':sorted(anchors)}
    return None
