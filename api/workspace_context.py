"""Question-directed literal passages within an explicitly selected file."""
import math
import re
from collections import Counter
from pathlib import Path

from text_normalization import lexical_key
from source_grounding import factual_source_text

_WORDS = re.compile(r'[\w-]+', re.UNICODE)
_STOP = set('''что кто где как какой какая какие какого сколько когда знаешь скажи
    расскажи про обо это этот эта есть мне можешь файл файле файла документ
    документе указан указана указано указаны прочитай покажи пожалуйста
    the what who where when which about tell does know file document read'''.split())
_STOP = {lexical_key(word) for word in _STOP}
_OVERVIEW = re.compile(r'\b(?:обзор\w*|перескаж\w*|кратк\w*\s+содержани\w*|'
                       r'суммари\w*|summary|summari[sz]\w*|overview)\b', re.I)


def _terms(text):
    return {lexical_key(word) for word in _WORDS.findall(text) if len(word) >= 3} - _STOP


def _safe_ranges(content, spans):
    # Check whole source lines before clipping: a window may start after an
    # instruction's prefix, which would otherwise hide it from the usual guard.
    blocked = [(match.start(), match.end()) for match in re.finditer(r'[^\n]*(?:\n|$)', content)
               if match[0] and not factual_source_text(match[0])]
    output = []
    for start, end in spans:
        for left, right in blocked:
            if right <= start:
                continue
            if left >= end:
                break
            if start < left:
                output.append((start, left))
            start = max(start, right)
        if start < end or not content:
            output.append((start, end))
    return output


def file_passages(question, path, content, limit, *, broad=False, score=20.0,
                  found_by='workspace_file_reference', preferred_text=''):
    """Keep source offsets and literal text; never join distant passages as a quote."""
    limit = max(1, limit)
    length = len(content)
    if length <= limit:
        spans, selection = [(0, length)], 'complete'
    else:
        slots = min(6 if broad or _OVERVIEW.search(question) else 3, max(1, limit // 240))
        width = max(1, limit // slots)
        width = min(width, 2400)
        # Overlap supplies context for a fact sitting at a window boundary.
        step = max(1, width - min(240, width // 4))
        starts = list(range(0, max(1, length - width + 1), step))
        if starts[-1] != length - width:
            starts.append(length - width)
        windows = [(start, min(length, start + width)) for start in starts]
        wanted = _terms(question) - _terms(Path(path).name)
        terms = [_terms(content[start:end]) & wanted for start, end in windows]
        frequency = Counter(term for group in terms for term in group)
        weights = {term: 1 + math.log((len(windows) + 1) / count)
                   for term, count in frequency.items()}
        ranked = sorted(range(len(windows)),
                        key=lambda i: (-sum(weights[t] for t in terms[i]), i))
        chosen = []
        anchor = content.find(preferred_text) if preferred_text else -1
        if anchor >= 0 and not (broad or _OVERVIEW.search(question)):
            # A semantic match may use none of the question's words. Preserve
            # its literal position instead of falling back to the file's start.
            start = max(0, min(anchor - min(120, width // 4), length - width))
            chosen.append((start, start + width))
        if wanted and frequency and not (broad or _OVERVIEW.search(question)):
            for index in ranked:
                if len(chosen) >= slots:
                    break
                if not terms[index]:
                    break
                start, end = windows[index]
                # Adjacent overlapping windows should not crowd out other facts.
                if all(min(end, b) <= max(start, a) for a, b in chosen):
                    chosen.append((start, end))
                if len(chosen) >= slots:
                    break
            selection = 'hybrid_passages' if anchor >= 0 else 'question_passages'
        else:
            selection = 'semantic_passages' if chosen else 'distributed_sample'
        if not chosen:
            # Unknown terminology still gets a distributed sample, including the end.
            chosen = [windows[round(i * (len(windows) - 1) / max(1, slots - 1))]
                      for i in range(slots)]
        spans = sorted(set(chosen))
        # Distributed windows may overlap in a file only slightly over its budget.
        merged = []
        for start, end in spans:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        spans = merged
    spans = _safe_ranges(content, spans)
    selected = sum(end - start for start, end in spans)
    coverage = {'available': 1, 'inspected': 1, 'omitted': 0, 'failed': 0,
                'matched_files': 1, 'selected': len(spans), 'source_characters': length,
                'selected_characters': selected, 'content_truncated': selected < length,
                'selection': selection, 'source_ranges': [list(span) for span in spans]}
    return [{'path': path, 'content': content[start:end], 'score': score,
              'matched': [], 'found_by': [found_by], 'coverage': coverage,
              'source_range': [start, end],
             'chunk_id': f'workspace:{path}:{start}:{end}' if selected < length else f'workspace:{path}'}
            for start, end in spans]


def file_coverage_notice(coverage):
    if coverage.get('content_truncated'):
        return '\n\nОтвет основан на выбранных фрагментах файла; охват текста ограничен.'
    return ''
