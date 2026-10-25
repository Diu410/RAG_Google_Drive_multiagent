"""Rank fusion without comparing lexical counts with embedding distances."""
from __future__ import annotations


def fuse_files(lexical: list[dict], semantic: list[dict], limit: int = 3) -> list[dict]:
    candidates = {}
    for channel, rows in (('lexical', lexical), ('semantic', semantic)):
        seen = set()
        for row in rows:
            path = row['path']
            if path in seen:
                continue
            seen.add(path)
            rank = len(seen)
            candidate = candidates.setdefault(path, {**row, 'score': 0.0, 'found_by': [], '_ranks': {}})
            candidate['score'] += 1 / (60 + rank)
            candidate['_ranks'][channel] = rank
            methods = row.get('found_by') or [f'workspace_{channel}']
            candidate['found_by'] = list(dict.fromkeys(candidate['found_by'] + methods))
            if channel == 'semantic':
                candidate['preferred_text'] = row['content']
    ranked = sorted(candidates.values(), key=lambda item: (
        -item['score'], min(item['_ranks'].values()), item['_ranks'].get('semantic', float('inf')), item['path']))
    return [{key: value for key, value in item.items() if key != '_ranks'} for item in ranked[:limit]]
