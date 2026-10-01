"""Small dependency-free lexical normalization for local Russian search."""
from __future__ import annotations

import re
import unicodedata

_CYRILLIC_RE = re.compile(r"^[а-яё]+$", re.IGNORECASE)
_RU_ENDINGS = tuple(sorted({
    "иями", "ями", "ами", "ого", "его", "ому", "ему", "ыми", "ими",
    "ией", "ией", "ией", "ью", "ию", "ия", "ья", "ье", "ей",
    "ов", "ев", "ах", "ях", "ам", "ям", "ом", "ем",
    "а", "я", "у", "ю", "ы", "и", "е", "о",
}, key=len, reverse=True))


def normalize_term(term: str) -> str:
    return unicodedata.normalize("NFC", term).casefold().replace("ё", "е")


def lexical_key(term: str) -> str:
    """Return a conservative case-normalized key, chiefly for proper names.

    This is not a full morphological analyser.  It removes common Russian case
    endings while keeping at least four characters, which is sufficient for
    exact local-name retrieval without changing Latin identifiers or formulas.
    """
    normalized = normalize_term(term)
    if len(normalized) < 4 or not _CYRILLIC_RE.fullmatch(normalized):
        return normalized
    for ending in _RU_ENDINGS:
        if normalized.endswith(ending) and len(normalized) - len(ending) >= 3:
            return normalized[:-len(ending)]
    return normalized
