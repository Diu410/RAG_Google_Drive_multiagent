"""Conservative entity-name normalization used as a SQLite identity key."""
import re
import unicodedata


def normalize_entity_name(value: str) -> str:
    # NFC preserves meaningful superscript/subscript and compatibility symbols.
    value = unicodedata.normalize("NFC", value or "").strip().casefold()
    value = re.sub(r"[\u2010-\u2015]", "-", value)
    value = re.sub(r"\s*([.,;:!?()\[\]{}\-])\s*", r"\1", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip(" \t\r\n.,;:!?")
