"""Conservative device-action intent from the user's own words."""
import re


def command_text(text: str) -> str:
    return re.sub(r'''«[^»]*»|"[^"\n]*"|`[^`]*`|“[^”]*”|(?<!\w)'[^'\n]*'(?!\w)''', ' ', text).casefold()


def requests_unsupported_file_change(text: str) -> bool:
    if not re.search(r'\b(?:файл|документ|заметк|папк|директор|files?|documents?|notes?|folders?|director(?:y|ies))\w*\b|'
                     r'\b[^\s]+\.(?:md|txt|pdf|docx|json|ya?ml|py|js|ts|csv)\b', text, re.IGNORECASE):
        return False
    text = command_text(text)
    for match in re.finditer(r"\b(?:удали(?:те)?|удаляй(?:те)?|переименуй(?:те)?|перемести(?:те)?|"
                             r"delete|remove|rename|move)\b", text):
        if not re.search(r"\b(?:не|not|don't|never)\s*$", text[:match.start()]):
            return True
    return False


def requests_file_open(text: str) -> bool:
    # Quoted commands are discussed text, not authorization. Paths inside
    # quotes may still be used by the resolver after authorization is checked.
    text = command_text(text)
    if re.search(r"\b(?:не\s+(?:\w+\s+){0,3}откр\w*|без\s+откры\w*|"
                 r"(?:do\s+not|don't|never)\s+(?:\w+\s+){0,3}open|without\s+opening)\b", text):
        return False
    if re.search(r"\b(?:как\s+(?:мне\s+)?откры\w*|how\s+(?:do\s+i|to)\s+open)\b", text):
        return False
    return bool(re.search(r"\b(?:открой(?:те)?|открывай(?:те)?|open)\b|"
                          r"\b(?:можешь|можете|прошу|хочу|нужно)\s+открыть\b|"
                          r"\bя\s+(?:тебя|вас)\s+(?:уже\s+)?(?:просил|попросил)[аи]?\s+открыть\b|"
                          r"\bя\s+(?:уже\s+)?(?:просил|попросил)[аи]?\s+(?:тебя|вас)\s+открыть\b", text))


def negates_comparison(text: str) -> bool:
    return bool(re.search(r"\b(?:не\s+(?:\w+\s+){0,3}(?:сравн|сопостав)\w*|"
                          r"(?:do\s+not|don't|never)\s+(?:\w+\s+){0,3}compare)\b", command_text(text)))
