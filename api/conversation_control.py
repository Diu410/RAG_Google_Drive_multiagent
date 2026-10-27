"""Bounded dialogue announcements; never file requests or rename authority."""
import re


_SESSION_ANNOUNCEMENT = re.compile(
    r'\s*(?:это|начинаем)\s+(?:(?:наш|наше)\s+)?(?:нов(?:ый|ое)\s+)?'
    r'(?P<label>(?:чат|диалог|тест|испытание)\s+'
    r'(?:(?:[nн№]\s*)?\d+(?:[-.]\d+)*|«[^«»\r\n]{1,80}»|"[^"\r\n]{1,80}"))\s*[.!]*\s*', re.I)


def session_announcement_reply(question):
    """Acknowledge a complete label announcement, not a question about that label."""
    match = _SESSION_ANNOUNCEMENT.fullmatch(question)
    return f'Понял, это {match["label"]}.' if match else None
