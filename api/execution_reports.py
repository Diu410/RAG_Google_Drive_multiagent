"""Explain our own completed work from its journal, never from document contents."""
import re
from assistant_models import ToolResult
from command_intent import command_text


def summary_feedback(question):
    text = command_text(question).strip()
    # Explicit document attribution must not be mistaken for our own operation.
    if re.search(r'\b(?:в\s+(?:тексте|документе|письме)|автор|персонаж|он|она)\b',text):
        return None
    if re.search(r'\b(?:продолжи|доработай|заверши|дополни|повтори)\b',text) and re.search(
            r'\b(?:пересказ|обзор|пропущенн\w*|необработанн\w*|неполн\w*)\b',text):
        return 'continue'
    if (re.search(r'\b(?:почему|зачем|что|какие|где|сколько|чем)\b',text)
            and re.search(r'\b(?:пропустил\w*|пропущен\w*|обработал\w*|не\s+обработан\w*|'
                          r'остановил\w*|не\s+завершил\w*|не\s+пересказал\w*|не\s+прошл\w*\s+сверк\w*)\b',text)):
        return 'describe'
    return None


def file_reason(row):
    if row.get('complete'):
        return 'complete'
    parts = row.get('parts', [])
    if any(p.get('error') == 'GenerationDeadlineExceeded' for p in parts):
        return 'deadline'
    if any(p.get('error') for p in parts):
        return 'technical_failure'
    if any(p.get('status') == 'UNSUPPORTED' for p in parts):
        return 'review_incomplete'
    return 'not_completed'


REASONS = {'deadline': 'закончился бюджет времени',
    'technical_failure': 'технический сбой обработки или сверки',
    'review_incomplete': 'не все предложения прошли сверку; это не означает, что сам файл неверен',
    'not_completed': 'обработка не завершена', 'unsupported_format': 'формат не поддерживается',
    'source_budget': 'достигнут лимит объёма источников'}


def summary_status(report):
    files = report.get('files', [])
    lines = [f"В обработку включено документов: {report.get('total_files', len(files))}. "
             f"Полностью проверенный пересказ получен для {report.get('covered_files', 0)}."]
    gaps = [(row['path'], file_reason(row)) for row in files if not row.get('complete')]
    gaps += [(row['path'], row.get('reason', 'not_completed')) for row in report.get('omitted', [])]
    if gaps:
        lines.append('Причины неполного результата по журналу:')
        lines.extend(f"- `{path}` — {REASONS.get(reason, REASONS['technical_failure'])}." for path,reason in gaps)
        lines.append('Проверенные части этих документов могли попасть в ответ. «Неполностью» не означает «не читался».')
    else:
        lines.append('В журнале нет пропущенных документов.')
    if report.get('scan_truncated') or report.get('scan_errors'):
        lines.append('Обход был ограничен; полный состав источников не установлен.')
    return '\n\n'.join(lines[:2])+ ('\n'+'\n'.join(lines[2:]) if len(lines)>2 else '')


def status_result(text):
    return ToolResult(success=True,tool='workspace',summary='Execution journal report',data={
        'final':text,'context':'','results':[],'strategy':'WORKSPACE_ACTION_STATUS'})
