"""Corrections to our own effects must not become questions about source facts."""
import re
import asyncio
from pathlib import PurePosixPath
from assistant_models import AssistantError, ToolResult
from command_intent import command_text
from workspace_intent import open_file_target


def action_feedback(question):
    text = command_text(question).strip()
    if re.match(r'^(?:почему|зачем|как|в\s+(?:тексте|документе|заметке)|по\s+документам)\b',text):return None
    if re.search(r'\b(?:не\s+то(?:т|т?\s+файл)?\s+(?:ты\s+)?открыл\w*|открыл\w*\s+не\s+то\w*)\b', text):
        return 'wrong_open'
    if (re.fullmatch(r'(?:слишком\s+(?:коротко|кратко)|огрызок|обрывок)[.!? ]*',text)
        or re.search(r'\b(?:это|ответ|результат|пересказ|документ|получилось)\b', text)
            and re.search(r'\b(?:огрызок|слишком\s+(?:коротк\w*|кратк\w*)|не\s+пересказ|'
                          r'неполны\w*|обрывок|обрезан\w*)\b', text)):
        return 'incomplete_document'
    return None


async def correct_open(prior, workspace_id, repository, filesystem, execution):
    """Retry the recorded literal target after a correction, without guessing."""
    if not prior:return None
    task=await asyncio.to_thread(repository.get_task,prior['task_id'])
    target=open_file_target(task.user_input) if task else ''
    if not target or not re.search(r'(?:\.|\s)(?:md|markdown|txt|pdf|docx|html?|csv|tsv|json)$',target,re.I):
        return None
    normalized=re.sub(r'\s+(md|markdown|txt|pdf|docx|html?|csv|tsv|json)$',r'.\1',target,flags=re.I).casefold()
    if normalized in {prior['path'].casefold(),PurePosixPath(prior['path']).name.casefold()}:return None
    try:
        async with execution.action('RESOLVE_CORRECTED_FILE',{'question':task.user_input}) as action:
            path=await asyncio.to_thread(filesystem.find_text_file,workspace_id,task.user_input,None,'')
            action.output={'path':path}
        if path==prior['path']:return None
        async with execution.action('OPEN_FILE',{'workspace_id':workspace_id,'path':path}) as action:
            opened=await asyncio.to_thread(filesystem.open_file,workspace_id,path)
            action.output={'path':opened}
        return ToolResult(success=True,tool='workspace',summary='Corrected file opened',data={
            'final':f"Ранее был открыт `{prior['path']}`. Теперь открыт запрошенный файл `{opened}`.",
            'context':'','results':[],'strategy':'WORKSPACE_OPEN','active_file':opened})
    except (AssistantError,OSError,ValueError):return None
