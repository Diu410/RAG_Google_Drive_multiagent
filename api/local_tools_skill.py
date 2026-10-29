"""Deterministic user-authorized document and repository operations in chat."""
from __future__ import annotations
import asyncio
import json
from pathlib import Path
import re

from assistant_models import AssistantError, ToolResult
from command_intent import command_text
from file_policy import FilePolicyError
from local_tool_intent import (permitted_local_operations, resolve_local_intent, requested_format,
                              document_content_question, directory_summary_request, directory_summary_chat_request,
                              repeated_user_request, summary_directory_reference)


_QUOTED = r'(?:«([^»]*)»|"([^"]*)"|`([^`]*)`)'
_PATH = r'(.+?\.(?:txt|md|markdown|csv|tsv|pdf|docx|html?|rs|py|ts|tsx|js|go|java))'


def _unquote(value):
    return value.strip().strip('«»"`')


async def _file_call(function, *args, **kwargs):
    try:
        return await asyncio.to_thread(function, *args, **kwargs)
    except AssistantError:
        raise
    except FilePolicyError as exc:
        raise AssistantError(str(exc), 403) from exc
    except FileNotFoundError as exc:
        raise AssistantError('Файл или папка не найдены. Укажите существующий относительный путь.', 404) from exc
    except ValueError as exc:
        raise AssistantError(str(exc), 422) from exc
    except LookupError as exc:
        raise AssistantError('Рабочая папка не найдена.', 404) from exc
    except OSError as exc:
        raise AssistantError('Не удалось выполнить файловую операцию. Проверьте доступ к файлу.', 400) from exc


def parse_local_command(question):
    """Only complete, explicit commands become tool calls; never source prose."""
    question = re.sub(r'^(?:предпросмотр|предварительно)\s*:?\s*','',question.strip(),flags=re.I)
    visible = command_text(question).strip()
    if re.match(r'^(?:не\b|как\b|объясни\b|do not\b|how\b)', visible):
        return None
    if re.fullmatch(r'(?:/repo|обзор\s+репозитория|покажи\s+структуру\s+(?:репозитория|проекта))[.!? ]*', visible):
        return {'operation': 'repository_overview'}
    match = re.fullmatch(r'(?:прочитай|покажи)\s+строки\s+(\d+)\s*[-–—]\s*(\d+)\s+(?:файла|в\s+файле)\s+(.+?)\s*', question.strip(), re.I)
    if match and re.match(r'^(?:прочитай|покажи)\b', visible):
        return {'operation': 'read_lines', 'path': _unquote(match[3]), 'start': int(match[1]), 'end': int(match[2])}
    match = re.fullmatch(r'(?:найди\s+текст|ищи\s+текст|поиск\s+текста|'
        r'где\s+(?:встречается|используется)\s+(?:фраза|текст|строка))\s+' + _QUOTED + r'\s*[.!?]?', question, re.I | re.S)
    if match:
        return {'operation': 'search_text', 'query': next(value for value in match.groups() if value is not None)}
    match = re.fullmatch(r'(?:конвертируй|преобразуй|convert)\s+(?:файл\s+)?' + _PATH +
        r'[»"`]?\s+(?:в|to)\s+(.+?)\s*[.!]?\s*', question.strip(), re.I | re.S)
    if match and re.match(r'^(?:конвертируй|преобразуй|convert)\b', visible):
        source, target = map(_unquote, match.groups())
        if re.fullmatch(r'(?i)\.?txt|\.?md|\.?markdown|\.?html|\.?docx|\.?pdf|\.?csv|\.?tsv', target):
            target = str(Path(source).with_suffix('.' + target.lstrip('.').lower())).replace('\\', '/')
        return {'operation': 'convert_document', 'path': source, 'target_path': target}
    match = re.fullmatch(r'создай\s+(?:файл|документ)\s+' + _PATH + r'[»"`]?\s+с\s+текстом\s+' + _QUOTED +
                         r'\s*[.!]?\s*', question.strip(), re.I | re.S)
    if match and visible.startswith('создай'):
        return {'operation': 'create_document', 'path': _unquote(match[1]),
                'text': next(value for value in match.groups()[1:] if value is not None)}
    match = re.fullmatch(r'замени\s+в\s+(?:файле|документе)\s+' + _PATH + r'[»"`]?\s+(?:текст\s+)?' +
        _QUOTED + r'\s+на\s+' + _QUOTED + r'\s*[.!]?\s*', question.strip(), re.I | re.S)
    if match and visible.startswith('замени'):
        return {'operation': 'edit_document', 'path': _unquote(match[1]),
            'old': next(value for value in match.groups()[1:4] if value is not None),
            'new': next(value for value in match.groups()[4:7] if value is not None)}
    return None


class LocalToolsSkill:
    name = 'local_tools'
    description = ('Act on ordinary user goals in the active folder: create a document about a topic '
        'by retrieving and checking sources, save supplied text or the previous answer, convert a '
        'named/active document, replace exact text, search literal text, read lines and inspect a project. '
        'Infer a useful new filename when unspecified. Accept indirect requests such as needing a PDF. '
        'Writes follow user intent, retain backups and never overwrite a new output. No code execution, '
        'deletion or renaming. Conversion exports extracted text, not layout. Preview only when requested.')
    permissions = ('workspace:read', 'workspace:write_documents')
    input_schema = {'type': 'object', 'properties': {'workspace_id': {'type': 'string'}}}
    operations = ('execute', 'repository_overview', 'search_text', 'read_lines', 'create_document', 'edit_document', 'convert_document')
    operation_contracts = {
        'execute': {'inputs': ['question'], 'outputs': ['final']},
        'repository_overview': {'inputs': [], 'outputs': ['paths', 'extensions']},
        'search_text': {'inputs': ['query'], 'outputs': ['matches', 'truncated']},
        'read_lines': {'inputs': ['path', 'start', 'end'], 'outputs': ['lines', 'total_lines', 'partial']},
        'create_document': {'inputs': ['question', 'path?', 'text?', 'content_request?'], 'outputs': ['path', 'final']},
        'edit_document': {'inputs': ['path', 'old', 'new'], 'outputs': ['proposal_id', 'path']},
        'convert_document': {'inputs': ['path', 'target_path'], 'outputs': ['proposal_id', 'path']},
    }

    def __init__(self, documents, repositories, workspace_skill=None):
        self.documents, self.repositories = documents, repositories
        self.workspace_skill = workspace_skill

    def match(self, question):
        return parse_local_command(question)

    async def execute(self, inputs, execution):
        result = await self.try_execute(inputs, execution)
        if result is None:
            raise AssistantError('Укажите, какой результат нужен от документа или файлов.', 409)
        return result

    async def execute_agent_step(self, operation, inputs, execution):
        from local_tool_intent import allowed_local_operations
        question = repeated_user_request(inputs.get('_local_tool_request', execution.task.user_input))
        allowed = allowed_local_operations(question,inputs)
        command = self.match(question)
        if command and '_local_allowed_operations' not in inputs:
            allowed.add(command['operation'])
        if not allowed or operation not in {'execute', *allowed}:
            raise AssistantError('Операция не разрешена исходным запросом пользователя.', 409)
        # Resolve the goal against the user's turn, not arbitrary plan arguments.
        return await self.execute(inputs, execution)

    @staticmethod
    def _clarify(question, inputs, message, expected_input='parameters', command=None):
        from local_tool_intent import allowed_local_operations, previous_answer_export
        allowed = allowed_local_operations(question,inputs)
        if command:
            allowed &= {command['operation']}
        pending = {'kind':'local_tool_input','workspace_id':inputs['workspace_id'],'question':question,
            'allowed_operations':sorted(allowed), 'expected_input':expected_input,
            'source_scope':inputs.get('source_scope')}
        if command:
            pending['command'] = dict(command)
        if previous_answer_export(question) or command and command.get('content_mode')=='previous_answer':
            # Freeze the literal source before another assistant clarification is saved.
            previous = inputs.get('_previous_answer') or (command or {}).get('text')
            if previous:
                pending['previous_answer'] = previous
        return ToolResult(success=True,tool='local_tools',summary='Tool input needed',data={
            'final':message,'context':'','results':[],'strategy':'LOCAL_TOOLS',
            'pending_interaction':pending,'pending_step':pending,'plan_status':'WAITING_FOR_USER'})

    async def try_execute(self, inputs, execution):
        from local_tool_intent import allowed_local_operations, previous_answer_export
        question = repeated_user_request(inputs.get('_local_tool_request', execution.task.user_input))
        command = inputs.get('_local_command') or previous_answer_export(question) or self.match(question)
        command = dict(command) if command else None
        directory_summary = directory_summary_request(question)
        if not command and directory_summary_chat_request(question):
            text, report = await self._compose_directory(question, inputs, execution, chat=True)
            async with execution.action('FINALIZE_SUMMARY') as action:
                action.output = report
            return ToolResult(success=True,tool=self.name,summary='Directory summary in chat',data={
                'final':text,'context':'','results':[],'strategy':'WORKSPACE_SUMMARY',
                'plan_status':'COMPLETED' if report['complete'] else 'PARTIAL'})
        if directory_summary:
            target=re.search(r'\b(?:создай|создать|сохрани|подготовь)\s+(?:(?:файл|документ)\s+)?'
                r'[«"`]?([^\n,;:«»"`]{1,120}?\.(?:md|txt|docx|pdf|html))'
                r'[»"`]?(?=\s+(?:с|со|для|в\s+котором)\b|\s*[,.;!?]|$)',question,re.I)
            command={'operation':'create_document','content_mode':'answer',
                'content_request':question,'path':target[1].strip() if target else 'Общий пересказ.md'}
        literal_command = command is not None
        allowed = allowed_local_operations(question, inputs)
        if command and command['operation'] not in allowed and '_local_allowed_operations' in inputs:
            return self._clarify(question,inputs,'Уточнение не разрешает другую операцию. Укажите отдельную новую просьбу.')
        if not command and not allowed:
            return None
        if not command:
            try:
                async with execution.action('RESOLVE_LOCAL_TOOL', {'allowed_operations':sorted(allowed)}) as action:
                    intent = await asyncio.to_thread(resolve_local_intent,question,inputs,self.operation_contracts)
                    action.output = intent.model_dump() if intent else {'operation':'none'}
            except (ValueError, RuntimeError) as exc:
                return self._clarify(question,inputs,'Не удалось определить параметры действия. '
                    'Укажите нужный документ и результат: например, новый текст по теме или копию в PDF.')
            if intent is None:
                return self._clarify(question,inputs,'Какой документ и результат вам нужен? '
                    'Можно описать задачу обычными словами, указать тему или исходный файл.')
            command = intent.model_dump()
        operation = command['operation']
        if not literal_command and operation not in allowed:
            if '_local_allowed_operations' in inputs:
                return self._clarify(question,inputs,'Уточнение не разрешает другую операцию. Укажите отдельную новую просьбу.')
            raise AssistantError('Операция не разрешена исходным запросом пользователя.',409)
        preview = bool(re.search(r'\b(?:предпросмотр|предварительно|не\s+сохраняй|только\s+черновик)\b',
                                 command_text(question)))
        command['preview_only'] = preview
        workspace_id = inputs['workspace_id']
        if operation in {'convert_document','edit_document','read_lines'}:
            reference = command.get('source_ref') or (command.get('path') if operation!='convert_document' else '')
            if 'source_ref' not in command and operation=='convert_document':
                reference = command['path']
            if not reference or re.fullmatch(r'(?:этот|этого|этой|данный|данного)?\s*(?:документ|документа|файл|файла|заметка|заметки)?|его|е[её]|это',reference.strip(),re.I):
                reference = inputs.get('_workspace_active_file','')
            if not reference:
                return self._clarify(question,inputs,'Какой исходный файл использовать? Напишите его имя или относительный путь.', 'source_ref', command)
            try:
                source = await _file_call(self.documents.filesystem.find_text_file,workspace_id,
                    'Прочитай файл '+reference, inputs.get('model'), inputs.get('_workspace_current_directory',''))
            except AssistantError as exc:
                if exc.status_code==409:
                    return self._clarify(question,inputs,str(exc),'source_ref',command)
                raise
            if operation=='convert_document':
                target = command.get('target_path') or (command.get('path') if 'source_ref' in command else '')
                suffix = requested_format(question) or Path(target or '').suffix.lstrip('.')
                if not suffix:
                    return self._clarify(question,inputs,'В какой формат преобразовать документ: PDF, DOCX, MD, TXT, HTML, CSV или TSV?',
                        'format', {**command,'source_ref':source})
                if not target:
                    target = str(Path(source).with_suffix('.'+suffix)).replace('\\','/')
                elif Path(target).suffix.lstrip('.').casefold()!=suffix:
                    if target in question:
                        return self._clarify(question,inputs,'Расширение нового файла и запрошенный формат различаются. Какой формат использовать?')
                    target = Path(target).with_suffix('.'+suffix).as_posix()
                command.update(path=source,target_path=target)
            else:
                command['path'] = source
        if operation=='edit_document':
            old,new = command.get('old',''),command.get('new','')
            if not old or old not in question or new and new not in question:
                return self._clarify(question,inputs,'Укажите исходный фрагмент и замену, например: замените «старый текст» на «новый текст».')
        if operation=='search_text' and (not command.get('query') or command['query'].casefold() not in question.casefold()):
            return self._clarify(question,inputs,'Какой точный текст найти в файлах?')
        if operation=='create_document':
            mode = command.get('content_mode','literal')
            if mode=='previous_answer':
                from conversation_context import _message
                previous = [content for role,content in map(_message,inputs.get('_chat_history') or [])
                            if role in {'assistant','ai'} and content.strip()]
                if not previous and not inputs.get('_previous_answer'):
                    return self._clarify(question,inputs,'Какой текст сохранить? В этом диалоге ещё нет предыдущего ответа.', 'previous_answer', command)
                # Explicitly copying text is not a new claim that it is factual.
                command['text'] = inputs.get('_previous_answer') or previous[-1]
            elif mode=='answer':
                query = command.get('content_request','')
                if not query:
                    return self._clarify(question,inputs,'О чём должен быть новый документ?', 'content_request', command)
                if directory_summary:
                    composed=await self._compose_directory(question,inputs,execution)
                else:
                    composed = await self._compose(query, inputs, execution)
                if isinstance(composed, ToolResult):
                    return composed
                command['text'],command['_document_evidence'] = composed
            elif command.get('text','') not in question:
                return self._clarify(question,inputs,'Для буквального сохранения укажите сам текст. Для нового описания достаточно назвать тему.')
            elif mode=='literal' and not command.get('text') and not literal_command and not re.search(r'\bпуст\w*\b',question,re.I):
                return self._clarify(question,inputs,'Какое содержимое нужно сохранить? Можно дать текст или описать тему нового документа.')
            path = command.get('path') or 'Новый документ.md'
            output_format=requested_format(question)
            if output_format and Path(path).suffix and Path(path).suffix.lstrip('.').casefold()!=output_format:
                if path in question:
                    return self._clarify(question,inputs,'Имя документа и требуемый формат различаются. Какое расширение использовать?',
                        'format',command)
                path=Path(path).with_suffix('.'+output_format).as_posix()
            if not Path(path).suffix:
                path += '.'+(output_format or 'md')
            directory = inputs.get('_workspace_current_directory','')
            explicit_path = path in question
            explicit_directory = await asyncio.to_thread(self.documents.filesystem.find_directory,
                workspace_id,question)
            if explicit_directory:
                directory=explicit_directory
            if '/' in path and not explicit_path:
                path=Path(path).name
            if '/' not in path and directory:
                path = directory.rstrip('/')+'/'+path
            # Generated names may choose a free suffix; explicitly requested
            # paths retain the existing no-overwrite conflict instead.
            explicit = path in question or Path(path).name in question
            if not explicit:
                root = self.documents._root(workspace_id)
                original = path
                for number in range(2,102):
                    if not (root/path).exists():break
                    value = Path(original)
                    path = value.with_name(f'{value.stem} ({number}){value.suffix}').as_posix()
            command['path'] = path
        result = await self._run(command,workspace_id,execution)
        result.data.setdefault('current_directory',inputs.get('_workspace_current_directory',''))
        return result

    async def _compose_directory(self, question, inputs, execution, chat=False):
        from directory_summary import DirectorySummary
        summary_inputs=dict(inputs)
        # Resolve a name next to the folder marker, not incidental topic words
        # elsewhere in the request (e.g. "странный архив" is not folder "Архив").
        label=summary_directory_reference(question)
        if label:
            scope=await asyncio.to_thread(self.documents.filesystem.find_directory,
                inputs['workspace_id'],'папка '+label)
            if not scope:
                raise AssistantError('Указанная папка не найдена. Уточните её относительный путь.',404)
            summary_inputs['_workspace_current_directory']=scope
            summary_inputs['_summary_source_directory']=scope
        summary_inputs['_summary_in_chat']=chat
        return await DirectorySummary(self.documents.filesystem).compose(summary_inputs,execution)

    async def try_document_search(self, inputs, execution):
        """Called before history compaction; shares the text/Live executor."""
        from document_finder import DocumentFinder, result
        from local_tool_intent import document_search_query, document_search_selection
        question = execution.task.user_input
        query = document_search_query(question)
        selection = document_search_selection(question)
        if query is not None:
            return await DocumentFinder(self.workspace_skill).execute_search(query, inputs, execution)
        pending = await asyncio.to_thread(execution.repository.pending_plan, execution.task.session_id)
        if pending and pending.get('kind') == 'workspace_file_choice':
            # An unresolved newer file choice takes precedence over an old search.
            return None
        if selection is None and re.fullmatch(r'\s*(открой(?:те)?|покажи(?:те)?|прочитай(?:те)?)\s+'
                r'(?:его|е[её]|этот\s+(?:файл|документ))\s*[.!?]*',question,re.I):
            if pending:
                return None
            state = await asyncio.to_thread(execution.repository.last_document_search, execution.task.session_id)
            active = inputs.get('_workspace_active_file')
            index = next((i for i,row in enumerate((state or {}).get('candidates',[]),1) if row['path']==active),None)
            if index is not None:
                operation = 'open' if question.strip().casefold().startswith('открой') else 'read'
                return await DocumentFinder(self.workspace_skill).execute_selection(
                    {'index':index,'operation':operation},state,inputs,execution)
        if selection is not None:
            # Do not steal bare numeric replies from existing pending workflows.
            if selection['operation']=='select' and pending:
                return None
            state = await asyncio.to_thread(execution.repository.last_document_search, execution.task.session_id)
            if state is None and selection['operation']=='select':
                return None
            return await DocumentFinder(self.workspace_skill).execute_selection(selection, state, inputs, execution)
        if re.fullmatch(r'\s*(?:отмена|отмени\s+поиск|не\s+надо)\s*[.!]*',question,re.I):
            state = await asyncio.to_thread(execution.repository.last_document_search, execution.task.session_id)
            if state and not pending:
                async with execution.action('CLEAR_DOCUMENT_SEARCH') as action:
                    action.output = {'candidates':[]}
                selected = inputs.get('_workspace_active_file') in {r['path'] for r in state.get('candidates',[])}
                return result('Выбор документа отменён.', **({'active_file':''} if selected else {}))
        return None

    async def _compose(self, query, inputs, execution):
        if self.workspace_skill is None:
            raise AssistantError('Навык чтения источников не подключён.',503)
        model = inputs.get('model','BALANCED')
        retrieval_question=document_content_question(query)
        query = retrieval_question.rstrip(' .?!')+'?\nСоставь содержательный документ по источникам, сохрани атрибуцию говорящего, условия и отрицания.'
        budget = self.workspace_skill.response_budget(query,model)
        async with execution.action('PLAN_DOCUMENT',{'content_request':query}) as action:
            action.output = {'steps':['SEARCH_WORKSPACE','GENERATE_RESPONSE','VERIFY_RESPONSE','CREATE_DOCUMENT','APPLY_DOCUMENT_CHANGE']}
        matches = await _file_call(self.workspace_skill.relevant_context,retrieval_question,inputs['workspace_id'],
            False,budget.max_source_characters,inputs.get('_workspace_active_file',''),
            source_scope=inputs.get('source_scope'))
        authored = await self.workspace_skill.execute({
            'question':query,'model':model,'workspace_id':inputs['workspace_id'],
            '_retrieval_question':retrieval_question,
            'session_id':inputs.get('session_id'),'use_multi_agent':True,
            'source_scope':inputs.get('source_scope'),
            '_document_authoring':True,'_authoring_request':execution.task.user_input,
            '_workspace_matches':matches,'_workspace_context_requested':True,'_response_budget':budget},execution)
        data = authored.data
        check = data.get('verification')
        if isinstance(check,dict):
            from pydantic_models import VerificationResult
            check = VerificationResult.model_validate(check)
        if not check or check.checked_answer!=data.get('final'):
            return ToolResult(success=True,tool=self.name,summary='Source review unavailable',data={
                'final':'Не удалось завершить проверку содержимого по источникам. Документ не сохранён.',
                'context':'','results':data.get('results',[]),'strategy':'LOCAL_TOOLS'})
        text = data['final']
        if check.status!='SUPPORTED':
            from verifier import checked_sentences_after_timeout
            selected = checked_sentences_after_timeout(text,data.get('results',[]),check)
            if not selected or selected[1].status!='SUPPORTED':
                return ToolResult(success=True,tool=self.name,summary='Document content not supported',data={
                    'final':'Источники найдены, но получить подтверждённое описание пока не удалось. Документ не сохранён.',
                    'context':'','results':data.get('results',[]),'strategy':'LOCAL_TOOLS'})
            text,check = selected
        paths = list(dict.fromkeys(s.get('metadata',{}).get('source','') for s in data.get('results',[])
            if s.get('chunk_id') in check.source_chunk_ids))
        if paths and not re.search(r'(?im)^Источники:',text):
            text += '\n\nИсточники: '+', '.join('`'+path+'`' for path in paths if path)+'.'
        evidence = check.model_dump(mode='json')
        evidence['source_snapshots'] = [{'path':s['metadata']['source'],'revision':s['metadata']['revision']}
            for s in data.get('results',[]) if s.get('metadata',{}).get('revision')]
        return text,evidence

    async def _run(self, command, workspace_id, execution):
        operation = command['operation']
        async with execution.action(operation.upper(), {'workspace_id': workspace_id}) as action:
            if operation == 'repository_overview':
                data = await _file_call(self.repositories.overview, workspace_id)
                final = (f"Доступных файлов: {data['readable_files']}. Учитываются .gitignore.\n\n" +
                    '\n'.join('- `' + path + '`' for path in data['paths']))
                if data['truncated'] or data['paths_truncated'] or data['errors']:
                    final += '\n\nПоказана часть структуры; обзор ограничен.'
            elif operation == 'read_lines':
                data = await _file_call(self.documents.read_lines, workspace_id, command['path'], command['start'], command['end'])
                final = '\n'.join(f"{row['line']}: {row['text']}" for row in data['lines']) or 'В указанном диапазоне строк нет.'
                final = f"`{data['path']}` · всего строк: {data['total_lines']}\n\n```text\n{final}\n```"
                if data['truncated']:
                    final += '\n\nВывод ограничен по объёму.'
            elif operation == 'search_text':
                data = await _file_call(self.repositories.search, workspace_id, command['query'])
                final = '\n'.join(f"`{row['path']}:{row['line']}` — {row['text']}" for row in data['matches'])
                final = final or 'Совпадений в просмотренных файлах не найдено.'
                if data['truncated'] or data['errors'] or data['scan_errors']:
                    final += '\n\nПоиск ограничен; часть файлов или результатов не показана.'
            else:
                if operation == 'convert_document':
                    data = await _file_call(self.documents.prepare_convert, workspace_id,
                                                   command['path'], command['target_path'])
                elif operation == 'edit_document':
                    data = await _file_call(self.documents.prepare_edit, workspace_id,
                                                   command['path'], command['old'], command['new'])
                else:
                    data = await _file_call(self.documents.prepare_create, workspace_id,
                                                    command['path'], command['text'])
                    evidence = command.get('_document_evidence')
                    if evidence:
                        parents = ([{'path':r['path'],'revision':r['revision']} for r in evidence.get('files',[])]
                            if evidence.get('mode')=='DIRECTORY_SUMMARY' else evidence.get('source_snapshots',[]))
                        self.documents.attach_provenance(data['proposal_id'],'generated',parents,execution.task.id)
                    elif command.get('content_mode')=='previous_answer':
                        self.documents.attach_provenance(data['proposal_id'],'chat_export',task_id=execution.task.id)
                final = (f"Подготовлен документ `{data['path']}` ({data['bytes']} байт).\n\n" + data['note'])
                if data['diff']:
                    final += '\n\n```diff\n' + data['diff'] + '\n```'
                elif operation == 'create_document':
                    final += '\n\n' + command['text'][:4000]
                if data['diff_truncated']:
                    final += '\n\nПоказана часть различий.'
                if command.get('_document_evidence'):
                    data['document_verification'] = command['_document_evidence']
                if command.get('preview_only'):
                    final += '\n\nОтветьте «да», чтобы сохранить, или «отмена».'
                    data['pending_interaction'] = {'kind': 'document_proposal', 'workspace_id': workspace_id,
                                                   'proposal_id': data['proposal_id']}
                    data['plan_status'] = 'WAITING_FOR_USER'
                    data['pending_step'] = data['pending_interaction']
            action.output = {key: value for key, value in data.items() if key not in {'diff', 'matches', 'paths'}}
        if operation in {'create_document','edit_document','convert_document'} and not command.get('preview_only'):
            saved = await self.apply(workspace_id,data['proposal_id'],execution)
            saved.data['document_verification'] = data.get('document_verification')
            if operation=='create_document':
                text=command.get('text','')
                if len(text)>6000:
                    saved.data['final'] += f'\n\nВ файле {len(text)} символов. Ниже показано начало; полный текст сохранён в документе.'
                evidence=command.get('_document_evidence') or {}
                if evidence.get('mode')=='DIRECTORY_SUMMARY' and not evidence['complete']:
                    saved.data['final'] += '\n\nПересказ не завершён: сохранены проверенные части, пропуски перечислены в документе.'
                    saved.data['plan_status']='PARTIAL'
                saved.data['final'] += '\n\n'+text[:6000]
            return saved
        return ToolResult(success=True, tool=self.name, summary=operation, data={**data, 'final': final,
            'context': '', 'results': [], 'strategy': 'LOCAL_TOOLS'})

    async def apply(self, workspace_id, proposal_id, execution):
        async with execution.action('APPLY_DOCUMENT_CHANGE', {'workspace_id': workspace_id}) as action:
            data = await _file_call(self.documents.apply, workspace_id, proposal_id)
            action.output = data
        final = f"Сохранён файл `{data['path']}`."
        if data['backup_path']:
            final += f"\n\nРезервная копия: `{data['backup_path']}`."
        if data.get('provenance_warning'):final += '\n\n'+data['provenance_warning']
        return ToolResult(success=True, tool=self.name, summary='Document saved',
            data={**data, 'final': final, 'context': '', 'results': [], 'strategy': 'LOCAL_TOOLS',
                  'current_directory':'' if Path(data['path']).parent.as_posix()=='.' else Path(data['path']).parent.as_posix(),
                  'active_file':data['path'],'plan_status': 'COMPLETED'})
