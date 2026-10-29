"""Assistant entry point for text, voice transcripts and future event adapters."""
import asyncio
import logging
import re
import time
from pathlib import PurePosixPath
from contextlib import asynccontextmanager

from assistant_models import AssistantError, AssistantEvent, EventType, TaskStatus, ToolResult
from assistant_skills import AgentExecutor, SkillRegistry
from agent_planner import AgentPlan, AgentPlanStep, is_single_information_request
from model_runtime import get_llm_provider
from llm_contract import (ProviderError, GenerationCancelled, GenerationDeadlineExceeded, GenerationScope, current_scope, current_action)
from pydantic_models import QueryResponse
from query_router import QueryStrategy, route_by_rules
from command_intent import requests_file_open, requests_unsupported_file_change, negates_comparison
from source_grounding import requests_document_library
from workspace_intent import (display_file_target, open_file_target, permits_workspace_operation,
                              references_active_file, references_created_file, asks_last_opened_file,
                              requests_active_file_read)
from workspace_router import WorkspaceRoute, WorkspaceRouteDecision
from workspace_source_scope import WorkspaceSourceScope

logger = logging.getLogger(__name__)


def short_error(exc):
    # Deliberately exclude arbitrary exception payloads (which may contain source
    # text, prompts or credentials). Full diagnostics stay in backend logs.
    return f'{type(exc).__name__}: execution failed'[:500]


class ExecutionContext:
    def __init__(self, repository, task, event_sink=None, progress_sink=None):
        self.repository, self.task = repository, task
        self.event_sink = event_sink
        self.events = []
        self.progress_sink=progress_sink

    def progress(self, message):
        if self.progress_sink:self.progress_sink(message)

    async def configure_policy(self, policy):
        from task_policy import apply_policy
        apply_policy(current_scope.get(),policy)
        await asyncio.to_thread(self.repository.update_task_metadata,self.task,execution_policy=policy.as_dict())

    def emit(self, kind, action_id=None):
        event = AssistantEvent(type=kind, task_id=self.task.id,
                               session_id=self.task.session_id, action_id=action_id)
        self.events.append(event)
        if self.event_sink:
            try:
                self.event_sink(event)
            except Exception:
                logger.exception('Assistant event observer failed')

    @asynccontextmanager
    async def action(self, action_type, inputs=None):
        scope = current_scope.get()
        if scope and scope.cancelled.is_set():
            raise GenerationCancelled()
        action = await asyncio.to_thread(self.repository.create_action, self.task.id, action_type, inputs)
        await asyncio.to_thread(self.repository.transition, action, TaskStatus.RUNNING)
        self.emit(EventType.ACTION_STARTED, action.id)
        token = current_action.set(action.id)
        try:
            yield action
            committed_write = (action.action_type in {'APPLY_DOCUMENT_CHANGE', 'REPOSITORY_EDIT',
                'REPOSITORY_REWRITE', 'REPOSITORY_WRITE'} and action.output.get('applied') is True)
            if (scope and scope.cancelled.is_set() and self.task.status != TaskStatus.COMPLETED
                    and not committed_write):
                raise GenerationCancelled()
        except BaseException as exc:
            cancelled = isinstance(exc, asyncio.CancelledError)
            await asyncio.to_thread(self.repository.transition, action,
                                    TaskStatus.CANCELLED if cancelled else TaskStatus.FAILED,
                                    error=short_error(exc))
            self.emit(EventType.ACTION_CANCELLED if cancelled else EventType.ACTION_FAILED, action.id)
            raise
        else:
            await asyncio.to_thread(self.repository.transition, action, TaskStatus.COMPLETED,
                output={'success': True, 'tool': action_type, 'summary': 'Completed',
                        'data': action.output, 'error': None})
            self.emit(EventType.ACTION_COMPLETED, action.id)
        finally:
            current_action.reset(token)


class AssistantCore:
    def __init__(self, repository, knowledge_skill, save_history, event_sink=None, workspace_skill=None,
                 history_loader=None, conversation_resolver=None, agent_planner=None, response_saver=None,
                 local_tools_skill=None, repository_agent_skill=None, conversation_memory=None):
        self.repository = repository
        self.active_tasks = {}
        self.deployment_lock = asyncio.Lock()
        self.skills = SkillRegistry()
        self.skills.register(knowledge_skill)
        if workspace_skill:
            self.skills.register(workspace_skill)
        self.workspace_skill = workspace_skill
        self.local_tools_skill = local_tools_skill
        if local_tools_skill:
            self.skills.register(local_tools_skill)
            local_tools_skill.workspace_skill = workspace_skill
        self.repository_agent_skill = repository_agent_skill
        if repository_agent_skill:
            self.skills.register(repository_agent_skill)
        self.agent_executor = AgentExecutor(self.skills)
        self.save_history = save_history
        self.response_saver = response_saver
        self.history_loader = history_loader
        self.conversation_memory=conversation_memory
        self.conversation_resolver = conversation_resolver
        self.agent_planner = agent_planner
        self.event_sink = event_sink
        # Locks exist only while turns are queued/running; idle sessions do not
        # accumulate in memory. The API owns a single event loop.
        self._session_locks = {}

    @staticmethod
    def compile_agent_plan(plan: AgentPlan) -> AgentPlan:
        """Repair safe, mechanically inferable links without granting new capabilities."""
        compiled = plan.model_copy(deep=True)
        safe_workspace_operations = {
            'execute', 'find_latest', 'compare_versions', 'open_file', 'read_file', 'list_file_dates',
        }
        previous_workspace_result = None
        workspace_results = set()
        for step in compiled.steps:
            if step.skill == 'workspace' and step.operation in safe_workspace_operations:
                # Reading metadata and opening an explicitly requested file are
                # reversible operations and never need a second confirmation.
                step.requires_confirmation = False
            if step.skill == 'workspace' and step.operation in {'find_latest', 'compare_versions'}:
                # Keep directory names and temporal qualifiers from the complete
                # user request even when the LLM shortened its step argument.
                step.arguments['target_query'] = compiled.objective
                previous_workspace_result = step.id
                workspace_results.add(step.id)
            elif step.skill == 'workspace' and step.operation == 'open_file':
                dependency = next((value for value in reversed(step.depends_on)
                                   if value in workspace_results), previous_workspace_result)
                if dependency and not step.arguments.get('path'):
                    if dependency not in step.depends_on:
                        step.depends_on.append(dependency)
                    step.arguments['path'] = f'${{{dependency}.data.selected_path}}'
                previous_workspace_result = step.id
                workspace_results.add(step.id)
        return AgentPlan.model_validate(compiled.model_dump(mode='json'))

    def cancel_task(self, task_id):
        scope = self.active_tasks.get(task_id)
        if scope is None:
            return False
        scope.cancelled.set()
        get_llm_provider().cancel(task_id)
        return True

    async def process_input(self, request, *, source='text', on_task_started=None, on_progress=None):
        try:
            session = await asyncio.to_thread(self.repository.ensure_session, request.session_id, request.chat_id)
        except ValueError as exc:
            raise AssistantError(str(exc), 409) from exc
        request = request.model_copy(update={'session_id': session.id})
        key = session.id
        entry = self._session_locks.setdefault(key, [asyncio.Lock(), 0])
        entry[1] += 1
        try:
            async with entry[0]:
                if self.response_saver and request.request_id:
                    try:
                        saved = await asyncio.to_thread(self.repository.saved_response, session.id, request)
                    except ValueError as exc:
                        raise AssistantError(str(exc), 409) from exc
                    if saved:
                        return saved, 0
                return await self._process_input(request, source=source, on_task_started=on_task_started,on_progress=on_progress)
        finally:
            entry[1] -= 1
            if not entry[1]:
                self._session_locks.pop(key, None)

    async def _process_input(self, request, *, source='text', on_task_started=None, on_progress=None):
        started = time.perf_counter()
        async with self.deployment_lock:
            try:
                session = await asyncio.to_thread(self.repository.ensure_session, request.session_id, request.chat_id)
            except ValueError as exc:
                raise AssistantError(str(exc), 409) from exc
            task = await asyncio.to_thread(self.repository.create_task, session.id, request.question, {'source': source})
            execution = ExecutionContext(self.repository, task, self.event_sink,on_progress)
            await asyncio.to_thread(self.repository.transition, task, TaskStatus.RUNNING)
            execution.emit(EventType.TASK_STARTED)
            from model_profiles import get_profile
            scope = GenerationScope(task.id, deadline=time.perf_counter() + get_profile(request.model).request_seconds)
            from task_policy import classify_task,apply_policy
            policy=classify_task(request.question,request.model,bool(request.workspace_id))
            apply_policy(scope,policy)
            await asyncio.to_thread(self.repository.update_task_metadata,task,execution_policy=policy.as_dict())
            token = current_scope.set(scope)
            from workspace import workspace_scan_cache
            scan_token = workspace_scan_cache.set({})
            from answer_contract import current_contract, make_contract
            contract_token = current_contract.set(make_contract(request.question, request.workspace_id,
                request.source_scope.model_dump(mode='json') if request.source_scope else None))
            self.active_tasks[task.id] = scope
        try:
            if on_task_started is not None:
                on_task_started(task)
            if policy.kind=='LONG':execution.progress('Длительная задача: выполняю по шагам, сохраняю результаты и проверки.')
            inputs = request.model_dump()
            # Pending state is JSON-persisted. Keep the immutable selection's
            # tuple/list representation identical across confirmation turns.
            inputs['source_scope'] = request.source_scope.model_dump(mode='json') if request.source_scope else None
            await asyncio.to_thread(self.repository.update_task_metadata, task,
                                   answer_contract=current_contract.get().as_dict())
            if request.workspace_id and self.workspace_skill:
                current_directory = await asyncio.to_thread(self.workspace_skill.repository.session_directory,
                    session.id, request.workspace_id
                )
                inputs['_workspace_current_directory'] = (
                    current_directory if isinstance(current_directory, str) else ''
                )
                active_file = await asyncio.to_thread(self.workspace_skill.repository.session_file,
                    session.id, request.workspace_id
                )
                inputs['_workspace_active_file'] = active_file if isinstance(active_file, str) else ''
                if getattr(request,'workspace_directory',None) is not None:
                    listing = await asyncio.to_thread(self.workspace_skill.filesystem.list_directory,
                        request.workspace_id,request.workspace_directory)
                    inputs['_workspace_current_directory'] = listing.path
                if request.source_scope is not None:
                    await asyncio.to_thread(self.workspace_skill.filesystem.validate_source_scope,
                        request.workspace_id, request.source_scope)
                    await asyncio.to_thread(self.repository.update_task_metadata, task,
                        source_scope=request.source_scope.fingerprint())
            skill = self.skills.get('knowledge')
            history = []
            precomputed_result = None
            resume_plan = None
            resume_results = None
            resume_file_question = None
            confirmed_steps = set()
            pending = await asyncio.to_thread(self.repository.pending_plan, session.id)
            from execution_reports import summary_feedback, summary_status, status_result
            feedback = summary_feedback(request.question) if request.workspace_id and not request.graph_selection else None
            if feedback:
                # A new read-only request must not become approval or a missing
                # filename for an unrelated previous write.
                await asyncio.to_thread(self.repository.clear_pending_plan, session.id)
                pending = None
                from dataclasses import replace
                current_contract.set(replace(current_contract.get(), source='EXECUTION'))
                await asyncio.to_thread(self.repository.update_task_metadata,task,
                                       answer_contract=current_contract.get().as_dict())
                prior = await asyncio.to_thread(self.repository.last_summary, session.id,
                                               request.workspace_id, inputs.get('source_scope'))
                async with execution.action('DESCRIBE_SUMMARY', {'workspace_id':request.workspace_id}) as action:
                    action.output = {'task_id':prior['task_id'] if prior else None,
                                     'scope_matches':prior['scope_matches'] if prior else None}
                if not prior:
                    precomputed_result = status_result('В журнале этого чата и рабочей папки нет завершённого пересказа для объяснения или продолжения.')
                elif not prior['scope_matches']:
                    precomputed_result = status_result('Область источников изменилась. Для продолжения выберите прежнюю область или запросите новый пересказ.')
                elif feedback == 'describe':
                    precomputed_result = status_result(summary_status(prior['report']))
                elif self.local_tools_skill:
                    report = prior['report']
                    resume_inputs = {**inputs, '_summary_previous_review':report,
                        '_summary_exclude':report.get('excluded',[]),
                        '_summary_source_directory':report.get('directory',''),
                        '_workspace_current_directory':report.get('directory',''), '_summary_in_chat':True}
                    from directory_summary import DirectorySummary
                    text, updated = await DirectorySummary(self.workspace_skill.filesystem).compose(resume_inputs, execution)
                    async with execution.action('FINALIZE_SUMMARY') as action:action.output = updated
                    precomputed_result = ToolResult(success=True,tool='workspace',summary='Summary continued',data={
                        'final':text,'context':'','results':[],'strategy':'WORKSPACE_SUMMARY',
                        'plan_status':'COMPLETED' if updated['complete'] else 'PARTIAL'})
            from conversation_control import session_announcement_reply
            announcement = session_announcement_reply(request.question) if not request.graph_selection else None
            if announcement:
                # A new dialogue label is neither a file parameter nor approval
                # of a pending write. Do not compact/rewrite it into a RAG query.
                await asyncio.to_thread(self.repository.clear_pending_plan, session.id)
                pending = None
                precomputed_result = ToolResult(success=True, tool='knowledge', summary='Dialogue announcement',
                    data={'final': announcement, 'context': '', 'results': [], 'strategy': 'GENERAL_CHAT'})
            if (precomputed_result is None and self.local_tools_skill and self.workspace_skill and request.workspace_id
                    and not request.graph_selection):
                precomputed_result = await self.local_tools_skill.try_document_search(inputs, execution)
                if precomputed_result is not None:
                    await asyncio.to_thread(self.repository.clear_pending_plan, session.id)
                    pending = None
            if request.graph_selection:
                selection = request.graph_selection
                if selection.workspace_id != request.workspace_id:
                    raise AssistantError('Источник вопроса должен совпадать с выбранным графом.', 422)
                from graph_workbench import graph_workbench
                async with execution.action('SELECT_GRAPH_SOURCES', {'relation_ids': selection.relation_ids,
                                                                  'workspace_id': selection.workspace_id}) as action:
                    workbench = await asyncio.to_thread(graph_workbench, selection.workspace_id,
                        filesystem=self.workspace_skill.filesystem if self.workspace_skill else None,
                        connection_factory=self.repository.connection_factory)
                    inputs['_graph_focus'] = await asyncio.to_thread(workbench.selection, selection.relation_ids)
                    inputs['_chat_history'] = []
                    action.output = {'source_count': len(inputs['_graph_focus']['sources'])}
                # An explicit graph question starts an independent information
                # request; it must not resume old commands or change sources.
                await asyncio.to_thread(self.repository.clear_pending_plan, session.id)
                pending = None
                precomputed_result = await skill.execute(inputs, execution)
            if pending:
                def scope_key(value):
                    return WorkspaceSourceScope.model_validate(value).fingerprint() if value is not None else None
                pending_workspace = pending.get('workspace_id') or (pending.get('base_inputs') or {}).get('workspace_id')
                if pending_workspace and pending_workspace != request.workspace_id:
                    await asyncio.to_thread(self.repository.clear_pending_plan, session.id)
                    pending = None
                elif scope_key(pending.get('source_scope')) != scope_key(inputs.get('source_scope')):
                    # A confirmation must not resume an operation prepared for
                    # a different set of sources in the same workspace.
                    await asyncio.to_thread(self.repository.clear_pending_plan, session.id)
                    pending = None
            if pending and pending.get('kind') == 'local_tool_input':
                from local_tool_intent import permitted_local_operations, local_tool_followup
                from action_feedback import action_feedback
                if re.fullmatch(r'(?:нет|отмена|отмени|не надо)[.! ]*',request.question.strip(),re.I):
                    precomputed_result = ToolResult(success=True,tool='local_tools',summary='Cancelled',data={
                        'final':'Действие отменено.','context':'','results':[],'strategy':'LOCAL_TOOLS'})
                elif (pending.get('workspace_id')==request.workspace_id
                        and not requests_file_open(request.question) and not asks_last_opened_file(request.question)
                        and not action_feedback(request.question)
                        and not permitted_local_operations(request.question)
                        and not self.local_tools_skill.match(request.question)
                        and local_tool_followup(pending,request.question)):
                    inputs['_local_tool_request'] = pending['question']+'\nУточнение пользователя: '+request.question
                    inputs['_local_allowed_operations'] = sorted(set(pending.get('allowed_operations',
                        permitted_local_operations(pending['question']))) & permitted_local_operations(pending['question']))
                    if pending.get('command'):
                        command = dict(pending['command'])
                        slot = pending.get('expected_input')
                        if slot in {'source_ref','path','content_request'}:
                            command[slot] = request.question.strip().strip('«»"`')
                        inputs['_local_command'] = command
                    if pending.get('previous_answer'):
                        inputs['_previous_answer'] = pending['previous_answer']
                await asyncio.to_thread(self.repository.clear_pending_plan, session.id)
                pending = None
            if pending and pending.get('kind') == 'document_proposal':
                reply = request.question.strip().casefold()
                if self.local_tools_skill and re.fullmatch(r'(?:да|подтверждаю|сохрани|сохранить|применить)[.! ]*', reply):
                    precomputed_result = await self.local_tools_skill.apply(
                        request.workspace_id, pending['proposal_id'], execution)
                elif re.fullmatch(r'(?:нет|отмена|отмени|не надо|не сохраняй)[.! ]*', reply):
                    precomputed_result = ToolResult(success=True, tool='local_tools', summary='Cancelled', data={
                        'final': 'Изменение отменено.', 'context': '', 'results': [], 'strategy': 'LOCAL_TOOLS'})
                # A new request leaves the previous proposal unapplied.
                await asyncio.to_thread(self.repository.clear_pending_plan, session.id)
                pending = None
            if pending:
                reply = request.question.strip().casefold()
                replacement_command = bool(re.search(
                    r'\b(?:перейди|найди|открой|прочитай|сравни|покажи|скажи|создай|переименуй)\w*\b',
                    request.question, re.IGNORECASE,
                ))
                if pending.get('kind') != 'workspace_file_choice' and replacement_command:
                    await asyncio.to_thread(self.repository.clear_pending_plan, session.id)
                    pending = None
                if pending and pending.get('kind') == 'workspace_file_choice':
                    choices = pending.get('choices') or []
                    ordinal = {'первый': 1, 'первая': 1, 'второй': 2, 'вторая': 2,
                               'третий': 3, 'третья': 3}
                    selected = None
                    refers_to_pending = bool(re.search(
                        r'\b(?:его|е[её]|этот|эту|данн\w*|перв\w*|втор\w*|трет\w*)\b', reply
                    ))
                    replacement_command = bool(re.search(
                        r'\b(?:открой|покажи|прочитай|найди)\w*\b', reply
                    )) and not refers_to_pending
                    if replacement_command:
                        await asyncio.to_thread(self.repository.clear_pending_plan, session.id)
                        pending = None
                    if pending and len(choices) == 1 and refers_to_pending:
                        selected = choices[0]
                    number = re.fullmatch(r'\s*(\d+)\s*[.!]?\s*', reply)
                    index = int(number.group(1)) if number else ordinal.get(reply.rstrip('.!'))
                    if index and 1 <= index <= len(choices):
                        selected = choices[index - 1]
                    if selected is None:
                        exact = [item for item in choices if item.casefold() == reply.strip('` .!')]
                        selected = exact[0] if len(exact) == 1 else None
                    if pending and selected:
                        original = pending.get('original_question', '')
                        verb = 'Открой файл' if re.search(r'\bоткрой\w*\b', original, re.I) else 'Прочитай файл'
                        resume_file_question = f'{verb} {selected}'
                        if requests_file_open(original):
                            inputs['_workspace_open_authorization'] = original
                        await asyncio.to_thread(self.repository.clear_pending_plan, session.id)
                    elif pending:
                        rendered = '\n'.join(f'{i}. `{item}`' for i, item in enumerate(choices, 1))
                        precomputed_result = ToolResult(
                            success=True, tool='agent', summary='Waiting for file choice', data={
                                'final': 'Выберите файл номером или относительным путём:\n' + rendered,
                                'context': '', 'results': [], 'strategy': 'AGENT_WAITING_FOR_USER',
                                'plan_status': 'WAITING_FOR_USER', 'pending_step': pending,
                            })
                elif pending:
                    affirmative = bool(re.fullmatch(
                        r'(?:да|подтверждаю|продолжай|выполняй|согласен|ок(?:ей)?)[.! ]*', reply
                    ))
                    negative = bool(re.fullmatch(
                        r'(?:нет|отмена|отмени|не надо|не выполняй)[.! ]*', reply
                    ))
                    if affirmative:
                        resume_plan = self.compile_agent_plan(AgentPlan.model_validate(pending['plan']))
                        resume_results = pending.get('completed_results') or {}
                        confirmed_steps = {pending['pending_step_id']}
                        inputs.update(pending.get('base_inputs') or {})
                    elif negative:
                        await asyncio.to_thread(self.repository.clear_pending_plan, session.id)
                        precomputed_result = ToolResult(success=True, tool='agent', summary='Plan cancelled', data={
                            'final': 'Хорошо, действие отменено.', 'context': '', 'results': [],
                            'strategy': 'AGENT_CANCELLED', 'plan_status': 'CANCELLED',
                        })
                    else:
                        step = pending.get('pending_step') or {'operation': pending.get('pending_step_id')}
                        precomputed_result = ToolResult(success=True, tool='agent', summary='Waiting for confirmation', data={
                            'final': (f"План ожидает подтверждения шага «{step.get('operation')}». "
                                      "Ответьте «да», чтобы продолжить, или «отмена», чтобы остановить задачу."),
                                'context': '', 'results': [], 'strategy': 'AGENT_WAITING_FOR_USER',
                                'plan_status': 'WAITING_FOR_USER', 'pending_step': step,
                            })
            if request.workspace_id and self.workspace_skill and not request.graph_selection:
                from action_feedback import action_feedback
                feedback = action_feedback(request.question)
                if precomputed_result is None and feedback:
                    async with execution.action('REVIEW_PREVIOUS_ACTION', {'feedback':feedback}) as action:
                        prior = await asyncio.to_thread(self.repository.last_file_action,session.id,
                            request.workspace_id,created=feedback=='incomplete_document')
                        action.output = prior or {'path':None}
                    if feedback == 'wrong_open':
                        from action_feedback import correct_open
                        corrected=await correct_open(prior,request.workspace_id,self.repository,
                            self.workspace_skill.filesystem,execution)
                        message = (f"Был открыт `{prior['path']}`. Это не тот файл, который вы имели в виду. "
                            'Укажите имя нужного файла; можно написать расширение без точки.' if prior else
                            'В журнале этого чата нет успешного открытия. Укажите имя нужного файла.')
                        precomputed_result = corrected or ToolResult(success=True,tool='workspace',summary='Action correction',data={
                            'final':message,'context':'','results':[],'strategy':'WORKSPACE_ACTION_STATUS'})
                    elif prior and self.local_tools_skill:
                        original_task = await asyncio.to_thread(self.repository.get_task,prior['task_id'])
                        from local_tool_intent import directory_summary_request
                        if original_task and directory_summary_request(original_task.user_input):
                            inputs['_local_tool_request'] = original_task.user_input
                            inputs['_summary_exclude'] = [prior['path']]
                            prior_actions=await asyncio.to_thread(self.repository.actions,prior['task_id'])
                            previous_review=next((a.output.get('data',{}).get('document_verification')
                                for a in reversed(prior_actions) if a.action_type=='CREATE_DOCUMENT'
                                and a.output.get('data',{}).get('document_verification')),None)
                            if previous_review and previous_review.get('mode')!='DIRECTORY_SUMMARY':previous_review=None
                            if previous_review:
                                inputs['_summary_previous_review']=previous_review
                                inputs['_summary_exclude']=list(dict.fromkeys(previous_review.get('excluded',[])+[prior['path']]))
                                inputs['_summary_expand']=previous_review.get('complete') is True
                            # Restore the source scope, not the currently active output.
                            inputs['_workspace_current_directory'] = (previous_review.get('directory','') if previous_review
                                else str(PurePosixPath(prior['path']).parent))
                            if inputs['_workspace_current_directory']=='.':inputs['_workspace_current_directory']=''
                        else:
                            precomputed_result = ToolResult(success=True,tool='local_tools',summary='Document correction needed',data={
                                'final':f"Замечание относится к `{prior['path']}`. Укажите, какие части нужно раскрыть или дополнить.",
                                'context':'','results':[],'strategy':'LOCAL_TOOLS'})
                    else:
                        precomputed_result = ToolResult(success=True,tool='local_tools',summary='No prior document',data={
                            'final':'В этом чате нет сохранённого документа, который можно дополнить. Укажите нужный файл или задачу.',
                            'context':'','results':[],'strategy':'LOCAL_TOOLS'})
                explicit_target = display_file_target(request.question) or open_file_target(request.question)
                if references_created_file(request.question) and not explicit_target:
                    created = await asyncio.to_thread(self.repository.last_file_action,
                        session.id,request.workspace_id,created=True)
                    inputs['_workspace_created_reference'] = True
                    inputs['_workspace_active_file'] = created['path'] if created else ''
                if precomputed_result is None and asks_last_opened_file(request.question):
                    async with execution.action('DESCRIBE_FILE_ACTION') as action:
                        opened = await asyncio.to_thread(self.repository.last_file_action,session.id,request.workspace_id)
                        action.output = opened or {'path':None}
                    precomputed_result = ToolResult(success=True,tool='workspace',summary='Last file action',data={
                        'final':f"Последним был открыт файл `{opened['path']}`." if opened else
                            'В этом чате ещё нет успешного открытия файла из выбранной папки.',
                        'context':'','results':[],'strategy':'WORKSPACE_ACTION_STATUS'})
            from local_tool_intent import previous_answer_export, directory_summary_chat_request
            copy_answer = previous_answer_export(inputs.get('_local_tool_request',request.question))
            if precomputed_result is None and copy_answer and self.conversation_memory and request.workspace_id:
                if '_previous_answer' not in inputs:
                    inputs['_previous_answer'] = await asyncio.to_thread(
                        self.conversation_memory.literal_previous_answer,session.id,request.workspace_id)
            direct_context_free=bool(self.conversation_memory and request.workspace_id and (copy_answer or
                directory_summary_chat_request(request.question) or
                display_file_target(request.question) or open_file_target(request.question)
                or inputs.get('_workspace_active_file') and references_active_file(request.question)
                or self.local_tools_skill and self.local_tools_skill.match(request.question)))
            if self.history_loader and not request.graph_selection and precomputed_result is None and not direct_context_free:
                if self.conversation_memory:
                    view=await asyncio.to_thread(self.conversation_memory.prepare,session.id,request.workspace_id,
                        request.question,request.model.value,execution.progress)
                    history=view.history
                    inputs.setdefault('_previous_answer',view.previous_answer)
                    await asyncio.to_thread(self.repository.update_task_metadata,task,context_memory=view.info)
                    if view.info['compacted']:
                        execution.progress('Контекст сжат, оригиналы сохранены. '+
                            ('Продолжаю длительную задачу по шагам.' if policy.kind=='LONG' else 'Готовлю ответ.'))
                else:
                    history = await asyncio.to_thread(self.history_loader, session.id)
                inputs['_chat_history'] = history if self.conversation_memory else history[-6:]
                from conversation_context import archive_marker_explanation
                explanation = archive_marker_explanation(request.question,history)
                if explanation:
                    precomputed_result = ToolResult(success=True,tool='knowledge',summary='Explain internal marker',
                        data={'final':explanation,'context':'','strategy':'GENERAL_CHAT'})
                from conversation_context import asks_history_question
                if self.conversation_memory and asks_history_question(request.question):
                    inputs['_history_recall']=True
                    precomputed_result=await self.skills.get('knowledge').execute(inputs,execution)
            if (precomputed_result is None and self.repository_agent_skill and request.workspace_id
                    and not request.graph_selection):
                from repository_agent import repository_intent
                if repository_intent(request.question):
                    precomputed_result = await self.repository_agent_skill.execute(inputs, execution)
            if (precomputed_result is None and self.local_tools_skill and request.workspace_id
                    and not request.graph_selection):
                precomputed_result = await self.local_tools_skill.try_execute(inputs, execution)
            direct_workspace_command = bool(not request.graph_selection and self.workspace_skill and request.workspace_id
                and self.workspace_skill.has_direct_command(request.question))
            active_file_context = bool(inputs.get('_workspace_active_file') and references_active_file(request.question))
            named_workspace_query = ''
            if (precomputed_result is None and self.workspace_skill and request.workspace_id
                    and not direct_workspace_command and not request.graph_selection):
                await asyncio.to_thread(self.workspace_skill.filesystem.sync_index, request.workspace_id)
                named_workspace_query = await asyncio.to_thread(
                    self.workspace_skill.named_file_query, request.question, request.workspace_id)
            initial_route = route_by_rules(request.question)
            if (precomputed_result is None and resume_plan is None and resume_file_question is None
                    and not direct_workspace_command and not named_workspace_query and not active_file_context
                    and history and self.conversation_resolver
                    and not (initial_route and initial_route.strategy == QueryStrategy.GENERAL_CHAT)):
                try:
                    async with execution.action('CONTEXTUALIZE_QUERY') as action:
                        resolved = await asyncio.to_thread(
                            self.conversation_resolver.resolve,
                            request.question,
                            history,
                            request.model.value,
                        )
                        inputs['_retrieval_question'] = resolved.standalone_question
                        inputs['_depends_on_history'] = resolved.depends_on_history
                        contract = current_contract.get().resolved(resolved.standalone_question, resolved.referent)
                        current_contract.set(contract)
                        await asyncio.to_thread(self.repository.update_task_metadata, task, answer_contract=contract.as_dict())
                        action.output = {'depends_on_history': resolved.depends_on_history, 'topic':contract.topic}
                except Exception:
                    logger.warning('Conversation contextualization unavailable; using the original query')
            agent_plan = resume_plan
            # A failed resolver may not erase the named topic of this same
            # workspace/selection. Store names only, never old factual prose.
            contract = current_contract.get()
            if (precomputed_result is None and not contract.topic and contract.source not in {'CONVERSATION','HISTORY','LIBRARY','EXECUTION'}
                    and request.workspace_id and not direct_workspace_command and not named_workspace_query
                    and re.search(r'\b(?:он|она|оно|его|е[её]|ему|ней|н[её]м)\b',request.question,re.I)):
                topic = await asyncio.to_thread(self.repository.last_answer_topic,session.id,
                                               request.workspace_id,inputs.get('source_scope'))
                if topic:
                    standalone = re.sub(r'\b(?:он|она|оно|его|е[её]|ему|ней|н[её]м)\b',topic,request.question,flags=re.I)
                    contract = contract.resolved(standalone,topic)
                    current_contract.set(contract)
                    inputs['_retrieval_question'] = standalone
                    inputs['_depends_on_history'] = True
                    await asyncio.to_thread(self.repository.update_task_metadata,task,answer_contract=contract.as_dict())
            planning_text = inputs.get('_retrieval_question', request.question)
            if not request.graph_selection and self.workspace_skill and request.workspace_id and requests_unsupported_file_change(request.question):
                precomputed_result = self.workspace_skill.clarification_result(request.workspace_id, unsupported=True)
            obvious_route = route_by_rules(planning_text)
            if (agent_plan is None and precomputed_result is None and self.agent_planner
                    and resume_file_question is None and not pending
                    and not (is_single_information_request(request.question)
                              and is_single_information_request(planning_text))
                    and current_contract.get().source != 'WORKSPACE'
                    and not (self.workspace_skill and request.workspace_id
                             and (direct_workspace_command or named_workspace_query or active_file_context
                                  or self.workspace_skill.plan(request.question, planning_text)))
                    and not (obvious_route and obvious_route.strategy == QueryStrategy.GENERAL_CHAT)):
                try:
                    async with execution.action('BUILD_AGENT_PLAN', {'objective': planning_text}) as action:
                        decision = await asyncio.to_thread(
                            self.agent_planner.plan, planning_text, request.model.value,
                            [item for item in self.skills.catalog()
                             if item['name'] not in {'workspace', 'local_tools', 'repository_agent'} or request.workspace_id],
                            {'workspace_id': request.workspace_id},
                        )
                        action.output = {'should_plan': decision.should_plan,
                                         'confidence': decision.confidence,
                                         'reason': decision.reason}
                    if decision.should_plan and decision.confidence >= 0.65:
                        candidate = self.compile_agent_plan(decision.plan)
                        if any(step.skill == 'workspace' and not permits_workspace_operation(
                                step.operation, request.question) for step in candidate.steps):
                            logger.warning('Discarded a plan containing an unrequested workspace operation')
                        elif any(step.skill == 'workspace' and (
                                (step.operation == 'open_file' and not requests_file_open(request.question))
                                or (step.operation == 'compare_versions' and negates_comparison(request.question)))
                               for step in candidate.steps):
                            logger.warning('Discarded a plan that contradicts explicit action intent')
                        elif (self.workspace_skill and request.workspace_id
                              and any(step.skill == 'knowledge' for step in candidate.steps)
                              and not requests_document_library(request.question)):
                            # The tool router owns source selection. A planner
                            # cannot silently bypass an active folder; explicit
                            # library requests still route to DOCUMENT_LIBRARY.
                            logger.warning('Routing source selection before accepting a knowledge-only plan')
                        elif (self.workspace_skill and request.workspace_id
                              and callable(getattr(self.workspace_skill, 'plan', None))
                              and self.workspace_skill.plan(request.question, planning_text)
                              and all(step.skill == 'workspace' for step in candidate.steps)
                              and any(step.operation == 'execute' and not step.arguments.get('question')
                                      for step in candidate.steps)
                              and any(step.operation != 'execute' for step in candidate.steps)):
                            logger.warning('Using the typed workspace plan instead of repeating a whole workflow')
                        else:
                            agent_plan = candidate
                except Exception:
                    logger.warning('Agent planning unavailable; using existing routing')
            if (self.workspace_skill and request.workspace_id and agent_plan is None
                    and resume_file_question is None
                    and precomputed_result is None):
                routing_question = inputs.get('_retrieval_question', request.question)
                workspace_plan = self.workspace_skill.plan(request.question, routing_question)
                if named_workspace_query or self.workspace_skill.matches(request.question, routing_question):
                    skill = self.workspace_skill
                    inputs['_workspace_command_question'] = routing_question
                    if workspace_plan:
                        inputs['_workspace_plan'] = workspace_plan.model_dump(mode='json')
                    if (not display_file_target(request.question) and not requests_file_open(request.question)
                            and not requests_active_file_read(request.question)
                            and (named_workspace_query or self.workspace_skill.requires_context(routing_question))):
                        budget = await asyncio.to_thread(
                            self.workspace_skill.response_budget, routing_question, request.model.value,
                        )
                        inputs['_response_budget'] = budget
                        try:
                            matches = await asyncio.to_thread(
                                self.workspace_skill.relevant_context,
                                inputs.get('_retrieval_question', request.question),
                                request.workspace_id,
                                False,
                                budget.max_source_characters,
                                inputs.get('_workspace_active_file', ''),
                                source_scope=inputs.get('source_scope'),
                            )
                        except AssistantError as exc:
                            if exc.status_code not in {404, 409}:
                                raise
                            matches = []
                            inputs['_workspace_resolution_error'] = str(exc)
                        inputs["_workspace_matches"] = matches
                        inputs["_workspace_context_requested"] = True
                else:
                    route = None
                    try:
                        async with execution.action('ROUTE_TOOL', {'question': routing_question}) as action:
                            if obvious_route and obvious_route.strategy == QueryStrategy.GENERAL_CHAT:
                                route = WorkspaceRouteDecision(route=WorkspaceRoute.GENERAL_CHAT,
                                    confidence=1, reason='Разговорная реплика без файловых действий')
                            elif current_contract.get().source == 'WORKSPACE' or (is_single_information_request(request.question)
                                    and is_single_information_request(routing_question)
                                    and not requests_document_library(request.question)
                                    and not (obvious_route and obvious_route.strategy == QueryStrategy.GENERAL_CHAT)):
                                route = WorkspaceRouteDecision(route=WorkspaceRoute.WORKSPACE_CONTENT,
                                    confidence=1, reason='Информационный вопрос использует активную папку')
                            else:
                                route = await asyncio.to_thread(
                                    self.workspace_skill.route_ambiguous,
                                    routing_question,
                                    request.model.value,
                                )
                            action.output = route.model_dump(mode='json')
                    except Exception:
                        logger.warning('Workspace tool routing unavailable; using lexical fallback')
                    if (route and route.route.value == 'DOCUMENT_LIBRARY'
                            and not requests_document_library(request.question)):
                        route = None
                    if route and (not permits_workspace_operation(route.operation, request.question)
                                  or (route.route.value == 'WORKSPACE_METADATA'
                                      and not permits_workspace_operation('list_file_dates', request.question))):
                        route = route.model_copy(update={'route': WorkspaceRoute.WORKSPACE_CONTENT, 'operation': 'answer_question'})
                    if route and (route.operation != 'answer_question' or route.confidence < .65):
                        inputs['_workspace_semantic_decision'] = route.model_dump(mode='json')
                        inputs['_workspace_command_question'] = routing_question
                        skill = self.workspace_skill
                    elif route and route.route.value == 'WORKSPACE_METADATA':
                        inputs['_workspace_plan'] = self.workspace_skill.metadata_plan(routing_question).model_dump(mode='json')
                        inputs['_workspace_command_question'] = routing_question
                        skill = self.workspace_skill
                    elif not route or route.route.value != 'DOCUMENT_LIBRARY':
                        # A workspace is the default factual source. A mistaken
                        # KNOWLEDGE classification must not skip its document search.
                        conversational = bool(obvious_route and obvious_route.strategy == QueryStrategy.GENERAL_CHAT)
                        if not conversational and (not route or route.route.value in {'KNOWLEDGE', 'GENERAL_CHAT'}):
                            try:
                                decision = await asyncio.to_thread(
                                    self.skills.get('knowledge').services.route,
                                    routing_question, request.model.value,
                                )
                                conversational = decision.strategy == QueryStrategy.GENERAL_CHAT
                            except Exception:
                                logger.warning('Conversation routing unavailable; searching active workspace')
                        if not conversational:
                            budget = await asyncio.to_thread(
                                self.workspace_skill.response_budget, routing_question, request.model.value,
                            )
                            inputs['_response_budget'] = budget
                            try:
                                matches = await asyncio.to_thread(
                                    self.workspace_skill.relevant_context,
                                    routing_question, request.workspace_id, False,
                                    budget.max_source_characters,
                                    inputs.get('_workspace_active_file', ''),
                                    source_scope=inputs.get('source_scope'),
                                )
                            except AssistantError as exc:
                                if exc.status_code not in {404, 409}:
                                    raise
                                matches = []
                                inputs['_workspace_resolution_error'] = str(exc)
                            inputs['_workspace_matches'] = matches
                            inputs['_workspace_context_requested'] = True
                            skill = self.workspace_skill
            if precomputed_result is not None:
                result = precomputed_result
            elif resume_file_question is not None:
                resumed_inputs = dict(inputs)
                resumed_inputs['question'] = resume_file_question
                resumed_inputs['_workspace_command_question'] = resume_file_question
                result = await self.workspace_skill.execute(resumed_inputs, execution)
            elif agent_plan is not None:
                if len(agent_plan.steps)>1:
                    from task_policy import TaskPolicy
                    import settings
                    await execution.configure_policy(TaskPolicy('LONG',settings.LONG_TASK_SECONDS,'Несколько зависимых шагов плана'))
                    execution.progress('Выполняю многошаговую задачу; результаты шагов сохраняются.')
                result = await self.agent_executor.execute(
                    agent_plan, execution, inputs, resume_results, confirmed_steps,
                )
            elif inputs.get('_workspace_plan'):
                workspace_plan = inputs['_workspace_plan']
                operation_names = {
                    'FIND_LATEST_FILE': 'find_latest',
                    'COMPARE_FILE_VERSIONS': 'compare_versions',
                    'OPEN_FILE': 'open_file',
                    'READ_FILE': 'read_file',
                    'LIST_FILE_DATES': 'list_file_dates',
                }
                agent_steps = []
                for raw_step in workspace_plan['steps']:
                    arguments = {'target_query': workspace_plan['target_query']}
                    if raw_step['operation'] in {'OPEN_FILE', 'READ_FILE'}:
                        dependency = raw_step.get('depends_on', [])[-1]
                        arguments['path'] = f"${{{dependency}.data.selected_path}}"
                    agent_steps.append(AgentPlanStep(
                        id=raw_step['id'], skill='workspace',
                        operation=operation_names[raw_step['operation']],
                        arguments=arguments, depends_on=raw_step.get('depends_on', []),
                    ))
                result = await self.agent_executor.execute(AgentPlan(
                    objective=workspace_plan['objective'], steps=agent_steps,
                ), execution, inputs)
            else:
                result = await skill.execute(inputs, execution)
            if not result.success:
                raise AssistantError(result.error or 'Knowledge query failed')
            data = result.data
            if data.get('resume_state'):
                state = dict(data['resume_state'])
                state['pending_step'] = data.get('pending_step')
                state['source_scope'] = inputs.get('source_scope')
                await asyncio.to_thread(self.repository.save_pending_plan, session.id, state)
            elif data.get('pending_interaction'):
                await asyncio.to_thread(self.repository.save_pending_plan, session.id,
                    {**data['pending_interaction'], 'source_scope': inputs.get('source_scope')})
            elif pending or resume_plan is not None:
                await asyncio.to_thread(self.repository.clear_pending_plan, session.id)
            if request.workspace_id and self.workspace_skill:
                await asyncio.to_thread(self.workspace_skill.repository.bind_session,
                    session.id, request.workspace_id, data.get('current_directory'), data.get('active_file')
                )
            return await self._complete(request,session,task,execution,data,started)
        except BaseException as exc:
            from conversation_memory import MemoryBudgetExceeded
            if isinstance(exc,(GenerationDeadlineExceeded,MemoryBudgetExceeded)) and not scope.cancelled.is_set():
                actions=await asyncio.to_thread(self.repository.actions,task.id)
                paths=list(dict.fromkeys(a.output.get('data',{}).get('path') for a in actions
                    if a.status==TaskStatus.COMPLETED and a.output.get('data',{}).get('applied') is True))
                text=('Достигнут лимит времени задачи. Выполненные действия сохранены в журнале; '
                    'незавершённая работа не считается выполненной.' if isinstance(exc,GenerationDeadlineExceeded) else
                    'Не удалось безопасно вместить контекст диалога: '+str(exc)+' Полная история сохранена.')
                if any(paths):text+='\n\nУже сохранены файлы: '+', '.join('`'+p+'`' for p in paths if p)+'.'
                return await self._complete(request,session,task,execution,{'final':text,
                    'plan_status':'INCOMPLETE','strategy':'TASK_INCOMPLETE'},started)
            cancelled = isinstance(exc, asyncio.CancelledError)
            if task.status != TaskStatus.COMPLETED:
                await asyncio.to_thread(self.repository.transition, task, TaskStatus.CANCELLED if cancelled else TaskStatus.FAILED,
                                           error=short_error(exc))
            execution.emit(EventType.TASK_CANCELLED if cancelled else EventType.TASK_FAILED)
            if isinstance(exc, GenerationCancelled):
                raise AssistantError("Задача отменена", 409) from exc
            if cancelled or not isinstance(exc, Exception):
                get_llm_provider().cancel(task.id)
                raise
            from file_policy import FilePolicyError
            if isinstance(exc, FilePolicyError):
                raise AssistantError(str(exc), 403) from exc
            logger.exception('Assistant task %s failed', task.id)
            if isinstance(exc, AssistantError):
                raise
            if isinstance(exc, ProviderError):
                raise AssistantError(str(exc), 503) from exc
            raise AssistantError('Query failed. Check backend logs.') from exc
        finally:
            self.active_tasks.pop(task.id, None)
            current_scope.reset(token)
            workspace_scan_cache.reset(scan_token)
            current_contract.reset(contract_token)

    async def _complete(self,request,session,task,execution,data,started):
        from answer_contract import current_contract
        from answer_publication import publish_reviewed_answer
        data = dict(data)
        contract = current_contract.get()
        if data.get('verification') and contract:
            data['final'], data['verification'] = publish_reviewed_answer(data['final'],data['verification'])
        if contract and contract.allow_interpretation and data.get('results'):
            from answer_interpretation import append_interpretation
            data = await append_interpretation(data, contract, request.model.value, execution)
        # Validate before touching conversation history. This also commits a
        # recoverable timeout as an explicit incomplete result, not an HTTP crash.
        response=QueryResponse(answer=data['final'],session_id=session.id,model=request.model,
            context=data.get('context',''),draft=data.get('draft',''),edited=data.get('edited',''),
            enhanced=data.get('enhanced',[]),results=data.get('results',[]),strategy=data.get('strategy'),
            verification=data.get('verification'),graph=data.get('graph'),plan_status=data.get('plan_status'),
            pending_step=data.get('pending_step'),task_policy=task.metadata.get('execution_policy'),
            context_memory=task.metadata.get('context_memory'),source_scope=request.source_scope,
            answer_contract=contract.as_dict() if contract else None, interpretation=data.get('interpretation'))
        response.task_id=task.id
        async with execution.action('SAVE_HISTORY'):
            if self.response_saver:await asyncio.to_thread(self.response_saver,request,response,task)
            else:await asyncio.to_thread(self.save_history,session.id,request.question,response.answer,request.model.value)
        response.actions=await asyncio.to_thread(self.repository.actions,task.id)
        if task.status!=TaskStatus.COMPLETED:await asyncio.to_thread(self.repository.transition,task,TaskStatus.COMPLETED)
        execution.emit(EventType.TASK_COMPLETED)
        response.task_status=task.status
        response.events=execution.events
        return response,round((time.perf_counter()-started)*1000)
