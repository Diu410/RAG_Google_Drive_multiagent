"""Assistant entry point for text, voice transcripts and future event adapters."""
import asyncio
import logging
import re
import time
from contextlib import asynccontextmanager

from assistant_models import AssistantError, AssistantEvent, EventType, TaskStatus, ToolResult
from assistant_skills import AgentExecutor, SkillRegistry
from agent_planner import AgentPlan, AgentPlanStep, is_single_information_request
from model_runtime import get_llm_provider
from llm_contract import (ProviderError, GenerationCancelled, GenerationScope, current_scope, current_action)
from pydantic_models import QueryResponse
from query_router import QueryStrategy, route_by_rules
from command_intent import requests_file_open, requests_unsupported_file_change, negates_comparison
from source_grounding import requests_document_library
from workspace_intent import display_file_target, permits_workspace_operation, references_active_file
from workspace_router import WorkspaceRoute, WorkspaceRouteDecision

logger = logging.getLogger(__name__)


def short_error(exc):
    # Deliberately exclude arbitrary exception payloads (which may contain source
    # text, prompts or credentials). Full diagnostics stay in backend logs.
    return f'{type(exc).__name__}: execution failed'[:500]


class ExecutionContext:
    def __init__(self, repository, task, event_sink=None):
        self.repository, self.task = repository, task
        self.event_sink = event_sink
        self.events = []

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
            if scope and scope.cancelled.is_set() and self.task.status != TaskStatus.COMPLETED:
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
                 history_loader=None, conversation_resolver=None, agent_planner=None, response_saver=None):
        self.repository = repository
        self.active_tasks = {}
        self.deployment_lock = asyncio.Lock()
        self.skills = SkillRegistry()
        self.skills.register(knowledge_skill)
        if workspace_skill:
            self.skills.register(workspace_skill)
        self.workspace_skill = workspace_skill
        self.agent_executor = AgentExecutor(self.skills)
        self.save_history = save_history
        self.response_saver = response_saver
        self.history_loader = history_loader
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

    async def process_input(self, request, *, source='text', on_task_started=None):
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
                return await self._process_input(request, source=source, on_task_started=on_task_started)
        finally:
            entry[1] -= 1
            if not entry[1]:
                self._session_locks.pop(key, None)

    async def _process_input(self, request, *, source='text', on_task_started=None):
        started = time.perf_counter()
        async with self.deployment_lock:
            try:
                session = await asyncio.to_thread(self.repository.ensure_session, request.session_id, request.chat_id)
            except ValueError as exc:
                raise AssistantError(str(exc), 409) from exc
            task = await asyncio.to_thread(self.repository.create_task, session.id, request.question, {'source': source})
            execution = ExecutionContext(self.repository, task, self.event_sink)
            await asyncio.to_thread(self.repository.transition, task, TaskStatus.RUNNING)
            execution.emit(EventType.TASK_STARTED)
            from model_profiles import get_profile
            scope = GenerationScope(task.id, deadline=time.perf_counter() + get_profile(request.model).request_seconds)
            token = current_scope.set(scope)
            from workspace import workspace_scan_cache
            scan_token = workspace_scan_cache.set({})
            self.active_tasks[task.id] = scope
        try:
            if on_task_started is not None:
                on_task_started(task)
            inputs = request.model_dump()
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
            skill = self.skills.get('knowledge')
            history = []
            precomputed_result = None
            resume_plan = None
            resume_results = None
            resume_file_question = None
            confirmed_steps = set()
            pending = await asyncio.to_thread(self.repository.pending_plan, session.id)
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
                pending_workspace = pending.get('workspace_id') or (pending.get('base_inputs') or {}).get('workspace_id')
                if pending_workspace and pending_workspace != request.workspace_id:
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
            if self.history_loader and not request.graph_selection:
                history = await asyncio.to_thread(self.history_loader, session.id)
                inputs['_chat_history'] = history[-6:]
            direct_workspace_command = bool(not request.graph_selection and self.workspace_skill and request.workspace_id
                and self.workspace_skill.has_direct_command(request.question))
            active_file_context = bool(inputs.get('_workspace_active_file') and references_active_file(request.question))
            named_workspace_query = ''
            if self.workspace_skill and request.workspace_id and not direct_workspace_command and not request.graph_selection:
                await asyncio.to_thread(self.workspace_skill.filesystem.sync_index, request.workspace_id)
                named_workspace_query = await asyncio.to_thread(
                    self.workspace_skill.named_file_query, request.question, request.workspace_id)
            if (precomputed_result is None and resume_plan is None and resume_file_question is None
                    and not direct_workspace_command and not named_workspace_query and not active_file_context
                    and history and self.conversation_resolver):
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
                        action.output = {'depends_on_history': resolved.depends_on_history}
                except Exception:
                    logger.warning('Conversation contextualization unavailable; using the original query')
            agent_plan = resume_plan
            planning_text = inputs.get('_retrieval_question', request.question)
            if not request.graph_selection and self.workspace_skill and request.workspace_id and requests_unsupported_file_change(request.question):
                precomputed_result = self.workspace_skill.clarification_result(request.workspace_id, unsupported=True)
            obvious_route = route_by_rules(planning_text)
            if (agent_plan is None and precomputed_result is None and self.agent_planner
                    and resume_file_question is None and not pending
                    and not (is_single_information_request(request.question)
                             and is_single_information_request(planning_text))
                    and not (self.workspace_skill and request.workspace_id
                             and (direct_workspace_command or named_workspace_query or active_file_context
                                  or self.workspace_skill.plan(request.question, planning_text)))
                    and not (obvious_route and obvious_route.strategy == QueryStrategy.GENERAL_CHAT)):
                try:
                    async with execution.action('BUILD_AGENT_PLAN', {'objective': planning_text}) as action:
                        decision = await asyncio.to_thread(
                            self.agent_planner.plan, planning_text, request.model.value,
                            [item for item in self.skills.catalog()
                             if item['name'] != 'workspace' or request.workspace_id],
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
                    if not display_file_target(request.question) and (named_workspace_query or self.workspace_skill.requires_context(routing_question)):
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
                            if (is_single_information_request(request.question)
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
                        conversational = False
                        if not route or route.route.value in {'KNOWLEDGE', 'GENERAL_CHAT'}:
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
                await asyncio.to_thread(self.repository.save_pending_plan, session.id, state)
            elif data.get('pending_interaction'):
                await asyncio.to_thread(self.repository.save_pending_plan, session.id, data['pending_interaction'])
            elif pending or resume_plan is not None:
                await asyncio.to_thread(self.repository.clear_pending_plan, session.id)
            if request.workspace_id and self.workspace_skill:
                await asyncio.to_thread(self.workspace_skill.repository.bind_session,
                    session.id, request.workspace_id, data.get('current_directory'), data.get('active_file')
                )
            # Validate before touching conversation history.
            response = QueryResponse(answer=data['final'], session_id=session.id, model=request.model,
                context=data.get('context', ''), draft=data.get('draft', ''), edited=data.get('edited', ''),
                enhanced=data.get('enhanced', []), results=data.get('results', []),
                strategy=data.get('strategy'), verification=data.get('verification'), graph=data.get('graph'),
                plan_status=data.get('plan_status'), pending_step=data.get('pending_step'))
            response.task_id = task.id
            async with execution.action('SAVE_HISTORY'):
                if self.response_saver:
                    await asyncio.to_thread(self.response_saver, request, response, task)
                else:
                    await asyncio.to_thread(self.save_history, session.id, request.question, response.answer, request.model.value)
            response.actions = await asyncio.to_thread(self.repository.actions, task.id)
            if task.status != TaskStatus.COMPLETED:
                await asyncio.to_thread(self.repository.transition, task, TaskStatus.COMPLETED)
            execution.emit(EventType.TASK_COMPLETED)
            response.task_id = task.id
            response.task_status = task.status
            response.events = execution.events
            return response, round((time.perf_counter() - started) * 1000)
        except BaseException as exc:
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
