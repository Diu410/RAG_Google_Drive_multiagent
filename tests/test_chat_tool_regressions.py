"""The remaining real chat exposed a planner and router agreeing on wrong tools."""
import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from support import UserEnvironment
from agent_planner import AgentPlan, AgentPlanDecision, AgentPlanStep
from workspace_router import WorkspaceRouteDecision
from assistant_models import AssistantError, ToolResult


class ChatToolRegressionTests(unittest.TestCase):
    def setUp(self):
        self.env = UserEnvironment()
        self.addCleanup(self.env.close)

    def forbid_model_routing(self):
        self.env.core.agent_planner = SimpleNamespace(plan=lambda *args: self.fail('Unexpected model planner'))
        self.env.core.conversation_resolver = SimpleNamespace(resolve=lambda *args: self.fail('Explicit target rewritten'))
        self.env.skill.route_ambiguous = lambda *args: self.fail('Unexpected model tool router')

    def test_actual_chat_sequence_reads_named_note_without_dates_or_listing(self):
        async def run():
            self.forbid_model_routing()
            first = await self.env.ask('Что можешь рассказать о директории?')
            response = await self.env.ask('что внутри без названия 1?', first.session_id)
            self.assertIn('NOTE_ONE', response.answer)
            self.assertEqual([a.action_type for a in response.actions],
                             ['RESOLVE_FILE_REFERENCE', 'READ_FILE', 'SAVE_HISTORY'])
            self.assertEqual(self.env.repository.session_file(first.session_id, self.env.workspace.id), 'Без названия 1.md')
        asyncio.run(run())

    def test_content_variants_use_file_operation_even_when_the_word_file_matches_listing(self):
        self.forbid_model_routing()
        for question in ('что внутри без названия 1?', 'Что в Без названия 1?',
                         'Что написано в «Без названия 1.md»?', 'Покажи содержимое файла Без названия 1.md',
                         'Покажи текст Без названия 1', 'Прочитай «Без названия 1.md»',
                         'Что внутри файла Без названия 1?', 'what is inside Без названия 1?',
                         'что внутри Без  названия 1?', 'Что внутри План.md в папке Проект Б?'):
            with self.subTest(question=question):
                response = asyncio.run(self.env.ask(question))
                self.assertIn('PROJECT_B' if 'План' in question else 'NOTE_ONE', response.answer)
                actions = [a.action_type for a in response.actions]
                self.assertIn('READ_FILE', actions)
                self.assertFalse(any('LIST_' in a or 'PLAN' in a for a in actions))
                self.assertEqual(self.env.opened, [])

    def test_missing_numeric_content_target_never_reads_unnumbered_or_neighbor(self):
        self.forbid_model_routing()
        for number in (6, 10, 120):
            response = asyncio.run(self.env.ask(f'Что внутри Без названия {number}?'))
            self.assertIn('не найден', response.answer)
            self.assertNotIn('READ_FILE', [a.action_type for a in response.actions])
            self.assertNotIn('ROOT_ZERO', response.answer)

    def test_bare_filename_open_means_system_open_and_sets_active_file(self):
        self.forbid_model_routing()
        for question in ('Открой Без названия 1', 'Можешь открыть «Без названия 1.md»?'):
            with self.subTest(question=question):
                count = len(self.env.opened)
                response = asyncio.run(self.env.ask(question))
                self.assertEqual(len(self.env.opened) - count, 1)
                self.assertIn('OPEN_FILE', [a.action_type for a in response.actions])
                self.assertNotIn('READ_FILE', [a.action_type for a in response.actions])

    def test_duplicate_content_target_requires_choice_then_reads_selected_file(self):
        async def run():
            self.forbid_model_routing()
            first = await self.env.ask('Что внутри План.md?')
            self.assertEqual(first.plan_status, 'WAITING_FOR_USER')
            second = await self.env.ask('2', first.session_id)
            self.assertIn('PROJECT_B', second.answer)
            self.assertNotIn('PROJECT_A', second.answer)
            self.assertEqual(self.env.opened, [])
        asyncio.run(run())

    def test_known_folder_without_the_word_folder_is_listed(self):
        self.forbid_model_routing()
        response = asyncio.run(self.env.ask('Что внутри Архив?'))
        self.assertIn('Хранитель.md', response.answer)
        self.assertNotIn('Без названия 1.md', response.answer)
        self.assertNotIn('READ_FILE', [a.action_type for a in response.actions])
        self.assertEqual(self.env.repository.session_directory(response.session_id, self.env.workspace.id), 'Архив')

    def test_ordinary_subject_is_not_turned_into_a_filename(self):
        self.forbid_model_routing()
        response = asyncio.run(self.env.ask('Что внутри Вселенной?'))
        self.assertFalse(any(a.action_type in ('READ_FILE', 'RESOLVE_FILE_REFERENCE', 'LIST_DIRECTORY') for a in response.actions))
        self.assertNotIn('Файл `Вселен', response.answer)

    def test_explicit_title_overrides_active_file_for_explanation(self):
        self.env.filesystem.sync_index(self.env.workspace.id)
        for question in ('Объясни Без названия 11.md', 'Перескажи содержание Без названия 11'):
            matches = self.env.skill.relevant_context(question, self.env.workspace.id, active_file='Без названия 4.md')
            self.assertEqual([m['path'] for m in matches], ['Без названия 11.md'])

    def test_explicit_active_file_reference_does_not_reach_model_rewriters(self):
        from support import NoModel
        async def run():
            first = await self.env.ask('Прочитай Без названия 11.md')
            self.forbid_model_routing()
            model = NoModel()
            model.generate = lambda *args, **kwargs: SimpleNamespace(text='Код в этом файле: 4817.')
            with patch('model_runtime.get_llm_provider', return_value=model):
                response = await self.env.ask('Что в этом файле?', first.session_id)
            self.assertEqual({s.metadata.get('source') for s in response.results}, {'Без названия 11.md'})
            self.assertFalse(any(a.action_type in ('BUILD_AGENT_PLAN', 'CONTEXTUALIZE_QUERY', 'ROUTE_WORKSPACE', 'ROUTE_TOOL')
                                 for a in response.actions))
        asyncio.run(run())

    def test_deleted_active_file_is_reported_instead_of_replaced(self):
        self.env.filesystem.sync_index(self.env.workspace.id)
        (self.env.root / 'Без названия 11.md').unlink()
        with self.assertRaises(AssistantError) as error:
            self.env.skill.relevant_context('Что в этом файле?', self.env.workspace.id, active_file='Без названия 11.md')
        self.assertEqual(error.exception.status_code, 404)

    def test_plain_question_cannot_gain_metadata_step_from_planner(self):
        plan = AgentPlan(objective='Who?', steps=[
            AgentPlanStep(id='dates', skill='workspace', operation='list_file_dates'),
            AgentPlanStep(id='answer', skill='workspace', arguments={'question': 'Как зовут хранителя?'})])
        self.env.core.agent_planner = SimpleNamespace(plan=lambda *args: AgentPlanDecision(
            should_plan=True, confidence=1, plan=plan))
        result = ToolResult(success=True, tool='workspace', summary='ok', data={'final': 'WORKSPACE_ANSWER'})
        with patch.object(self.env.skill, 'execute', return_value=result):
            response = asyncio.run(self.env.ask('Как зовут хранителя северного архива?'))
        self.assertNotIn('LIST_FILE_DATES', [a.action_type for a in response.actions])
        self.assertNotIn('PLAN_AGENT', [a.action_type for a in response.actions])

    def test_incorrect_semantic_metadata_classification_falls_back_to_content(self):
        decision = WorkspaceRouteDecision(route='WORKSPACE_METADATA', operation='list_file_dates', confidence=1, reason='wrong')
        result = ToolResult(success=True, tool='workspace', summary='ok', data={'final': 'ANSWER'})
        with patch.object(self.env.skill, 'route_ambiguous', return_value=decision), \
             patch.object(self.env.skill, 'execute', return_value=result):
            response = asyncio.run(self.env.ask('Как зовут хранителя северного архива?'))
        self.assertNotIn('LIST_FILE_DATES', [a.action_type for a in response.actions])

    def test_granular_metadata_operation_checks_original_not_rewritten_question(self):
        with self.assertRaises(AssistantError):
            asyncio.run(self.env.execute('Что внутри заметки?', 'list_file_dates', target_query='Покажи системные даты файлов'))

    def test_quoted_metadata_command_does_not_authorize_metadata_plan(self):
        from workspace_planner import WorkspacePlanner
        self.assertIsNone(WorkspacePlanner().plan('Объясни фразу «покажи системные даты файлов»'))

    def test_metadata_in_execute_subtask_cannot_bypass_original_intent(self):
        with self.assertRaises(AssistantError):
            asyncio.run(self.env.execute('Как зовут хранителя?', _workspace_command_question='Покажи системные даты файлов'))

    def test_step_question_is_not_overridden_by_cached_objective_routing(self):
        from assistant_core import ExecutionContext
        question = 'Покажи системные даты файлов и прочитай Без названия 1'
        session = self.env.assistant.ensure_session()
        task = self.env.assistant.create_task(session.id, question)
        plan = AgentPlan(objective=question, steps=[
            AgentPlanStep(id='dates', skill='workspace', operation='list_file_dates'),
            AgentPlanStep(id='read', skill='workspace', arguments={'question': 'Покажи содержимое файла Без названия 1'},
                          depends_on=['dates'])])
        result = asyncio.run(self.env.core.agent_executor.execute(plan, ExecutionContext(self.env.assistant, task),
            {'question': question, 'model': 'FAST', 'workspace_id': self.env.workspace.id,
             '_retrieval_question': question, '_workspace_command_question': question,
             '_workspace_semantic_decision': {'route': 'WORKSPACE_CONTENT', 'operation': 'list_directory',
                                             'confidence': 1, 'reason': 'wrong'},
             '_workspace_matches': [{'path': 'wrong.md', 'content': 'STALE_RESULT', 'score': 1}]}))
        self.assertIn('NOTE_ONE', result.data['final'])
        self.assertNotIn('STALE_RESULT', result.data['final'])
        actions = self.env.assistant.actions(task.id)
        self.assertEqual(sum(a.action_type == 'LIST_FILE_DATES' for a in actions), 1)
        self.assertEqual(sum(a.action_type == 'READ_FILE' for a in actions), 1)

    def test_last_metadata_step_preserves_the_document_read_earlier(self):
        from assistant_core import ExecutionContext
        question = 'Прочитай Без названия 1 и покажи системные даты файлов'
        session = self.env.assistant.ensure_session()
        task = self.env.assistant.create_task(session.id, question)
        plan = AgentPlan(objective=question, steps=[
            AgentPlanStep(id='read', skill='workspace', arguments={'question': 'Прочитай Без названия 1'}),
            AgentPlanStep(id='dates', skill='workspace', operation='list_file_dates', depends_on=['read'])])
        result = asyncio.run(self.env.core.agent_executor.execute(plan, ExecutionContext(self.env.assistant, task),
            {'question': question, 'model': 'FAST', 'workspace_id': self.env.workspace.id}))
        self.assertEqual(result.data['active_file'], 'Без названия 1.md')
        self.assertEqual(result.data['current_directory'], '')
