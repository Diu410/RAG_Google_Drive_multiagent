"""Repository agent effects, recovery, authority and actual child processes."""
import asyncio
import copy
from contextlib import closing
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import Mock

from support import UserEnvironment
from repository_agent import RepositoryAgentSkill, RepositoryStep, RepositoryCall, SourceChanges, repository_intent
from repository_process import ProjectChecks
from llm_contract import current_scope, GenerationScope, GenerationCancelled, GenerationDeadlineExceeded, ProviderError
from workspace_documents import WorkspaceDocuments


def step(tool, **values):
    return RepositoryStep(**{'tool': tool, 'path': '', 'text': '', 'replacement': '', 'check': '', 'note': '', **values})


class RepositoryAgentTests(unittest.TestCase):
    def setUp(self):
        self.env = UserEnvironment()
        self.addCleanup(self.env.close)
        (self.env.root/'calc.py').write_text('def add(a, b):\n    return a - b\n', encoding='utf-8')
        (self.env.root/'tests').mkdir()
        (self.env.root/'tests/test_calc.py').write_text(
            'import unittest\nfrom calc import add\nclass AdditionTests(unittest.TestCase):\n'
            '    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n', encoding='utf-8')
        self.steps, self.states = [], []
        self.agent = RepositoryAgentSkill(self.env.filesystem, decide=self.decide)
        self.env.core.repository_agent_skill = self.agent
        self.env.core.skills.register(self.agent)
        self.env.core.agent_planner = Mock()
        self.env.core.conversation_resolver = Mock()

    def decide(self, state, profile):
        self.states.append(copy.deepcopy(state))
        value = self.steps.pop(0)
        if isinstance(value, BaseException):
            raise value
        if callable(value):
            value = value()
        return value

    def ask(self, question='Исправь ошибку в коде проекта и запусти тесты'):
        return asyncio.run(self.env.ask(question))

    def test_real_failed_check_guides_a_repair_and_a_successful_rerun(self):
        self.steps = [step('read', path='calc.py'), step('check', check='python-tests'),
            step('edit', path='calc.py', text='return a - b', replacement='return a * b'),
            step('check', check='python-tests'), step('read', path='calc.py'),
            step('edit', path='calc.py', text='return a * b', replacement='return a + b'),
            step('check', check='python-tests'), step('finish', note='Исправлено сложение.')]
        response = self.ask()
        self.assertEqual(response.strategy, 'REPOSITORY_AGENT')
        self.assertEqual(response.plan_status, 'COMPLETED')
        self.assertIn('return a + b', (self.env.root/'calc.py').read_text())
        results = [a.output['data'] for a in response.actions if a.action_type == 'REPOSITORY_CHECK']
        self.assertFalse(results[0]['passed'])
        self.assertTrue(results[-1]['passed'])
        self.assertGreaterEqual(sum(not r['passed'] for r in results),2)
        self.assertIn('AssertionError', self.states[2]['observations'][-1]['result']['output'])
        self.assertEqual(len(list((self.env.root/'.fortuna-backups').glob('*'))), 2)
        self.env.core.agent_planner.plan.assert_not_called()
        self.env.core.conversation_resolver.resolve.assert_not_called()

    def test_a_passing_check_before_an_edit_does_not_verify_the_edit(self):
        (self.env.root/'calc.py').write_text('def add(a, b):\n    return a + b\n')
        self.steps = [step('check', check='python-tests'), step('read', path='calc.py'),
            step('edit', path='calc.py', text='return a + b', replacement='return a - b'), step('finish')]
        response = self.ask()
        self.assertEqual(response.plan_status, 'INCOMPLETE')
        self.assertIn('не пройдена', response.answer)

    def test_model_finish_cannot_invent_a_passed_check(self):
        self.steps = [step('finish', note='Все тесты прошли.')]
        response = self.ask()
        self.assertEqual(response.plan_status, 'INCOMPLETE')
        self.assertIn('Проверки не выполнены', response.answer)
        self.assertNotIn('Все тесты прошли', response.answer)

    def test_missing_catalog_check_is_an_observation_instead_of_shell_execution(self):
        self.steps = [step('check', check='python -c "print(1)"'), step('finish')]
        response = self.ask()
        self.assertEqual(response.plan_status, 'INCOMPLETE')
        self.assertIn('не разрешена', self.states[1]['observations'][-1]['error'])

    def test_read_only_analysis_cannot_write_or_run_commands(self):
        original = (self.env.root/'calc.py').read_bytes()
        self.steps = [step('edit', path='calc.py', text='return a - b', replacement='return a + b'),
            step('check', check='python-tests'), step('finish', note='Нужна правка сложения.')]
        response = self.ask('Проанализируй код проекта, только план без изменений')
        self.assertEqual((self.env.root/'calc.py').read_bytes(), original)
        self.assertEqual(self.states[0]['checks'], {})
        self.assertEqual(response.plan_status, 'INCOMPLETE')

    def test_user_can_explicitly_prohibit_execution_during_a_repair(self):
        self.steps = [step('read', path='calc.py'),
            step('edit', path='calc.py', text='return a - b', replacement='return a + b'), step('finish')]
        response = self.ask('Исправь ошибку в коде проекта, но не запускай тесты')
        self.assertIn('Проверки не запускались', response.answer)
        self.assertFalse(any(a.action_type == 'REPOSITORY_CHECK' for a in response.actions))

    def test_edit_requires_reading_current_bytes_and_reports_the_failure(self):
        self.steps = [step('edit', path='calc.py', text='return a - b', replacement='return a + b'), step('finish')]
        response = self.ask('Исправь код проекта без запуска тестов')
        self.assertEqual(response.plan_status, 'INCOMPLETE')
        self.assertIn('return a - b', (self.env.root/'calc.py').read_text())

    def test_external_edit_after_read_is_preserved(self):
        def changed():
            (self.env.root/'calc.py').write_text('USER_EXTERNAL_EDIT\n')
            return step('edit', path='calc.py', text='return a - b', replacement='return a + b')
        self.steps = [step('read', path='calc.py'), changed, step('finish')]
        response = self.ask('Исправь код проекта без запуска тестов')
        self.assertEqual(response.plan_status, 'INCOMPLETE')
        self.assertEqual((self.env.root/'calc.py').read_text(), 'USER_EXTERNAL_EDIT\n')

    def test_path_traversal_and_secret_paths_are_denied(self):
        self.steps = [step('write', path='../outside.py', text='print(1)'),
                      step('write', path='.env', text='SECRET=blocked'), step('finish')]
        self.ask()
        self.assertFalse((self.env.root.parent/'outside.py').exists())
        self.assertFalse((self.env.root/'.env').exists())

    def test_new_source_is_really_created_and_existing_sources_are_not_overwritten(self):
        self.steps = [step('write', path='new_module.py', text='VALUE = 6248\n'),
            step('write', path='calc.py', text='DO_NOT_OVERWRITE'), step('finish')]
        self.ask('Добавь функцию в код проекта без запуска тестов')
        self.assertEqual((self.env.root/'new_module.py').read_text(), 'VALUE = 6248\n')
        self.assertIn('return a - b', (self.env.root/'calc.py').read_text())

    def test_source_writes_do_not_expand_the_document_tool_policy(self):
        documents = WorkspaceDocuments(self.env.filesystem)
        with self.assertRaises(ValueError):
            documents.prepare_create(self.env.workspace.id, 'blocked.py', 'print(1)')

    def test_small_file_rewrite_uses_lines_and_preserves_line_endings(self):
        (self.env.root/'calc.py').write_bytes(b'def add(a, b):\r\n    return a - b\r\n')
        self.steps = [step('read',path='calc.py'),
            step('rewrite',path='calc.py',lines=['def add(a, b):','    return a + b']),step('finish')]
        response = self.ask('Исправь код проекта без запуска тестов')
        self.assertEqual((self.env.root/'calc.py').read_bytes(), b'def add(a, b):\r\n    return a + b\r\n')
        self.assertIn('calc.py',response.answer)

    def test_rewrite_cannot_replace_an_omitted_source(self):
        self.steps = [step('read',path='calc.py'),
            step('rewrite',path='calc.py',lines=['BAD_REPLACEMENT'],source_complete=False),step('finish')]
        response = self.ask('Исправь код проекта без запуска тестов')
        self.assertIn('return a - b',(self.env.root/'calc.py').read_text())
        self.assertEqual(response.plan_status,'INCOMPLETE')

    def test_rewrite_cannot_drop_a_python_function_and_can_recover(self):
        self.steps = [step('read',path='calc.py'),step('rewrite',path='calc.py',lines=['import math']),
            step('rewrite',path='calc.py',lines=['def add(a, b):','    return a + b']),step('finish')]
        response = self.ask()
        self.assertEqual(response.plan_status,'COMPLETED')
        self.assertIn('потеряла определения',self.states[2]['observations'][-1]['error'])
        self.assertIn('def add',(self.env.root/'calc.py').read_text())

    def test_external_edit_during_preparation_is_not_overwritten(self):
        original = self.agent.changes.prepare_edit
        def external_edit(*args, **kwargs):
            (self.env.root/'calc.py').write_text('USER_EDIT_DURING_PREPARATION\n')
            return original(*args, **kwargs)
        self.agent.changes.prepare_edit = external_edit
        self.steps = [step('read',path='calc.py'),
            step('edit',path='calc.py',text='return a - b',replacement='return a + b'),step('finish')]
        self.ask('Исправь код проекта без запуска тестов')
        self.assertEqual((self.env.root/'calc.py').read_text(),'USER_EDIT_DURING_PREPARATION\n')

    def test_tool_schema_does_not_accept_edit_fields_on_a_read_call(self):
        with self.assertRaises(ValueError):
            RepositoryCall.model_validate({'action':{'tool':'read','path':'calc.py','replacement':'new code'}})
        call = RepositoryCall.model_validate({'action':{'tool':'edit','path':'calc.py',
            'old_text':'return a - b','new_text':'return a + b'}})
        self.assertEqual(call.action.old_text,'return a - b')

    def test_preview_and_no_save_requests_do_not_authorize_source_writes(self):
        for question in ('Предпросмотр: исправь код проекта', 'Исправь код проекта, но не сохраняй',
                         'Fix the code, do not edit files'):
            intent = repository_intent(question)
            self.assertFalse(intent['write'], question)
            self.assertFalse(intent['check'], question)

    def test_repeating_an_identical_action_stops_before_the_step_limit(self):
        self.steps = [step('read', path='calc.py')]*8
        response = self.ask()
        self.assertIn('повторил действие', response.answer)
        self.assertEqual(len(self.states), 3)

    def test_budget_or_model_failure_preserves_applied_changes_and_their_journal(self):
        self.steps = [step('read', path='calc.py'),
            step('edit', path='calc.py', text='return a - b', replacement='return a + b'), ProviderError('offline')]
        response = self.ask()
        self.assertEqual(response.plan_status, 'INCOMPLETE')
        self.assertIn('calc.py', response.answer)
        self.assertIn('return a + b', (self.env.root/'calc.py').read_text())
        self.assertTrue(any(a.action_type == 'REPOSITORY_EDIT' for a in response.actions))

    def test_cancellation_after_atomic_write_keeps_the_committed_effect_in_the_journal(self):
        original = self.agent.changes.apply
        def apply_then_cancel(*args):
            result = original(*args)
            current_scope.get().cancelled.set()
            return result
        self.agent.changes.apply = apply_then_cancel
        self.steps = [step('read',path='calc.py'),
            step('edit',path='calc.py',text='return a - b',replacement='return a + b')]
        from assistant_models import AssistantError
        with self.assertRaises(AssistantError):
            self.ask()
        with closing(self.env.connection()) as conn:
            row = conn.execute("SELECT status,output FROM assistant_actions WHERE action_type='REPOSITORY_EDIT'").fetchone()
            self.assertEqual(row['status'],'COMPLETED')
            import json
            self.assertTrue(json.loads(row['output'])['data']['applied'])
        self.assertIn('return a + b',(self.env.root/'calc.py').read_text())

    def test_bundled_skill_and_project_rules_reach_the_tool_loop(self):
        (self.env.root/'AGENTS.md').write_text('PROJECT_RULE: preserve compatibility\n')
        self.steps = [step('skill', path='repository-maintenance'), step('finish')]
        self.ask('Проанализируй код проекта, только план')
        self.assertIn('PROJECT_RULE', self.states[0]['project_rules'])
        self.assertIn('Never change a test expectation', self.states[1]['observations'][-1]['result']['text'])

    def test_preserving_tests_does_not_turn_a_repair_into_global_read_only_mode(self):
        for question in ('Исправь код проекта. Тесты не меняй.', 'Почини проект, но не меняй тесты.'):
            intent = repository_intent(question)
            self.assertTrue(intent['write'], question)
            self.assertTrue(intent['check'], question)
            self.assertTrue(intent['preserve_tests'], question)
        self.steps = [step('read',path='tests/test_calc.py'),
            step('edit',path='tests/test_calc.py',text='5)',replacement='-1)'),step('finish')]
        original = (self.env.root/'tests/test_calc.py').read_bytes()
        self.ask('Исправь код проекта. Тесты не меняй.')
        self.assertEqual((self.env.root/'tests/test_calc.py').read_bytes(), original)

    def test_empty_unittest_discovery_is_not_success(self):
        (self.env.root/'tests/test_calc.py').unlink()
        result = self.agent.checks.run(self.env.workspace.id, 'python-tests')
        self.assertFalse(result['passed'])
        result = self.agent.checks._run(self.env.root,
            [sys.executable, '-c', 'print("Ran 0 tests in 0.000s")'], 'python-tests')
        self.assertEqual(result['exit_code'], 0)
        self.assertFalse(result['passed'])

    def test_invalid_package_metadata_does_not_abort_the_agent_before_it_can_read_sources(self):
        (self.env.root/'package.json').write_text('{BROKEN_JSON')
        self.steps = [step('read',path='package.json'),step('finish')]
        response = self.ask()
        self.assertEqual(response.strategy,'REPOSITORY_AGENT')
        self.assertEqual(response.plan_status,'INCOMPLETE')
        self.assertTrue(self.states[0]['check_discovery_errors'])

    def test_plain_pytest_functions_are_not_mistaken_for_unittest(self):
        (self.env.root/'tests/test_calc.py').write_text('def test_add():\n    assert 1 == 1\n')
        checks = self.agent.checks.discover(self.env.workspace.id)
        self.assertIn('pytest', checks['python-tests']['argv'])

    def test_timed_out_process_is_stopped_and_not_reported_passed(self):
        runner = ProjectChecks(self.env.filesystem, timeout=.2)
        result = runner._run(self.env.root, [sys.executable, '-c', 'import time;time.sleep(10)'], 'slow')
        self.assertTrue(result['timed_out'])
        self.assertFalse(result['passed'])
        self.assertLess(result['seconds'], 4)

    def test_cancellation_and_deadline_interrupt_a_running_process(self):
        for cancelled in (True, False):
            scope = GenerationScope('test', deadline=None if cancelled else time.perf_counter()+.2)
            token = current_scope.set(scope)
            timer = threading.Timer(.2, scope.cancelled.set) if cancelled else None
            if timer: timer.start()
            try:
                with self.assertRaises(GenerationCancelled if cancelled else GenerationDeadlineExceeded):
                    self.agent.checks._run(self.env.root, [sys.executable, '-c', 'import time;time.sleep(10)'], 'slow')
            finally:
                if timer: timer.join()
                current_scope.reset(token)

    def test_process_output_is_bounded_and_nonzero_exit_is_preserved(self):
        runner = ProjectChecks(self.env.filesystem, output_limit=1024)
        result = runner._run(self.env.root, [sys.executable, '-c', 'print("x"*100000);raise SystemExit(7)'], 'large')
        self.assertTrue(result['truncated'])
        self.assertEqual(result['exit_code'], 7)
        self.assertLessEqual(len(result['output']), 1024)
        self.assertFalse(result['passed'])

    def test_natural_task_intent_excludes_questions_quotes_negation_and_document_requests(self):
        for question in ('Как исправить ошибку в коде?', '«Исправь код проекта»',
                         'Не исправляй код проекта', 'Кто исправил код проекта?',
                         'Открой созданный документ', 'Создай документ с описанием проекта'):
            self.assertIsNone(repository_intent(question), question)
        for question in ('Исправь документ с описанием проекта', 'Проанализируй документы проекта',
                         'Исправь код проекта и удали файл old.py'):
            self.assertIsNone(repository_intent(question),question)
        for question in ('В проекте падают тесты, разбери и почини', 'Fix the bug in calc.py',
                         'Можешь исправить код проекта?'):
            self.assertIsNotNone(repository_intent(question), question)


if __name__ == '__main__':
    unittest.main()
