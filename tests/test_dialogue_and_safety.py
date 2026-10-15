"""End-user conversations, boundaries and failures with controlled inference."""
import asyncio
from contextlib import closing
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from support import UserEnvironment
from assistant_models import AssistantError, TaskStatus, EventType
from assistant_core import ExecutionContext
from pydantic_models import QueryInput, SearchRequest, ReindexRequest
from pydantic import ValidationError
from llm_contract import GenerationScope, GenerationCancelled, current_scope


class DialogueTests(unittest.TestCase):
    def setUp(self):
        self.env = UserEnvironment()
        self.addCleanup(self.env.close)

    def test_duplicate_choice_reads_selected_document(self):
        async def scenario():
            first = await self.env.ask('Прочитай План.md')
            self.assertEqual(first.plan_status, 'WAITING_FOR_USER')
            self.assertIn('Проект Б', first.answer)
            second = await self.env.ask('2', first.session_id)
            self.assertIn('PROJECT_B', second.answer)
            self.assertNotIn('PROJECT_A', second.answer)
            self.assertIsNone(self.env.assistant.pending_plan(first.session_id))
        asyncio.run(scenario())

    def test_open_choice_preserves_original_authorization(self):
        async def scenario():
            first = await self.env.ask('Открой План.md')
            self.assertEqual(self.env.opened, [])
            second = await self.env.ask('2', first.session_id)
            self.assertEqual(len(self.env.opened), 1)
            self.assertTrue(self.env.opened[0].endswith('План.md'))
            self.assertIn('открыт', second.answer)
            self.assertEqual(self.env.repository.session_file(first.session_id, self.env.workspace.id), 'Проект Б/План.md')
        asyncio.run(scenario())

    def test_selection_never_adds_open_authorization_to_read_request(self):
        async def scenario():
            first = await self.env.ask('Прочитай План.md')
            await self.env.ask('2', first.session_id)
            self.assertEqual(self.env.opened, [])
        asyncio.run(scenario())

    def test_workspace_switch_discards_pending_choices_from_previous_folder(self):
        async def scenario():
            first = await self.env.ask('Прочитай План.md')
            other = self.env.directory / 'Второй проект'
            other.mkdir()
            (other / 'другой.md').write_text('NEW_WORKSPACE', encoding='utf-8')
            workspace = self.env.repository.select(str(other))
            response = await self.env.ask('Покажи список файлов', first.session_id, workspace.id)
            self.assertIsNone(self.env.assistant.pending_plan(first.session_id))
            self.assertIn('другой.md', response.answer)
            self.assertNotIn('Проект А', response.answer)
        asyncio.run(scenario())

    def test_open_sets_active_document_for_followup(self):
        async def scenario():
            first = await self.env.ask('Открой Без названия 11.md')
            self.assertEqual(len(self.env.opened), 1)
            self.assertEqual(self.env.repository.session_file(first.session_id, self.env.workspace.id),
                             'Без названия 11.md')
            # Retrieval itself must include the active source, independently of generation.
            context = self.env.skill.relevant_context('Что в этом файле?', self.env.workspace.id,
                active_file=self.env.repository.session_file(first.session_id, self.env.workspace.id))
            self.assertIn('4817', str(context))
        asyncio.run(scenario())

    def test_two_chats_have_independent_active_files(self):
        async def scenario():
            first = await self.env.ask('Прочитай Без названия 4.md')
            second = await self.env.ask('Прочитай Без названия 11.md')
            self.assertNotEqual(first.session_id, second.session_id)
            self.assertEqual(self.env.repository.session_file(first.session_id, self.env.workspace.id),
                             'Без названия 4.md')
            self.assertEqual(self.env.repository.session_file(second.session_id, self.env.workspace.id),
                             'Без названия 11.md')
        asyncio.run(scenario())

    def test_workspace_switch_clears_previous_file_and_directory(self):
        async def scenario():
            first = await self.env.ask('Прочитай Без названия 4.md')
            other = self.env.directory / 'Другой проект'
            other.mkdir()
            (other / 'Без названия 4.md').write_text('SECOND_WORKSPACE_ONLY', encoding='utf-8')
            selected = self.env.repository.select(str(other))
            second = await self.env.ask('Прочитай Без названия 4.md', first.session_id, selected.id)
            self.assertIn('SECOND_WORKSPACE_ONLY', second.answer)
            self.assertNotIn('NOTE_FOUR', second.answer)
            self.assertEqual(self.env.repository.session_file(first.session_id, self.env.workspace.id), '')
        asyncio.run(scenario())

    def test_failed_open_reports_failure_without_success_event(self):
        async def scenario():
            with patch.object(self.env.filesystem, 'opener', side_effect=OSError('PRIVATE_SOURCE_TEXT')):
                with self.assertRaises(AssistantError) as caught:
                    await self.env.ask('Открой Без названия 4.md')
            self.assertNotIn('PRIVATE_SOURCE_TEXT', str(caught.exception))
            with closing(self.env.connection()) as conn:
                task = conn.execute('SELECT * FROM assistant_tasks ORDER BY rowid DESC LIMIT 1').fetchone()
            self.assertEqual(task['status'], 'FAILED')
            actions = self.env.assistant.actions(task['id'])
            opened = [a for a in actions if a.action_type == 'OPEN_FILE']
            self.assertEqual(opened[0].status, TaskStatus.FAILED)
        asyncio.run(scenario())

    def test_success_response_agrees_with_journal(self):
        response = asyncio.run(self.env.ask('Прочитай Без названия 4.md'))
        self.assertEqual(response.task_status, TaskStatus.COMPLETED)
        self.assertEqual(self.env.assistant.get_task(response.task_id).status, TaskStatus.COMPLETED)
        self.assertEqual(response.events[-1].type, EventType.TASK_COMPLETED)
        self.assertTrue(all(a.status == TaskStatus.COMPLETED for a in response.actions))
        self.assertEqual(len(self.env.history[response.session_id]), 2)

    def test_two_simultaneous_turns_in_same_chat_are_serialized(self):
        async def scenario():
            session = self.env.assistant.ensure_session()
            responses = await asyncio.gather(
                self.env.ask('Прочитай Без названия 4.md', session.id),
                self.env.ask('Прочитай Без названия 11.md', session.id))
            self.assertIn('NOTE_FOUR', responses[0].answer)
            self.assertIn('NOTE_ELEVEN', responses[1].answer)
            self.assertEqual(len(self.env.history[session.id]), 4)
            self.assertEqual(self.env.core._session_locks, {})
            self.assertEqual(self.env.core.active_tasks, {})
        asyncio.run(scenario())

    def test_cancellation_is_recorded_and_not_saved_as_answer(self):
        async def scenario():
            started = asyncio.Event()
            release = asyncio.Event()
            task_id = []
            async def waiting(inputs, execution):
                started.set()
                await release.wait()
                if current_scope.get().cancelled.is_set():
                    raise GenerationCancelled()
                self.fail('Cancellation did not reach the execution scope')
            with patch.object(self.env.knowledge, 'execute', waiting):
                runner = asyncio.create_task(self.env.core.process_input(QueryInput(question='Привет'),
                    on_task_started=lambda task: task_id.append(task.id)))
                await asyncio.wait_for(started.wait(), 3)
                self.assertTrue(self.env.core.cancel_task(task_id[0]))
                release.set()
                with self.assertRaises(AssistantError) as error:
                    await runner
                self.assertEqual(error.exception.status_code, 409)
            self.assertEqual(self.env.assistant.get_task(task_id[0]).status, TaskStatus.CANCELLED)
            self.assertEqual(self.env.history, {})
            self.assertEqual(self.env.core.active_tasks, {})
        asyncio.run(scenario())

    def test_missing_file_does_not_replace_active_document(self):
        async def scenario():
            first = await self.env.ask('Прочитай Без названия 4.md')
            missing = await self.env.ask('Прочитай Без названия 120.md', first.session_id)
            self.assertIn('не найден', missing.answer)
            self.assertEqual(self.env.repository.session_file(first.session_id, self.env.workspace.id),
                             'Без названия 4.md')
        asyncio.run(scenario())

    def test_directory_navigation_is_per_session(self):
        async def scenario():
            first = await self.env.ask('Покажи содержимое папки Архив')
            other = await self.env.ask('Что находится внутри текущей директории?')
            self.assertEqual(self.env.repository.session_directory(first.session_id, self.env.workspace.id), 'Архив')
            self.assertEqual(self.env.repository.session_directory(other.session_id, self.env.workspace.id), '')
            self.assertIn('Без названия', other.answer)
        asyncio.run(scenario())

    def test_open_folder_is_navigation_not_a_question_about_its_files(self):
        response = asyncio.run(self.env.ask('Открой папку Архив'))
        self.assertEqual(self.env.repository.session_directory(response.session_id, self.env.workspace.id), 'Архив')
        self.assertIn('Хранитель.md', response.answer)
        self.assertNotIn('READ_FILE', [a.action_type for a in response.actions])
        self.assertEqual(self.env.opened, [])

    def test_find_latest_and_read_performs_both_operations(self):
        response = asyncio.run(self.env.ask('Найди последнюю версию отчёта и прочитай её'))
        self.assertIn('87%', response.answer)
        self.assertIn('Отчёт 11.md', response.answer)
        self.assertIn('FIND_LATEST_FILE', [a.action_type for a in response.actions])
        self.assertIn('READ_FILE', [a.action_type for a in response.actions])
        self.assertEqual(self.env.opened, [])

    def test_explicit_comparison_turn_uses_only_requested_versions(self):
        response = asyncio.run(self.env.ask('Сравни версии отчёта 2 и 9'))
        self.assertIn('42%', response.answer)
        self.assertIn('64%', response.answer)
        self.assertNotIn('87%', response.answer)
        self.assertNotIn('FIND_LATEST_FILE', [a.action_type for a in response.actions])

    def test_explicit_file_command_cannot_be_replaced_by_model_directory_plan(self):
        planner = SimpleNamespace(plan=lambda *args: self.fail('Explicit commands must not reach the LLM planner'))
        self.env.core.agent_planner = planner
        response = asyncio.run(self.env.ask('Открой без названия6'))
        self.assertIn('не найден', response.answer)
        self.assertNotIn('LIST_DIRECTORY', [a.action_type for a in response.actions])
        self.assertEqual(self.env.opened, [])

    def test_active_file_is_used_for_time_and_content_followups(self):
        for question in ('Что в этом файле?', 'Какое время указано в этом файле?', 'Перескажи его содержание'):
            with self.subTest(question=question):
                context = self.env.skill.relevant_context(question, self.env.workspace.id,
                    active_file='Без названия 4.md')
                self.assertEqual([c['path'] for c in context], ['Без названия 4.md'])
                self.assertIn('09:45', context[0]['content'])


class AccessBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.env = UserEnvironment()
        self.addCleanup(self.env.close)

    def test_paths_cannot_escape_workspace(self):
        outside = self.env.directory / 'private.txt'
        outside.write_text('PRIVATE_OUTSIDE', encoding='utf-8')
        for path in ('../private.txt', '..\\private.txt', str(outside), '../../private.txt'):
            with self.subTest(path=path), self.assertRaises((PermissionError, ValueError, FileNotFoundError)):
                self.env.filesystem.read_text_file(self.env.workspace.id, path)

    def test_junction_component_cannot_be_entered_or_indexed(self):
        original = Path.is_junction
        junction = self.env.root / 'Архив'
        with patch.object(Path, 'is_junction', lambda path: path == junction or original(path)):
            with self.assertRaises(PermissionError):
                self.env.filesystem.read_text_file(self.env.workspace.id, 'Архив/Хранитель.md')
            self.env.filesystem.sync_index(self.env.workspace.id)
            self.assertFalse(any(f.relative_path.startswith('Архив/') for f in
                                 self.env.repository.indexed_files(self.env.workspace.id)))

    def test_symlink_component_cannot_be_entered_or_indexed(self):
        original = Path.is_symlink
        link = self.env.root / 'Архив'
        with patch.object(Path, 'is_symlink', lambda path: path == link or original(path)):
            with self.assertRaises(PermissionError):
                self.env.filesystem.read_text_file(self.env.workspace.id, 'Архив/Хранитель.md')
            self.env.filesystem.sync_index(self.env.workspace.id)
            self.assertFalse(any(f.relative_path.startswith('Архив/') for f in
                                 self.env.repository.indexed_files(self.env.workspace.id)))

    def test_directory_listing_cannot_escape_workspace(self):
        with self.assertRaises((PermissionError, ValueError)):
            self.env.filesystem.list_directory(self.env.workspace.id, '..')

    def test_large_file_has_explicit_error(self):
        self.env.filesystem.max_text_bytes = 20
        with self.assertRaisesRegex(ValueError, 'размер'):
            self.env.filesystem.read_text_file(self.env.workspace.id, 'Без названия 4.md')

    def test_unknown_extension_is_not_read_as_document(self):
        (self.env.root / 'program.exe').write_bytes(b'MZPRIVATE')
        with self.assertRaisesRegex(ValueError, 'Формат'):
            self.env.filesystem.read_text_file(self.env.workspace.id, 'program.exe')

    def test_empty_file_is_readable_without_substituting_another_source(self):
        _, path, content = self.env.filesystem.read_text_file(self.env.workspace.id, 'пустой.md')
        self.assertEqual(path, 'пустой.md')
        self.assertEqual(content, '')

    def test_broken_pdf_produces_error_not_unrelated_text(self):
        (self.env.root / 'broken.pdf').write_bytes(b'not a PDF')
        with self.assertRaises(Exception):
            self.env.filesystem.read_text_file(self.env.workspace.id, 'broken.pdf')

    def test_deleted_source_is_removed_from_metadata_index(self):
        self.env.filesystem.sync_index(self.env.workspace.id)
        (self.env.root / 'Без названия 4.md').unlink()
        self.env.filesystem.sync_index(self.env.workspace.id)
        self.assertNotIn('Без названия 4.md',
                         [f.relative_path for f in self.env.repository.indexed_files(self.env.workspace.id)])

    def test_changed_source_refreshes_content_hash(self):
        self.env.filesystem.sync_index(self.env.workspace.id)
        before = {f.relative_path: f.content_hash for f in self.env.repository.indexed_files(self.env.workspace.id)}
        (self.env.root / 'Без названия 4.md').write_text('UPDATED_FACT_790', encoding='utf-8')
        self.env.filesystem.sync_index(self.env.workspace.id)
        after = {f.relative_path: f.content_hash for f in self.env.repository.indexed_files(self.env.workspace.id)}
        self.assertNotEqual(before['Без названия 4.md'], after['Без названия 4.md'])


class InputValidationTests(unittest.TestCase):
    def test_blank_input_is_rejected(self):
        for question in ('', ' ', '\n\t'):
            with self.subTest(question=question), self.assertRaises(ValidationError):
                QueryInput(question=question)

    def test_long_question_is_rejected(self):
        with self.assertRaises(ValidationError):
            QueryInput(question='x' * 32001)

    def test_search_limits_cannot_be_unbounded(self):
        for limit in (0, -1, 101):
            with self.subTest(limit=limit), self.assertRaises(ValidationError):
                SearchRequest(query='архив', limit=limit)

    def test_search_whitespace_is_rejected(self):
        with self.assertRaises(ValidationError):
            SearchRequest(query='\t ')

    def test_reindex_ids_are_positive_and_deduplicated(self):
        self.assertEqual(ReindexRequest(file_ids=[1, 2, 1]).file_ids, [1, 2])
        for ids in ([], [-1], [0], list(range(1, 102))):
            with self.subTest(ids=ids), self.assertRaises(ValidationError):
                ReindexRequest(file_ids=ids)
