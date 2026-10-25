"""Real file round trips, chat authorization, stale edits and repository boundaries."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import support
from support import UserEnvironment
from assistant_models import AssistantError
from local_tools_skill import LocalToolsSkill, parse_local_command
from workspace_documents import WorkspaceDocuments, convert_text
from workspace_repository_tools import RepositoryTools


class LocalToolsTests(unittest.TestCase):
    def setUp(self):
        self.env = UserEnvironment()
        self.documents = WorkspaceDocuments(self.env.filesystem)
        self.repositories = RepositoryTools(self.env.filesystem)
        self.skill = LocalToolsSkill(self.documents, self.repositories, self.env.skill)
        self.env.core.local_tools_skill = self.skill
        self.env.core.skills.register(self.skill)
        self.id = self.env.workspace.id
        self.path = self.env.root / 'Тест.md'
        self.path.write_bytes('Первая строка\r\nКод: 4817\r\n'.encode('utf-8'))

    def tearDown(self):
        self.env.close()

    def test_preview_changes_nothing_apply_keeps_exact_backup_and_newlines(self):
        before = self.path.read_bytes()
        preview = self.documents.prepare_edit(self.id, 'Тест.md', '4817', '6248')
        self.assertEqual(self.path.read_bytes(), before)
        self.assertIn('-Код: 4817', preview['diff'])
        result = self.documents.apply(self.id, preview['proposal_id'])
        self.assertEqual((self.env.root / result['backup_path']).read_bytes(), before)
        self.assertEqual(self.path.read_bytes(), before.replace(b'4817', b'6248'))
        self.assertIn('6248', self.env.filesystem.read_text_file(self.id, 'Тест.md')[2])
        with self.assertRaises(AssistantError):
            self.documents.apply(self.id, preview['proposal_id'])

    def test_cp1251_and_utf8_bom_preserved(self):
        for encoding in ('cp1251', 'utf-8-sig'):
            with self.subTest(encoding=encoding):
                self.path.write_bytes('Статус: готово\r\n'.encode(encoding))
                preview = self.documents.prepare_edit(self.id, 'Тест.md', 'готово', 'проверено')
                self.documents.apply(self.id, preview['proposal_id'])
                self.assertEqual(self.path.read_bytes(), 'Статус: проверено\r\n'.encode(encoding))

    def test_changed_file_cannot_receive_stale_edit(self):
        preview = self.documents.prepare_edit(self.id, 'Тест.md', '4817', '6248')
        self.path.write_text('Новая версия', encoding='utf-8')
        with self.assertRaisesRegex(AssistantError, 'изменился'):
            self.documents.apply(self.id, preview['proposal_id'])
        self.assertEqual(self.path.read_text(encoding='utf-8'), 'Новая версия')

    def test_edit_requires_exact_match_count(self):
        self.path.write_text('один один', encoding='utf-8')
        with self.assertRaisesRegex(AssistantError, 'совпадений: 2'):
            self.documents.prepare_edit(self.id, 'Тест.md', 'один', 'два')
        preview = self.documents.prepare_edit(self.id, 'Тест.md', 'один', 'два', 2)
        self.documents.apply(self.id, preview['proposal_id'])
        self.assertEqual(self.path.read_text(encoding='utf-8'), 'два два')

    def test_create_and_conversion_never_overwrite(self):
        preview = self.documents.prepare_create(self.id, 'Новый.md', 'Документ')
        target = self.env.root / 'Новый.md'
        target.write_text('Другой автор', encoding='utf-8')
        with self.assertRaises(AssistantError):
            self.documents.apply(self.id, preview['proposal_id'])
        self.assertEqual(target.read_text(encoding='utf-8'), 'Другой автор')
        with self.assertRaises(AssistantError):
            self.documents.prepare_convert(self.id, 'Тест.md', 'Новый.md')

    def test_two_proposals_for_same_output_only_one_wins(self):
        previews = [self.documents.prepare_create(self.id, 'Один.txt', str(i)) for i in (1, 2)]
        def apply(preview):
            try:
                return self.documents.apply(self.id, preview['proposal_id'])['applied']
            except AssistantError:
                return False
        with ThreadPoolExecutor(2) as pool:
            self.assertEqual(sorted(pool.map(apply, previews)), [False, True])

    def test_conversion_source_revision_is_checked_again_at_apply(self):
        preview = self.documents.prepare_convert(self.id, 'Тест.md', 'Копия.txt')
        self.path.write_text('Обновлено', encoding='utf-8')
        with self.assertRaises(AssistantError):
            self.documents.apply(self.id, preview['proposal_id'])
        self.assertFalse((self.env.root / 'Копия.txt').exists())

    def test_expiration_and_workspace_binding(self):
        preview = self.documents.prepare_create(self.id, 'Новый.txt', 'Текст')
        with self.assertRaises(AssistantError):
            self.documents.apply('other', preview['proposal_id'])
        with patch('workspace_documents.time.monotonic', return_value=time.monotonic() + 601):
            with self.assertRaisesRegex(AssistantError, 'истёк'):
                self.documents.apply(self.id, preview['proposal_id'])

    def test_write_path_policy(self):
        for path in ('../outside.txt', '/outside.txt', 'C:/outside.txt', 'x.txt:stream', '.env',
                     '.git/test.md', '.fortuna-backups/test.txt', 'CON.txt', 'new.py', 'missing/test.txt',
                     'test./file.md', 'new\\file.md'):
            with self.subTest(path=path), self.assertRaises((ValueError, OSError)):
                self.documents.prepare_create(self.id, path, 'Нельзя')

    def test_junction_target_and_replaced_parent_are_rejected(self):
        # _is_link covers native junctions and symlinks; exercise both traversal
        # and a parent that became a link after the proposal was produced.
        directory = self.env.root / 'Папка'
        directory.mkdir()
        preview = self.documents.prepare_create(self.id, 'Папка/Новый.md', 'Текст')
        original = self.env.filesystem._is_link
        with patch.object(self.env.filesystem, '_is_link', side_effect=lambda p: p == directory or original(p)):
            # _resolve intentionally uses the class static policy.
            with patch('workspace.FilesystemProvider._is_link', side_effect=lambda p: p == directory):
                with self.assertRaises(PermissionError):
                    self.documents.apply(self.id, preview['proposal_id'])

    def test_round_trip_all_exports_with_cyrillic_and_escaped_markup(self):
        text = 'Отчёт Фортуны\nКод 4817; нагрузка 27 кг.\n<script>данные & текст</script>'
        self.path.write_text(text, encoding='utf-8')
        for suffix in ('.txt', '.md', '.html', '.docx', '.pdf'):
            with self.subTest(suffix=suffix):
                target = 'Копия' + suffix
                preview = self.documents.prepare_convert(self.id, 'Тест.md', target)
                self.assertFalse((self.env.root / target).exists())
                result = self.documents.apply(self.id, preview['proposal_id'])
                self.assertIsNone(result['backup_path'])
                extracted = self.env.filesystem.read_text_file(self.id, target)[2]
                for phrase in ('Отчёт Фортуны', '4817', '27 кг', 'данные & текст'):
                    self.assertIn(phrase, extracted)
                if suffix == '.html':
                    self.assertNotIn('<script>', (self.env.root / target).read_text(encoding='utf-8'))

    def test_read_lines_returns_ranges_and_rejects_unbounded_requests(self):
        result = self.documents.read_lines(self.id, 'Тест.md', 2, 2)
        self.assertEqual(result['lines'], [{'line': 2, 'text': 'Код: 4817'}])
        self.assertTrue(result['partial'])
        for start, end in ((0, 1), (2, 1), (1, 501)):
            with self.assertRaises(ValueError):
                self.documents.read_lines(self.id, 'Тест.md', start, end)

    def test_search_and_nested_gitignore_negations(self):
        (self.env.root / '.gitignore').write_text('*.rs\n!keep.rs\ncache/\n', encoding='utf-8')
        (self.env.root / 'keep.rs').write_text('const TOKEN: i32 = 4817;', encoding='utf-8')
        (self.env.root / 'skip.rs').write_text('TOKEN = 9999', encoding='utf-8')
        sub = self.env.root / 'src'
        sub.mkdir()
        (sub / '.gitignore').write_text('!local.rs\n', encoding='utf-8')
        (sub / 'local.rs').write_text('TOKEN = 6248', encoding='utf-8')
        (self.env.root / 'cache').mkdir()
        (self.env.root / 'cache/x.md').write_text('TOKEN hidden', encoding='utf-8')
        result = self.repositories.search(self.id, 'token')
        paths = {row['path'] for row in result['matches']}
        self.assertEqual(paths, {'keep.rs', 'src/local.rs'})
        self.assertEqual({row['line'] for row in result['matches']}, {1})
        self.assertIn('.rs', self.repositories.overview(self.id)['extensions'])
        self.assertNotIn('cache/x.md', self.repositories.overview(self.id)['paths'])

    def test_new_ignore_rules_remove_previously_indexed_files(self):
        self.env.filesystem.sync_index(self.id)
        (self.env.root / '.gitignore').write_text('Тест.md\n', encoding='utf-8')
        self.env.filesystem.sync_index(self.id)
        self.assertNotIn('Тест.md', [item.relative_path for item in self.env.repository.indexed_files(self.id)])
        # Explicit read is distinct from discovery and still works.
        self.assertIn('4817', self.env.filesystem.read_text_file(self.id, 'Тест.md')[2])

    def test_search_limit_and_secret_exclusion(self):
        (self.env.root / '.env').write_text('CODE=SECRET', encoding='utf-8')
        result = self.repositories.search(self.id, 'Код', max_results=1)
        self.assertEqual(len(result['matches']), 1)
        self.assertFalse(self.repositories.search(self.id, 'SECRET')['matches'])
        with self.assertRaises(PermissionError):
            self.repositories.search(self.id, 'CODE', '../')

    def test_quoted_negated_and_explanatory_commands_are_not_actions(self):
        for question in ('Не конвертируй Тест.md в pdf', 'Как конвертировать Тест.md в pdf?',
                         '«Конвертируй Тест.md в pdf»', 'Объясни: замени в файле Тест.md «а» на «б»'):
            self.assertIsNone(parse_local_command(question), question)

    def test_chat_edit_preview_and_apply_no_model_calls(self):
        preview = asyncio.run(self.env.ask('Предпросмотр: Замени в файле «Тест.md» текст «4817» на «6248»'))
        self.assertIn('4817', self.path.read_text(encoding='utf-8'))
        self.assertEqual(preview.plan_status, 'WAITING_FOR_USER')
        response = asyncio.run(self.env.ask('да', session_id=preview.session_id))
        self.assertIn('Сохранён файл', response.answer)
        self.assertIn('6248', self.path.read_text(encoding='utf-8'))
        self.assertTrue(any(a.action_type == 'APPLY_DOCUMENT_CHANGE' for a in response.actions))
        self.assertFalse(self.env.knowledge.calls)

    def test_chat_conversion_cancel_and_create(self):
        response = asyncio.run(self.env.ask('Предпросмотр: Конвертируй «Тест.md» в PDF'))
        self.assertEqual(response.plan_status, 'WAITING_FOR_USER')
        asyncio.run(self.env.ask('отмена', session_id=response.session_id))
        self.assertFalse((self.env.root / 'Тест.pdf').exists())
        response = asyncio.run(self.env.ask('Создай файл Новое.md с текстом «Привет, Фортуна!»'))
        self.assertEqual(response.plan_status,'COMPLETED')
        self.assertEqual((self.env.root / 'Новое.md').read_text(encoding='utf-8'), 'Привет, Фортуна!')

    def test_workspace_switch_cannot_apply_old_proposal(self):
        preview = asyncio.run(self.env.ask('Предпросмотр: Конвертируй Тест.md в pdf'))
        other_root = self.env.directory / 'other'
        other_root.mkdir()
        other = self.env.repository.select(str(other_root))
        asyncio.run(self.env.ask('да', session_id=preview.session_id, workspace_id=other.id))
        self.assertFalse((self.env.root / 'Тест.pdf').exists())
        self.assertFalse((other_root / 'Тест.pdf').exists())

    def test_planner_cannot_invent_write_command(self):
        session = self.env.assistant.ensure_session()
        task = self.env.assistant.create_task(session.id, 'Что написано в Тест.md?')
        execution = support.ExecutionContext(self.env.assistant, task)
        with self.assertRaises(AssistantError):
            asyncio.run(self.skill.execute_agent_step('create_document',
                {'workspace_id': self.id, 'path': 'Взлом.md', 'text': 'Запись'}, execution))
        self.assertFalse((self.env.root / 'Взлом.md').exists())

    def test_extensionless_repository_files_and_chat_line_ranges(self):
        for name in ('Dockerfile', 'Makefile', 'README', '.gitignore'):
            (self.env.root / name).write_text('LOCAL_FILE_4817', encoding='utf-8')
            self.assertIn('LOCAL_FILE_4817', self.env.filesystem.read_text_file(self.id, name)[2])
        response = asyncio.run(self.env.ask('Покажи строки 2–2 файла Тест.md'))
        self.assertIn('2: Код: 4817', response.answer)
        self.assertNotIn('Первая строка', response.answer)
        response = asyncio.run(self.env.ask('Найди текст «Код: 4817»'))
        self.assertIn('Тест.md:2', response.answer)

    def test_read_source_file_in_chat(self):
        (self.env.root / 'lib.rs').write_text('pub const VALUE: u32 = 4817;', encoding='utf-8')
        response = asyncio.run(self.env.ask('Прочитай lib.rs'))
        self.assertIn('pub const VALUE: u32 = 4817;', response.answer)

    def test_invalid_tool_input_has_actionable_chat_error(self):
        with self.assertRaisesRegex(AssistantError, 'диапазон до 500 строк') as error:
            asyncio.run(self.env.ask('Покажи строки 1-900 файла Тест.md'))
        self.assertEqual(error.exception.status_code, 422)
        with self.assertRaisesRegex(AssistantError, 'не найден') as error:
            asyncio.run(self.env.ask('Конвертируй отсутствует.md в PDF'))
        self.assertEqual(error.exception.status_code, 404)


if __name__ == '__main__':
    unittest.main()
