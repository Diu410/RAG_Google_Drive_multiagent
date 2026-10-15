"""User actions and file resolution derived from patterns in actual chat history."""
import asyncio
from pathlib import Path
import os
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from support import UserEnvironment
from assistant_models import AssistantError
from command_intent import requests_file_open, requests_unsupported_file_change, negates_comparison
from workspace_planner import WorkspacePlanner, WorkspaceOperation
from fixtures import create_docx, create_pdf


class UserToolsTests(unittest.TestCase):
    def setUp(self):
        self.env = UserEnvironment()
        self.addCleanup(self.env.close)

    def test_explicit_version_pair_does_not_substitute_latest_version(self):
        result = self.env.filesystem.compare_latest(self.env.workspace.id, 'Сравни версии отчёта 2 и 9')
        self.assertEqual(result['current'].relative_path, 'Отчёты/Отчёт 9.md')
        self.assertEqual(result['previous'].relative_path, 'Отчёты/Отчёт 2.md')
        self.assertIn('64%', result['diff'])
        self.assertIn('42%', result['diff'])
        self.assertNotIn('87%', result['diff'])

    def test_missing_comparison_version_is_not_replaced_with_another(self):
        with self.assertRaises(AssistantError) as error:
            self.env.filesystem.compare_latest(self.env.workspace.id, 'Сравни версии отчёта 2 и 120')
        self.assertEqual(error.exception.status_code, 404)

    def test_exact_numeric_names_never_substitute_another_note(self):
        for number in (1, 4, 9, 11):
            for question in (f'Прочитай файл Без названия {number}.md', f'открой без названия {number}'):
                with self.subTest(question=question):
                    path = self.env.filesystem.find_text_file(self.env.workspace.id, question)
                    self.assertEqual(path, f'Без названия {number}.md')
        for number in (6, 120):
            with self.subTest(missing=number), self.assertRaises(AssistantError) as caught:
                self.env.filesystem.find_text_file(self.env.workspace.id, f'Прочитай Без названия {number}.md')
            self.assertEqual(caught.exception.status_code, 404)

    def test_unnumbered_file_is_not_the_newest_numbered_note(self):
        self.assertEqual(self.env.filesystem.find_text_file(self.env.workspace.id, 'прочитай Без названия'), 'Без названия.md')

    def test_filename_with_quotes_and_extra_spaces(self):
        for question in ('Прочитай файл "Алчное письмо.md"', 'Прочитай файл Алчное  письмо.md',
                         'Прочитай файл «Алчное письмо.md»', 'Прочитай файл АЛЧНОЕ ПИСЬМО.MD'):
            with self.subTest(question=question):
                self.assertEqual(self.env.filesystem.find_text_file(self.env.workspace.id, question), 'Архив/Алчное письмо.md')

    def test_duplicate_filename_requires_choice_instead_of_arbitrary_selection(self):
        with self.assertRaises(AssistantError) as caught:
            self.env.filesystem.find_text_file(self.env.workspace.id, 'прочитай План.md')
        self.assertEqual(caught.exception.status_code, 409)
        self.assertIn('Проект А/План.md', str(caught.exception))
        self.assertIn('Проект Б/План.md', str(caught.exception))

    def test_current_directory_disambiguates_duplicate_name(self):
        self.assertEqual(self.env.filesystem.find_text_file(self.env.workspace.id,
            'прочитай План.md', current_directory='Проект А'), 'Проект А/План.md')

    def test_explicit_folder_overrides_current_directory(self):
        self.assertEqual(self.env.filesystem.find_text_file(self.env.workspace.id,
            'прочитай План.md в папке Проект Б', current_directory='Проект А'), 'Проект Б/План.md')

    def test_hidden_folder_is_readable_when_explicitly_requested(self):
        self.assertEqual(self.env.filesystem.find_text_file(self.env.workspace.id,
            'прочитай app.json в папке .obsidian'), '.obsidian/app.json')
        self.assertNotIn('.obsidian/app.json', [record.relative_path for record in
                         self.env.repository.indexed_files(self.env.workspace.id)])

    def test_letter_filter_selects_real_file(self):
        self.assertEqual(self.env.filesystem.find_text_file(self.env.workspace.id,
            'открой файл название которого начинается на букву "Ф"'), 'Физический стандарт.md')

    def test_model_cannot_invent_candidate_or_change_numeric_target(self):
        resolver = self.env.filesystem.file_reference_resolver
        for selection in ('outside.md', '../private.md', 'Без названия 11.md'):
            with self.subTest(selection=selection), patch.object(resolver, 'resolve', return_value=SimpleNamespace(
                    selected_path=selection, confidence=1.0)), self.assertRaises(AssistantError):
                self.env.filesystem.find_text_file(self.env.workspace.id, 'Прочитай без назвния 120', model='FAST')

    def test_new_file_is_discovered_without_reselecting_workspace(self):
        self.env.filesystem.sync_index(self.env.workspace.id)
        (self.env.root / 'Новый.md').write_text('NEW_FACT', encoding='utf-8')
        self.assertEqual(self.env.filesystem.find_text_file(self.env.workspace.id, 'Прочитай Новый.md'), 'Новый.md')

    def test_latest_numeric_version_ignores_newer_mtime_of_old_version(self):
        record, reason = self.env.filesystem.latest_file(self.env.workspace.id, 'Найди последнюю версию Отчёт в папке Отчёты')
        self.assertEqual(record.relative_path, 'Отчёты/Отчёт 11.md')

    def test_compare_reports_shows_actual_difference(self):
        result = self.env.filesystem.compare_latest(self.env.workspace.id, 'Сравни версии Отчёт в папке Отчёты')
        self.assertEqual(result['current'].relative_path, 'Отчёты/Отчёт 11.md')
        self.assertEqual(result['previous'].relative_path, 'Отчёты/Отчёт 9.md')
        self.assertIn('87%', result['diff'])
        self.assertIn('64%', result['diff'])

    def test_display_does_not_launch_system_application(self):
        response = asyncio.run(self.env.ask('Прочитай Без названия 4.md'))
        self.assertIn('NOTE_FOUR', response.answer)
        self.assertFalse(self.env.opened)
        self.assertNotIn('OPEN_FILE', [action.action_type for action in response.actions])

    def test_explicit_open_launches_exactly_one_selected_file(self):
        response = asyncio.run(self.env.ask('Открой файл Без названия 4.md'))
        self.assertEqual(len(self.env.opened), 1)
        self.assertEqual(Path(self.env.opened[0]).name, 'Без названия 4.md')
        self.assertEqual([action.action_type for action in response.actions].count('OPEN_FILE'), 1)

    def test_agent_cannot_open_without_users_own_consent(self):
        for question in ('Что в Без названия 4.md?', 'Не открывай файл Без названия 4.md',
                         'Объясни фразу «открой файл Без названия 4.md»'):
            with self.subTest(question=question), self.assertRaises(AssistantError):
                asyncio.run(self.env.execute(question, 'open_file', path='Без названия 4.md'))
        self.assertFalse(self.env.opened)

    def test_unsupported_changes_do_not_touch_file_bytes(self):
        path = self.env.root / 'Без названия 4.md'
        before = path.read_bytes()
        for question in ('Удали файл Без названия 4.md', 'Переименуй файл Без названия 4.md',
                         'Перемести файл Без названия 4.md'):
            with self.subTest(question=question):
                response = asyncio.run(self.env.ask(question))
                self.assertIn('не', response.answer.casefold())
                self.assertEqual(path.read_bytes(), before)
                self.assertFalse(self.env.opened)

    def test_pdf_docx_html_and_legacy_text_encoding_are_readable(self):
        create_docx(self.env.root / 'Документ.docx')
        create_pdf(self.env.root / 'Документ.pdf')
        (self.env.root / 'Страница.html').write_text('<p>HTML_FACT 640 рублей</p>', encoding='utf-8')
        (self.env.root / 'Кодировка.txt').write_bytes('Цена — 570 рублей'.encode('cp1251'))
        for relative, expected in (('Документ.docx','DOCX_FACT'), ('Документ.pdf','PDF_FACT'),
                                   ('Страница.html','HTML_FACT'), ('Кодировка.txt','570')):
            with self.subTest(relative=relative):
                _, _, text = self.env.filesystem.read_text_file(self.env.workspace.id, relative)
                self.assertIn(expected, text)


class IntentTests(unittest.TestCase):
    def test_open_requests(self):
        for question in ('Открой файл notes.md', 'Можешь открыть notes.md?', 'open notes.md',
                         'Прошу открыть notes.md', 'Откройте notes.md'):
            with self.subTest(question=question):
                self.assertTrue(requests_file_open(question))

    def test_no_open_without_consent(self):
        for question in ('не открывай notes.md', 'не нужно открывать notes.md', 'прочитай notes.md',
                         'Как мне открыть notes.md?', 'How do I open notes.md?', 'do not open notes.md',
                         "don't open notes.md", 'never open notes.md', 'without opening notes.md',
                         'Объясни фразу «открой notes.md»', 'Что означает "open notes.md"?',
                         'Открытие notes.md обсуждается в документе'):
            with self.subTest(question=question):
                self.assertFalse(requests_file_open(question))

    def test_negation_is_preserved_in_version_plan(self):
        for question in ('Найди последнюю версию отчёта, не открывай и не сравнивай её',
                         'Найди последнюю версию отчёта без открытия'):
            plan = WorkspacePlanner().plan(question)
            self.assertEqual([step.operation for step in plan.steps], [WorkspaceOperation.FIND_LATEST])

    def test_source_dates_are_not_filesystem_metadata(self):
        for question in ('Когда написан текст, согласно содержимому документов?',
                         'Какая дата указана в тексте файла?', 'Расскажи о дате внутри документа'):
            with self.subTest(question=question):
                plan = WorkspacePlanner().plan(question)
                self.assertTrue(plan is None or all(step.operation != WorkspaceOperation.LIST_FILE_DATES for step in plan.steps))

    def test_quoted_and_negated_deletion_is_not_an_action(self):
        for question in ('Не удаляй файл notes.md', 'Объясни фразу «удали файл notes.md»',
                         "don't delete file notes.md", 'never remove file notes.md'):
            with self.subTest(question=question):
                self.assertFalse(requests_unsupported_file_change(question))


if __name__ == '__main__':
    unittest.main()
