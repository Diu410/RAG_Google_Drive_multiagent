"""Real file effects and durable artifact references through the chat entry point."""
import asyncio
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from support import UserEnvironment
from assistant_models import AssistantError
from local_tools_skill import LocalToolsSkill
from local_tool_intent import permitted_local_operations
from workspace_documents import WorkspaceDocuments
from workspace_repository_tools import RepositoryTools
from command_intent import requests_file_open
from pydantic_models import QueryInput


class CreatedDocumentFollowupTests(unittest.TestCase):
    def setUp(self):
        self.env = UserEnvironment()
        self.addCleanup(self.env.close)
        self.tools = LocalToolsSkill(WorkspaceDocuments(self.env.filesystem),RepositoryTools(self.env.filesystem),self.env.skill)
        self.env.core.local_tools_skill = self.tools
        self.env.core.skills.register(self.tools)
        self.env.core.conversation_resolver = Mock()
        self.env.core.agent_planner = Mock()
        (self.env.root/'Материалы').mkdir()
        (self.env.root/'Материалы/Другой.md').write_text('OTHER_FILE_MUST_NOT_REPLACE_TARGET',encoding='utf-8')

    def ask(self,question,session=None,workspace=None):
        return asyncio.run(self.env.ask(question,session,workspace))

    def create(self):
        return self.ask('Создай файл Новый документ.md с текстом «CREATED_DOCUMENT_CONTENT»')

    def assert_opened(self,path):
        self.assertEqual([Path(p).relative_to(self.env.root).as_posix() for p in self.env.opened],[path])

    def test_create_then_open_it_uses_the_written_file_without_any_model(self):
        created = self.create()
        response = self.ask('открой его',created.session_id)
        self.assertEqual(response.strategy,'WORKSPACE_OPEN')
        self.assert_opened('Новый документ.md')
        self.assertNotIn('READ_FILE',[a.action_type for a in response.actions])
        self.env.core.conversation_resolver.resolve.assert_not_called()
        self.env.core.agent_planner.plan.assert_not_called()

    def test_creation_persists_the_selected_directory_instead_of_an_old_nested_folder(self):
        session = self.env.assistant.ensure_session()
        self.env.repository.bind_session(session.id,self.env.workspace.id,'Материалы','Материалы/Другой.md')
        response = asyncio.run(self.env.core.process_input(QueryInput(
            question='Создай файл Новый документ.md с текстом «ROOT_DOCUMENT»',model='FAST',
            session_id=session.id,workspace_id=self.env.workspace.id,workspace_directory='')))[0]
        self.assertTrue((self.env.root/'Новый документ.md').is_file())
        self.assertEqual(self.env.repository.session_directory(session.id,self.env.workspace.id),'')
        self.ask('открой его',response.session_id)
        self.assert_opened('Новый документ.md')

    def test_explicit_filename_in_root_wins_over_unrelated_current_directory(self):
        created = self.create()
        self.env.repository.bind_session(created.session_id,self.env.workspace.id,'Материалы','Материалы/Другой.md')
        with patch.object(self.env.filesystem.file_reference_resolver,'resolve') as resolver:
            self.ask('открой Новый документ.md',created.session_id)
        self.assert_opened('Новый документ.md')
        resolver.assert_not_called()

    def test_literal_missing_filename_cannot_be_replaced_by_confident_model_choice(self):
        with patch.object(self.env.filesystem.file_reference_resolver,'resolve',return_value=Mock(
                selected_path='Материалы/Другой.md',confidence=1)) as resolver:
            response = self.ask('открой Новый документ.md')
        self.assertFalse(self.env.opened)
        self.assertIn('не найден',response.answer)
        resolver.assert_not_called()

    def test_explicit_folder_remains_a_constraint_for_an_exact_filename(self):
        self.create()
        response = self.ask('открой Новый документ.md в папке Материалы')
        self.assertFalse(self.env.opened)
        self.assertIn('не найден',response.answer)

    def test_active_path_is_not_reresolved_to_a_same_named_file_in_another_folder(self):
        created = self.create()
        (self.env.root/'Материалы/Новый документ.md').write_text('DIFFERENT_FILE',encoding='utf-8')
        self.env.repository.bind_session(created.session_id,self.env.workspace.id,'Материалы','Новый документ.md')
        self.ask('открой его',created.session_id)
        self.assert_opened('Новый документ.md')

    def test_created_file_is_recovered_from_actions_after_active_file_and_history_change(self):
        created = self.create()
        self.ask('Прочитай файл Материалы/Другой.md',created.session_id)
        self.env.history[created.session_id] = [{'role':'assistant','content':'Обсуждение другой темы.'}]*8
        with patch('local_tools_skill.resolve_local_intent') as resolver:
            response = self.ask('я тебя попросил открыть созданный тобой документы',created.session_id)
        self.assertEqual(response.strategy,'WORKSPACE_OPEN')
        self.assert_opened('Новый документ.md')
        resolver.assert_not_called()
        self.assertEqual(list(self.env.root.glob('Новый документ*.md')),[self.env.root/'Новый документ.md'])

    def test_last_opened_question_reports_the_actual_action_without_reading_again(self):
        first = self.ask('Открой файл Материалы/Другой.md')
        response = self.ask('ты чё открыл',first.session_id)
        self.assertEqual(response.strategy,'WORKSPACE_ACTION_STATUS')
        self.assertIn('Материалы/Другой.md',response.answer)
        self.assertNotIn('OTHER_FILE_MUST_NOT_REPLACE_TARGET',response.answer)
        self.assertEqual(len(self.env.opened),1)
        self.assertNotIn('READ_FILE',[a.action_type for a in response.actions])

    def test_last_opened_question_does_not_invent_an_action(self):
        response = self.ask('что ты открыл?')
        self.assertEqual(response.strategy,'WORKSPACE_ACTION_STATUS')
        self.assertIn('нет успешного открытия',response.answer)
        self.assertFalse(self.env.opened)

    def test_created_artifact_does_not_cross_chats_or_workspaces(self):
        created = self.create()
        response = self.ask('открой созданный тобой документ')
        self.assertEqual(response.strategy,'WORKSPACE_CLARIFICATION')
        other = self.env.directory/'Другой workspace'
        other.mkdir()
        (other/'Новый документ.md').write_text('NOT_CREATED_IN_THIS_WORKSPACE',encoding='utf-8')
        workspace = self.env.repository.select(str(other))
        response = self.ask('открой созданный тобой документ',created.session_id,workspace.id)
        self.assertEqual(response.strategy,'WORKSPACE_CLARIFICATION')
        self.assertFalse(self.env.opened)

    def test_open_request_replaces_a_pending_conversion_clarification(self):
        created = self.create()
        self.env.assistant.save_pending_plan(created.session_id,{'kind':'local_tool_input',
            'workspace_id':self.env.workspace.id,'question':'Мне нужен PDF'})
        response = self.ask('открой Новый документ.md',created.session_id)
        self.assertEqual(response.strategy,'WORKSPACE_OPEN')
        self.assert_opened('Новый документ.md')
        self.assertFalse(list(self.env.root.glob('*.pdf')))

    def test_removed_artifact_does_not_open_another_document(self):
        created = self.create()
        (self.env.root/'Новый документ.md').unlink()
        response = self.ask('открой его',created.session_id)
        self.assertIn('больше недоступен',response.answer)
        self.assertFalse(self.env.opened)

    def test_read_it_uses_the_active_document_without_a_generation_or_os_open(self):
        created = self.create()
        response = self.ask('прочитай его',created.session_id)
        self.assertEqual(response.strategy,'WORKSPACE_READ')
        self.assertIn('CREATED_DOCUMENT_CONTENT',response.answer)
        self.assertFalse(self.env.opened)
        self.assertNotIn('GENERATE_RESPONSE',[a.action_type for a in response.actions])

    def test_read_created_document_recovers_it_after_reading_another_file(self):
        created = self.create()
        self.ask('Прочитай файл Материалы/Другой.md',created.session_id)
        response = self.ask('Прочитай созданный тобой документ',created.session_id)
        self.assertEqual(response.strategy,'WORKSPACE_READ')
        self.assertIn('CREATED_DOCUMENT_CONTENT',response.answer)
        self.assertNotIn('OTHER_FILE_MUST_NOT_REPLACE_TARGET',response.answer)
        self.assertFalse(self.env.opened)

    def test_a_literal_filename_starting_with_a_pronoun_is_still_a_literal(self):
        created = self.create()
        (self.env.root/'Этот файл.md').write_text('LITERAL_PRONOUN_FILENAME',encoding='utf-8')
        response = self.ask('Прочитай Этот файл.md',created.session_id)
        self.assertIn('LITERAL_PRONOUN_FILENAME',response.answer)
        self.ask('Открой Этот файл.md',created.session_id)
        self.assert_opened('Этот файл.md')

    def test_unapplied_preview_and_later_edit_do_not_replace_created_artifact(self):
        created = self.create()
        self.ask('Предпросмотр: Создай файл Предпросмотр.md с текстом «UNSAVED_CONTENT»',created.session_id)
        self.ask('отмена',created.session_id)
        self.ask('Замени в файле Материалы/Другой.md текст «OTHER_FILE_MUST_NOT_REPLACE_TARGET» на «EDITED_OTHER»',
                 created.session_id)
        self.ask('открой созданный тобой документ',created.session_id)
        self.assert_opened('Новый документ.md')
        self.assertFalse((self.env.root/'Предпросмотр.md').exists())

    def test_confirmed_preview_is_a_created_artifact_across_turns(self):
        first = self.ask('Предпросмотр: Создай файл Сохранённый.md с текстом «CONFIRMED_CONTENT»')
        self.ask('да',first.session_id)
        self.ask('Прочитай файл Материалы/Другой.md',first.session_id)
        self.ask('открой созданный тобой документ',first.session_id)
        self.assert_opened('Сохранённый.md')

    def test_a_new_conversion_remains_the_created_artifact_after_other_reads(self):
        created = self.create()
        self.ask('Конвертируй Новый документ.md в Копия.txt',created.session_id)
        self.ask('Прочитай файл Материалы/Другой.md',created.session_id)
        self.ask('открой созданный тобой документ',created.session_id)
        self.assert_opened('Копия.txt')

    def test_literal_missing_filename_is_not_replaced_by_a_lexical_match(self):
        created = self.create()
        with patch.object(self.env.filesystem.file_reference_resolver,'resolve') as resolver:
            response = self.ask('Открой Новый документ потерян.md',created.session_id)
        self.assertIn('не найден',response.answer)
        self.assertFalse(self.env.opened)
        resolver.assert_not_called()

    def test_past_participles_and_quoted_or_negated_openings_do_not_grant_mutation(self):
        for question in ('открой созданный тобой документ','кто создал этот документ?',
                         'открой исправленный документ','открой конвертированный документ'):
            self.assertFalse(permitted_local_operations(question),question)
        for question in ('я не просил тебя открыть созданный документ',
                         'я тебя просил не открывать созданный документ',
                         '«я тебя попросил открыть созданный тобой документ»'):
            self.assertFalse(requests_file_open(question),question)
            with self.assertRaises(AssistantError):
                asyncio.run(self.env.execute(question,'open_file',path='Материалы/Другой.md'))
        self.assertFalse(self.env.opened)


if __name__ == '__main__':
    unittest.main()
