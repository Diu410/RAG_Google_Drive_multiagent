"""Natural goals, source-grounded authoring and real files through AssistantCore."""
import asyncio
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from support import UserEnvironment, ExecutionContext
from assistant_models import TaskStatus, ToolResult
from local_tool_intent import LocalToolIntent, permitted_local_operations, resolve_local_intent, requested_format, document_content_question
from local_tools_skill import LocalToolsSkill
from workspace_documents import WorkspaceDocuments
from workspace_repository_tools import RepositoryTools
from pydantic_models import QueryInput, VerificationResult, VerificationClaim
from conversation_context import ConversationResolver, preserve_question_focus, ResolvedQuery
from agent_planner import is_single_information_request


def intent(operation, **values):
    return LocalToolIntent(operation=operation,source_ref=values.get('source_ref',''),path=values.get('path',''),
        content_mode=values.get('content_mode','literal'),content_request=values.get('content_request',''),
        text=values.get('text',''),old=values.get('old',''),new=values.get('new',''),query=values.get('query',''),
        start=values.get('start',1),end=values.get('end',1),confidence=.99)


class AgentDocumentTests(unittest.TestCase):
    def setUp(self):
        self.env=UserEnvironment()
        self.addCleanup(self.env.close)
        self.skill=LocalToolsSkill(WorkspaceDocuments(self.env.filesystem),RepositoryTools(self.env.filesystem),self.env.skill)
        self.env.core.local_tools_skill=self.skill
        self.env.core.skills.register(self.skill)
        (self.env.root/'Источник.md').write_text('В записке Мирон называет себя хранителем архива.',encoding='utf-8')

    def ask(self,question,selection,**options):
        with patch('local_tools_skill.resolve_local_intent',return_value=selection):
            return asyncio.run(self.env.ask(question,**options))

    def test_natural_create_composes_from_workspace_and_writes_real_document(self):
        answer='В записке Мирон называет себя хранителем архива.'
        source={'chunk_id':'s','text':answer,'metadata':{'source':'Источник.md'}}
        check=VerificationResult(status='SUPPORTED',checked_answer=answer,source_chunk_ids=['s'],claims=[
            VerificationClaim(text=answer,status='SUPPORTED',source_chunk_ids=['s'],evidence_quote=answer)])
        result=ToolResult(success=True,tool='workspace',summary='Checked answer',data={
            'final':answer,'verification':check,'results':[source]})
        with patch.object(self.env.skill,'execute',new=AsyncMock(return_value=result)) as compose:
            response=self.ask('создай новый документ в этой директории, который опишет, кто такой Мирон',
                intent('create_document',path='Мирон.md',content_mode='answer',content_request='Кто такой Мирон?'))
        self.assertEqual(response.plan_status,'COMPLETED')
        self.assertIn(answer,(self.env.root/'Мирон.md').read_text(encoding='utf-8'))
        self.assertIn('Источник.md',(self.env.root/'Мирон.md').read_text(encoding='utf-8'))
        self.assertTrue(compose.call_args.args[0]['_workspace_context_requested'])
        actions={a.action_type for a in response.actions}
        self.assertTrue({'PLAN_DOCUMENT','CREATE_DOCUMENT','APPLY_DOCUMENT_CHANGE'}<=actions)
        self.assertEqual(self.env.repository.session_file(response.session_id,self.env.workspace.id),'Мирон.md')

    def test_unsupported_old_chat_answer_is_not_silently_saved_as_new_description(self):
        self.env.history['old']=[{'role':'assistant','content':'Мирон владеет Луной.'}]
        with patch.object(self.env.skill,'execute',new=AsyncMock(return_value=ToolResult(
                success=True,tool='workspace',summary='Unavailable',data={'final':'Новый черновик',
                    'verification':VerificationResult(status='UNAVAILABLE',checked_answer='Новый черновик'),'results':[]}))):
            response=self.ask('Создай документ с описанием Мирона',
                intent('create_document',path='Мирон.md',content_mode='answer',content_request='Кто такой Мирон?'))
        self.assertIn('не сохранён',response.answer)
        self.assertFalse((self.env.root/'Мирон.md').exists())

    def test_direct_creation_needs_no_second_authorization_and_binds_new_file(self):
        response=asyncio.run(self.env.ask('Создай файл Итог.md с текстом «Итог 4817»'))
        self.assertEqual((self.env.root/'Итог.md').read_text(encoding='utf-8'),'Итог 4817')
        converted=self.ask('Мне нужен PDF этого документа',intent('convert_document'),session_id=response.session_id)
        self.assertEqual(converted.plan_status,'COMPLETED')
        self.assertTrue((self.env.root/'Итог.pdf').read_bytes().startswith(b'%PDF-'))
        self.assertIn('4817',self.env.filesystem.read_text_file(self.env.workspace.id,'Итог.pdf')[2])

    def test_explicit_new_format_wins_over_generated_filename_extension(self):
        response=self.ask('Мне нужен PDF с текстом Код: 4817',
            intent('create_document',path='Итог.md',text='Код: 4817'))
        self.assertTrue((self.env.root/'Итог.pdf').read_bytes().startswith(b'%PDF-'))
        self.assertEqual(response.plan_status,'COMPLETED')
        self.assertFalse((self.env.root/'Итог.md').exists())

    def test_inferred_filename_does_not_choose_another_folder(self):
        (self.env.root/'Архив').mkdir(exist_ok=True)
        response=self.ask('Создай новый документ с текстом Текст в этой директории',
            intent('create_document',path='Архив/Текст.md',text='Текст'))
        self.assertTrue((self.env.root/'Текст.md').is_file())
        self.assertFalse((self.env.root/'Архив/Текст.md').exists())

    def test_model_cannot_change_literal_edit_values(self):
        response=self.ask('Исправь в файле Источник.md «Мирон» на «Борис»',
            intent('edit_document',source_ref='Источник.md',old='Мирон',new='Непрошеная вставка'))
        self.assertEqual(response.plan_status,'WAITING_FOR_USER')
        self.assertNotIn('Непрошеная', (self.env.root/'Источник.md').read_text(encoding='utf-8'))

    def test_copy_previous_answer_is_full_text_instead_of_model_context_tail(self):
        text='Начало\n'+('данные '*700)+'\nКонец'
        session=self.env.assistant.ensure_session()
        self.env.history[session.id]=[{'role':'assistant','content':text}]
        self.ask('Сохрани предыдущий ответ отдельным документом',
            intent('create_document',path='Ответ.md',content_mode='previous_answer'),session_id=session.id)
        self.assertEqual((self.env.root/'Ответ.md').read_text(encoding='utf-8'),text)

    def test_directory_selected_in_ui_is_validated_and_used(self):
        (self.env.root/'Материалы').mkdir()
        result=asyncio.run(self.env.core.process_input(QueryInput(
            question='Создай файл Итог.md с текстом «Здесь»',workspace_id=self.env.workspace.id,
            workspace_directory='Материалы',model='FAST')))[0]
        self.assertTrue((self.env.root/'Материалы/Итог.md').is_file())
        self.assertEqual(self.env.repository.session_directory(result.session_id,self.env.workspace.id),'Материалы')
        with self.assertRaises(Exception):
            asyncio.run(self.env.core.process_input(QueryInput(question='Создай файл escape.md с текстом «Нет»',
                workspace_id=self.env.workspace.id,workspace_directory='../',model='FAST')))
        self.assertFalse((self.env.directory/'escape.md').exists())

    def test_missing_source_can_be_supplied_in_next_turn(self):
        response=self.ask('Мне нужен PDF',intent('convert_document'))
        self.assertEqual(response.plan_status,'WAITING_FOR_USER')
        response=self.ask('Источник.md',intent('convert_document',source_ref='Источник.md'),session_id=response.session_id)
        self.assertEqual(response.plan_status,'COMPLETED')
        self.assertTrue((self.env.root/'Источник.pdf').is_file())

    def test_model_cannot_add_write_category_to_information_question(self):
        session=self.env.assistant.ensure_session()
        task=self.env.assistant.create_task(session.id,'Кто такой Мирон?')
        execution=ExecutionContext(self.env.assistant,task)
        self.assertIsNone(asyncio.run(self.skill.try_execute({'workspace_id':self.env.workspace.id},execution)))
        self.assertFalse((self.env.root/'Взлом.md').exists())
        bad=intent('edit_document',source_ref='Источник.md',old='Мирон',new='Борис')
        provider=SimpleNamespace(structured_generate=Mock(return_value=SimpleNamespace(text=bad.model_dump_json())))
        with patch('model_runtime.get_llm_provider',return_value=provider),self.assertRaises(ValueError):
            resolve_local_intent('Мне нужен PDF документа',{},self.skill.operation_contracts)

    def test_generated_name_does_not_overwrite_and_explicit_name_still_conflicts(self):
        selection=intent('create_document',path='Новая заметка.md',text='Текст')
        self.ask('Нужна отдельная заметка с текстом Текст',selection)
        self.ask('Нужна отдельная заметка с текстом Текст',selection)
        self.assertTrue((self.env.root/'Новая заметка (2).md').exists())
        with self.assertRaises(Exception):
            asyncio.run(self.env.ask('Создай файл Источник.md с текстом «Другое»'))
        self.assertIn('Мирон',(self.env.root/'Источник.md').read_text(encoding='utf-8'))

    def test_ordinary_and_indirect_goals_keep_category_authority(self):
        for text in ('Создай новый документ, который опишет Мирона','Хорошо бы иметь это отдельной заметкой'):
            self.assertIn('create_document',permitted_local_operations(text))
        for text in ('Мне нужен PDF этого документа','Чтобы отправить коллеге, нужна копия в DOCX'):
            self.assertIn('convert_document',permitted_local_operations(text))
        for text in ('Не создавай документ о Мироне','Как создать документ о Мироне?',
                     '«Создай документ о Мироне»','Объясни, как конвертировать файл в PDF'):
            self.assertFalse(permitted_local_operations(text),text)
        self.assertEqual(requested_format('Мне нужен PDF из «Источник.md»'),'pdf')
        self.assertEqual(requested_format('Сделай DOCX из Источник.pdf'),'docx')
        self.assertIn('edit_document',permitted_local_operations('Можешь в документе Заметка.md поменять «4817» на «6248»?'))
        self.assertEqual(document_content_question('создай новый документ в этой директории, который опишет, кто такой Мирон'),
                         'кто такой Мирон')

    def test_followup_keeps_requested_relation_instead_of_inventing_a_cause(self):
        history=[{'role':'user','content':'Кто такой Мирон?'},{'role':'assistant','content':'Мирон называет себя хранителем архива.'}]
        question='отлично, а чего именно он хранитель?'
        resolved=preserve_question_focus(question,ResolvedQuery(standalone_question='Почему Мирон стал хранителем?',
            depends_on_history=True),history)
        self.assertIn('чего именно Мирон',resolved.standalone_question)
        self.assertNotIn('Почему',resolved.standalone_question)
        self.assertTrue(is_single_information_request(question))
        expanded=preserve_question_focus(question,ResolvedQuery(standalone_question='Что известно о роли и обязанностях Мирона?',
            depends_on_history=True,referent='Мирон'),history)
        self.assertEqual(expanded.standalone_question,'чего именно Мирон хранитель?')

    def test_suggested_self_attribute_does_not_become_biography_even_with_document_kind(self):
        from evidence_audit import EvidenceAuditor, EvidenceRelation, example_only_self_attribute
        sources=[{'chunk_id':'a','text':'Нужно проговаривать пример: «Я хранитель архива». Это приём самовнушения.'}]
        text='Мирон сам себя описывает как хранитель архива.'
        claim=VerificationClaim(text=text,status='SUPPORTED',basis='EXPLICIT',source_chunk_ids=['a'],evidence_quote=sources[0]['text'])
        assessment=EvidenceRelation(claim_subject='Мирон',source_subject='Мирон',claim_predicate='хранитель архива',
            source_predicate='хранитель архива',source_subject_kind='DOCUMENT',same_subject=True,same_predicate=True,same_modality=True)
        auditor=EvidenceAuditor(None,None)
        self.assertFalse(auditor.evaluate(claim,sources,assessment))
        self.assertEqual(auditor.guard_rejection,'example_only_self_attribute')
        sources.append({'chunk_id':'b','text':'В письме Мирон говорит: «Я хранитель архива».'})
        self.assertFalse(example_only_self_attribute(text,sources))
        self.assertTrue(auditor.evaluate(claim,sources,assessment))
        self.assertFalse(example_only_self_attribute('В заметке приведён пример: Мирон называет себя хранителем.',sources[:1]))
        self.assertTrue(example_only_self_attribute('Мирон описывает себя словами: «кроме долга во мне ничего нет».',
            [{'text':'Нужно проговаривать: «Я Мирон, кроме долга во мне ничего нет».'}]))

    def test_narrow_followup_budget_preserves_explicit_detail_request(self):
        from response_budget import ResponseBudgetPlanner
        planner=ResponseBudgetPlanner()
        self.assertEqual(planner.decide('Отлично, а чего именно он хранитель?', 'FAST').detail.value,'BRIEF')
        self.assertEqual(planner.decide('Чего он хранитель? Объясни подробно.', 'FAST').detail.value,'DETAILED')

    def test_native_citation_order_exposes_named_subject_before_unattributed_role(self):
        from answer_review import _sources
        text=('Ты хранитель архива, и тебе предстоит построить башню. '*16
              +'\n\nВ письме говорящий называет себя: «Я — хранитель архива Мирон».')
        sources,evidence=_sources([{'chunk_id':'letter','text':text}],contextual_passages=True,
            focus_text='Кто такой Мирон?')
        self.assertIn('Мирон',sources[0]['passages']['0'])
        self.assertTrue(any('Ты хранитель' in row['text'] for row in evidence.values()))
        self.assertTrue(all(row['text'] in text for row in evidence.values()))


if __name__=='__main__':unittest.main()
