"""Real filesystem scenarios from the latest chat; no private source contents."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from support import UserEnvironment
from local_tools_skill import LocalToolsSkill
from workspace_documents import WorkspaceDocuments
from workspace_repository_tools import RepositoryTools
from directory_summary import DirectorySummary, parts
from action_feedback import action_feedback
from local_tool_intent import directory_summary_request, repeated_user_request


class PracticalChatTests(unittest.TestCase):
    def setUp(self):
        self.env=UserEnvironment()
        self.addCleanup(self.env.close)
        self.tools=LocalToolsSkill(WorkspaceDocuments(self.env.filesystem),
            RepositoryTools(self.env.filesystem),self.env.skill)
        self.env.core.local_tools_skill=self.tools
        self.env.core.skills.register(self.tools)
        self.env.core.conversation_resolver=Mock()
        self.env.core.agent_planner=Mock()

    def ask(self,question,session=None):return asyncio.run(self.env.ask(question,session))

    def test_filename_without_dot_is_exact_even_when_model_would_open_unrelated_file(self):
        (self.env.root/'Новый документ.md').write_text('RIGHT',encoding='utf-8')
        with patch.object(self.env.filesystem.file_reference_resolver,'resolve') as resolver:
            response=self.ask('Открой новый документ md')
        self.assertEqual([Path(p).name for p in self.env.opened],['Новый документ.md'])
        resolver.assert_not_called()
        self.assertEqual(response.strategy,'WORKSPACE_OPEN')

    def test_missing_no_dot_filename_cannot_turn_into_a_fuzzy_match(self):
        with patch.object(self.env.filesystem.file_reference_resolver,'resolve') as resolver:
            response=self.ask('Открой отсутствующий документ md')
        resolver.assert_not_called()
        self.assertFalse(self.env.opened)
        self.assertIn('не найден',response.answer)

    def test_an_actual_filename_without_a_dot_wins_over_extension_alias(self):
        (self.env.root/'Заметка md.txt').write_text('LITERAL',encoding='utf-8')
        (self.env.root/'Заметка.md').write_text('ALIAS',encoding='utf-8')
        response=self.ask('Прочитай Заметка md')
        self.assertIn('LITERAL',response.answer)
        self.assertNotIn('ALIAS',response.answer)

    def test_wrong_open_feedback_recovers_original_literal_target_from_journal(self):
        (self.env.root/'Новый документ.md').write_text('RIGHT',encoding='utf-8')
        (self.env.root/'Другой.md').write_text('WRONG',encoding='utf-8')
        with patch.object(self.env.filesystem,'find_text_file',return_value='Другой.md'):
            first=self.ask('Открой новый документ md')
        response=self.ask('не то открыл',first.session_id)
        self.assertEqual([Path(p).name for p in self.env.opened],['Другой.md','Новый документ.md'])
        self.assertEqual(response.strategy,'WORKSPACE_OPEN')
        self.env.core.conversation_resolver.resolve.assert_not_called()

    def test_feedback_uses_actual_action_and_does_not_read_or_query_documents(self):
        (self.env.root/'Черновик.md').write_text('PRIVATE_SOURCE',encoding='utf-8')
        opened=self.ask('Открой Черновик.md')
        self.env.assistant.save_pending_plan(opened.session_id,{'kind':'local_tool_input',
            'workspace_id':self.env.workspace.id,'question':'Создай новый документ'})
        response=self.ask('не то открыл',opened.session_id)
        self.assertIn('Черновик.md',response.answer)
        self.assertEqual(response.strategy,'WORKSPACE_ACTION_STATUS')
        self.assertNotIn('PRIVATE_SOURCE',response.answer)
        self.assertEqual(len(self.env.opened),1)
        self.assertEqual([a.action_type for a in response.actions if a.action_type!='SAVE_HISTORY'],['REVIEW_PREVIOUS_ACTION'])
        self.env.core.conversation_resolver.resolve.assert_not_called()

    def test_clear_directory_summary_does_not_need_the_model_to_choose_a_tool(self):
        async def compose(instance,inputs,execution):
            return 'Пересказ всех материалов.',{'mode':'DIRECTORY_SUMMARY','complete':True}
        with (patch('local_tools_skill.resolve_local_intent',side_effect=AssertionError('unnecessary routing')),
                patch.object(DirectorySummary,'compose',compose)):
            result=self.ask('Создай файл в котором будет содержаться общий пересказ всего, что есть в директории, назови сам')
        self.assertEqual(result.plan_status,'COMPLETED')
        self.assertEqual((self.env.root/'Общий пересказ.md').read_text(encoding='utf-8'),'Пересказ всех материалов.')

    def test_correction_reuses_original_goal_not_history_rewrite_or_active_output(self):
        observed=[]
        async def compose(instance,inputs,execution):
            observed.append(dict(inputs))
            return 'Полный результат.',{'mode':'DIRECTORY_SUMMARY','complete':True}
        with patch.object(DirectorySummary,'compose',compose):
            first=self.ask('Создай документ с общим пересказом всего в директории, назови сам')
            second=self.ask('это не пересказ тварь, это огрызок',first.session_id)
        self.assertEqual(len(observed),2)
        self.assertEqual(observed[1]['_summary_exclude'],['Общий пересказ.md'])
        self.assertEqual(observed[1]['_workspace_current_directory'],'')
        self.assertTrue((self.env.root/'Общий пересказ (2).md').is_file())
        self.assertNotIn('READ_FILE',[a.action_type for a in second.actions])
        self.env.core.conversation_resolver.resolve.assert_not_called()

    def test_partial_summary_is_visible_and_not_reported_as_completed(self):
        async def compose(instance,inputs,execution):
            return 'Неполный пересказ. Пропуски: файл 2.',{'mode':'DIRECTORY_SUMMARY','complete':False}
        with patch.object(DirectorySummary,'compose',compose):
            response=self.ask('Создай файл с общим пересказом всей папки, назови сам')
        self.assertEqual(response.plan_status,'PARTIAL')
        self.assertIn('не завершён',response.answer)

    def test_repeat_of_own_request_is_authorized_but_cited_source_is_not(self):
        original='Создай файл с общим пересказом всей папки, назови сам'
        self.assertTrue(directory_summary_request('Я сказал "'+original+'"'))
        self.assertFalse(directory_summary_request('В источнике написано: "'+original+'"'))
        self.assertFalse(directory_summary_request('Не создавай документ с пересказом всей папки'))
        self.assertEqual(repeated_user_request('Как понять «я сказал»?'),'Как понять «я сказал»?')
        self.assertFalse(directory_summary_request('В источнике есть строка я сказал "'+original+'"'))

    def test_criticism_is_not_inferred_from_a_question_about_a_character(self):
        self.assertIsNone(action_feedback('Почему герой открыл не тот файл?'))
        self.assertIsNone(action_feedback('В документе сказано, что ответ слишком короткий'))

    def test_missing_named_summary_scope_does_not_fall_back_to_whole_workspace(self):
        with patch.object(DirectorySummary,'compose') as compose:
            with self.assertRaises(Exception) as error:
                self.ask('Создай файл с общим пересказом всего в папке «Несуществующая»')
        self.assertIn('не найдена',str(error.exception))
        compose.assert_not_called()

    def test_explicit_summary_filename_is_preserved(self):
        async def compose(instance,inputs,execution):return 'Тематический пересказ.',{'mode':'DIRECTORY_SUMMARY','complete':True}
        with patch.object(DirectorySummary,'compose',compose):
            self.ask('Создай файл Обзор.md с общим пересказом всей папки')
        self.assertTrue((self.env.root/'Обзор.md').is_file())

    def test_technical_failure_during_correction_keeps_already_checked_sentence(self):
        folder=self.env.root/'Корпус'
        folder.mkdir()
        (folder/'Один.md').write_text('Confirmed content. Incorrect conclusion.',encoding='utf-8')
        from pydantic_models import VerificationClaim, QueryInput
        def draft(instance,source,index,text,model,client,findings=None):
            if findings:raise RuntimeError('repair failed')
            return [VerificationClaim(text='Confirmed content.',status='SUPPORTED',basis='EXPLICIT'),
                VerificationClaim(text='Incorrect conclusion.',status='SUPPORTED',basis='EXPLICIT')],[]
        def audit(instance,claims,results,model,client):
            claims[1].status='UNSUPPORTED'
            return [{'claim':c.model_dump(mode='json')} for c in claims]
        with patch.object(DirectorySummary,'draft',draft),patch.object(DirectorySummary,'audit',audit):
            response=asyncio.run(self.env.core.process_input(QueryInput(
                question='Создай файл с общим пересказом всей папки',model='FAST',
                workspace_id=self.env.workspace.id,workspace_directory='Корпус')))[0]
        saved=(folder/'Общий пересказ.md').read_text(encoding='utf-8')
        self.assertIn('Confirmed content.',saved)
        self.assertNotIn('Incorrect conclusion.',saved)
        self.assertEqual(response.plan_status,'PARTIAL')

    def test_draft_output_truncation_cannot_create_a_completed_document(self):
        from llm_contract import GenerationMetrics, StructuredOutputTruncated
        client=Mock()
        client.count_prompt.return_value=100
        client.structured_generate.return_value=SimpleNamespace(metrics=GenerationMetrics(finish_reason='length'),text='{}')
        with self.assertRaises(StructuredOutputTruncated):
            DirectorySummary(self.env.filesystem).draft({'path':'Doc.md'},0,'A long enough factual source sentence.', 'FAST',client)

    def test_source_change_after_generation_prevents_save(self):
        from assistant_models import AssistantError
        with patch.object(DirectorySummary,'unchanged',return_value=False),patch.object(DirectorySummary,'draft',return_value=(None,[])):
            with self.assertRaises(AssistantError) as error:
                self.ask('Создай файл с общим пересказом всей папки')
        self.assertIn('изменились',str(error.exception))
        self.assertFalse((self.env.root/'Общий пересказ.md').exists())

    def test_feedback_resumes_partial_summary_and_reuses_only_unchanged_completed_sources(self):
        folder=self.env.root/'Корпус'
        folder.mkdir()
        for name in ('Готовый','Незавершённый'):
            (folder/(name+'.md')).write_text(name+' содержит самостоятельную важную идею.',encoding='utf-8')
        from pydantic_models import VerificationClaim, QueryInput
        calls=[]
        allow_second=False
        def draft(instance,source,index,text,model,client,findings=None):
            calls.append(source['path'])
            if not allow_second and source['path'].endswith('Незавершённый.md'):raise RuntimeError('temporary failure')
            return [VerificationClaim(text=text,status='SUPPORTED',basis='EXPLICIT')],[]
        def audit(instance,claims,results,model,client):
            return [{'claim':c.model_dump(mode='json')} for c in claims]
        with patch.object(DirectorySummary,'draft',draft),patch.object(DirectorySummary,'audit',audit):
            first=asyncio.run(self.env.core.process_input(QueryInput(
                question='Создай файл с общим пересказом всей папки, назови сам',model='FAST',
                workspace_id=self.env.workspace.id,workspace_directory='Корпус')))[0]
            self.assertEqual(first.plan_status,'PARTIAL')
            allow_second=True
            second=self.ask('Это не пересказ, это огрызок',first.session_id)
        self.assertEqual(second.plan_status,'COMPLETED')
        self.assertEqual(calls.count('Корпус/Готовый.md'),1)
        self.assertEqual(calls.count('Корпус/Незавершённый.md'),2)
        self.assertIn('REUSE_DOCUMENT_SUMMARY',[a.action_type for a in second.actions])
        output=(folder/'Общий пересказ (2).md').read_text(encoding='utf-8')
        self.assertIn('2 из 2',output)
        self.assertNotIn('Общий пересказ.md',output)

    def test_malformed_subject_clause_is_not_accepted_as_a_semantic_comparison(self):
        from evidence_audit import _decode_comparison
        row=dict(claim_subject='В заметке автор подробно описывает развитие одного события и утверждает что всё это было настоящим фактом биографии.',
            source_subject='автор',claim_predicate='описывает событие',source_predicate='описывает событие',
            claim_modality='FACT',source_modality='FACT',source_subject_kind='DOCUMENT',
            relevant_to_question=True,same_subject=True,same_predicate=True,same_modality=True)
        with self.assertRaises(ValueError):_decode_comparison(row)

    def _sentence_resume(self, duplicate=False):
        folder=self.env.root/'Корпус'
        folder.mkdir()
        (folder/'Тема.md').write_text('Подтверждённая мысль. Вторая мысль требует отдельной формулировки.',encoding='utf-8')
        from pydantic_models import VerificationClaim, QueryInput
        resumed=False
        audited=[]
        def claim(text):return VerificationClaim(text=text,status='SUPPORTED',basis='EXPLICIT',
            evidence_quote='Подтверждённая мысль. Вторая мысль требует отдельной формулировки.')
        def draft(instance,source,index,text,model,client,findings=None):
            if resumed:
                self.assertTrue(findings)
                self.assertEqual(len(findings),1)
                return [claim('Подтверждённая мысль.' if duplicate else 'Исправленная вторая мысль.')],[]
            if findings:raise RuntimeError('first correction unavailable')
            return [claim('Подтверждённая мысль.'),claim('Ошибочная мысль.')],[]
        def audit(instance,claims,results,model,client):
            audited.append([c.text for c in claims])
            for c in claims:
                if c.text=='Ошибочная мысль.':c.status='UNSUPPORTED'
            return [{'claim':c.model_dump(mode='json')} for c in claims]
        with patch.object(DirectorySummary,'draft',draft),patch.object(DirectorySummary,'audit',audit):
            first=asyncio.run(self.env.core.process_input(QueryInput(
                question='Создай файл с общим пересказом всей папки, назови сам',model='FAST',
                workspace_id=self.env.workspace.id,workspace_directory='Корпус')))[0]
            self.assertEqual(first.plan_status,'PARTIAL')
            resumed=True
            second=self.ask('Это не пересказ, это огрызок',first.session_id)
        return second,(folder/'Общий пересказ (2).md').read_text(encoding='utf-8'),audited

    def test_partial_file_continuation_keeps_checked_sentence_and_only_checks_replacement(self):
        response,text,audited=self._sentence_resume()
        self.assertEqual(response.plan_status,'COMPLETED')
        self.assertIn('Подтверждённая мысль.',text)
        self.assertIn('Исправленная вторая мысль.',text)
        self.assertNotIn('Ошибочная мысль.',text)
        self.assertEqual(audited[-1],['Исправленная вторая мысль.'])
        self.assertIn('REUSE_DOCUMENT_SENTENCES',[a.action_type for a in response.actions])

    def test_repeating_a_retained_good_sentence_cannot_cover_a_rejected_one(self):
        response,text,audited=self._sentence_resume(duplicate=True)
        self.assertEqual(response.plan_status,'PARTIAL')
        self.assertIn('Пропуски',text)

    def test_full_chunk_coverage_preserves_middle_and_tail_of_long_unbroken_text(self):
        text='a'*10000+'TAIL'
        chunks=list(parts(text,1200))
        self.assertEqual(''.join(chunks),text)
        self.assertTrue(chunks[-1].endswith('TAIL'))
        self.assertTrue(all(len(c)<=1200 for c in chunks))

    def test_inventory_has_all_files_nested_scope_and_detects_new_source_before_write(self):
        folder=self.env.root/'Сводка'
        folder.mkdir()
        for i in range(12):(folder/f'{i}.md').write_text(f'Distinct source {i}.',encoding='utf-8')
        (folder/'Вложенные').mkdir()
        (folder/'Вложенные/Конец.txt').write_text('LAST SOURCE',encoding='utf-8')
        summary=DirectorySummary(self.env.filesystem)
        inventory=summary.inventory(self.env.workspace.id,'Сводка')
        self.assertEqual(len(inventory['sources']),13)
        self.assertTrue(all(s['path'].startswith('Сводка/') for s in inventory['sources']))
        self.assertTrue(summary.unchanged(self.env.workspace.id,inventory,[]))
        (folder/'Добавлен.md').write_text('ADDED',encoding='utf-8')
        self.assertFalse(summary.unchanged(self.env.workspace.id,inventory,[]))

    def test_partial_parts_and_model_failure_remain_in_coverage_report(self):
        folder=self.env.root/'Корпус'
        folder.mkdir()
        for i in range(6):(folder/f'{i}.md').write_text(f'Topic {i}. Unique fact {i}.',encoding='utf-8')
        def draft(instance,source,index,text,model,client):
            from pydantic_models import VerificationClaim
            if source['path'].endswith('5.md'):raise RuntimeError('inference failed')
            return [VerificationClaim(text=text,status='SUPPORTED',basis='EXPLICIT')],[]
        with patch.object(DirectorySummary,'draft',draft),patch.object(DirectorySummary,'audit',return_value=[]):
            from pydantic_models import QueryInput
            response=asyncio.run(self.env.core.process_input(QueryInput(
                question='Создай файл с общим пересказом всей папки, назови сам',model='FAST',
                workspace_id=self.env.workspace.id,workspace_directory='Корпус')))[0]
        saved=(folder/'Общий пересказ.md').read_text(encoding='utf-8')
        self.assertIn('5 из 6',saved)
        self.assertIn('Пропуски',saved)
        self.assertIn('Корпус/5.md',saved)
        self.assertEqual(response.plan_status,'PARTIAL')


if __name__=='__main__':unittest.main()
