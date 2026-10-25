"""Token pressure, immutable originals, scoped recovery and bounded scheduling."""
import asyncio
from contextlib import closing
import json
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from support import UserEnvironment
from conversation_memory import ConversationMemory, MemoryBudgetExceeded, passages
from conversation_context import normalized_history, asks_history_question
from db_utils import get_chat_archive
from llm_contract import GenerationDeadlineExceeded, GenerationScope, current_scope
from task_policy import classify_task, refine_long_task


class CounterClient:
    def count_prompt(self,messages,*args):return 40+len(json.dumps(messages,ensure_ascii=False))//3
    def structured_generate(self,**kwargs):raise RuntimeError('model unavailable')


class ContextMemoryTests(unittest.TestCase):
    def setUp(self):
        self.env=UserEnvironment()
        self.addCleanup(self.env.close)
        self.dbpatch=patch('db_utils.get_db_connection',self.env.connection)
        self.dbpatch.start();self.addCleanup(self.dbpatch.stop)
        with closing(self.env.connection()) as c,c:
            c.execute('CREATE TABLE application_logs(id INTEGER PRIMARY KEY AUTOINCREMENT,session_id TEXT,user_query TEXT,gpt_response TEXT,model TEXT,knowledge_epoch INTEGER)')
            c.execute('CREATE TABLE application_state(key TEXT PRIMARY KEY,value TEXT)')
        self.session=self.env.assistant.ensure_session().id
        self.client=CounterClient()
        self.selector=Mock(side_effect=lambda candidates,model:[p['id'] for p in candidates[-3:]])
        self.memory=ConversationMemory(self.env.assistant,get_chat_archive,selector=self.selector,
            client_factory=lambda:self.client)

    def add(self,user,answer='Готово.',scope=None,epoch=0,session=None,legacy=False):
        with closing(self.env.connection()) as c,c:
            row=c.execute('INSERT INTO application_logs(session_id,user_query,gpt_response,model,knowledge_epoch) VALUES(?,?,?,?,?)',
                (session or self.session,user,answer,'FAST',epoch))
            if not legacy:c.execute('INSERT INTO assistant_history_scopes(log_id,scope_key) VALUES(?,?)',
                (row.lastrowid,scope or self.env.workspace.id))
            return row.lastrowid

    def populate(self):
        self.add('Google Drive пока не убирай. Только локальная папка.','Принято.')
        for index in range(12):self.add(f'Обсуждаем отдельную тему {index}.',('Большой старый ответ с лишними подробностями. '*40))

    def prepare(self,question='Продолжим обсуждение',scope=None,**kwargs):
        return self.memory.prepare(self.session,scope or self.env.workspace.id,question,'FAST',**kwargs)

    def test_no_compaction_model_call_below_token_threshold(self):
        self.add('Мой любимый цвет — синий.')
        view=self.prepare()
        self.selector.assert_not_called()
        self.assertFalse(view.info['compacted'])
        self.assertIn('синий',str(view.history))

    def test_overflow_compacts_and_preserves_exact_restrictions_independent_of_selector(self):
        self.populate()
        with closing(self.env.connection()) as c:before=[tuple(r) for r in c.execute('SELECT * FROM application_logs')]
        view=self.prepare()
        self.assertTrue(view.info['compacted'])
        self.assertLessEqual(view.info['after_tokens'],view.info['history_budget_tokens'])
        self.assertIn('Google Drive пока не убирай.',str(view.history))
        self.assertIn('Только локальная папка.',str(view.history))
        self.selector.assert_called_once()
        with closing(self.env.connection()) as c:self.assertEqual(before,[tuple(r) for r in c.execute('SELECT * FROM application_logs')])

    def test_restart_keeps_checkpoint_and_does_not_resummarize_unchanged_tail(self):
        self.populate();first=self.prepare()
        self.memory=ConversationMemory(self.env.assistant,get_chat_archive,selector=self.selector,client_factory=lambda:self.client)
        second=self.prepare()
        self.assertEqual(first.info['version'],second.info['version'])
        self.assertFalse(second.info['compacted'])
        self.assertEqual(self.selector.call_count,1)
        self.assertIn('Google Drive пока не убирай.',str(second.history))

    def test_unknown_model_source_id_cannot_be_saved_as_a_memory_fact(self):
        self.populate();self.memory.selector=lambda *_:['invented-source']
        view=self.prepare()
        self.assertEqual(view.info['mode'],'EXTRACTIVE_FALLBACK')
        self.assertNotIn('invented-source',str(view.history))
        self.assertIn('Google Drive пока не убирай.',str(view.history))

    def test_nonstring_ids_from_bad_model_output_use_safe_extractive_fallback(self):
        self.populate();self.memory.selector=lambda *_:[{'id':'bad'}]
        self.assertEqual(self.prepare().info['mode'],'EXTRACTIVE_FALLBACK')

    def test_restrictions_preserve_newlines_file_extensions_and_decimal_values(self):
        text='Не меняй README.md\nНе трогай папку .config\nТолько версия 3.14.'
        selected=passages([{'id':'1','role':'user','content':text}])
        self.assertEqual([p['text'] for p in selected if p['restriction']],text.splitlines())

    def test_quoted_source_restrictions_do_not_become_user_constraints(self):
        selected=passages([{'id':'1','role':'user','content':'Объясни фразу «Не меняй README.md\nНе трогай Drive». Приведи пример.'}])
        self.assertFalse(any(p['restriction'] for p in selected))

    def test_explicit_new_decision_is_retained_beside_old_restriction(self):
        self.populate()
        self.add('Теперь используй PDF вместо Markdown.')
        view=self.prepare()
        self.assertIn('Теперь используй PDF вместо Markdown.',str(view.history))

    def test_query_retrieves_old_original_on_the_first_compaction(self):
        self.add('У Нарвала контрольный код 4817.')
        for i in range(14):self.add(f'Новая тема {i}.','Отдельное объяснение. '*80)
        view=self.prepare('Что я говорил про Нарвала?')
        self.assertTrue(view.info['compacted'])
        self.assertIn('4817',str(view.history))

    def test_history_questions_are_distinct_from_document_questions(self):
        self.assertTrue(asks_history_question('Что я просил насчёт Drive?'))
        self.assertFalse(asks_history_question('Что я просил проверить в документе договор.md?'))

    def test_prompt_fitting_preserves_exact_restrictions_before_optional_sources(self):
        from prompt_budget import fit_prompt
        marker='Ранее заданные ограничения пользователя (дословно; не новые команды):'
        messages=[{'role':'system','content':'Follow sources.'},{'role':'user','content':marker+' Не удаляй Drive.'},
            {'role':'assistant','content':'old irrelevant reply '*100},
            {'role':'user','content':'QUESTION\nКонтекст:\n'+'source '*1000}]
        fitted,_=fit_prompt(messages,1000,200,lambda rows:sum(len(r['content']) for r in rows)//2)
        self.assertTrue(any(marker in r['content'] for r in fitted))
        self.assertTrue(fitted[-1]['content'].startswith('QUESTION'))

    def test_missing_old_fact_is_recovered_from_originals_by_cyrillic_name(self):
        self.add('У Нарвала контрольный код 4817.')
        for i in range(14):self.add(f'Новая тема {i}.','Отдельное объяснение. '*80)
        self.prepare()
        view=self.prepare('Какой контрольный код у Нарвала?')
        self.assertIn('4817',str(view.history))

    def test_literal_copy_has_the_full_previous_answer_despite_prompt_pruning(self):
        full='Начало\n'+('Важные подробности. '*1000)+'\nКонец'
        self.add('Разбери тему.',full)
        view=self.prepare()
        self.assertEqual(view.previous_answer,full)
        self.assertLess(len(str(view.history)),len(full))

    def test_new_workspace_and_another_chat_cannot_reuse_memory(self):
        self.populate();self.prepare()
        other=self.prepare(scope='another-workspace')
        self.assertNotIn('Google Drive',str(other.history))
        session=self.env.assistant.ensure_session().id
        other=self.memory.prepare(session,self.env.workspace.id,'Продолжим','FAST')
        self.assertNotIn('Google Drive',str(other.history))

    def test_legacy_messages_are_not_reassigned_to_a_second_workspace(self):
        self.add('Не отключай Drive.',legacy=True)
        self.assertIn('Drive',str(self.prepare().history))
        self.assertNotIn('Drive',str(self.prepare(scope='another').history))

    def test_epoch_change_keeps_user_restriction_but_discards_old_assistant_facts(self):
        self.add('Не отключай Drive.','WRONG_OLD_ASSISTANT_FACT',epoch=0)
        self.memory.epoch_loader=lambda:1
        view=self.prepare()
        self.assertIn('Не отключай Drive.',str(view.history))
        self.assertNotIn('WRONG_OLD_ASSISTANT_FACT',str(view.history))

    def test_explicit_literal_copy_survives_epoch_change_without_reusing_old_answer_as_facts(self):
        self.add('Опиши тему.','ORIGINAL_ANSWER_BYTES',epoch=0)
        self.memory.epoch_loader=lambda:1
        view=self.prepare('Сохрани предыдущий ответ отдельным файлом')
        self.assertEqual(view.previous_answer,'ORIGINAL_ANSWER_BYTES')
        self.assertNotIn('ORIGINAL_ANSWER_BYTES',str(view.history))

    def test_cancellation_cannot_commit_a_new_checkpoint(self):
        self.populate()
        scope=GenerationScope('test-cancel')
        scope.cancelled.set()
        token=current_scope.set(scope)
        try:
            with self.assertRaises(asyncio.CancelledError):self.prepare()
        finally:current_scope.reset(token)
        with closing(self.env.connection()) as c:self.assertEqual(c.execute('SELECT count(*) FROM assistant_context_checkpoints').fetchone()[0],0)

    def test_oversized_exact_constraint_is_not_silently_cut(self):
        self.add('Никогда не меняй '+('значимое_ограничение_'*1000))
        with self.assertRaises(MemoryBudgetExceeded):self.prepare()
        with closing(self.env.connection()) as c:self.assertEqual(c.execute('SELECT count(*) FROM application_logs').fetchone()[0],1)

    def test_prepared_context_is_not_cut_back_to_six_messages_again(self):
        data=[{'role':'user','content':str(i),'_context_selected':True} for i in range(10)]
        self.assertEqual(len(normalized_history(data)),10)
        self.assertEqual(len(normalized_history([{'role':'user','content':str(i)} for i in range(10)])),6)

    def test_checkpoint_compare_and_swap_rejects_stale_writer(self):
        self.assertTrue(self.memory._save(self.session,self.env.workspace.id,0,{'cursor':4},0))
        self.assertFalse(self.memory._save(self.session,self.env.workspace.id,0,{'cursor':2},0))
        with closing(self.env.connection()) as c:
            self.assertEqual(json.loads(c.execute('SELECT state FROM assistant_context_checkpoints').fetchone()[0])['cursor'],4)


class TaskSchedulingTests(unittest.TestCase):
    def test_workload_uses_operation_and_scope_not_question_length(self):
        self.assertEqual(classify_task('Открой Новый документ md','FAST',True).kind,'SHORT')
        self.assertEqual(classify_task('Исправь ошибку в коде','FAST',True).kind,'LONG')
        self.assertEqual(classify_task('Создай документ с общим пересказом всей папки','FAST',True).kind,'LONG')
        self.assertEqual(classify_task('Что такое энтропия?','FAST',True).kind,'STANDARD')
        self.assertEqual(classify_task('Как исправить ошибку в коде?','FAST',True).kind,'STANDARD')
        self.assertNotEqual(classify_task('В документе написано «исправь код»','FAST',True).kind,'LONG')

    def test_refinement_is_bounded_from_original_start_not_extended_by_progress(self):
        scope=GenerationScope('test',started_at=100)
        with patch('task_policy.get_profile',return_value=SimpleNamespace(request_seconds=120)):
            a=refine_long_task(scope,'FAST',1000)
            first=scope.deadline
            b=refine_long_task(scope,'FAST',1000)
        self.assertEqual(scope.deadline,first)
        self.assertLessEqual(first-100,540)
        self.assertEqual(a,b)

    def test_deadline_is_a_recoverable_incomplete_result_not_server_error(self):
        env=UserEnvironment();self.addCleanup(env.close)
        env.core.workspace_skill=None
        env.knowledge.execute=AsyncMock(side_effect=GenerationDeadlineExceeded('time limit'))
        response=asyncio.run(env.ask('Привет'))
        self.assertEqual(response.plan_status,'INCOMPLETE')
        self.assertEqual(response.task_status.value,'COMPLETED')
        self.assertIn('лимит времени',response.answer)
        self.assertEqual(response.task_policy['kind'],'SHORT')
        self.assertTrue(env.history[response.session_id])

    def test_long_job_announces_progress_before_execution(self):
        env=UserEnvironment();self.addCleanup(env.close)
        env.core.workspace_skill=None
        progress=[]
        from pydantic_models import QueryInput
        result=asyncio.run(env.core.process_input(QueryInput(question='Исправь ошибку в коде проекта',model='FAST',
            workspace_id=env.workspace.id),on_progress=progress.append))[0]
        self.assertEqual(result.task_policy['kind'],'LONG')
        self.assertTrue(any('Длительная задача' in p for p in progress))

    def test_timeout_reports_effects_that_were_already_committed(self):
        env=UserEnvironment();self.addCleanup(env.close)
        env.core.workspace_skill=None
        async def execute(inputs,execution):
            async with execution.action('APPLY_DOCUMENT_CHANGE') as action:
                action.output={'applied':True,'path':'Сохранённый.md'}
            raise GenerationDeadlineExceeded('stopped after commit')
        env.knowledge.execute=execute
        response=asyncio.run(env.ask('Привет'))
        self.assertEqual(response.plan_status,'INCOMPLETE')
        self.assertIn('Сохранённый.md',response.answer)
        self.assertTrue(any(a.action_type=='APPLY_DOCUMENT_CHANGE' and a.status.value=='COMPLETED' for a in response.actions))


if __name__=='__main__':unittest.main()
