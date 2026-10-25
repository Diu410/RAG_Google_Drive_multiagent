"""Corpus integrity and independent expectations for hard real-model runs."""
import asyncio
from pathlib import Path
import re
from types import SimpleNamespace
import unittest

import support
from complex_workspace_cases import CASES, REVIEW_CASES, EXAMPLE_ROOT
from benchmark_complex_workspace import answer_checks
from agent_planner import is_single_information_request
from evidence_calculations import utc_normalizations, arithmetic_checks, table_sums, source_time_comparison


class ComplexWorkspaceTests(unittest.TestCase):
    def test_cases_have_real_sources_and_unique_ids(self):
        self.assertGreaterEqual(len(CASES), 11)
        self.assertGreaterEqual(len(REVIEW_CASES), 8)
        self.assertEqual(len({c['id'] for c in CASES}),len(CASES))
        self.assertEqual(len({c['id'] for c in REVIEW_CASES}),len(REVIEW_CASES))
        for case in CASES:
            self.assertTrue(case['facts'])
            for path in case['required_sources']:
                self.assertTrue((EXAMPLE_ROOT/path).is_file(),path)
            for patterns in case['facts'].values():
                for pattern in patterns:re.compile(pattern)
        self.assertNotIn('CASES', '\n'.join(path.read_text(encoding='utf-8') for path in EXAMPLE_ROOT.iterdir()))

    def test_long_report_has_separated_obsolete_and_current_evidence(self):
        text=(EXAMPLE_ROOT/'Длинный_отчёт_L90.md').read_text(encoding='utf-8')
        self.assertGreater(len(text),20_000)
        self.assertLess(text.index('LEG-104'),1000)
        self.assertGreater(text.index('ZX-682'),len(text)-800)
        self.assertIn('Для ночной смены этот допуск не действует',text[-800:])

    def test_checker_does_not_treat_correct_words_as_completed_verification(self):
        case=CASES[0]
        response=SimpleNamespace(answer='Лимит 120 000, но закупать нельзя.',
            task_status=SimpleNamespace(value='COMPLETED'),verification=SimpleNamespace(status='UNAVAILABLE',checked_answer=''),
            results=[SimpleNamespace(metadata={'source':path}) for path in case['required_sources']])
        checks=answer_checks(case,response)
        self.assertTrue(checks['current_limit'])
        self.assertFalse(checks['review_completed'])
        self.assertFalse(checks['answer_supported'])
        self.assertFalse(all(checks.values()))

    def test_checker_requires_both_documents_and_same_checked_answer(self):
        case=next(c for c in CASES if c['id']=='two-hop-identifiers')
        answer='Контракт C-19, партия B-620 из Северного архива.'
        response=SimpleNamespace(answer=answer,task_status=SimpleNamespace(value='COMPLETED'),
            verification=SimpleNamespace(status='SUPPORTED',checked_answer='Другой ответ'),
            results=[SimpleNamespace(metadata={'source':case['required_sources'][0]})])
        checks=answer_checks(case,response)
        self.assertFalse(checks['answer_supported'])
        self.assertFalse(checks['source:'+case['required_sources'][1]])

    def test_labeled_operands_are_the_same_calculation_not_a_content_failure(self):
        case=next(c for c in CASES if c['id']=='filtered-table-sum')
        answer='К выдаче готовы 17 штук. Расчёт: 12 (партия L-11) + 5 (партия L-13) = 17.'
        response=SimpleNamespace(answer=answer,task_status=SimpleNamespace(value='COMPLETED'),
            verification=SimpleNamespace(status='SUPPORTED',checked_answer=answer),
            results=[SimpleNamespace(metadata={'source':case['required_sources'][0]})])
        self.assertTrue(all(answer_checks(case,response).values()))
        response.answer=response.answer.replace('+ 5','+ 7')
        response.verification.checked_answer=response.answer
        self.assertFalse(answer_checks(case,response)['arithmetic'])

    def test_checker_accepts_only_expected_missing_fact_with_all_positive_claims_checked(self):
        case=next(c for c in CASES if c['id']=='injection-and-missing')
        positive=SimpleNamespace(text='Код Vela — VELA-773. Ответственная Ольга Ильина.',status='SUPPORTED',basis='EXPLICIT')
        missing=SimpleNamespace(text='Телефон не указан.',status='INSUFFICIENT_INFORMATION',basis='UNANSWERED')
        answer=positive.text+' '+missing.text
        check=SimpleNamespace(status='PARTIALLY_SUPPORTED',checked_answer=answer,claims=[positive,missing])
        response=SimpleNamespace(answer=answer,verification=check,task_status=SimpleNamespace(value='COMPLETED'),
            results=[SimpleNamespace(metadata={'source':case['required_sources'][0]})])
        self.assertTrue(all(answer_checks(case,response).values()))
        positive.status='UNSUPPORTED'
        self.assertFalse(answer_checks(case,response)['answer_supported'])
        positive.status='SUPPORTED'
        missing.text='Неизвестна погода на завтра.'
        self.assertFalse(answer_checks(case,response)['answer_supported'])

    def test_information_questions_do_not_become_repeated_workflows(self):
        for case in CASES:
            if case['id']=='unresolved-conflict':continue  # Two explicit questions keep workflow detection conservative.
            with self.subTest(case=case['id']):
                self.assertTrue(is_single_information_request(case['question']))
        for question in ('Какой код? Сохрани ответ в файл Итог.md',
                         'Кто подписал? Затем открой файл',
                         'Какое правило? Сохрани условие в файл Итог.md',
                         'Кто отправил письмо? Покажи файлы',
                         'Что известно? Найди документ и сравни версии'):
            with self.subTest(question=question):self.assertFalse(is_single_information_request(question))

    def test_native_passages_bind_headings_and_table_headers_to_content(self):
        from answer_review import _sources
        source={'chunk_id':'csv','text':'позиция,количество,статус\nмодуль A,12,готово\nмодуль A,7,карантин\nмодуль A,5,готово\n'}
        sources,evidence=_sources([source],contextual_passages=True)
        self.assertEqual(len(evidence),1)
        self.assertEqual(sources[0]['passages']['0'],source['text'].strip())
        source={'chunk_id':'title','text':'# Длинное название проекта\n\nУтверждённый код проекта — ACT-77.'}
        _,evidence=_sources([source],contextual_passages=True)
        self.assertIn('ACT-77',evidence['0']['text'])
        self.assertIn(evidence['0']['text'],source['text'])

    def test_literal_name_case_binding_preserves_identifiers_and_does_not_name_anonymous_voice(self):
        from evidence_audit import EvidenceRelation, _speaker_binding_is_explicit
        assessment=EvidenceRelation(claim_subject='Анна Петрова P-22',source_subject='Анна Петрова P-22',
            claim_predicate='должна подписать макет',source_predicate='должна подписать макет',
            same_subject=True,same_predicate=True,same_modality=True,source_subject_kind='PERSON')
        self.assertTrue(_speaker_binding_is_explicit(assessment,'Схему передали дизайнеру Анне Петровой P-22.',[]))
        self.assertFalse(_speaker_binding_is_explicit(assessment,'Схему передали дизайнеру Анне Петровой P-23.',[]))
        self.assertFalse(_speaker_binding_is_explicit(assessment,'Я должна подписать макет.',[]))

    def test_explicit_utc_normalization_keeps_day_shift_and_rejects_ambiguous_local_times(self):
        rows=utc_normalizations('12:15 по UTC+03:00; 09:30 UTC; 00:20 UTC+03:00; 23:50 UTC-02:00.')
        self.assertEqual([(r['utc'],r['day_shift']) for r in rows],
            [('09:15 UTC',0),('09:30 UTC',0),('21:20 UTC',-1),('01:50 UTC',1)])
        self.assertFalse(utc_normalizations('12:15 местного времени; 27:00 UTC; 12:70 UTC; 12:00 UTC+15:00'))

    def test_arithmetic_checks_are_exact_and_never_execute_input(self):
        rows=arithmetic_checks('12 + 5 = 17; 12 + 7 = 17; 0,1 + 0,2 = 0,3; 5 ÷ 2 = 2,5')
        self.assertEqual([r['correct'] for r in rows],[True,False,True,True])
        self.assertFalse(arithmetic_checks("__import__('os').system('anything')"))
        self.assertFalse(arithmetic_checks('1 / 0 = 3'))

    def test_bad_arithmetic_is_rejected_even_when_model_comparison_agrees(self):
        from evidence_audit import EvidenceAuditor, EvidenceRelation
        from pydantic_models import VerificationClaim
        claim=VerificationClaim(text='Итого 12 + 7 = 17.',status='SUPPORTED',basis='EXPLICIT',
            source_chunk_ids=['table'],evidence_quote='Первая партия: 12; вторая партия: 7.')
        assessment=EvidenceRelation(claim_subject='итог',source_subject='итог',claim_predicate='17',source_predicate='17',
            source_subject_kind='DOCUMENT',same_subject=True,same_predicate=True,same_modality=True)
        auditor=EvidenceAuditor(None,None)
        self.assertFalse(auditor.evaluate(claim,[],assessment))
        self.assertEqual(auditor.guard_rejection,'arithmetic_mismatch')

    def test_table_sum_labels_filters_and_keeps_quarantined_rows_separate(self):
        text=(EXAMPLE_ROOT/'Склад_Q8.csv').read_text(encoding='utf-8')
        rows=table_sums(text)
        selected=[row for row in rows if set(row['group'].values())=={'модуль Q-8','готово к выдаче'}]
        self.assertEqual(len(selected),1)
        self.assertEqual(selected[0]['sum'],'17')
        self.assertEqual(selected[0]['source_lines'],[2,4])
        self.assertTrue(any(row['sum']=='24' and list(row['group'].values())==['модуль Q-8'] for row in rows))
        self.assertFalse(table_sums('колонка,число\nA,2\nB,неизвестно'))
        self.assertFalse(table_sums('колонка,число\nA,2\nA,3,лишнее'))

    def test_audit_context_keeps_literal_table_rows_and_headers(self):
        from evidence_audit import EvidenceAuditor
        text=(EXAMPLE_ROOT/'Склад_Q8.csv').read_text(encoding='utf-8')
        context=EvidenceAuditor.source_context(text,text)
        self.assertEqual(context,text)
        self.assertTrue(table_sums(context))

    def test_source_footer_is_metadata_only_when_it_contains_known_paths(self):
        from answer_review import _statements
        sources=[{'metadata':{'source':'a.md'}},{'metadata':{'source':'b.csv'}}]
        for footer in ('Использованные файлы: a.md.', 'Источники: `a.md`, `b.csv`.',
                       '**Sources:** `a.md`.'):
            with self.subTest(footer=footer):
                # The bold label is split as a heading by the sentence parser.
                self.assertEqual(len(_statements('Пакет доставлен.\n\n'+footer,sources)),1)
        for footer in ('Источники: a.md утверждает, что все партии приняты.',
                       'Использованные файлы: неизвестный.md.'):
            with self.subTest(footer=footer):
                self.assertGreater(len(_statements('Пакет доставлен.\n\n'+footer,sources)),1)

    @staticmethod
    def time_source(text=None):
        return {'chunk_id':'time:1','metadata':{'source':'Хронология_P3.md'},
                'text':text or (EXAMPLE_ROOT/'Хронология_P3.md').read_text(encoding='utf-8')}

    def test_explicit_source_time_comparison_proves_order_and_day_shift(self):
        source=self.time_source()
        result=source_time_comparison('Что раньше для P-3?', [source])
        self.assertEqual(result['difference_minutes'],15)
        self.assertIn('опубликован',result['earlier']['quote'])
        self.assertIn(result['earlier']['quote'],source['text'])
        text=source['text'].replace('12:15 по UTC+03:00','00:20 по UTC+03:00')
        result=source_time_comparison('Что раньше для P-3?', [self.time_source(text)])
        self.assertEqual(result['earlier']['day_shift'],-1)
        self.assertEqual(result['difference_minutes'],730)

    def test_source_time_calculator_refuses_ambiguous_or_invalid_evidence(self):
        source=self.time_source()
        self.assertIsNone(source_time_comparison('Что раньше?', [source]))
        self.assertIsNone(source_time_comparison('Что раньше для P-30?', [source]))
        self.assertIsNone(source_time_comparison('Что раньше для P-3?', [source,{**source,'chunk_id':'time:2'}]))
        for old,new in [('12 сентября','31 февраля'),('Все записи относятся к','Первая запись относится к'),
                        ('Все записи относятся к','Все записи относятся не к'),
                        ('Все записи относятся к','Все записи относятся к отчёту за'),
                        ('09:30 UTC','09:30 местного времени'),('09:30 UTC','09:30 UTC; 10:00 UTC'),
                        ('Отмена выпуска P-3','Отмена выпуска P-4'),
                        ('Отмена выпуска P-3','На следующий день отмена выпуска P-3')]:
            with self.subTest(new=new):
                self.assertIsNone(source_time_comparison('Что раньше для P-3?',
                    [self.time_source(source['text'].replace(old,new))]))

    def test_calculator_runs_through_chat_without_model_and_journals_its_proof(self):
        env=support.UserEnvironment()
        self.addCleanup(env.close)
        (env.root/'Хронология_P3.md').write_text(self.time_source()['text'],encoding='utf-8')
        result=asyncio.run(env.ask('Что раньше для P-3: публикация или регистрация отмены? Кратко.'))
        self.assertEqual(result.strategy,'WORKSPACE_CALCULATION')
        self.assertEqual(result.verification.review_mode,'CALCULATION')
        self.assertEqual(result.verification.checked_answer,result.answer)
        self.assertEqual(result.verification.review_details['difference_minutes'],15)
        self.assertTrue(any(a.action_type=='CALCULATE_SOURCE_TIMES' for a in result.actions))


if __name__=='__main__':unittest.main()
