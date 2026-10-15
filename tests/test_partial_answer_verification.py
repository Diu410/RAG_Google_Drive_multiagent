"""A slow check must preserve exact assessed sentences without approving the rest."""
import asyncio
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import support
from evidence_audit import EvidenceAuditor, EvidenceRelation
from llm_contract import GenerationDeadlineExceeded
from pydantic_models import VerificationClaim, VerificationResult
from verifier import Verifier, checked_sentences_after_timeout, unavailable_result, verify_answer


SOURCE = {'chunk_id': 'workspace:Письмо.md', 'text':
    'Жди, я приду за тобой. Это всё я — Хранитель архива Мирон Лисов. Я храню карты.',
    'metadata': {'source': 'Письмо.md'}}
FACT = 'В письме Мирон сообщает, что хранит карты.'
UNKNOWN = 'В найденных фрагментах личность адресата не указана.'


class SelectionProvider:
    supports_evidence_ids = True

    def __init__(self, malformed=None, missing_only=None, evidence=None):
        self.calls = []
        self.malformed = malformed
        self.missing_only = set(missing_only or [UNKNOWN, 'В тексте имя адресата не раскрывается.'])
        self.evidence = evidence or (SOURCE['chunk_id'], 'Я храню карты.')

    def count_prompt(self, *args):
        return 100

    def structured_generate(self, **kwargs):
        self.calls.append(kwargs)
        packet = json.loads(kwargs['messages'][-1]['content'])
        assessments = {}
        evidence_id = next(key for key, item in packet['evidence'].items()
                           if (item['source'], item['text']) == self.evidence)
        for key, text in packet['statements'].items():
            missing = 'не указана' in text
            assessments[key] = {'contains_positive_facts': text not in self.missing_only,
                'status': 'INSUFFICIENT_INFORMATION' if missing else 'SUPPORTED',
                'basis': 'UNANSWERED' if missing else 'EXPLICIT',
                'evidence_ids': [] if missing else [evidence_id], 'relevant_to_question': True}
        if self.malformed == 'unknown':
            next(iter(assessments.values()))['evidence_ids'] = ['invented']
        if self.malformed == 'omitted':
            assessments.pop(next(iter(assessments)))
        if self.malformed == 'missing_type':
            next(iter(assessments.values())).pop('contains_positive_facts')
        return SimpleNamespace(text=json.dumps({'assessments': assessments}, ensure_ascii=False))


class PartialAnswerVerificationTests(unittest.TestCase):
    def test_workspace_final_preserves_checked_answer_and_cause_without_recovery_calls(self):
        environment = support.UserEnvironment()
        self.addCleanup(environment.close)
        rejected = 'В письме Мирон требует деньги.'
        draft = UNKNOWN + '\n' + FACT + '\n' + rejected
        partial = VerificationResult(status='UNAVAILABLE', warning=unavailable_result(
            GenerationDeadlineExceeded('deadline')).warning, claims=[
                VerificationClaim(text=UNKNOWN, status='INSUFFICIENT_INFORMATION', basis='UNANSWERED'),
                VerificationClaim(text=FACT, status='SUPPORTED', basis='EXPLICIT',
                    source_chunk_ids=[SOURCE['chunk_id']], evidence_quote='Я храню карты.'),
                VerificationClaim(text=rejected, status='UNAVAILABLE', basis='EXPLICIT')])
        provider = SimpleNamespace(generate=lambda *args, **kwargs: SimpleNamespace(text=draft))
        with patch('model_runtime.get_llm_provider', return_value=provider), \
             patch('settings.VERIFIER_ENABLED', True), \
             patch('verifier.verify_answer', return_value=(draft, partial)) as verify:
            result = asyncio.run(environment.execute('К кому обращается Мирон?', _workspace_matches=[{
                'path': 'Письмо.md', 'content': SOURCE['text'], 'score': 1}]))
        final = result.data['final']
        self.assertIn(UNKNOWN, final)
        self.assertIn(FACT, final)
        self.assertNotIn(rejected, final)
        self.assertNotIn('проверка ответа сейчас недоступна', final)
        self.assertEqual(result.data['verification'].checked_answer, final)
        self.assertEqual(result.data['verification'].status, 'PARTIALLY_SUPPORTED')
        self.assertIn('лимиту времени', result.data['verification'].warning)
        verify.assert_called_once()

    def test_native_assessment_resolves_exact_source_and_retains_unchanged_sentences(self):
        provider = SelectionProvider()
        result = Verifier(provider, audit_evidence=False).verify('К кому обращается Мирон?',
            UNKNOWN + '\n' + FACT, [SOURCE], 'WORKSPACE_QA', 'FAST')
        self.assertEqual(result.status, 'PARTIALLY_SUPPORTED')
        self.assertEqual([c.text for c in result.claims], [UNKNOWN, FACT])
        self.assertEqual(result.claims[1].evidence_quote, 'Я храню карты.')
        self.assertEqual(result.claims[1].source_chunk_ids, [SOURCE['chunk_id']])
        self.assertEqual(len(provider.calls), 1)
        packet = json.loads(provider.calls[0]['messages'][-1]['content'])
        self.assertEqual(list(packet['statements'].values()), [UNKNOWN, FACT])

    def test_bad_native_id_or_omitted_sentence_never_inherits_proof(self):
        for malformed in ('unknown', 'omitted', 'missing_type'):
            with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                Verifier(SelectionProvider(malformed)).verify('Что хранит Мирон?', FACT,
                    [SOURCE], 'WORKSPACE_QA', 'FAST')

    def test_native_selection_still_requires_independent_semantic_audit(self):
        relation = EvidenceRelation(claim_subject='Мирон', source_subject='Мирон',
            claim_predicate='хранит деньги', source_predicate='хранит карты',
            claim_modality='TEXTUAL_STATEMENT', source_modality='TEXTUAL_STATEMENT',
            source_subject_kind='PERSON', same_subject=True, same_predicate=False, same_modality=True)
        auditor = SimpleNamespace(last_assessment=relation, guard_rejection=None, guard_adjustment=None)
        with patch.object(EvidenceAuditor, 'supports_many', return_value=[(False, auditor)]) as audit:
            result = Verifier(SelectionProvider()).verify('Что хранит Мирон?',
                'В письме Мирон сообщает, что хранит деньги.', [SOURCE], 'WORKSPACE_QA', 'FAST')
        audit.assert_called_once()
        self.assertEqual(result.status, 'UNSUPPORTED')

    def test_native_citation_does_not_move_to_another_document_with_the_same_quote(self):
        other = {**SOURCE, 'chunk_id': 'workspace:Другой голос.md',
                 'text': 'Я храню карты. Это всё я — Смотритель Лев.'}
        result = Verifier(SelectionProvider(), audit_evidence=False).verify('Что хранит Мирон?',
            FACT, [other, SOURCE], 'WORKSPACE_QA', 'FAST')
        self.assertEqual(result.claims[0].source_chunk_ids, [SOURCE['chunk_id']])

    def test_mixed_positive_and_missing_information_sentence_is_not_exempted_from_audit(self):
        mixed = 'Имя адресата не указано, но Мирон хранит деньги.'
        with patch.object(EvidenceAuditor, 'supports_many', side_effect=GenerationDeadlineExceeded('deadline')) as audit:
            result = Verifier(SelectionProvider()).verify('Что известно об адресате?', mixed,
                [SOURCE], 'WORKSPACE_QA', 'FAST')
        audit.assert_called_once()
        self.assertEqual(result.claims[0].status, 'UNAVAILABLE')
        self.assertIsNone(checked_sentences_after_timeout(mixed, [SOURCE], result))

    def test_standalone_absence_with_wrong_basis_is_normalized_without_audit(self):
        sentence = 'В тексте имя адресата не раскрывается.'
        with patch.object(EvidenceAuditor, 'supports_many') as audit:
            result = Verifier(SelectionProvider()).verify('К кому обращается Мирон?', sentence,
                [SOURCE], 'WORKSPACE_QA', 'FAST')
        audit.assert_not_called()
        self.assertEqual(result.claims[0].basis, 'UNANSWERED')
        self.assertEqual(result.claims[0].status, 'INSUFFICIENT_INFORMATION')

    def test_short_preserved_answer_can_include_literal_context_with_its_own_proof(self):
        quote = 'Жди, я приду за тобой.'
        result = VerificationResult(status='UNAVAILABLE', warning='deadline', claims=[VerificationClaim(
            text=UNKNOWN, status='INSUFFICIENT_INFORMATION', basis='UNANSWERED',
            source_chunk_ids=[SOURCE['chunk_id']], evidence_quote=quote)])
        selected = checked_sentences_after_timeout(UNKNOWN, [SOURCE], result)
        self.assertIn(UNKNOWN, selected[0])
        self.assertIn(f'В тексте сказано: «{quote.rstrip(".")}».', selected[0])
        self.assertEqual([c.status for c in selected[1].claims], ['INSUFFICIENT_INFORMATION', 'SUPPORTED'])
        self.assertEqual(selected[1].answer_scope, 'LIMITED')
        self.assertEqual(selected[1].checked_answer, selected[0])

    def test_forged_context_quote_cannot_enter_answer_or_verification(self):
        result = VerificationResult(status='UNAVAILABLE', claims=[VerificationClaim(text=UNKNOWN,
            status='INSUFFICIENT_INFORMATION', basis='UNANSWERED', source_chunk_ids=[SOURCE['chunk_id']],
            evidence_quote='Жди, я приду за тобой с деньгами.')])
        selected = checked_sentences_after_timeout(UNKNOWN, [SOURCE], result)
        self.assertEqual(selected[0], UNKNOWN)
        self.assertIsNone(selected[1].claims[0].evidence_quote)
        self.assertEqual(selected[1].claims[0].source_chunk_ids, [])

    def test_absence_with_forged_title_keeps_independent_audit(self):
        sentence = 'В тексте не указано, к кому обращается Мирон (Князь драконов).'
        with patch.object(EvidenceAuditor, 'supports_many', side_effect=GenerationDeadlineExceeded('deadline')) as audit:
            result = Verifier(SelectionProvider()).verify('К кому обращается Мирон?', sentence,
                [SOURCE], 'WORKSPACE_QA', 'FAST')
        audit.assert_called_once()
        self.assertEqual(result.claims[0].status, 'UNAVAILABLE')

    def test_timeout_during_audit_keeps_absence_statement_but_does_not_approve_pending_fact(self):
        with patch.object(EvidenceAuditor, 'supports_many', side_effect=GenerationDeadlineExceeded('deadline')) as audit:
            result = Verifier(SelectionProvider()).verify('К кому обращается Мирон?',
                UNKNOWN + '\n' + FACT, [SOURCE], 'WORKSPACE_QA', 'FAST')
        self.assertEqual(result.status, 'UNAVAILABLE')
        self.assertEqual([c.status for c in result.claims], ['INSUFFICIENT_INFORMATION', 'UNAVAILABLE'])
        selected = checked_sentences_after_timeout(UNKNOWN + '\n' + FACT, [SOURCE], result)
        self.assertEqual(selected[0], UNKNOWN)
        self.assertEqual(selected[1].checked_answer, UNKNOWN)
        self.assertIn('лимиту времени', selected[1].warning)
        audit.assert_called_once()

    def test_timeout_after_independent_audit_preserves_only_completed_results(self):
        third = 'В письме Мирон намерен открыть зал.'
        relation = EvidenceRelation(claim_subject='Мирон', source_subject='Мирон',
            claim_predicate='хранит карты', source_predicate='хранит карты',
            claim_modality='TEXTUAL_STATEMENT', source_modality='TEXTUAL_STATEMENT',
            source_subject_kind='PERSON', same_subject=True, same_predicate=True, same_modality=True)
        auditor = SimpleNamespace(last_assessment=relation, guard_rejection=None, guard_adjustment=None)
        error = GenerationDeadlineExceeded('deadline')
        error.partial_results = [(True, auditor)]
        with patch.object(EvidenceAuditor, 'supports_many', side_effect=error):
            result = Verifier(SelectionProvider()).verify('Что говорит Мирон?',
                FACT + '\n' + third, [SOURCE], 'WORKSPACE_QA', 'FAST')
        self.assertEqual([c.status for c in result.claims], ['SUPPORTED', 'UNAVAILABLE'])
        selected = checked_sentences_after_timeout(FACT + '\n' + third, [SOURCE], result)
        self.assertIn(FACT, selected[0])
        self.assertNotIn(third, selected[0])
        self.assertEqual(selected[1].status, 'SUPPORTED')
        self.assertIn('лимиту времени', selected[1].warning)

    def test_timeout_in_later_part_preserves_previously_assessed_part(self):
        long_fact = 'В письме Мирон хранит карты ' + 'с подробными пометками ' * 40 + '.'
        ready = VerificationResult(status='SUPPORTED', claims=[VerificationClaim(text=long_fact,
            status='SUPPORTED', source_chunk_ids=[SOURCE['chunk_id']], evidence_quote='Я храню карты.')])
        with patch.object(Verifier, '_verify_batch', side_effect=[ready, GenerationDeadlineExceeded('deadline')]) as check:
            result = Verifier(SelectionProvider()).verify('Что хранит Мирон?', long_fact + '\n' + FACT,
                [SOURCE], 'WORKSPACE_QA', 'FAST')
        self.assertEqual(result.status, 'UNAVAILABLE')
        self.assertEqual(result.claims, ready.claims)
        self.assertEqual(check.call_count, 2)

    def test_selection_cannot_keep_changed_text_forged_quote_or_orphaned_pronoun(self):
        claims = [VerificationClaim(text='Он хранит карты.', status='SUPPORTED',
                      source_chunk_ids=[SOURCE['chunk_id']], evidence_quote='Я храню карты.'),
                  VerificationClaim(text=FACT, status='SUPPORTED',
                      source_chunk_ids=[SOURCE['chunk_id']], evidence_quote='Я храню деньги.')]
        result = VerificationResult(status='UNAVAILABLE', claims=claims, warning='time')
        self.assertIsNone(checked_sentences_after_timeout('Он хранит карты.\n' + FACT, [SOURCE], result))
        result.claims = [claims[1].model_copy(update={'evidence_quote': 'Я храню карты.'})]
        self.assertIsNone(checked_sentences_after_timeout('Мирон хранит карты.', [SOURCE], result))

    def test_deadline_warning_identifies_cause_without_exposing_error_payload(self):
        result = unavailable_result(GenerationDeadlineExceeded('PRIVATE_SOURCE_TEXT'))
        self.assertIn('лимиту времени', result.warning)
        self.assertNotIn('PRIVATE_SOURCE_TEXT', result.warning)

    def test_parameter_absence_and_completed_project_fact_use_the_same_pipeline(self):
        source = {'chunk_id': 'workspace:Отчёт.md', 'text': 'Сборка проекта занимает 8 минут.',
                  'metadata': {'source': 'Отчёт.md'}}
        missing = 'В отчёте версия компилятора не указана.'
        fact = 'В отчёте сказано, что сборка проекта занимает 8 минут.'
        pending = 'Проект использует второй сервер.'
        provider = SelectionProvider(missing_only=[missing], evidence=(source['chunk_id'], source['text']))
        relation = EvidenceRelation(claim_subject='сборка проекта', source_subject='сборка проекта',
            claim_predicate='занимает 8 минут', source_predicate='занимает 8 минут',
            claim_modality='TEXTUAL_STATEMENT', source_modality='TEXTUAL_STATEMENT',
            source_subject_kind='CONCEPT', same_subject=True, same_predicate=True, same_modality=True)
        auditor = SimpleNamespace(last_assessment=relation, guard_rejection=None, guard_adjustment=None)
        error = GenerationDeadlineExceeded('deadline')
        error.partial_results = [(True, auditor)]
        with patch.object(EvidenceAuditor, 'supports_many', side_effect=error):
            checked = Verifier(provider).verify('Какие параметры сборки указаны в отчёте?',
                '\n'.join([missing, fact, pending]), [source], 'WORKSPACE_QA', 'FAST')
        self.assertEqual([c.status for c in checked.claims],
                         ['INSUFFICIENT_INFORMATION', 'SUPPORTED', 'UNAVAILABLE'])
        selected = checked_sentences_after_timeout('\n'.join([missing, fact, pending]), [source], checked)
        self.assertIn(missing, selected[0])
        self.assertIn(fact, selected[0])
        self.assertNotIn(pending, selected[0])
        self.assertEqual(len(selected[1].claims), 2)

    def test_mixed_parameter_and_absence_cannot_be_masked_as_unanswered(self):
        source = {'chunk_id': 'workspace:Параметры.md', 'text': 'Сборка проекта занимает 8 минут.',
                  'metadata': {'source': 'Параметры.md'}}
        mixed = 'Версия не указана, но сборка проекта занимает 80 минут.'
        provider = SelectionProvider(evidence=(source['chunk_id'], source['text']))
        result = Verifier(provider).verify('Каковы параметры сборки?', mixed,
            [source], 'WORKSPACE_QA', 'FAST')
        self.assertEqual(result.claims[0].basis, 'EXPLICIT')
        self.assertEqual(result.claims[0].status, 'UNSUPPORTED')

    def test_relations_between_services_are_not_approved_by_matching_quote_alone(self):
        source = {'chunk_id': 'workspace:Архитектура.md', 'text': 'Сервис А передаёт события сервису Б.',
                  'metadata': {'source': 'Архитектура.md'}}
        reversed_relation = 'Сервис Б передаёт события сервису А.'
        provider = SelectionProvider(evidence=(source['chunk_id'], source['text']))
        relation = EvidenceRelation(claim_subject='сервис Б', source_subject='сервис А',
            claim_predicate='передаёт события сервису А', source_predicate='передаёт события сервису Б',
            claim_modality='FACT', source_modality='FACT', source_subject_kind='CONCEPT',
            same_subject=False, same_predicate=False, same_modality=True)
        auditor = SimpleNamespace(last_assessment=relation, guard_rejection=None, guard_adjustment=None)
        with patch.object(EvidenceAuditor, 'supports_many', return_value=[(False, auditor)]):
            result = Verifier(provider).verify('Как связаны сервисы?', reversed_relation,
                [source], 'WORKSPACE_QA', 'FAST')
        self.assertEqual(result.status, 'UNSUPPORTED')
        self.assertIsNone(checked_sentences_after_timeout(reversed_relation, [source], result))

    def test_presentation_only_native_batch_does_not_request_fact_assessment(self):
        provider = SelectionProvider()
        result = Verifier(provider).verify('Какие документы найдены?', 'Источники:\nПисьмо.md',
            [SOURCE], 'WORKSPACE_QA', 'FAST')
        self.assertEqual(result.claims, [])
        self.assertEqual(provider.calls, [])

    def test_compound_statement_can_cite_multiple_documents_without_rewriting(self):
        sources = [
            {'chunk_id': 'workspace:Отправитель.md', 'text': 'Сервис А передаёт события сервису Б.',
             'metadata': {'source': 'Отправитель.md'}},
            {'chunk_id': 'workspace:Получатель.md', 'text': 'Сервис Б сохраняет полученные события.',
             'metadata': {'source': 'Получатель.md'}}]
        sentence = 'Сервис А передаёт события сервису Б, а сервис Б сохраняет полученные события.'

        class MultipleEvidenceProvider(SelectionProvider):
            def structured_generate(self, **kwargs):
                self.calls.append(kwargs)
                packet = json.loads(kwargs['messages'][-1]['content'])
                return SimpleNamespace(text=json.dumps({'assessments': {
                    key: {'contains_positive_facts': True, 'status': 'SUPPORTED', 'basis': 'EXPLICIT',
                          'evidence_ids': list(packet['evidence']), 'relevant_to_question': True}
                    for key in packet['statements']}}))

        provider = MultipleEvidenceProvider()
        result = Verifier(provider, audit_evidence=False).verify('Как сервисы обрабатывают события?',
            sentence, sources, 'WORKSPACE_QA', 'FAST')
        self.assertEqual(result.status, 'SUPPORTED')
        self.assertEqual(result.claims[0].text, sentence)
        self.assertEqual(result.claims[0].source_chunk_ids, [s['chunk_id'] for s in sources])
        self.assertEqual(result.claims[0].evidence_quote, ' … '.join(s['text'] for s in sources))

    def test_failed_correction_preserves_original_completed_checks_and_technical_cause(self):
        pending = 'Сервис запущен на другом компьютере.'
        draft = FACT + '\n' + pending
        ready = VerificationResult(status='PARTIALLY_SUPPORTED', checked_answer=draft,
            claims=[VerificationClaim(text=FACT, status='SUPPORTED', basis='EXPLICIT', source_chunk_ids=[SOURCE['chunk_id']],
                                      evidence_quote='Я храню карты.'),
                    VerificationClaim(text=pending, status='UNSUPPORTED')])
        service = SimpleNamespace(verify=lambda *args, **kwargs: ready,
                                  correct=unittest.mock.Mock(side_effect=GenerationDeadlineExceeded('PRIVATE_TEXT')))
        with patch('verifier.VERIFIER_ENABLED', True):
            answer, checked = verify_answer('Какие сведения есть в документе?', draft, [SOURCE],
                                             'WORKSPACE_QA', 'FAST', verifier=service)
        self.assertEqual(answer, draft)
        self.assertEqual(checked.status, 'UNAVAILABLE')
        self.assertEqual(checked.claims, ready.claims)
        self.assertIn('лимиту времени', checked.warning)
        self.assertNotIn('PRIVATE_TEXT', checked.warning)
        selected = checked_sentences_after_timeout(answer, [SOURCE], checked)
        self.assertIn(FACT, selected[0])
        self.assertNotIn(pending, selected[0])
        self.assertIn('лимиту времени', selected[1].warning)

    def test_unavailable_recheck_does_not_lose_timeout_or_transfer_proof_to_new_wording(self):
        pending = 'Сервис запущен на другом компьютере.'
        draft = FACT + '\n' + pending
        ready = VerificationResult(status='PARTIALLY_SUPPORTED', checked_answer=draft,
            claims=[VerificationClaim(text=FACT, status='SUPPORTED', basis='EXPLICIT', source_chunk_ids=[SOURCE['chunk_id']],
                                      evidence_quote='Я храню карты.'),
                    VerificationClaim(text=pending, status='UNSUPPORTED')])
        candidate = 'В документе описан другой факт.'
        failed_check = unavailable_result(GenerationDeadlineExceeded('PRIVATE_TEXT'))
        service = SimpleNamespace(verify=unittest.mock.Mock(side_effect=[ready, failed_check]),
                                  correct=unittest.mock.Mock(return_value=candidate))
        with patch('verifier.VERIFIER_ENABLED', True):
            answer, checked = verify_answer('Какие сведения есть в документе?', draft, [SOURCE],
                                             'WORKSPACE_QA', 'FAST', verifier=service)
        self.assertEqual(answer, draft)
        self.assertEqual(checked.checked_answer, draft)
        self.assertEqual(checked.claims, ready.claims)
        self.assertEqual(checked.status, 'UNAVAILABLE')
        self.assertEqual(checked.warning, failed_check.warning)
        self.assertNotIn(candidate, [c.text for c in checked.claims])


if __name__ == '__main__':
    unittest.main()
