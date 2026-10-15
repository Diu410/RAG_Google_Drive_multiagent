"""Long answers, nested source names and incomplete model assessments."""
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import support  # Isolate settings before importing application modules.
from llm_contract import StructuredOutputTruncated, GenerationCancelled
from verifier import Verifier, _verification_parts, verify_answer


class VerificationBatchTests(unittest.TestCase):
    source_id = 'workspace:Архив/Персонажи/Хранитель.md'
    fact = 'Хранитель северного архива хранит карты.'

    def sources(self):
        return [{'chunk_id': self.source_id, 'text': self.fact,
                 'metadata': {'source': 'Архив/Персонажи/Хранитель.md'}, 'found_by': ['workspace']}]

    def provider(self, fail=None, unsupported_marker=None):
        calls = []
        def generate(**kwargs):
            prompt = kwargs['messages'][-1]['content']
            fragment = prompt.split('\nОтвет: ', 1)[1].split('\nИсточники JSON:', 1)[0]
            calls.append(fragment)
            if fail:
                fail(fragment, len(calls))
            unsupported = unsupported_marker and unsupported_marker in fragment
            payload = {'status': 'UNSUPPORTED' if unsupported else 'SUPPORTED', 'claims': [{
                'text': fragment.strip(), 'status': 'UNSUPPORTED' if unsupported else 'SUPPORTED',
                'source_chunk_ids': [self.source_id], 'basis': 'EXPLICIT', 'relevant_to_question': True,
                'evidence_quote': '' if unsupported else self.fact}]}
            return SimpleNamespace(text=json.dumps(payload, ensure_ascii=False))
        return SimpleNamespace(structured_generate=generate), calls

    def verify(self, answer, provider):
        # Isolate batching from the separate semantic audit model pass.
        return Verifier(provider, audit_evidence=False).verify('Что хранит хранитель?', answer, self.sources(), 'WORKSPACE_QA', 'BALANCED')

    def test_sentence_partition_preserves_all_text_and_does_not_cut_negation(self):
        text = ('Сведения архива относятся к хранителю.\n' * 20 +
                'Он хранит карты, но не деньги.\n' + 'Очень длинное неделимое предложение ' * 30)
        parts = _verification_parts(text)
        self.assertEqual(''.join(parts), text)
        self.assertTrue(any('Он хранит карты, но не деньги.' in part for part in parts))
        self.assertTrue(any(len(part) > 700 for part in parts))

    def test_long_answer_with_nested_sources_checks_every_part(self):
        answer = (self.fact + '\n') * 35
        provider, calls = self.provider()
        result = self.verify(answer, provider)
        self.assertGreater(len(calls), 1)
        self.assertEqual(''.join(calls), answer)
        self.assertEqual(result.status, 'SUPPORTED')
        self.assertEqual(result.source_chunk_ids, [self.source_id])
        self.assertEqual(len(result.claims), len(calls))

    def test_unsupported_final_part_cannot_be_dropped(self):
        answer = (self.fact + '\n') * 35 + 'Он также работает банкиром.'
        provider, _ = self.provider(unsupported_marker='банкиром')
        result = self.verify(answer, provider)
        self.assertEqual(result.status, 'PARTIALLY_SUPPORTED')
        self.assertTrue(any('банкиром' in claim for claim in result.unsupported_claims))

    def test_truncated_batch_is_retried_as_smaller_complete_sentences(self):
        answer = 'Хранитель хранит карты. Хранитель не хранит деньги.'
        def fail(fragment, count):
            if count == 1:
                raise StructuredOutputTruncated('output budget exhausted')
        provider, calls = self.provider(fail=fail)
        result = self.verify(answer, provider)
        self.assertEqual(len(calls), 3)
        self.assertEqual(''.join(calls[1:]), answer)
        self.assertEqual(result.status, 'SUPPORTED')

    def test_malformed_json_is_retried_without_accepting_partial_claims(self):
        answer = 'Хранитель хранит карты. Хранитель не хранит деньги.'
        def fail(fragment, count):
            if count == 1:
                json.loads('{"claims": [{"text": "unfinished')
        provider, calls = self.provider(fail=fail)
        self.assertEqual(self.verify(answer, provider).status, 'SUPPORTED')
        self.assertEqual(len(calls), 3)

    def test_failed_final_batch_cannot_return_supported_prefix(self):
        answer = (self.fact + '\n') * 35 + 'Окончательное утверждение.'
        def fail(fragment, count):
            if 'Окончательное' in fragment:
                raise StructuredOutputTruncated('output budget exhausted')
        provider, calls = self.provider(fail=fail)
        with patch('verifier.VERIFIER_ENABLED', True):
            original, result = verify_answer('Что хранит?', answer, self.sources(), 'WORKSPACE_QA',
                                             'BALANCED', verifier=Verifier(provider, audit_evidence=False))
        self.assertEqual(original, answer)
        self.assertEqual(result.status, 'UNAVAILABLE')
        self.assertIn('лимит', result.warning)
        self.assertLessEqual(len(calls), 12)

    def test_cancellation_during_batching_is_not_retried_or_swallowed(self):
        def fail(fragment, count):
            if count == 2:
                raise GenerationCancelled()
        provider, calls = self.provider(fail=fail)
        with self.assertRaises(GenerationCancelled):
            self.verify((self.fact + '\n') * 35, provider)
        self.assertEqual(len(calls), 2)

    def test_empty_final_assessment_is_not_hidden_by_supported_parts(self):
        provider, calls = self.provider()
        generate = provider.structured_generate
        def empty_last(**kwargs):
            response = generate(**kwargs)
            if 'Последняя' in calls[-1]:
                return SimpleNamespace(text='{"status":"SUPPORTED","claims":[]}')
            return response
        provider.structured_generate = empty_last
        result = self.verify((self.fact + '\n') * 35 + 'Последняя часть.', provider)
        self.assertEqual(result.status, 'UNAVAILABLE')

    def test_positive_fact_mislabeled_unanswered_does_not_claim_information_is_missing(self):
        for status in ('UNSUPPORTED', 'SUPPORTED', 'INSUFFICIENT_INFORMATION'):
            with self.subTest(status=status):
                payload = {'status': status, 'claims': [{
                    'text': self.fact, 'status': status, 'basis': 'UNANSWERED',
                    'evidence_quote': self.fact, 'source_chunk_ids': [self.source_id]}]}
                provider = SimpleNamespace(structured_generate=lambda **_: SimpleNamespace(text=json.dumps(payload)))
                result = self.verify(self.fact, provider)
                self.assertEqual(result.claims[0].basis, 'EXPLICIT')
                self.assertNotEqual(result.claims[0].status, 'INSUFFICIENT_INFORMATION')
                self.assertEqual(result.claims[0].evidence_quote, self.fact)

    def test_unanswered_positive_fact_without_evidence_remains_unsupported(self):
        payload = {'status': 'UNSUPPORTED', 'claims': [{
            'text': 'Хранитель работает банкиром.', 'status': 'UNSUPPORTED', 'basis': 'UNANSWERED',
            'evidence_quote': '', 'source_chunk_ids': []}]}
        provider = SimpleNamespace(structured_generate=lambda **_: SimpleNamespace(text=json.dumps(payload)))
        result = self.verify('Хранитель работает банкиром.', provider)
        self.assertEqual(result.status, 'UNSUPPORTED')
        self.assertEqual(result.claims[0].basis, 'EXPLICIT')

    def test_no_sources_remains_insufficient_for_a_long_answer(self):
        provider = SimpleNamespace(structured_generate=lambda **_: self.fail('No sources must not call model'))
        result = Verifier(provider).verify('Кто?', self.fact * 50, [], 'WORKSPACE_QA', 'BALANCED')
        self.assertEqual(result.status, 'INSUFFICIENT_INFORMATION')

    def test_budget_failure_is_unavailable_and_does_not_rewrite_the_answer(self):
        provider = SimpleNamespace(count_prompt=lambda *a: 10000,
            structured_generate=lambda **_: self.fail('Oversized prompt must not reach inference'))
        service = Verifier(provider, audit_evidence=False)
        with patch('verifier.VERIFIER_ENABLED', True), patch.object(service, 'correct') as correct:
            answer, result = verify_answer('Что хранит?', self.fact, self.sources(),
                                          'WORKSPACE_QA', 'BALANCED', verifier=service)
        self.assertEqual(answer, self.fact)
        self.assertEqual(result.status, 'UNAVAILABLE')
        self.assertIn('контекст проверки', result.warning)
        self.assertEqual(result.checked_answer, self.fact)
        correct.assert_not_called()

    def test_system_instructions_and_repeated_answer_fit_the_same_counted_budget(self):
        provider, calls = self.provider()
        requests = []
        generate = provider.structured_generate
        def capture(**kwargs):
            requests.append(kwargs)
            return generate(**kwargs)
        provider.structured_generate = capture
        provider.count_prompt = lambda messages, alias, think: sum(len(m['content']) for m in messages) // 3
        sources = self.sources()
        sources[0]['text'] += ' Дополнительный материал источника.' * 400
        original_text = sources[0]['text']
        Verifier(provider, audit_evidence=False).verify('Что хранит?', self.fact, sources, 'WORKSPACE_QA', 'BALANCED')
        self.assertEqual(sources[0]['text'], original_text)
        for request in requests:
            messages = request['messages']
            self.assertEqual(messages[0]['role'], 'system')
            self.assertIn('Не добавляй утверждения из источников', messages[0]['content'])
            self.assertTrue(messages[-1]['content'].endswith(self.fact))
            count = provider.count_prompt(messages, 'assistant', False)
            self.assertLessEqual(count + request['max_tokens'] + 32, 4096)
