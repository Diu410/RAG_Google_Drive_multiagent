"""Regressions: a grounded answer must still answer the user's question."""
import asyncio
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from support import UserEnvironment
from model_profiles import get_profile
from pydantic_models import VerificationClaim, VerificationResult
from response_budget import ResponseBudgetPlanner
from verifier import Verifier, verify_answer
from workspace import _confirmed_statements_for_fallback


SOURCE = {'chunk_id': 'workspace:Архив/Хранитель.md',
          'text': 'Хранитель — смотритель архива. Его зовут Мирон.',
          'metadata': {'source': 'Архив/Хранитель.md'}}


def supported(text='Хранитель — смотритель архива.'):
    return VerificationClaim(text=text, status='SUPPORTED', basis='EXPLICIT',
                             source_chunk_ids=[SOURCE['chunk_id']], evidence_quote=text)


def missing():
    return VerificationClaim(text='Полная биография не указана.', status='INSUFFICIENT_INFORMATION',
                             basis='UNANSWERED', relevant_to_question=True)


class AnswerRelevanceTests(unittest.TestCase):
    def test_identity_rewrite_cannot_trade_the_direct_answer_for_supported_secondary_traits(self):
        role = supported()
        trait = supported('Хранитель хранит карты.')
        initial = VerificationResult(status='PARTIALLY_SUPPORTED', claims=[role, trait,
            VerificationClaim(text='Хранитель бессмертен.', status='UNSUPPORTED')])
        candidate = VerificationResult(status='SUPPORTED', claims=[trait])
        service = SimpleNamespace(verify=Mock(side_effect=[initial, candidate]),
                                  correct=Mock(return_value=trait.text))
        with patch('verifier.VERIFIER_ENABLED', True), patch('verifier.MAX_VERIFICATION_RETRIES', 1):
            answer, result = verify_answer('Кто такой хранитель?', 'Он смотритель. Он бессмертен.',
                [SOURCE], 'WORKSPACE_QA', 'BALANCED', verifier=service)
        self.assertEqual(answer, 'Он смотритель. Он бессмертен.')
        self.assertIs(result, initial)

    def test_a_true_but_unrelated_definition_cannot_be_the_only_overview_answer(self):
        source = {**SOURCE, 'text': 'Волевое действие совершается без рассмотрения альтернатив.'}
        claim = supported(source['text'])
        # Even an optimistic model relevance flag cannot make an unrelated
        # definition into an explanation of the explicitly named organization.
        result = VerificationResult(status='PARTIALLY_SUPPORTED', claims=[claim])
        self.assertIsNone(_confirmed_statements_for_fallback('Что за культ Терпения?', [source], result))

    def test_generation_does_not_import_old_assistant_facts_into_an_independent_question(self):
        env = UserEnvironment()
        self.addCleanup(env.close)
        calls = []
        provider = SimpleNamespace(generate=lambda *a, **kw: calls.append(a[0]) or SimpleNamespace(text=SOURCE['text']))
        with patch('model_runtime.get_llm_provider', return_value=provider):
            asyncio.run(env.execute('Кто такой хранитель?', _workspace_matches=[{
                'path': SOURCE['metadata']['source'], 'content': SOURCE['text'], 'score': 1}],
                _chat_history=[{'role': 'assistant', 'content': 'Хранитель — банкир.'}], _depends_on_history=False))
        self.assertNotIn('банкир', str(calls))

    def test_a_follow_up_preserves_the_resolved_question_and_reference_context(self):
        env = UserEnvironment()
        self.addCleanup(env.close)
        calls = []
        provider = SimpleNamespace(generate=lambda *a, **kw: calls.append(a[0]) or SimpleNamespace(text=SOURCE['text']))
        with patch('model_runtime.get_llm_provider', return_value=provider):
            asyncio.run(env.execute('Кто он?', _workspace_matches=[{
                'path': SOURCE['metadata']['source'], 'content': SOURCE['text'], 'score': 1}],
                _chat_history=[{'role': 'user', 'content': 'В письме упомянут хранитель.'}],
                _depends_on_history=True, _retrieval_question='Кто такой хранитель?'))
        self.assertIn('В письме упомянут хранитель.', str(calls))
        self.assertIn('Самостоятельная формулировка: Кто такой хранитель?', str(calls))

    def test_failed_rewrite_does_not_erase_original_supported_facts(self):
        original = VerificationResult(status='PARTIALLY_SUPPORTED', claims=[supported(), missing()])
        rejected = VerificationResult(status='UNSUPPORTED', claims=[VerificationClaim(
            text='Хранитель — банкир.', status='UNSUPPORTED', basis='EXPLICIT')])
        service = SimpleNamespace(verify=Mock(side_effect=[original, rejected]),
                                  correct=Mock(return_value='Хранитель — банкир.'))
        with patch('verifier.VERIFIER_ENABLED', True), patch('verifier.MAX_VERIFICATION_RETRIES', 1):
            answer, result = verify_answer('Кто такой хранитель?', 'Он смотритель. Биография неизвестна.',
                [SOURCE], 'WORKSPACE_QA', 'BALANCED', verifier=service)
        self.assertEqual(answer, 'Он смотритель. Биография неизвестна.')
        self.assertIs(result, original)

    def test_fallback_reuses_a_complete_checked_answer_instead_of_only_a_quote(self):
        result = VerificationResult(status='PARTIALLY_SUPPORTED', claims=[supported(), VerificationClaim(
            text='Хранитель — банкир.', status='UNSUPPORTED', basis='EXPLICIT')])
        answer = _confirmed_statements_for_fallback('Кто такой хранитель?', [SOURCE], result)
        self.assertTrue(answer.startswith('Хранитель — смотритель архива.'))
        self.assertIn('Архив/Хранитель.md', answer)
        self.assertNotIn('банкир', answer)

    def test_fallback_does_not_present_fragments_or_uncited_claims_as_explanations(self):
        fragment = supported()
        fragment.text = 'утверждая, что он смотритель архива'
        unrelated = supported()
        unrelated.relevant_to_question = False
        wrong_source = supported()
        wrong_source.source_chunk_ids = ['invented-source']
        result = VerificationResult(status='PARTIALLY_SUPPORTED', claims=[fragment, unrelated, wrong_source])
        self.assertIsNone(_confirmed_statements_for_fallback('Кто такой хранитель?', [SOURCE], result))

    def test_correction_keeps_evidence_in_context_despite_a_large_failed_draft(self):
        calls = []
        def count(messages, profile, think):
            return sum(len(m['content']) for m in messages) // 3
        def generate(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(text='Хранитель — смотритель архива.')
        provider = SimpleNamespace(count_prompt=count, generate=generate)
        result = VerificationResult(status='PARTIALLY_SUPPORTED', claims=[supported()] + [
            VerificationClaim(text='Он банкир. ' * 300, status='UNSUPPORTED', basis='EXPLICIT') for _ in range(40)])
        Verifier(provider).correct('Кто такой хранитель?', 'Ошибочный ответ. ' * 2000,
                                  [SOURCE], result, 'BALANCED')
        request = calls[0]
        self.assertEqual(request['messages'][0]['role'], 'system')
        self.assertIn('Хранитель — смотритель архива.', request['messages'][1]['content'])
        payload = request['messages'][1]['content'].split('Доказательства (JSON): ', 1)[1]
        self.assertNotIn('банкир', payload)
        self.assertIn('Не повторяй эти отвергнутые утверждения', request['messages'][1]['content'])
        self.assertNotIn('Ошибочный ответ', request['messages'][1]['content'])
        self.assertLessEqual(count(request['messages'], 'BALANCED', False) + request['max_tokens'] + 32,
                             get_profile('BALANCED').context_size)

    def test_correction_does_not_reuse_supported_but_unrelated_evidence(self):
        calls = []
        unrelated = supported()
        unrelated.relevant_to_question = False
        provider = SimpleNamespace(generate=lambda **kw: calls.append(kw) or SimpleNamespace(text='Не указано.'))
        Verifier(provider).correct('Какой телефон?', 'Он смотритель.', [SOURCE],
            VerificationResult(status='PARTIALLY_SUPPORTED', claims=[unrelated, missing()]), 'BALANCED')
        payload = json.loads(calls[0]['messages'][1]['content'].split('Доказательства (JSON): ', 1)[1])
        self.assertTrue(all('claim' not in item for item in payload))

    def test_correction_can_recover_identity_missing_from_accepted_fragments(self):
        calls = []
        provider = SimpleNamespace(generate=lambda **kw: calls.append(kw) or SimpleNamespace(text='Его зовут Мирон.'))
        Verifier(provider).correct('Как зовут хранителя?', 'Он смотритель и банкир.', [SOURCE],
            VerificationResult(status='PARTIALLY_SUPPORTED', claims=[supported(), VerificationClaim(
                text='Его зовут Мирон и он банкир.', status='UNSUPPORTED', basis='EXPLICIT')]), 'BALANCED')
        payload = json.loads(calls[0]['messages'][1]['content'].split('Доказательства (JSON): ', 1)[1])
        self.assertIn('Его зовут Мирон.', payload[0]['text'])
        self.assertEqual(payload[0]['checked_claims'], ['Хранитель — смотритель архива.'])
        self.assertNotIn('банкир', json.dumps(payload, ensure_ascii=False))

    def test_an_overview_with_known_identity_and_missing_biography_gets_rewritten_and_rechecked(self):
        initial = VerificationResult(status='PARTIALLY_SUPPORTED', claims=[supported(), missing()])
        final = VerificationResult(status='SUPPORTED', claims=[supported()])
        service = SimpleNamespace(verify=Mock(side_effect=[initial, final]),
                                  correct=Mock(return_value='Хранитель — смотритель архива.'))
        with patch('verifier.VERIFIER_ENABLED', True), patch('verifier.MAX_VERIFICATION_RETRIES', 1):
            answer, verification = verify_answer('Кто такой хранитель?', 'Полной биографии нет.',
                [SOURCE], 'WORKSPACE_QA', 'BALANCED', verifier=service)
        self.assertEqual(answer, 'Хранитель — смотритель архива.')
        self.assertEqual(verification.status, 'SUPPORTED')
        self.assertEqual(service.verify.call_count, 2)
        self.assertEqual(service.verify.call_args.args[1], answer)

    def test_safe_partial_answer_retains_its_explanation_and_both_sources(self):
        env = UserEnvironment()
        self.addCleanup(env.close)
        other = {'chunk_id': 'workspace:Архив/Карты.md', 'text': 'Мирон хранит карты.',
                 'metadata': {'source': 'Архив/Карты.md'}}
        fact = VerificationClaim(text=other['text'], status='SUPPORTED', basis='EXPLICIT',
            source_chunk_ids=[other['chunk_id']], evidence_quote=other['text'])
        result = VerificationResult(status='PARTIALLY_SUPPORTED', claims=[supported(), fact, missing()])
        answer = 'Хранитель — смотритель архива. Мирон хранит карты. Полная биография не указана.'
        matches = [{'path': s['metadata']['source'], 'content': s['text'], 'score': 1} for s in [SOURCE, other]]
        budget = ResponseBudgetPlanner().plan('Кто такой хранитель?', 'BALANCED')
        provider = SimpleNamespace(generate=lambda *args, **kwargs: SimpleNamespace(text=answer))
        with patch('model_runtime.get_llm_provider', return_value=provider), \
             patch('settings.VERIFIER_ENABLED', True), \
             patch('verifier.verify_answer', return_value=(answer, result)):
            response = asyncio.run(env.execute('Кто такой хранитель?',
                _workspace_matches=matches, _response_budget=budget))
        self.assertEqual(response.data['final'], answer)
        self.assertEqual(response.data['verification'].claims, result.claims)
        self.assertEqual(response.data['verification'].status, result.status)
        self.assertEqual(response.data['verification'].checked_answer, answer)

    def test_failed_correction_cannot_promote_unchecked_claims(self):
        result = VerificationResult(status='UNSUPPORTED', claims=[VerificationClaim(
            text='Хранитель — банкир.', status='UNSUPPORTED', basis='EXPLICIT')])
        service = SimpleNamespace(verify=Mock(return_value=result), correct=Mock(side_effect=ValueError('too large')))
        with patch('verifier.VERIFIER_ENABLED', True), patch('verifier.MAX_VERIFICATION_RETRIES', 1):
            _, verification = verify_answer('Кто хранитель?', 'Хранитель — банкир.', [SOURCE],
                'WORKSPACE_QA', 'BALANCED', verifier=service)
        self.assertEqual(verification.status, 'UNAVAILABLE')
        self.assertEqual(verification.claims[0].status, 'UNSUPPORTED')
        self.assertNotIn('SUPPORTED', [claim.status for claim in verification.claims])

    def test_short_identity_and_overview_do_not_request_an_unasked_extended_essay(self):
        provider = SimpleNamespace(structured_generate=Mock(side_effect=AssertionError('unneeded model call')))
        with patch('model_runtime.get_llm_provider', return_value=provider):
            for question in ('Кто такой хранитель?', 'Что ты знаешь о хранителе?', 'Who is the keeper?'):
                budget = ResponseBudgetPlanner().plan(question, 'BALANCED')
                self.assertEqual(budget.detail.value, 'NORMAL')
                self.assertEqual(budget.coverage.value, 'BALANCED')
            self.assertEqual(ResponseBudgetPlanner().plan('Подробно расскажи о хранителе.', 'BALANCED').detail.value,
                             'DETAILED')
        provider.structured_generate.assert_not_called()
