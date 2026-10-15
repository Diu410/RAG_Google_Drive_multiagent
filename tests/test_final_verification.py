"""The verification panel describes the shown answer, not rejected draft prose."""
import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from support import UserEnvironment
from pydantic_models import VerificationClaim, VerificationResult
from response_budget import ResponseBudgetPlanner
from verifier import verification_for_final_text


SOURCE = {'chunk_id': 'workspace:Архив/Хранитель.md',
          'text': 'Хранитель — смотритель архива. Его зовут Мирон.',
          'metadata': {'source': 'Архив/Хранитель.md'}, 'found_by': ['workspace']}
KNOWN = 'Хранитель — смотритель архива.'


def draft_result():
    return VerificationResult(status='PARTIALLY_SUPPORTED', checked_answer=KNOWN + ' Он банкир.',
        claims=[VerificationClaim(text=KNOWN, status='SUPPORTED', basis='EXPLICIT',
            source_chunk_ids=[SOURCE['chunk_id']], evidence_quote=KNOWN),
            VerificationClaim(text='Он банкир.', status='UNSUPPORTED', basis='EXPLICIT')],
        unsupported_claims=['Он банкир.'], confidence=.2)


class FinalVerificationTests(unittest.TestCase):
    def test_supported_aside_is_removed_without_regenerating_a_checked_explanation(self):
        env = UserEnvironment()
        self.addCleanup(env.close)
        sentences = ['Северный архив хранит карты.', 'Северный архив основан в 2001 году.',
                     'Мирон работает банкиром.']
        draft = ' '.join(sentences)
        source_id = 'workspace:Архив.md'
        checked = VerificationResult(status='SUPPORTED', checked_answer=draft, claims=[
            VerificationClaim(text=text, status='SUPPORTED', basis='EXPLICIT', evidence_quote=text,
                source_chunk_ids=[source_id], relevant_to_question=index < 2)
            for index, text in enumerate(sentences)])
        provider = SimpleNamespace(generate=lambda *a, **kw: self.fail('Must retain already checked sentences'))
        with patch('model_runtime.get_llm_provider', return_value=provider), \
             patch('settings.VERIFIER_ENABLED', True), \
             patch('overview_synthesis.OverviewSynthesizer.compose', return_value=draft) as compose, \
             patch('verifier.Verifier.verify', return_value=checked) as verify:
            result = asyncio.run(env.execute('Что за Северный архив?', _workspace_matches=[{
                'path': 'Архив.md', 'content': draft, 'score': 1}]))
        compose.assert_called_once()
        verify.assert_called_once()
        for text in sentences[:2]:
            self.assertIn(text, result.data['final'])
        self.assertNotIn(sentences[-1], result.data['final'])
        self.assertEqual(result.data['verification'].answer_scope, 'EXPLANATORY')
        self.assertEqual(len(result.data['verification'].claims), 2)

    def test_technical_failure_keeps_its_cause_without_retrying_synthesis(self):
        env = UserEnvironment()
        self.addCleanup(env.close)
        draft = 'Северный архив хранит карты.'
        warning = 'Не удалось вместить источники в контекст проверки модели.'
        unavailable = VerificationResult(status='UNAVAILABLE', warning=warning)
        provider = SimpleNamespace(generate=lambda *a, **kw: SimpleNamespace(text=draft))
        with patch('model_runtime.get_llm_provider', return_value=provider), \
             patch('settings.VERIFIER_ENABLED', True), \
             patch('overview_synthesis.OverviewSynthesizer.compose', return_value=draft) as compose, \
             patch('verifier.Verifier.verify', return_value=unavailable) as verify:
            result = asyncio.run(env.execute('Что за Северный архив?', _workspace_matches=[{
                'path': 'Архив.md', 'content': draft, 'score': 1}]))
        compose.assert_called_once()
        verify.assert_called_once()
        self.assertIn(warning, result.data['final'])
        self.assertNotIn('Уточните вопрос', result.data['final'])
        self.assertEqual(result.data['verification'].status, 'UNAVAILABLE')
        self.assertEqual(result.data['verification'].checked_answer, result.data['final'])
        self.assertEqual(result.data['draft'], draft)

    def test_documentary_recovery_requires_successful_verification_and_journals_the_candidate(self):
        from contextlib import closing
        import json
        text = 'Это всё я — Хранитель Северного архива Мирон.'
        source = {'path': 'Письмо.md', 'content': text, 'score': 1}
        original = VerificationResult(status='UNSUPPORTED', claims=[VerificationClaim(
            text='Мирон бессмертен.', status='UNSUPPORTED')])
        for status in ('SUPPORTED', 'UNSUPPORTED', 'UNAVAILABLE'):
            with self.subTest(status=status):
                env = UserEnvironment()
                try:
                    provider = SimpleNamespace(generate=lambda *a, **kw: SimpleNamespace(text='Мирон бессмертен.'))
                    recovered = VerificationResult(status=status, claims=[VerificationClaim(
                        text='Мирон называет себя «Хранитель Северного архива Мирон».', status=status,
                        basis='EXPLICIT', source_chunk_ids=['workspace:Письмо.md'], evidence_quote=text)])
                    with patch('model_runtime.get_llm_provider', return_value=provider), \
                         patch('settings.VERIFIER_ENABLED', True), \
                         patch('verifier.verify_answer', return_value=('Мирон бессмертен.', original)), \
                         patch('verifier.Verifier.verify', return_value=recovered) as check:
                        response = asyncio.run(env.execute('Кто такой Мирон?', _workspace_matches=[source]))
                    candidate = check.call_args.args[1]
                    self.assertTrue(candidate.startswith('В тексте Мирон называет себя'))
                    if status == 'SUPPORTED':
                        self.assertEqual(response.data['final'], candidate)
                        self.assertEqual(response.data['verification'].checked_answer, candidate)
                    else:
                        self.assertNotEqual(response.data['final'], candidate)
                        self.assertNotEqual(response.data['verification'].status, 'SUPPORTED')
                    with closing(env.connection()) as conn:
                        row = conn.execute("SELECT output FROM assistant_actions WHERE action_type='VERIFY_DOCUMENTARY_OVERVIEW'").fetchone()
                    self.assertEqual(json.loads(row['output'])['data']['candidate'], candidate)
                finally:
                    env.close()

    def test_trimmed_answer_does_not_display_rejected_draft_claims(self):
        result = verification_for_final_text(KNOWN, [SOURCE], draft_result())
        self.assertEqual(result.checked_answer, KNOWN)
        self.assertEqual(result.status, 'SUPPORTED')
        self.assertEqual([c.text for c in result.claims], [KNOWN])
        self.assertEqual(result.unsupported_claims, [])
        self.assertEqual(result.source_chunk_ids, [SOURCE['chunk_id']])
        self.assertGreater(result.confidence, .2)

    def test_paraphrase_or_new_fact_cannot_inherit_a_previous_verification(self):
        result = verification_for_final_text('Хранитель — директор банка.', [SOURCE], draft_result())
        self.assertEqual(result.status, 'UNAVAILABLE')
        self.assertEqual(result.claims, [])

    def test_retaining_a_rejected_fact_or_adding_an_unchecked_fact_never_turns_green(self):
        kept = verification_for_final_text(KNOWN + ' Он банкир.', [SOURCE], draft_result())
        self.assertEqual(kept.status, 'PARTIALLY_SUPPORTED')
        self.assertEqual(kept.unsupported_claims, ['Он банкир.'])
        added = verification_for_final_text(KNOWN + ' Он обладает магическими способностями.', [SOURCE], draft_result())
        self.assertEqual(added.status, 'UNAVAILABLE')

    def test_a_quote_with_a_new_interpretation_is_not_certified_as_a_literal_excerpt(self):
        final = f'В файле сказано: «{KNOWN}». Следовательно, он бессмертен.'
        result = verification_for_final_text(final, [SOURCE], draft_result(), literal_quote=KNOWN)
        self.assertEqual(result.status, 'UNAVAILABLE')

    def test_a_missing_or_changed_source_cannot_certify_old_proof(self):
        altered = {**SOURCE, 'text': 'Файл сейчас содержит другую информацию.'}
        result = verification_for_final_text(KNOWN, [altered], draft_result())
        self.assertEqual(result.status, 'UNAVAILABLE')

    def test_literal_fallback_certifies_only_what_the_document_says(self):
        final = f'В файле сказано: «{KNOWN}».'
        result = verification_for_final_text(final, [SOURCE], draft_result(), literal_quote=KNOWN)
        self.assertEqual(result.checked_answer, final)
        self.assertTrue(all(c.status == 'SUPPORTED' for c in result.claims))
        self.assertFalse(any('банкир' in c.text for c in result.claims))

    def test_the_actual_response_and_journal_preserve_different_draft_and_final_assessments(self):
        env = UserEnvironment()
        self.addCleanup(env.close)
        result = draft_result()
        provider = SimpleNamespace(generate=lambda *a, **kw: SimpleNamespace(text=result.checked_answer))
        budget = ResponseBudgetPlanner().plan('Кто такой хранитель?', 'BALANCED')
        with patch('model_runtime.get_llm_provider', return_value=provider), \
             patch('settings.VERIFIER_ENABLED', True), \
             patch('verifier.verify_answer', return_value=(result.checked_answer, result)):
            response = asyncio.run(env.execute('Кто такой хранитель?', _response_budget=budget,
                _workspace_matches=[{'path': SOURCE['metadata']['source'], 'content': SOURCE['text'], 'score': 1}]))
        final = response.data['final']
        self.assertTrue(final.startswith(KNOWN))
        self.assertNotIn('банкир', final)
        self.assertEqual(response.data['verification'].checked_answer, final)
        self.assertEqual(response.data['verification'].status, 'SUPPORTED')
        self.assertIn('банкир', response.data['draft'])
        from contextlib import closing
        with closing(env.connection()) as conn:
            rows = conn.execute('SELECT action_type,output FROM assistant_actions').fetchall()
        import json
        outputs = {r['action_type']: json.loads(r['output']) for r in rows}
        # ExecutionContext wraps action data into the standard tool result.
        checked = outputs['VERIFY_RESPONSE']['data']
        self.assertEqual(checked['verification']['status'], 'PARTIALLY_SUPPORTED')
        self.assertIn('банкир', checked['checked_answer'])
        displayed = outputs['FINALIZE_RESPONSE']['data']
        self.assertEqual(displayed['verification']['status'], 'SUPPORTED')

    def test_short_what_is_and_what_kind_questions_have_normal_budget(self):
        for question in ('Что за культ Терпения?', 'Что такое Северный архив?'):
            budget = ResponseBudgetPlanner().plan(question, 'BALANCED')
            self.assertEqual(budget.detail.value, 'NORMAL')
            self.assertEqual(budget.coverage.value, 'BALANCED')
