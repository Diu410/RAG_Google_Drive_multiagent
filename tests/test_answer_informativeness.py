"""Keep useful source descriptions instead of reducing a character answer to a title."""
import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import support
from evidence_audit import EvidenceAuditor, EvidenceRelation, is_document_report
from overview_synthesis import merge_verified_answers, overview_evidence, has_unresolved_reference
from pydantic_models import VerificationClaim, VerificationResult
from source_grounding import focused_source_text


class AnswerInformativenessTests(unittest.TestCase):
    def test_named_speaker_is_not_discarded_because_of_a_local_reported_pronoun(self):
        statement = 'Мирон заявляет, что он хранитель архива.'
        self.assertFalse(has_unresolved_reference(statement))
        for dependent in ['Он заявляет, что архив закрыт.', 'Мирон хранит их.',
                          'В письме он заявляет, что архив закрыт.', 'Мирон заявляет, что он хранит их.']:
            self.assertTrue(has_unresolved_reference(dependent))
        source = {'chunk_id': 'letter', 'text': 'Меня зовут Мирон. Я хранитель архива.'}
        checked = VerificationResult(status='SUPPORTED', checked_answer=statement, claims=[VerificationClaim(
            text=statement, status='SUPPORTED', basis='EXPLICIT', evidence_quote='Я хранитель архива.',
            source_chunk_ids=['letter'])])
        self.assertEqual(merge_verified_answers(statement, checked, '', None, [source])[0], statement)

    def test_self_description_report_accepts_plural_frames_and_reflexive_attribution(self):
        quote = 'Я хранитель архива.'
        source = {'chunk_id': 'letter', 'text': 'Меня зовут Мирон. ' + quote}
        relation = EvidenceRelation(claim_subject='Мирон', source_subject='Мирон',
            claim_predicate='описывает себя как хранителя архива', source_predicate='называет себя хранителем архива',
            claim_modality='FACT', source_modality='SELF_DESCRIPTION',
            same_subject=True, same_predicate=True, same_modality=False)
        for statement in ['В текстах он описывает себя как хранителя архива.',
                          'Согласно письмам, Мирон говорит, что он хранитель архива.',
                          'Мирон описывает себя как хранителя архива.',
                          'Мирон заявляет, что он хранитель архива.']:
            with self.subTest(statement=statement):
                claim = VerificationClaim(text=statement, status='SUPPORTED', evidence_quote=quote)
                auditor = EvidenceAuditor(SimpleNamespace(), None)
                self.assertTrue(is_document_report(statement))
                self.assertTrue(auditor.evaluate(claim, [source], relation))
                self.assertEqual(auditor.guard_adjustment, 'reported_self_description_modality')
        for statement, text in [('Мирон — хранитель архива.', source['text']),
                                ('В текстах он описывает себя как хранителя архива.',
                                 'Например: Меня зовут Мирон. ' + quote)]:
            with self.subTest(statement=statement, text=text):
                claim = VerificationClaim(text=statement, status='SUPPORTED', evidence_quote=quote)
                self.assertFalse(EvidenceAuditor(SimpleNamespace(), None).evaluate(claim, [{'text': text}], relation))
        optimistic = relation.model_copy(update={'same_modality': True})
        claim = VerificationClaim(text='Мирон заявляет, что он хранитель архива.',
                                  status='SUPPORTED', evidence_quote=quote)
        auditor = EvidenceAuditor(SimpleNamespace(), None)
        self.assertFalse(auditor.evaluate(claim, [{'text': 'Например: Меня зовут Мирон. ' + quote}], optimistic))
        self.assertEqual(auditor.guard_rejection, 'self_description_from_example')

    def test_named_speaker_keeps_description_before_the_closing_identity(self):
        letter = ('Я храню карты и хочу стать главным хранителем. '
                  + 'Рассказываю о своей работе в архиве. ' * 33
                  + 'Это всё я — Хранитель Северного архива Мирон Лисов.')
        self.assertGreater(len(letter), 900)
        self.assertEqual(focused_source_text('Кто такой Мирон?', letter), letter)
        records = overview_evidence('Кто такой Мирон?', [{'chunk_id': 'letter', 'text': letter}])
        self.assertEqual(len(records), 1)
        self.assertIn('Я храню карты', records[0]['text'])
        self.assertIn('Мирон Лисов', records[0]['text'])

    def test_recovery_keeps_whole_markdown_quote_with_internal_sentence_boundaries(self):
        quote = 'Я хранитель архива. Я храню карты.'
        text = 'Мирон бессмертен.\n\n> «' + quote + '»'
        source = {'chunk_id': 'letter', 'text': quote, 'metadata': {'source': 'Письмо.md'}}
        quoted = '«' + quote + '»'
        original = VerificationResult(status='PARTIALLY_SUPPORTED', checked_answer=text, claims=[
            VerificationClaim(text='Мирон бессмертен.', status='UNSUPPORTED'),
            VerificationClaim(text=quoted, status='SUPPORTED', basis='EXPLICIT', evidence_quote=quote, source_chunk_ids=['letter'])])
        role = 'В тексте Мирон называет себя «Хранитель Северного архива Мирон Лисов».'
        source['text'] += ' Это всё я — Хранитель Северного архива Мирон Лисов.'
        recovered = VerificationResult(status='SUPPORTED', checked_answer=role, claims=[VerificationClaim(
            text=role, status='SUPPORTED', basis='EXPLICIT', evidence_quote='Хранитель Северного архива Мирон Лисов', source_chunk_ids=['letter'])])
        merged = merge_verified_answers(text, original, role, recovered, [source])
        self.assertIsNotNone(merged)
        self.assertIn(quote, merged[0])
        self.assertIn(role, merged[0])
        self.assertNotIn('бессмертен', merged[0])
        self.assertEqual(merged[1].status, 'SUPPORTED')
        self.assertEqual(len(merged[1].claims), 2)

    def test_who_is_question_uses_explanatory_recovery_and_preserves_checked_quote(self):
        env = support.UserEnvironment(); self.addCleanup(env.close)
        quote = 'Я хранитель архива.'
        draft = 'Мирон бессмертен.\n\n> «' + quote + '»'
        source_text = ('Меня зовут Мирон. ' + quote + ' Я храню карты. '
                       'Это всё я — Хранитель Северного архива Мирон Лисов.')
        initial = VerificationResult(status='PARTIALLY_SUPPORTED', checked_answer=draft, claims=[
            VerificationClaim(text='Мирон бессмертен.', status='UNSUPPORTED'),
            VerificationClaim(text='«' + quote + '»', status='SUPPORTED', basis='EXPLICIT',
                              evidence_quote=quote, source_chunk_ids=['workspace:Письмо.md'])])
        statements = ['В письме Мирон описывает себя как хранителя архива.',
                      'В письме Мирон сообщает, что хранит карты.']
        candidate = ' '.join(statements)
        recovered = VerificationResult(status='SUPPORTED', checked_answer=candidate, claims=[
            VerificationClaim(text=statement, status='SUPPORTED', basis='EXPLICIT', evidence_quote=evidence,
                              source_chunk_ids=['workspace:Письмо.md'])
            for statement, evidence in zip(statements, [quote, 'Я храню карты.'])])
        prompts = []
        def generate(messages, **kw):
            prompts.append(messages)
            return SimpleNamespace(text=draft)
        provider = SimpleNamespace(generate=generate)
        def compose(synthesizer, *args, **kwargs):
            synthesizer.last_verification = recovered
            return candidate
        with patch('model_runtime.get_llm_provider', return_value=provider), \
             patch('settings.VERIFIER_ENABLED', True), \
             patch('verifier.verify_answer', return_value=(draft, initial)), \
             patch('overview_synthesis.OverviewSynthesizer.compose', autospec=True, side_effect=compose) as composition, \
             patch('verifier.Verifier.verify', side_effect=AssertionError('Do not recheck the unchanged audited recovery')):
            result = asyncio.run(env.execute('Кто такой Мирон?', _workspace_matches=[{
                'path': 'Письмо.md', 'content': source_text, 'score': 1}]))
        composition.assert_called_once()
        for statement in statements:
            self.assertIn(statement, result.data['final'])
        self.assertIn(quote, result.data['final'])
        self.assertNotIn('бессмертен', result.data['final'])
        self.assertNotIn('Не более четырёх предложений', prompts[0][0]['content'])
        self.assertEqual(result.data['verification'].checked_answer, result.data['final'])
        self.assertEqual(result.data['verification'].answer_scope, 'EXPLANATORY')
