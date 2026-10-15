"""Explanations retain attribution; successful citation checks alone are insufficient."""
import json
from types import SimpleNamespace
import unittest

import support
from overview_synthesis import (overview_evidence, OverviewSynthesizer, accepted_overview_evidence,
                               is_grounded_explanation, verified_explanation_sentences, is_direct_topic_description)
from pydantic_models import VerificationClaim, VerificationResult
from verifier import Verifier, verification_for_final_text


class OverviewSynthesisTests(unittest.TestCase):
    question = 'Что за Культ Терпения?'
    source = {'chunk_id': 'workspace:Архив/Письмо.md', 'text':
        'Культ Терпения зависит от того, как ты представишь идею. '
        'Развивай навыки, чтобы стать лучшим в культе. '
        'Это всё я — Архиепископ Культа Терпения Мирон.',
        'metadata': {'source': 'Архив/Письмо.md'}}

    def claim(self, text, **extra):
        return VerificationClaim(text=text, status='SUPPORTED', basis='EXPLICIT',
            source_chunk_ids=[self.source['chunk_id']], evidence_quote=self.source['text'], **extra)

    def test_packet_preserves_example_and_recipient_context(self):
        example = {'chunk_id': 'workspace:Архив/Воля.md', 'text':
            'Например — Я Архиепископ Культа Терпения Мирон. Кроме Терпения внутри меня ничего нет.'}
        records = overview_evidence(self.question, [self.source, example])
        self.assertTrue(any('адресате' in r['scope'] for r in records))
        self.assertTrue(any('Пример' in r['scope'] and 'Например' in r['text'] for r in records))
        for r in records:
            original = next(s['text'] for s in [self.source, example] if s['chunk_id'] == r['chunk_id'])
            self.assertIn(r['text'], original)

    def test_unrelated_game_setting_is_not_evidence_for_organization(self):
        game = {'chunk_id': 'workspace:Игра.md', 'text': 'Действие игры происходит после мировой войны.'}
        self.assertEqual(overview_evidence(self.question, [game]), [])

    def test_many_mentions_in_one_file_do_not_hide_another_relevant_document(self):
        repeated = {**self.source, 'text': 'Культ Терпения представлен в письме. ' * 12}
        other = {'chunk_id': 'workspace:Воля.md', 'text': 'Например — Я Архиепископ Культа Терпения Мирон.'}
        records = overview_evidence(self.question, [repeated, other])
        self.assertEqual({r['chunk_id'] for r in records[:2]}, {repeated['chunk_id'], other['chunk_id']})

    def test_context_fitting_preserves_complete_evidence_from_both_documents(self):
        packets = []
        def count(messages, *args):
            packet = json.loads(messages[-1]['content'].split('Опорные фрагменты JSON: ', 1)[1]
                                .split('\nОтветь на вопрос:', 1)[0])
            packets.append(packet)
            return 5000 if len(packet) > 1 else 300
        calls = []
        def generate(messages, **kw):
            calls.append(kw)
            return SimpleNamespace(text='В письме описан культ.', metrics=SimpleNamespace(finish_reason='stop'))
        client = SimpleNamespace(count_prompt=count, generate=generate)
        other = {'chunk_id': 'workspace:Воля.md', 'text': 'Например — Я Архиепископ Культа Терпения Мирон.'}
        OverviewSynthesizer(SimpleNamespace(client=client)).compose(self.question, [self.source, other], 'BALANCED')
        self.assertTrue(all(len(p) == 1 for p in packets))
        self.assertIn('«Например — Я Архиепископ Культа Терпения Мирон»',
                      OverviewSynthesizer(SimpleNamespace(client=client)).compose(self.question, [self.source, other], 'BALANCED'))
        for r in [p[0] for p in packets]:
            self.assertIn(r['text'], next(s['text'] for s in [self.source, other] if s['chunk_id'] == r['chunk_id']))
        self.assertFalse(calls[0]['thinking'])
        self.assertNotIn('reasoning_budget_tokens', calls[0])

    def test_true_but_unrelated_fact_does_not_make_a_complete_explanation(self):
        result = VerificationResult(status='SUPPORTED', claims=[
            self.claim('В письме культ связывается с представлением идеи.'),
            self.claim('Адресату советуют развивать навыки.'),
            self.claim('В другом разделе описаны горы.', relevant_to_question=False)])
        self.assertFalse(is_grounded_explanation(result, [self.source]))

    def test_two_literal_mentions_do_not_count_as_an_explanation(self):
        result = VerificationResult(status='SUPPORTED', claims=[
            self.claim('Культ Терпения упоминается в самоназвании говорящего.'),
            self.claim('В тексте сказано: «Культ Терпения зависит от того, как ты представишь идею».')])
        self.assertFalse(is_grounded_explanation(result, [self.source]))

    def test_one_description_and_an_affirmation_quote_are_limited_evidence(self):
        example = {'chunk_id': 'workspace:Пример.md',
                   'text': 'Например — Я Архиепископ Культа Терпения Мирон.'}
        result = VerificationResult(status='SUPPORTED', claims=[
            self.claim('В письме культ связывается с представлением идеи.'),
            VerificationClaim(text='В тексте приведено: «Например — Я Архиепископ Культа Терпения Мирон».',
                              status='SUPPORTED', basis='EXPLICIT', source_chunk_ids=[example['chunk_id']],
                              evidence_quote=example['text'])])
        self.assertFalse(is_grounded_explanation(result, [self.source, example]))

    def test_reported_description_is_relevant_but_a_title_or_adjacent_fact_is_not(self):
        self.assertTrue(is_direct_topic_description(self.question,
            'Согласно тексту, говорящий называет Культ Терпения вопросом представления идеи.'))
        self.assertTrue(is_direct_topic_description(self.question,
            'Согласно тексту, Культ Терпения зависит от представления идеи.'))
        for text in ('Мирон называет себя Архиепископом Культа Терпения.',
                     'Архиепископ Культа Терпения тренируется ежедневно.',
                     'Говорящий называет Культ Ясности сообществом.',
                     'Мирон называет Культ Терпения, когда рассказывает о погоде.'):
            self.assertFalse(is_direct_topic_description(self.question, text), text)

    def test_explanation_can_include_an_honest_missing_detail(self):
        result = VerificationResult(status='PARTIALLY_SUPPORTED', claims=[
            self.claim('В письме культ связывается с представлением идеи.'),
            self.claim('Адресату советуют развивать навыки, чтобы стать лучшим в культе.'),
            VerificationClaim(text='В этих фрагментах не указаны ритуалы.', status='INSUFFICIENT_INFORMATION',
                              basis='UNANSWERED')])
        self.assertTrue(is_grounded_explanation(result, [self.source]))
        result.claims[1].status = 'UNSUPPORTED'
        self.assertFalse(is_grounded_explanation(result, [self.source]))

    def test_retry_rejects_invented_quote_and_unrelated_claim(self):
        good = self.claim('В письме советуют развивать навыки.')
        bad = self.claim('Участники обязаны соблюдать ритуалы.')
        bad.evidence_quote = 'Все участники обязаны соблюдать ритуалы.'
        irrelevant = self.claim('В игре есть горы.', relevant_to_question=False)
        records = accepted_overview_evidence(VerificationResult(status='SUPPORTED', claims=[good, bad, irrelevant]),
                                            [self.source])
        self.assertEqual([r['checked_statement'] for r in records], [good.text])

    def test_exact_check_rejects_changed_or_omitted_answer_claims(self):
        answer = 'В письме адресату советуют развивать навыки. Мирон называет себя архиепископом.'
        for texts in ([answer.split('. ')[0] + '.'], ['Все участники должны развивать навыки.'],
                      [answer.split('. ')[0] + '.'] * 2):
            with self.subTest(texts=texts):
                def generate(**kw):
                    self.assertEqual(kw['schema']['properties']['claims']['minItems'], 2)
                    self.assertIn(answer.split('. ')[0] + '.',
                        kw['schema']['$defs']['ClaimAssessment']['properties']['text']['enum'])
                    return SimpleNamespace(text=json.dumps({'status': 'SUPPORTED', 'claims': [
                        {'text': t, 'status': 'SUPPORTED', 'source_chunk_ids': [self.source['chunk_id']],
                         'basis': 'EXPLICIT', 'evidence_quote': self.source['text']} for t in texts]}))
                with self.assertRaisesRegex(ValueError, 'changed or omitted'):
                    Verifier(SimpleNamespace(structured_generate=generate), audit_evidence=False).verify(
                        self.question, answer, [self.source], 'WORKSPACE_QA', 'BALANCED', exact_statements=True)

    def test_exact_check_preserves_qualified_sentence_and_checked_text(self):
        answer = 'В письме адресату советуют развивать навыки, это не правило для всех участников.'
        def generate(**kw):
            return SimpleNamespace(text=json.dumps({'status': 'SUPPORTED', 'claims': [
                {'text': answer, 'status': 'SUPPORTED', 'source_chunk_ids': [self.source['chunk_id']],
                 'basis': 'EXPLICIT', 'evidence_quote': self.source['text']}]}))
        result = Verifier(SimpleNamespace(structured_generate=generate), audit_evidence=False).verify(
            self.question, answer, [self.source], 'WORKSPACE_QA', 'BALANCED', exact_statements=True)
        self.assertEqual(result.claims[0].text, answer)
        self.assertEqual(result.checked_answer, answer)

    def test_checker_cannot_turn_a_categorical_rule_into_a_cautious_inference(self):
        answer = 'Культ требует от всех участников тренировки тела.'
        def generate(**kw):
            return SimpleNamespace(text=json.dumps({'status': 'SUPPORTED', 'claims': [
                {'text': answer, 'status': 'SUPPORTED', 'source_chunk_ids': [self.source['chunk_id']],
                 'basis': 'INFERENCE', 'evidence_quote': self.source['text']}]}))
        result = Verifier(SimpleNamespace(structured_generate=generate), audit_evidence=False).verify(
            self.question, answer, [self.source], 'WORKSPACE_QA', 'BALANCED', exact_statements=True)
        self.assertEqual(result.status, 'UNSUPPORTED')

    def test_incomplete_synthesis_cannot_be_adopted(self):
        client = SimpleNamespace(generate=lambda *a, **kw: SimpleNamespace(
            text='Недописанный ответ', metrics=SimpleNamespace(finish_reason='length')))
        with self.assertRaisesRegex(ValueError, 'did not finish'):
            OverviewSynthesizer(SimpleNamespace(client=client)).compose(self.question, [self.source], 'BALANCED')

    def test_oversized_evidence_is_rejected_whole_instead_of_losing_its_context(self):
        client = SimpleNamespace(count_prompt=lambda *a: 5000,
                                 generate=lambda *a, **k: self.fail('Oversized evidence was sent'))
        with self.assertRaisesRegex(ValueError, 'does not fit'):
            OverviewSynthesizer(SimpleNamespace(client=client)).compose(self.question, [self.source], 'BALANCED')

    def test_a_multi_statement_card_cannot_be_silently_used_as_one_fact(self):
        client = SimpleNamespace(generate=lambda *a, **kw: SimpleNamespace(
            text='Культ зависит от харизмы. Автор является божеством.', metrics=SimpleNamespace(finish_reason='stop')))
        with self.assertRaisesRegex(ValueError, 'more than one statement'):
            OverviewSynthesizer(SimpleNamespace(client=client)).compose(self.question, [self.source], 'BALANCED')

    def test_copied_source_speech_is_presented_as_a_quote(self):
        def generate(messages, **kw):
            record = json.loads(messages[-1]['content'].split('Опорные фрагменты JSON: ', 1)[1]
                                .split('\nОтветь на вопрос:', 1)[0])[0]
            return SimpleNamespace(text=record['topic_statement'], metrics=SimpleNamespace(finish_reason='stop'))
        client = SimpleNamespace(generate=generate)
        answer = OverviewSynthesizer(SimpleNamespace(client=client)).compose(self.question, [self.source], 'BALANCED')
        self.assertTrue(answer.startswith('В тексте сказано: «'))
        self.assertIn('«Культ Терпения зависит', answer)

    def test_paraphrased_source_address_cannot_become_an_assertion_about_the_user(self):
        client = SimpleNamespace(generate=lambda *a, **kw: SimpleNamespace(
            text='Ты должен стать архиепископом.', metrics=SimpleNamespace(finish_reason='stop')))
        answer = OverviewSynthesizer(SimpleNamespace(client=client)).compose(self.question, [self.source], 'BALANCED')
        self.assertTrue(answer.startswith('В тексте сказано: «'))
        self.assertNotIn('Ты должен стать архиепископом', answer)

    def test_bad_address_paraphrase_does_not_discard_other_source_evidence(self):
        calls = []
        def generate(messages, **kw):
            calls.append(messages)
            return SimpleNamespace(text='Культ Терпения — это вопрос твоей харизмы.' if len(calls) == 1
                else 'В письме адресату советуют развивать навыки.', metrics=SimpleNamespace(finish_reason='stop'))
        answer = OverviewSynthesizer(SimpleNamespace(client=SimpleNamespace(generate=generate))).compose(
            self.question, [self.source], 'BALANCED')
        self.assertTrue(answer.startswith('В тексте сказано: «'))
        self.assertIn('В письме адресату советуют развивать навыки.', answer)
        self.assertNotIn('вопрос твоей харизмы', answer)
        self.assertGreater(len(calls), 1)

    def test_self_affirmation_example_is_quoted_without_a_model_inventing_its_author(self):
        source = {'chunk_id': 'workspace:Пример.md', 'text': 'Например — Я Архиепископ Культа Терпения Мирон.'}
        client = SimpleNamespace(generate=lambda *a, **kw: self.fail('Example attribution should be preserved literally'))
        answer = OverviewSynthesizer(SimpleNamespace(client=client)).compose(self.question, [source], 'BALANCED')
        self.assertEqual(answer, 'В тексте приведено: «Например — Я Архиепископ Культа Терпения Мирон».')

    def test_sources_footer_preserves_scope_but_removing_facts_limits_it(self):
        claims = [self.claim('В письме культ связывается с представлением идеи.'),
                  self.claim('Адресату советуют развивать навыки.')]
        answer = ' '.join(c.text for c in claims)
        previous = VerificationResult(status='SUPPORTED', claims=claims, checked_answer=answer,
                                      answer_scope='EXPLANATORY')
        final = answer + '\n\nИсточники: `Архив/Письмо.md`.'
        result = verification_for_final_text(final, [self.source], previous)
        self.assertEqual(result.answer_scope, 'EXPLANATORY')
        self.assertEqual(result.checked_answer, final)
        result = verification_for_final_text(claims[0].text, [self.source], previous)
        self.assertEqual(result.answer_scope, 'LIMITED')

    def test_one_invented_doctrine_does_not_erase_two_verified_explanations(self):
        claims = [self.claim('В письме культ связывается с представлением идеи.'),
                  self.claim('Адресату советуют развивать навыки.'),
                  self.claim('Участники обязаны соблюдать ритуалы.')]
        claims[-1].status = 'UNSUPPORTED'
        answer = ' '.join(c.text for c in claims)
        result = VerificationResult(status='PARTIALLY_SUPPORTED', claims=claims, checked_answer=answer)
        final, checked = verified_explanation_sentences(answer, result, [self.source])
        self.assertEqual(final, ' '.join(c.text for c in claims[:2]))
        self.assertEqual(checked.checked_answer, final)
        self.assertEqual(checked.answer_scope, 'EXPLANATORY')
        self.assertNotIn('ритуалы', final)

    def test_preserving_sentences_requires_complete_unchanged_assessment(self):
        claim = self.claim('В письме культ связывается с представлением идеи.')
        result = VerificationResult(status='SUPPORTED', claims=[claim])
        self.assertIsNone(verified_explanation_sentences(claim.text + ' Пропущенный факт.', result, [self.source]))
        self.assertIsNone(verified_explanation_sentences('Автор утверждает другое.', result, [self.source]))

    def test_removed_antecedent_cannot_leave_a_detached_supported_pronoun(self):
        claims = [self.claim('Мирон создал организацию.'), self.claim('Он тренирует тело.'),
                  self.claim('В письме культ связывается с представлением идеи.')]
        claims[0].status = 'UNSUPPORTED'
        result = VerificationResult(status='PARTIALLY_SUPPORTED', claims=claims)
        self.assertIsNone(verified_explanation_sentences(' '.join(c.text for c in claims), result, [self.source]))

    def test_copula_in_a_definition_does_not_depend_on_a_removed_antecedent(self):
        claims = [self.claim('Культ — это вопрос представления идеи.'),
                  self.claim('Архиепископ должен развивать навыки.'), self.claim('Автор является божеством.')]
        claims[-1].status = 'UNSUPPORTED'
        result = VerificationResult(status='PARTIALLY_SUPPORTED', claims=claims)
        final, _ = verified_explanation_sentences(' '.join(c.text for c in claims), result, [self.source])
        self.assertEqual(final, ' '.join(c.text for c in claims[:2]))


if __name__ == '__main__':
    unittest.main()
