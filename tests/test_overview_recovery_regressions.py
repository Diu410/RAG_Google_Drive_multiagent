"""Source selection, scoped attribution, and recovery must preserve true facts."""
import json
import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import support
from evidence_audit import EvidenceAuditor
from overview_synthesis import merge_verified_answers, OverviewSynthesizer
from pydantic_models import VerificationClaim, VerificationResult
from verifier import Verifier


class OverviewRecoveryTests(unittest.TestCase):
    def test_successful_explanatory_recovery_also_keeps_the_previous_useful_fact(self):
        env = support.UserEnvironment()
        self.addCleanup(env.close)
        role = 'В тексте сказано, что хранитель должен быть лучшим в архиве.'
        new_facts = ['Северный архив хранит карты.', 'Северный архив основан в 2001 году.']
        draft = role + ' Архив бессмертен.'
        source_text = 'Хранитель должен быть лучшим в архиве. ' + ' '.join(new_facts)
        original = VerificationResult(status='PARTIALLY_SUPPORTED', checked_answer=draft,
            claims=[VerificationClaim(text=role, status='SUPPORTED', basis='EXPLICIT',
                source_chunk_ids=['workspace:Письмо.md'], evidence_quote='Хранитель должен быть лучшим в архиве.'),
                VerificationClaim(text='Архив бессмертен.', status='UNSUPPORTED', basis='EXPLICIT')])
        recovered = VerificationResult(status='SUPPORTED', checked_answer=' '.join(new_facts), claims=[
            VerificationClaim(text=fact, status='SUPPORTED', basis='EXPLICIT',
                source_chunk_ids=['workspace:Письмо.md'], evidence_quote=fact) for fact in new_facts])
        provider = SimpleNamespace(generate=lambda *a, **kw: self.fail('Recovery uses the scoped synthesizer'))
        with patch('model_runtime.get_llm_provider', return_value=provider), \
             patch('settings.VERIFIER_ENABLED', True), \
             patch('overview_synthesis.OverviewSynthesizer.compose', side_effect=[draft, ' '.join(new_facts)]), \
             patch('verifier.Verifier.verify', side_effect=[original, recovered]):
            result = asyncio.run(env.execute('Что за Северный архив?', _workspace_matches=[{
                'path': 'Письмо.md', 'content': source_text, 'score': 1}]))
        for fact in [role, *new_facts]:
            self.assertIn(fact, result.data['final'])
        self.assertNotIn('бессмертен', result.data['final'])
        self.assertEqual(result.data['verification'].answer_scope, 'EXPLANATORY')
        self.assertEqual(result.data['verification'].checked_answer, result.data['final'])

    def test_workspace_documentary_recovery_preserves_the_previously_verified_role(self):
        env = support.UserEnvironment()
        self.addCleanup(env.close)
        role = 'В тексте сказано, что хранитель должен быть лучшим в архиве.'
        draft = role + ' Архив бессмертен.'
        source_text = ('Хранитель должен быть лучшим в архиве. '
                       'Это всё я — Хранитель Северного архива Мирон.')
        original = VerificationResult(status='PARTIALLY_SUPPORTED', checked_answer=draft,
            claims=[VerificationClaim(text=role, status='SUPPORTED', basis='EXPLICIT',
                source_chunk_ids=['workspace:Письмо.md'], evidence_quote='Хранитель должен быть лучшим в архиве.'),
                VerificationClaim(text='Архив бессмертен.', status='UNSUPPORTED', basis='EXPLICIT')])
        def verify(question, answer, *args, **kw):
            if answer == draft:
                return original
            fact = answer.split('\n\n')[0]
            return VerificationResult(status='SUPPORTED', checked_answer=answer, claims=[
                VerificationClaim(text=fact, status='SUPPORTED', basis='EXPLICIT',
                    source_chunk_ids=['workspace:Письмо.md'],
                    evidence_quote='Хранитель Северного архива Мирон')])
        provider = SimpleNamespace(generate=lambda *a, **kw: SimpleNamespace(text=draft))
        with patch('model_runtime.get_llm_provider', return_value=provider), \
             patch('settings.VERIFIER_ENABLED', True), \
             patch('overview_synthesis.OverviewSynthesizer.compose', return_value=draft), \
             patch('verifier.Verifier.verify', side_effect=verify):
            result = asyncio.run(env.execute('Что за Северный архив?', _workspace_matches=[{
                'path': 'Письмо.md', 'content': source_text, 'score': 1}]))
        self.assertIn(role, result.data['final'])
        self.assertIn('Хранитель Северного архива Мирон', result.data['final'])
        self.assertNotIn('бессмертен', result.data['final'])
        self.assertEqual(result.data['verification'].checked_answer, result.data['final'])
        self.assertEqual(result.data['verification'].status, 'SUPPORTED')

    def test_literal_report_does_not_ask_a_model_to_reject_or_rewrite_the_quote(self):
        source = {'chunk_id': 'workspace:Заметка.md', 'text': 'Например — Я хранитель Северного архива.'}
        answer = 'В тексте приведено: «Например — Я хранитель Северного архива».'
        provider = SimpleNamespace(structured_generate=lambda **kw: self.fail('Literal text needs no inference'))
        checked = Verifier(provider).verify('Что за Северный архив?', answer, [source], 'WORKSPACE_QA', 'BALANCED')
        self.assertEqual(checked.status, 'SUPPORTED')
        self.assertEqual(checked.checked_answer, answer)
        self.assertEqual(checked.claims[0].source_chunk_ids, [source['chunk_id']])

    def relation_provider(self, **changes):
        relation = dict(claim_subject='заметка', source_subject='заметка',
            claim_predicate='использует имя в примере', source_predicate='использует имя в примере',
            claim_modality='TEXTUAL_STATEMENT', source_modality='TEXTUAL_STATEMENT',
            source_subject_kind='DOCUMENT', relevant_to_question=True,
            same_subject=True, same_predicate=True, same_modality=True, **changes)
        return SimpleNamespace(structured_generate=lambda **kw: SimpleNamespace(text=json.dumps(relation)))

    def test_address_in_example_does_not_reject_a_document_report(self):
        quote = 'Повторяй утверждения, будто ты уже достиг цели. Например — Я хранитель архива.'
        claim = VerificationClaim(text='В заметке имя используется в примере.', status='SUPPORTED',
            evidence_quote=quote, source_chunk_ids=['id'])
        audit = EvidenceAuditor(self.relation_provider(), lambda prefix, sources, *a, **kw: sources)
        self.assertTrue(audit.supports(claim, [{'chunk_id': 'id', 'text': quote}], 'BALANCED'))
        self.assertIsNone(audit.guard_rejection)

    def test_named_person_still_cannot_take_an_anonymous_address(self):
        quote = 'Ты должен сохранять карты архива.'
        relation = dict(claim_subject='Мирон', source_subject='Мирон',
            claim_predicate='сохраняет карты', source_predicate='сохраняет карты',
            claim_modality='FACT', source_modality='FACT', source_subject_kind='PERSON',
            relevant_to_question=True, same_subject=True, same_predicate=True, same_modality=True)
        provider = SimpleNamespace(structured_generate=lambda **kw: SimpleNamespace(text=json.dumps(relation)))
        audit = EvidenceAuditor(provider, lambda prefix, sources, *a, **kw: sources)
        claim = VerificationClaim(text='Мирон сохраняет карты.', status='SUPPORTED',
            evidence_quote=quote, source_chunk_ids=['id'])
        self.assertFalse(audit.supports(claim, [{'chunk_id': 'id', 'text': quote}], 'BALANCED'))
        self.assertEqual(audit.guard_rejection, 'recipient_binding')

    def test_verified_recovery_keeps_a_useful_fact_and_drops_an_invention(self):
        role = 'В письме сказано, что хранитель должен быть лучшим в архиве.'
        mention = 'В тексте сказано: «Это всё я — Хранитель Северного архива Мирон».'
        invented = 'Хранитель бессмертен.'
        source = {'chunk_id': 'id', 'text': 'Хранитель должен быть лучшим в архиве. '
                  'Это всё я — Хранитель Северного архива Мирон.'}
        good = VerificationClaim(text=role, status='SUPPORTED', basis='EXPLICIT', source_chunk_ids=['id'],
            evidence_quote='Хранитель должен быть лучшим в архиве.')
        bad = VerificationClaim(text=invented, status='UNSUPPORTED')
        old = VerificationResult(status='PARTIALLY_SUPPORTED', claims=[good, bad])
        new = VerificationResult(status='SUPPORTED', claims=[VerificationClaim(text=mention,
            status='SUPPORTED', basis='EXPLICIT', source_chunk_ids=['id'], evidence_quote='Это всё я — Хранитель Северного архива Мирон.')])
        final, checked = merge_verified_answers(role + ' ' + invented, old, mention, new, [source])
        self.assertIn(role, final)
        self.assertIn(mention, final)
        self.assertNotIn(invented, final)
        self.assertEqual(checked.checked_answer, final)
        self.assertEqual(checked.status, 'SUPPORTED')

    def test_recovery_does_not_certify_a_rewritten_statement(self):
        claim = VerificationClaim(text='Архив хранит карты.', status='SUPPORTED', basis='EXPLICIT',
            source_chunk_ids=['id'], evidence_quote='Архив хранит карты.')
        result = merge_verified_answers('Архив продаёт карты.', VerificationResult(status='SUPPORTED', claims=[claim]),
            '', None, [{'chunk_id': 'id', 'text': 'Архив хранит карты.'}])
        self.assertIsNone(result)

    def test_example_with_context_can_explain_the_method_without_inventing_a_person(self):
        source = {'chunk_id': 'id', 'text': 'Повторяй утверждение о себе ежедневно. '
                  'Например — Я хранитель Северного архива.'}
        def generate(messages, **kw):
            self.fail('An explicit example can be reported without guessing its speaker')
        answer = OverviewSynthesizer(SimpleNamespace(client=SimpleNamespace(generate=generate))).compose(
            'Что за Северный архив?', [source], 'BALANCED')
        self.assertIn('используется в примере утверждения о себе', answer)
        self.assertIn('Например — Я хранитель Северного архива', answer)

    def test_reported_requirement_preserves_the_whole_proposition_and_modality(self):
        from evidence_audit import documentary_statement_evidence
        from verifier import _quote_source_ids
        source = {'id': 'Ты был прав, когда сказал, что хранитель должен быть лучшим в архиве.'}
        good = 'В тексте сказано, что хранитель должен быть лучшим в архиве.'
        self.assertIsNotNone(documentary_statement_evidence(good, source, _quote_source_ids))
        for bad in [good.replace('должен быть', 'является'),
                    good.replace('лучшим', 'самым мудрым'),
                    good.replace('хранитель', 'Мирон')]:
            self.assertIsNone(documentary_statement_evidence(bad, source, _quote_source_ids))
        contradicted = {'id': 'Ты ошибался, когда сказал, что хранитель должен быть лучшим в архиве.'}
        self.assertIsNone(documentary_statement_evidence(good, contradicted, _quote_source_ids))

    def test_example_report_proves_only_the_named_textual_use(self):
        from evidence_audit import documentary_statement_evidence
        from verifier import _quote_source_ids
        source = {'id': 'Например — Я хранитель Северного архива.'}
        good = ('В тексте название «Северный архив» используется в примере утверждения о себе: '
                '«Например — Я хранитель Северного архива».')
        self.assertIsNotNone(documentary_statement_evidence(good, source, _quote_source_ids))
        self.assertIsNone(documentary_statement_evidence(good.replace('«Северный архив»', '«Южный архив»'), source, _quote_source_ids))
        self.assertIsNone(documentary_statement_evidence(good + ' Мирон реально возглавляет архив.', source, _quote_source_ids))
        self.assertIsNone(documentary_statement_evidence(good, {'id': 'Я хранитель Северного архива.'}, _quote_source_ids))
        combined = ('В тексте название «Северный архив» используется в примере утверждения о себе: '
                    '«Например — Я хранитель ... Северного архива».')
        separate = {'a': 'Например — Я хранитель башни.', 'b': 'В Северного архива летописи.'}
        self.assertIsNone(documentary_statement_evidence(combined, separate, _quote_source_ids))
