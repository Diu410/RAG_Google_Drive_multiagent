"""Bound a whole model chain and retain source-backed, informative descriptions."""
import asyncio
from dataclasses import replace
import json
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
import support
from evidence_audit import EvidenceAuditor, EvidenceRelation
from llama_provider import LlamaCppProvider
from llm_contract import GenerationDeadlineExceeded, GenerationScope, current_scope
from model_profiles import PROFILES
from overview_facts import FactFirstOverview
from pydantic_models import VerificationClaim
from verifier import Verifier


SOURCE = {'chunk_id': 'workspace:Письмо.md', 'metadata': {'source': 'Письмо.md'},
          'text': 'Это всё я — Хранитель архива Мирон Лисов. Я храню карты. Я хочу открыть читальный зал.'}
STATEMENTS = ['Мирон называет себя хранителем архива.',
              'В письме Мирон сообщает, что хранит карты.',
              'В письме Мирон намерен открыть читальный зал.']


class ReferenceProvider:
    supports_batch_evidence = True
    supports_evidence_ids = True

    def __init__(self, invalid=False):
        self.calls = []
        self.invalid = invalid

    def count_prompt(self, *args):
        return 100

    def generate(self, *args, **kwargs):
        raise AssertionError('Do not duplicate preparation with an ordinary draft and correction')

    def structured_generate(self, **kwargs):
        self.calls.append(kwargs)
        packet = json.loads(kwargs['messages'][-1]['content'].split('\nSources JSON: ', 1)[0])
        properties = kwargs['schema']['properties']
        relation = dict(claim_subject='Мирон', source_subject='Мирон', claim_predicate='хранит карты',
            source_predicate='хранит карты', claim_modality='TEXTUAL_STATEMENT', source_modality='TEXTUAL_STATEMENT',
            source_subject_kind='PERSON', relevant_to_question=True,
            same_subject=True, same_predicate=True, same_modality=True)
        if 'facts' in properties:
            result = {'facts': {
                'unknown' if self.invalid else packet['evidence'][index]['evidence_id']: s
                for index, s in enumerate(STATEMENTS)}}
        elif 'assessments' in properties:
            result = {'assessments': [{**relation, 'claim_predicate': item['claim'],
                'source_predicate': item['claim'], 'id': item['id']} for item in packet['items']]}
        else:
            result = relation
        return SimpleNamespace(text=json.dumps(result, ensure_ascii=False), metrics=SimpleNamespace(finish_reason='stop'))


class ResponseLatencyTests(unittest.TestCase):
    def test_explicit_speaker_label_does_not_replace_a_name_with_the_whole_label(self):
        quote = 'Жди, я иду за тобой, это всё я — Хранитель архива Мирон Лисов.'
        source = {'chunk_id': 'letter', 'text': quote}
        relation = EvidenceRelation(claim_subject='Мирон', source_subject='Говорящий (Мирон Лисов)',
            claim_predicate='называет себя хранителем архива', source_predicate='называет себя хранителем архива',
            claim_modality='SELF_DESCRIPTION', source_modality='SELF_DESCRIPTION', source_subject_kind='PERSON',
            same_subject=True, same_predicate=True, same_modality=True)
        claim = VerificationClaim(text='Мирон называет себя хранителем архива.',
                                  status='SUPPORTED', evidence_quote=quote)
        auditor = EvidenceAuditor(SimpleNamespace(), None)
        self.assertTrue(auditor.evaluate(claim, [source], relation))
        self.assertFalse(auditor.evaluate(claim.model_copy(update={'evidence_quote': 'Я храню карты.'}),
            [{'text': 'Я храню карты.'}], relation))

    def test_reporting_to_an_addressee_keeps_source_modality(self):
        quote = 'Я храню карты. Это всё я — Хранитель архива Мирон Лисов.'
        relation = EvidenceRelation(claim_subject='Мирон', source_subject='Мирон',
            claim_predicate='хранит карты', source_predicate='хранит карты',
            claim_modality='FACT', source_modality='SELF_DESCRIPTION', source_subject_kind='PERSON',
            same_subject=True, same_predicate=True, same_modality=False)
        claim = VerificationClaim(text='Мирон сообщает адресату, что хранит карты.',
                                  status='SUPPORTED', evidence_quote=quote)
        auditor = EvidenceAuditor(SimpleNamespace(), None)
        self.assertTrue(auditor.evaluate(claim, [{'text': quote}], relation))
        self.assertEqual(auditor.guard_adjustment, 'reported_self_description_modality')

    def test_conditional_speaker_action_is_not_mistaken_for_an_addressee_action(self):
        quote = 'Если ты научишь меня, то я сохраню карты.'
        source = {'chunk_id': 'letter', 'text': quote + ' Это всё я — Хранитель архива Мирон Лисов.'}
        relation = EvidenceRelation(claim_subject='Мирон', source_subject='Мирон',
            claim_predicate='сохранит карты, если его научат', source_predicate='если ты научишь меня, то я сохраню карты',
            claim_modality='INTENTION', source_modality='INTENTION', source_subject_kind='PERSON',
            same_subject=True, same_predicate=True, same_modality=True)
        auditor = EvidenceAuditor(SimpleNamespace(), None)
        claim = VerificationClaim(text='Мирон обещает сохранить карты, если его научат.',
                                  status='SUPPORTED', evidence_quote=quote)
        self.assertTrue(auditor.evaluate(claim, [source], relation))
        for text in ['Мирон сохранит карты.', 'Мирон научит адресата сохранять карты.']:
            claim = claim.model_copy(update={'text': text})
            self.assertFalse(auditor.evaluate(claim, [source], relation))
        anonymous = {'text': quote}
        claim = claim.model_copy(update={'text': 'Мирон обещает сохранить карты, если его научат.'})
        self.assertFalse(auditor.evaluate(claim, [anonymous], relation))

    def test_live_tokens_do_not_extend_request_or_generation_deadline(self):
        # A read timeout would never fire: this stream continuously produces data.
        for limited_scope in (True, False):
            with self.subTest(limited_scope=limited_scope):
                closed = threading.Event()
                class Endless(httpx.AsyncByteStream):
                    async def __aiter__(self):
                        while True:
                            await asyncio.sleep(.015)
                            yield b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'

                    async def aclose(self):
                        closed.set()
                provider = LlamaCppProvider(transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, stream=Endless())))
                scope = GenerationScope('deadline-test', deadline=time.perf_counter() + .16 if limited_scope else None)
                token = current_scope.set(scope)
                profile = replace(PROFILES['FAST'], generation_seconds=3 if limited_scope else .16)
                try:
                    with patch.dict(PROFILES, FAST=profile), patch.object(provider, 'health', return_value={'ready': True}), \
                         patch.object(provider, 'count_prompt', return_value=3):
                        with self.assertRaises(GenerationDeadlineExceeded):
                            provider.structured_generate([{'role': 'user', 'content': 'hi'}],
                                profile='FAST', schema={'type': 'object'})
                    self.assertTrue(closed.wait(1), 'Timed-out request left the model stream open')
                    self.assertFalse(scope.cancelled.is_set(), 'Time limit must not masquerade as user cancellation')
                    self.assertEqual(provider._active, {})
                finally:
                    current_scope.reset(token)
                    provider.close()

    def test_followup_cannot_restart_an_expired_shared_budget(self):
        provider = LlamaCppProvider()
        self.addCleanup(provider.close)
        scope = GenerationScope('expired-task', deadline=time.perf_counter() - 1)
        token = current_scope.set(scope)
        try:
            with patch.object(provider, 'health', side_effect=AssertionError('Do not start another call')):
                for _ in range(2):
                    with self.assertRaises(GenerationDeadlineExceeded):
                        provider.generate([{'role': 'user', 'content': 'hi'}])
        finally:
            current_scope.reset(token)

    def test_reference_selection_preserves_exact_source_and_independent_audit(self):
        provider = ReferenceProvider()
        pipeline = FactFirstOverview(Verifier(provider))
        facts = pipeline.prepare('Кто такой Мирон?', [SOURCE], 'BALANCED')
        self.assertEqual(len(facts), 3)
        self.assertTrue(all(f.evidence_quote in SOURCE['text'] for f in facts))
        self.assertEqual([f.statement for f in facts], STATEMENTS)
        self.assertEqual(len(pipeline.verifier.audit_records), 3)
        first = provider.calls[0]
        selection = first['schema']['properties']['facts']
        self.assertEqual(list(selection['properties']), ['e000', 'e001', 'e002'])
        self.assertFalse(selection['additionalProperties'])
        self.assertEqual(selection['required'], [])
        answer, checked = pipeline.render('Кто такой Мирон?', facts, [SOURCE], 'BALANCED')
        self.assertTrue(all(s in answer for s in STATEMENTS))
        self.assertEqual(checked.checked_answer, answer)
        self.assertEqual(len(provider.calls), 3, 'Extraction, one batch and one single audit suffice')
        self.assertTrue(provider.calls[0]['thinking'])
        self.assertEqual(provider.calls[0]['reasoning_budget_tokens'], 128)
        self.assertTrue(all(not c['thinking'] for c in provider.calls[1:]))

    def test_unknown_reference_is_rejected_before_semantic_audit(self):
        provider = ReferenceProvider(invalid=True)
        pipeline = FactFirstOverview(Verifier(provider))
        pipeline.prepare('Кто такой Мирон?', [SOURCE], 'FAST')
        self.assertTrue(any(r['reason'] == 'unknown_evidence_id' for r in pipeline.report['rejected']))
        self.assertFalse(any('assessments' in c['schema']['properties'] for c in provider.calls))

    def test_addressee_advice_does_not_displace_the_named_speakers_description(self):
        source = {**SOURCE, 'text': 'Ты должен поддерживать архив. ' + SOURCE['text']}
        provider = ReferenceProvider()
        facts = FactFirstOverview(Verifier(provider)).prepare('Кто такой Мирон?', [source], 'BALANCED')
        self.assertEqual([fact.statement for fact in facts], STATEMENTS)
        self.assertEqual([fact.evidence_quote.strip() for fact in facts],
            ['Это всё я — Хранитель архива Мирон Лисов.', 'Я храню карты.', 'Я хочу открыть читальный зал.'])

    def test_failed_identity_paraphrase_does_not_leave_an_answer_without_its_subject(self):
        provider = ReferenceProvider()
        original = provider.structured_generate
        def response(**kwargs):
            result = original(**kwargs)
            if 'assessments' in kwargs['schema']['properties']:
                packet = json.loads(result.text)
                packet['assessments'][0]['same_predicate'] = False
                result.text = json.dumps(packet, ensure_ascii=False)
            return result
        pipeline = FactFirstOverview(Verifier(provider))
        with patch.object(provider, 'structured_generate', side_effect=response):
            facts = pipeline.prepare('Кто такой Мирон?', [SOURCE], 'BALANCED')
        self.assertEqual(facts[0].statement, 'В тексте Мирон называет себя «Хранитель архива Мирон Лисов».')
        self.assertTrue(all(s in [f.statement for f in facts] for s in STATEMENTS[1:]))
        self.assertNotIn(STATEMENTS[0], [f.statement for f in facts])
        self.assertEqual(len(provider.calls), 3)

    def test_failed_paraphrase_keeps_original_words_without_inheriting_its_verdict(self):
        provider = ReferenceProvider()
        original = provider.structured_generate
        def response(**kwargs):
            result = original(**kwargs)
            if 'assessments' in kwargs['schema']['properties']:
                packet = json.loads(result.text)
                packet['assessments'][1]['same_predicate'] = False
                result.text = json.dumps(packet, ensure_ascii=False)
            return result
        pipeline = FactFirstOverview(Verifier(provider))
        with patch.object(provider, 'structured_generate', side_effect=response):
            facts = pipeline.prepare('Кто такой Мирон?', [SOURCE], 'BALANCED')
        quoted = next(f for f in facts if f.scope.startswith('Дословная цитата'))
        self.assertEqual(quoted.statement, 'В тексте сказано: «Я храню карты».')
        self.assertEqual(quoted.evidence_quote, 'Я храню карты.')
        self.assertNotIn(STATEMENTS[1], [f.statement for f in facts])
        self.assertEqual(len(provider.calls), 3)

    def test_reasoning_is_explicit_and_has_a_finite_token_budget(self):
        provider = ReferenceProvider()
        FactFirstOverview(Verifier(provider)).prepare('Кто такой Мирон?', [SOURCE], 'REASONING')
        self.assertTrue(all(c['thinking'] for c in provider.calls))
        self.assertTrue(all(c['reasoning_budget_tokens'] + 128 <= c['max_tokens'] for c in provider.calls))

    def test_who_is_uses_one_general_preparation_chain_without_rechecking(self):
        env = support.UserEnvironment()
        self.addCleanup(env.close)
        provider = ReferenceProvider()
        with patch('model_runtime.get_llm_provider', return_value=provider), patch('settings.VERIFIER_ENABLED', True):
            result = asyncio.run(env.execute('Кто такой Мирон?', _workspace_matches=[{
                'path': 'Письмо.md', 'content': SOURCE['text'], 'score': 1}], model='BALANCED'))
        self.assertTrue(all(s in result.data['final'] for s in STATEMENTS))
        self.assertEqual(result.data['verification'].answer_scope, 'EXPLANATORY')
        self.assertEqual(result.data['verification'].checked_answer, result.data['final'])
        self.assertEqual(len(provider.calls), 3)

    def test_timed_out_batch_is_not_retried_with_fresh_single_calls(self):
        provider = ReferenceProvider()
        auditor = EvidenceAuditor(provider, Verifier(provider).bounded_sources)
        claims = [VerificationClaim(text=s, status='SUPPORTED', source_chunk_ids=[SOURCE['chunk_id']],
            evidence_quote=SOURCE['text'], basis='EXPLICIT') for s in STATEMENTS[:2]]
        with patch.object(auditor, '_batch', side_effect=GenerationDeadlineExceeded('time budget')) as batch, \
             patch.object(EvidenceAuditor, 'supports', side_effect=AssertionError('No repeated calls after deadline')):
            with self.assertRaises(GenerationDeadlineExceeded):
                auditor.supports_many(claims, [SOURCE], 'BALANCED')
        batch.assert_called_once()

    def test_copied_batch_comparisons_are_rechecked_independently(self):
        provider = ReferenceProvider()
        original = provider.structured_generate
        def response(**kwargs):
            result = original(**kwargs)
            if 'assessments' in kwargs['schema']['properties']:
                rows = json.loads(result.text)['assessments']
                rows[1] = {**rows[0], 'id': '1'}
                result.text = json.dumps({'assessments': rows}, ensure_ascii=False)
            return result
        auditor = EvidenceAuditor(provider, Verifier(provider).bounded_sources)
        claims = [VerificationClaim(text=s, status='SUPPORTED', source_chunk_ids=[SOURCE['chunk_id']],
            evidence_quote=q, basis='EXPLICIT') for s, q in zip(STATEMENTS[:2],
                ['Это всё я — Хранитель архива Мирон Лисов.', 'Я храню карты.'])]
        with patch.object(provider, 'structured_generate', side_effect=response):
            results = auditor.supports_many(claims, [SOURCE], 'BALANCED')
        self.assertTrue(all(supported for supported, _ in results))
        self.assertEqual(len(provider.calls), 3, 'A copied batch needs two independent assessments')

    def test_completed_audits_survive_a_later_deadline_without_retrying_unchecked_facts(self):
        provider = ReferenceProvider()
        original = provider.structured_generate
        def response(**kwargs):
            if 'claim_subject' in kwargs['schema']['properties']:
                provider.calls.append(kwargs)
                raise GenerationDeadlineExceeded('single audit time budget')
            return original(**kwargs)
        pipeline = FactFirstOverview(Verifier(provider))
        with patch.object(provider, 'structured_generate', side_effect=response):
            facts = pipeline.prepare('Кто такой Мирон?', [SOURCE], 'BALANCED')
        self.assertEqual([f.statement for f in facts], STATEMENTS[:2])
        self.assertEqual(len(provider.calls), 3)
        answer, checked = pipeline.render('Кто такой Мирон?', facts, [SOURCE], 'BALANCED')
        self.assertIn(STATEMENTS[1], answer)
        self.assertNotIn(STATEMENTS[2], answer)
        self.assertEqual(checked.status, 'SUPPORTED')
        self.assertIn('лимиту времени', checked.warning)
