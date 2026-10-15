"""Fact preparation precedes writing; a paraphrase never inherits verification."""
import asyncio
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import support
from evidence_audit import EvidenceAuditor
from llm_contract import GenerationCancelled, StructuredOutputTruncated
from overview_facts import FactFirstOverview
from overview_synthesis import OverviewSynthesizer, is_grounded_explanation
from pydantic_models import VerificationClaim, VerificationResult
from verifier import Verifier


SOURCE = {'chunk_id': 'workspace:Архив.md', 'metadata': {'source': 'Архив.md'},
          'text': 'Северный архив хранит карты. Северный архив основан в 2001 году.'}
STATEMENTS = ['Северный архив хранит карты.', 'Северный архив основан в 2001 году.']


class Provider:
    def __init__(self, proposals=None, writer=None):
        self.proposals = proposals or [dict(statement=t, evidence_quote=t) for t in STATEMENTS]
        self.writer = writer
        self.calls = []

    def structured_generate(self, **kw):
        self.calls.append(kw)
        schema = kw['schema']['properties']
        if 'facts' in schema:
            result = {'facts': self.proposals}
        elif 'sentences' in schema:
            packet = json.loads(kw['messages'][-1]['content'])
            result = {'sentences': self.writer(packet['facts']) if self.writer else [
                {'text': f['statement'], 'fact_ids': [f['fact_id']]} for f in packet['facts'][:4]]}
        else:
            result = dict(claim_subject='Северный архив', source_subject='Северный архив',
                claim_predicate='хранит карты', source_predicate='хранит карты',
                claim_modality='FACT', source_modality='FACT', source_subject_kind='ORGANIZATION',
                relevant_to_question=True, same_subject=True, same_predicate=True, same_modality=True)
        return SimpleNamespace(text=json.dumps(result, ensure_ascii=False), metrics=SimpleNamespace(finish_reason='stop'))


class FactFirstTests(unittest.TestCase):
    def test_checked_fact_does_not_hide_another_topic_passage_in_the_same_file(self):
        provider = Provider()
        pipeline = FactFirstOverview(Verifier(provider))
        literal = f'В тексте сказано: «{STATEMENTS[0].rstrip(".")}».'
        facts = pipeline.prepare('Что за Северный архив?', [SOURCE], 'BALANCED', accepted=[
            {'chunk_id': SOURCE['chunk_id'], 'text': STATEMENTS[0], 'checked_statement': literal, 'scope': 'EXPLICIT'},
            {'chunk_id': SOURCE['chunk_id'], 'text': STATEMENTS[1], 'topic_statement': STATEMENTS[1], 'scope': 'EXPLICIT'},
        ])
        self.assertTrue(any('2001' in fact.statement for fact in facts))
        self.assertTrue(any('хранит карты' in fact.statement for fact in facts))

    def prepare(self, provider, sources=None):
        pipeline = FactFirstOverview(Verifier(provider))
        facts = pipeline.prepare('Что за Северный архив?', sources or [SOURCE], 'BALANCED')
        return pipeline, facts

    def test_independent_audit_precedes_writer_and_uses_audited_subject_and_predicate(self):
        provider = Provider()
        pipeline, facts = self.prepare(provider)
        self.assertEqual(len(facts), 2)
        self.assertTrue(all(f.subject == 'Северный архив' and f.predicate == 'хранит карты' for f in facts))
        answer, checked = pipeline.render('Что за Северный архив?', facts, [SOURCE], 'BALANCED')
        phases = [list(c['schema']['properties'])[0] for c in provider.calls]
        self.assertEqual(phases, ['facts', 'claim_subject', 'claim_subject', 'sentences'])
        self.assertEqual(checked.checked_answer, answer)
        self.assertTrue(is_grounded_explanation(checked, [SOURCE]))
        packet = json.loads(provider.calls[-1]['messages'][-1]['content'])['facts']
        self.assertNotIn('evidence_quote', packet[0], 'Neighbouring source prose is not a writing input')
        self.assertEqual(packet[0]['source_path'], 'Архив.md')

    def test_invented_quote_cannot_reach_the_writer(self):
        fake = 'Северный архив выпускает ракеты.'
        provider = Provider(proposals=[dict(statement=fake, evidence_quote=fake)])
        pipeline, facts = self.prepare(provider)
        self.assertTrue(pipeline.report['rejected'])
        self.assertNotIn(fake, [f.statement for f in facts])
        pipeline.render('Что за Северный архив?', facts, [SOURCE], 'BALANCED')
        writer = next(c for c in provider.calls if 'sentences' in c['schema']['properties'])
        self.assertNotIn(fake, writer['messages'][-1]['content'])

    def test_rejected_evidence_does_not_block_a_later_valid_quote_for_the_same_statement(self):
        provider = Provider(proposals=[
            dict(statement=STATEMENTS[1], evidence_quote=STATEMENTS[0]),
            dict(statement=STATEMENTS[1], evidence_quote=STATEMENTS[1])])
        original = provider.structured_generate
        def check_quote(**kw):
            result = original(**kw)
            if 'same_subject' in kw['schema']['properties']:
                packet = json.loads(kw['messages'][-1]['content'].split('\nSources JSON: ')[0])
                if packet['quote'] == STATEMENTS[0]:
                    data = json.loads(result.text);data['same_predicate'] = False
                    result.text = json.dumps(data)
            return result
        provider.structured_generate = check_quote
        pipeline, facts = self.prepare(provider)
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].evidence_quote, STATEMENTS[1])
        self.assertTrue(pipeline.report['rejected'])

    def test_negative_semantic_audit_keeps_only_literal_evidence(self):
        provider = Provider()
        generate = provider.structured_generate
        def negative(**kw):
            result = generate(**kw)
            if 'same_subject' in kw['schema']['properties']:
                data = json.loads(result.text);data['same_subject'] = False
                result.text = json.dumps(data)
            return result
        provider.structured_generate = negative
        pipeline, facts = self.prepare(provider)
        self.assertTrue(all('В тексте сказано:' in f.statement for f in facts))
        self.assertTrue(pipeline.report['rejected'])

    def test_missing_terminal_punctuation_does_not_merge_separate_checked_facts(self):
        provider = Provider(proposals=[dict(statement=t.rstrip('.'), evidence_quote=t) for t in STATEMENTS])
        pipeline, facts = self.prepare(provider)
        answer, checked = pipeline.render('Что за Северный архив?', facts, [SOURCE], 'BALANCED')
        self.assertTrue(all(f.statement.endswith('.') for f in facts))
        self.assertEqual(len(checked.claims), 2)
        self.assertEqual(checked.checked_answer, answer)
        self.assertTrue(is_grounded_explanation(checked, [SOURCE]))

    def test_writer_invention_does_not_replace_checked_facts(self):
        provider = Provider(writer=lambda facts: [{'text': 'Северный архив бессмертен.', 'fact_ids': ['fact:1']}])
        pipeline, facts = self.prepare(provider)
        def reject(auditor, *args, **kw):
            auditor.guard_rejection = None;auditor.guard_adjustment = None
            auditor.last_assessment = SimpleNamespace(relevant_to_question=True, model_dump=lambda: {})
            return False
        with patch.object(EvidenceAuditor, 'supports', reject):
            answer, checked = pipeline.render('Что за Северный архив?', facts, [SOURCE], 'BALANCED')
        self.assertNotIn('бессмертен', answer)
        self.assertEqual(checked.status, 'SUPPORTED')
        self.assertEqual(pipeline.report['writer_attempts'], 1)

    def test_writer_cannot_add_another_true_but_unprepared_fact_from_the_same_source(self):
        provider = Provider(writer=lambda facts: [{'text': 'Северный архив основан в 2001 году.', 'fact_ids': ['fact:1']}])
        pipeline, facts = self.prepare(provider)
        original = provider.structured_generate
        def reject_other(**kw):
            result = original(**kw)
            if 'same_subject' in kw['schema']['properties']:
                packet = json.loads(kw['messages'][-1]['content'].split('\nSources JSON: ')[0])
                self.assertEqual(packet['quote'], facts[0].statement)
                self.assertNotIn('2001', packet['quote'])
                data = json.loads(result.text);data['same_predicate'] = False
                result.text = json.dumps(data)
            return result
        with patch.object(provider, 'structured_generate', side_effect=reject_other):
            answer, _ = pipeline.render('Что за Северный архив?', facts[:1], [SOURCE], 'BALANCED')
        self.assertEqual(answer, facts[0].statement)
        self.assertEqual(pipeline.report['rejected'][-1]['reason'], 'retelling_not_supported')

    def test_truncated_writer_returns_the_original_checked_statements(self):
        provider = Provider()
        pipeline, facts = self.prepare(provider)
        with patch.object(provider, 'structured_generate', side_effect=StructuredOutputTruncated('incomplete')):
            answer, checked = pipeline.render('Что за Северный архив?', facts, [SOURCE], 'BALANCED')
        self.assertEqual(answer, ' '.join(f.statement for f in facts))
        self.assertEqual(checked.status, 'SUPPORTED')
        self.assertEqual(pipeline.report['errors'][-1]['stage'], 'writer')

    def test_cancellation_is_not_a_fallback_answer(self):
        provider = Provider()
        pipeline, facts = self.prepare(provider)
        with patch.object(provider, 'structured_generate', side_effect=GenerationCancelled()):
            with self.assertRaises(GenerationCancelled):
                pipeline.render('Что за Северный архив?', facts, [SOURCE], 'BALANCED')

    def test_audited_paraphrase_is_used_and_unexpressed_facts_are_preserved(self):
        paraphrase = 'В документе «Архив.md» сообщается, что Северный архив хранит карты.'
        provider = Provider(writer=lambda facts: [{'text': paraphrase, 'fact_ids': ['fact:1']}])
        pipeline, facts = self.prepare(provider)
        original = provider.structured_generate
        def textual(**kw):
            result = original(**kw)
            if 'same_subject' in kw['schema']['properties']:
                data = json.loads(result.text)
                data.update(claim_modality='TEXTUAL_STATEMENT', source_modality='TEXTUAL_STATEMENT')
                result.text = json.dumps(data)
            return result
        with patch.object(provider, 'structured_generate', side_effect=textual):
            answer, checked = pipeline.render('Что за Северный архив?', facts, [SOURCE], 'BALANCED')
        self.assertIn(paraphrase, answer)
        self.assertIn(facts[1].statement, answer)
        self.assertNotIn(facts[0].statement, answer)
        self.assertEqual(checked.checked_answer, answer)
        self.assertEqual(pipeline.verifier.audit_records[-1]['phase'], 'RETELLING')

    def test_wrong_document_attribution_is_rejected(self):
        provider = Provider(writer=lambda facts: [{
            'text': 'В документе «Другой.md»: Северный архив хранит карты.', 'fact_ids': ['fact:1']}])
        pipeline, facts = self.prepare(provider)
        answer, _ = pipeline.render('Что за Северный архив?', facts, [SOURCE], 'BALANCED')
        self.assertNotIn('Другой.md', answer)
        self.assertEqual(pipeline.report['rejected'][-1]['reason'], 'wrong_source_attribution')

    def test_unqualified_world_fact_cannot_replace_a_documentary_report(self):
        provider = Provider(writer=lambda facts: [{'text': STATEMENTS[0], 'fact_ids': ['fact:1']}])
        pipeline, facts = self.prepare(provider)
        answer, _ = pipeline.render('Что за Северный архив?', facts, [SOURCE], 'BALANCED')
        self.assertEqual(answer, ' '.join(f.statement for f in facts))
        self.assertEqual(pipeline.report['rejected'][-1]['reason'], 'retelling_not_supported')

    def test_explicit_document_report_survives_wrong_fact_modality_after_successful_content_audit(self):
        for sentence in [
            'В документе «Архив.md» указано, что Северный архив хранит карты.',
            'Северный архив хранит карты, согласно документу «Архив.md».']:
            with self.subTest(sentence=sentence):
                provider = Provider(writer=lambda facts: [{'text': sentence, 'fact_ids': ['fact:1']}])
                pipeline, facts = self.prepare(provider)
                answer, checked = pipeline.render('Что за Северный архив?', facts, [SOURCE], 'BALANCED')
                self.assertIn(sentence, answer)
                self.assertEqual(checked.checked_answer, answer)
                self.assertEqual(pipeline.verifier.audit_records[-1]['scope_adjustment'], 'explicit_document_frame')

    def test_according_to_another_document_cannot_inherit_fact_attribution(self):
        provider = Provider(writer=lambda facts: [{
            'text': 'Северный архив хранит карты, согласно документу «Другой.md».', 'fact_ids': ['fact:1']}])
        pipeline, facts = self.prepare(provider)
        answer, _ = pipeline.render('Что за Северный архив?', facts, [SOURCE], 'BALANCED')
        self.assertNotIn('Другой.md', answer)
        self.assertEqual(pipeline.report['rejected'][-1]['reason'], 'wrong_source_attribution')

    def test_one_sentence_cannot_hide_an_unexpressed_fact_behind_multiple_ids(self):
        provider = Provider(writer=lambda facts: [{'text': facts[0]['statement'], 'fact_ids': ['fact:1', 'fact:2']}])
        pipeline, facts = self.prepare(provider)
        answer, _ = pipeline.render('Что за Северный архив?', facts, [SOURCE], 'BALANCED')
        self.assertEqual(answer, ' '.join(f.statement for f in facts))
        self.assertEqual(pipeline.report['errors'][-1]['stage'], 'writer')

    def test_intention_is_not_confirmed_as_an_event_even_with_three_positive_model_flags(self):
        source = {**SOURCE, 'text': 'Северный архив планирует открыть выставку.'}
        provider = Provider(proposals=[{'statement': 'Северный архив открыл выставку.',
                                       'evidence_quote': source['text']}])
        original = provider.structured_generate
        def intention(**kw):
            result = original(**kw)
            if 'same_subject' in kw['schema']['properties']:
                data = json.loads(result.text)
                data.update(claim_predicate='открыл выставку', source_predicate='планирует открыть выставку',
                            claim_modality='FACT', source_modality='INTENTION')
                result.text = json.dumps(data)
            return result
        provider.structured_generate = intention
        pipeline, facts = self.prepare(provider, [source])
        self.assertTrue(all('открыл выставку' not in f.statement for f in facts))
        answer, checked = pipeline.render('Что за Северный архив?', facts, [source], 'BALANCED')
        self.assertIn('планирует', answer)
        self.assertFalse(is_grounded_explanation(checked, [source]))

    def test_source_diversity_and_conflicting_dates_are_preserved(self):
        sources = [SOURCE, {**SOURCE, 'chunk_id': 'workspace:Другой.md',
            'metadata': {'source': 'Другой.md'}, 'text': 'Северный архив основан в 2010 году.'}]
        provider = Provider()
        original = provider.structured_generate
        def generate(**kw):
            if 'facts' in kw['schema']['properties']:
                packet = json.loads(kw['messages'][-1]['content'])
                source_id = packet['passages'][0]['chunk_id']
                provider.proposals = [dict(statement=sources[1]['text'], evidence_quote=sources[1]['text'])] \
                    if source_id == sources[1]['chunk_id'] else [dict(statement=t, evidence_quote=t) for t in STATEMENTS]
            return original(**kw)
        provider.structured_generate = generate
        pipeline, facts = self.prepare(provider, sources)
        self.assertEqual({f.source_chunk_id for f in facts}, {s['chunk_id'] for s in sources})
        self.assertTrue(any('2001' in f.statement for f in facts))
        self.assertTrue(any('2010' in f.statement for f in facts))
        self.assertEqual(len({f.fact_id for f in facts}), len(facts))
        answer, _ = pipeline.render('Что за Северный архив?', facts, sources, 'BALANCED')
        self.assertIn('В документе «Архив.md»: Северный архив основан в 2001 году.', answer)
        self.assertIn('В документе «Другой.md»: Северный архив основан в 2010 году.', answer)

    def test_runtime_synthesizer_uses_fact_first_protocol(self):
        synthesizer = OverviewSynthesizer(Verifier(Provider()))
        answer = synthesizer.compose('Что за Северный архив?', [SOURCE], 'BALANCED')
        self.assertEqual(synthesizer.last_verification.checked_answer, answer)
        self.assertEqual(len(synthesizer.fact_report['facts']), 2)

    def test_workspace_journals_fact_preparation_and_does_not_recheck_unchanged_answer(self):
        from contextlib import closing
        env = support.UserEnvironment()
        self.addCleanup(env.close)
        provider = Provider()
        with patch('model_runtime.get_llm_provider', return_value=provider), patch('settings.VERIFIER_ENABLED', True):
            result = asyncio.run(env.execute('Что за Северный архив?', _workspace_matches=[{
                'path': 'Архив.md', 'content': SOURCE['text'], 'score': 1}]))
        self.assertEqual(result.data['verification'].answer_scope, 'EXPLANATORY')
        self.assertEqual(result.data['verification'].checked_answer, result.data['final'])
        self.assertFalse(any('claims' in c['schema']['properties'] for c in provider.calls))
        with closing(env.connection()) as conn:
            action = json.loads(conn.execute("SELECT output FROM assistant_actions WHERE action_type='GENERATE_RESPONSE'").fetchone()[0])
        self.assertEqual(len(action['data']['fact_preparation']['facts']), 2)

    def test_workspace_preparation_failure_is_unavailable_without_free_generation(self):
        env = support.UserEnvironment()
        self.addCleanup(env.close)
        provider = Provider()
        with patch('model_runtime.get_llm_provider', return_value=provider), patch('settings.VERIFIER_ENABLED', True), \
                patch.object(FactFirstOverview, 'prepare', side_effect=RuntimeError('tokenizer temporarily unavailable')):
            result = asyncio.run(env.execute('Что за Северный архив?', _workspace_matches=[{
                'path': 'Архив.md', 'content': SOURCE['text'], 'score': 1}]))
        self.assertEqual(result.data['verification'].status, 'UNAVAILABLE')
        self.assertEqual(result.data['verification'].checked_answer, result.data['final'])
        self.assertIn('tokenizer temporarily unavailable', result.data['final'])
        self.assertEqual(provider.calls, [])
