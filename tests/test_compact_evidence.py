"""Compact inference transport preserves every independent audit comparison."""
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import support
from evidence_audit import (EvidenceAuditor, EvidenceRelation, _WIRE_FIELDS, _MODALITIES,
                            _KINDS, _comparison_schema, _decode_comparison,
                            conflicting_self_report_author)
from pydantic_models import VerificationClaim
from verifier import Verifier, _quote_source_ids


class CompactEvidenceTests(unittest.TestCase):
    def row(self):
        return {'cs': 'Мирон', 'ss': 'Мирон', 'cp': 'хранит карты', 'sp': 'хранит карты',
                'cm': 'F', 'sm': 'F', 'sk': 'P', 'rq': True, 'es': True, 'ep': True, 'em': True}

    def test_all_fields_are_required_and_expanded_without_defaults(self):
        schema = _comparison_schema(True)
        self.assertEqual(set(schema['required']), set(self.row()))
        self.assertEqual(len(schema['required']), len(EvidenceRelation.model_fields))
        for missing in self.row():
            with self.subTest(missing=missing):
                row = self.row();row.pop(missing)
                with self.assertRaises(ValueError):
                    _decode_comparison(row, True)
        expanded = _decode_comparison(self.row(), True)
        self.assertEqual(expanded.source_subject_kind, 'PERSON')
        self.assertEqual(expanded.source_modality, 'FACT')
        self.assertEqual(set(expanded.model_dump()), set(EvidenceRelation.model_fields))

    def test_all_modality_and_kind_codes_round_trip(self):
        for code, value in _MODALITIES.items():
            row = {**self.row(), 'sm': code, 'cm': code}
            result = _decode_comparison(row, True)
            self.assertEqual(result.source_modality, value)
            self.assertEqual(result.claim_modality, value)
        for code, value in _KINDS.items():
            self.assertEqual(_decode_comparison({**self.row(), 'sk': code}, True).source_subject_kind, value)

    def test_truthy_strings_and_unknown_codes_are_not_certification(self):
        for key, value in (('es', 'false'), ('ep', 1), ('rq', 'true'), ('em', None), ('cm', 'UNKNOWN')):
            with self.subTest(key=key), self.assertRaises((ValueError, KeyError)):
                _decode_comparison({**self.row(), key: value}, True)

    def test_independent_guards_still_reject_modality_strengthening(self):
        claim = VerificationClaim(text='Мирон хранит карты.', status='SUPPORTED',
            source_chunk_ids=['s'], evidence_quote='Мирон намерен хранить карты.', basis='EXPLICIT')
        auditor = EvidenceAuditor(None, None)
        result = auditor.evaluate(claim, [{'chunk_id': 's', 'text': claim.evidence_quote}],
            _decode_comparison({**self.row(), 'sm': 'I'}, True))
        self.assertFalse(result)
        self.assertEqual(auditor.guard_rejection, 'modality_strengthening')

    def test_single_wire_request_and_full_internal_assessment(self):
        calls = []
        def generate(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(text=json.dumps(self.row()))
        client = SimpleNamespace(supports_compact_evidence=True, structured_generate=generate)
        auditor = EvidenceAuditor(client, lambda prefix,sources,*a,**kw:sources)
        claim = VerificationClaim(text='Мирон хранит карты.', status='SUPPORTED',
            source_chunk_ids=['s'], evidence_quote='Мирон хранит карты.', basis='EXPLICIT')
        with patch('settings.EVIDENCE_AUDIT_COMPACT', True):
            self.assertTrue(auditor.supports(claim, [{'chunk_id':'s','text':claim.evidence_quote}], 'FAST'))
        self.assertIn('cp', calls[0]['schema']['properties'])
        self.assertEqual(auditor.last_assessment.source_predicate, 'хранит карты')

    def test_legacy_transport_remains_selectable(self):
        full = _decode_comparison(self.row(), True).model_dump()
        calls = []
        client = SimpleNamespace(supports_compact_evidence=True,
            structured_generate=lambda **kwargs:(calls.append(kwargs) or SimpleNamespace(text=json.dumps(full))))
        claim = VerificationClaim(text='Мирон хранит карты.', status='SUPPORTED',
            source_chunk_ids=['s'], evidence_quote='Мирон хранит карты.', basis='EXPLICIT')
        with patch('settings.EVIDENCE_AUDIT_COMPACT', False):
            self.assertTrue(EvidenceAuditor(client, lambda p,s,*a,**kw:s).supports(
                claim, [{'chunk_id':'s','text':claim.evidence_quote}], 'FAST'))
        self.assertIn('claim_predicate', calls[0]['schema']['properties'])

    def test_tight_context_keeps_literal_evidence_once_with_provenance(self):
        quote = ('Мирон хранит карты северного архива. ' * 110).strip()
        source = {'chunk_id': 's', 'text': 'Автор письма — Мирон.\n' + quote + '\nКонец письма.'}
        calls = []
        client = SimpleNamespace(supports_compact_evidence=True,
            count_prompt=lambda messages,*a:sum(len(m['content']) for m in messages)//3,
            structured_generate=lambda **kw:(calls.append(kw) or SimpleNamespace(text=json.dumps(self.row()))))
        claim = VerificationClaim(text='Мирон хранит карты.', status='SUPPORTED',
            source_chunk_ids=['s'], evidence_quote=quote, basis='EXPLICIT')
        with patch('settings.EVIDENCE_AUDIT_COMPACT', True):
            self.assertTrue(EvidenceAuditor(client, Verifier(client).bounded_sources).supports(claim,[source],'FAST'))
        raw = calls[0]['messages'][-1]['content']
        packet, content = map(json.loads,raw.split('\nSources JSON: '))
        sources = content['sources']
        self.assertEqual(content['claim_to_check'],claim.text)
        self.assertNotIn('quote',packet)
        self.assertEqual(set(packet['quote_ids']),set(sources[0]['quotes']))
        self.assertEqual(_quote_source_ids(quote,{'s':' … '.join(sources[0]['quotes'].values())}),['s'])
        self.assertEqual(raw.count(quote.strip(' .')),1)
        self.assertIn('Автор письма',sources[0]['text'])

    def test_missing_reference_never_calls_the_audit_or_certifies_a_claim(self):
        client = SimpleNamespace(supports_compact_evidence=True, structured_generate=lambda **kw:self.fail('No evidence'))
        claim = VerificationClaim(text='Мирон хранит карты.',status='SUPPORTED',basis='EXPLICIT',
            source_chunk_ids=['s'],evidence_quote='Мирон хранит карты.')
        def too_small(prefix,sources,*a,**kw):
            return []
        with patch('settings.EVIDENCE_AUDIT_COMPACT',True), self.assertRaises(ValueError):
            EvidenceAuditor(client,too_small).supports(claim,[{'chunk_id':'s','text':claim.evidence_quote}],'FAST')

    def test_named_author_cannot_be_replaced_by_addressee_in_direct_self_report(self):
        source = 'Борис, подготовь карты. Я называю себя хранителем архива. Автор письма — Мирон.'
        wrong = 'В письме Борис называет себя хранителем архива.'
        assessment = _decode_comparison({**self.row(),'cs':'Борис','ss':'Борис','sk':'D'},True)
        self.assertTrue(conflicting_self_report_author(wrong,assessment,[{'text':source}]))
        claim = VerificationClaim(text=wrong,status='SUPPORTED',basis='EXPLICIT',source_chunk_ids=['s'],evidence_quote=source)
        auditor = EvidenceAuditor(None,None)
        self.assertFalse(auditor.evaluate(claim,[{'chunk_id':'s','text':source}],assessment))
        self.assertEqual(auditor.guard_rejection,'self_report_author_conflict')
        self.assertFalse(conflicting_self_report_author('Мирон называет себя хранителем.',
            _decode_comparison(self.row(),True),[{'text':source}]))
        self.assertFalse(conflicting_self_report_author('Мирон называет себя хранителем.',
            _decode_comparison(self.row(),True),[{'text':source.replace('Автор письма — Мирон.','Автор письма — Мирон Орлов.') }]))
        self.assertFalse(conflicting_self_report_author('Автор называет себя хранителем.',
            _decode_comparison({**self.row(),'cs':'Автор'},True),[{'text':source}]))
        for other_voice in (
            'Борис сказал: «Я называю себя хранителем архива». Автор письма — Мирон.',
            'Борис, ты называешь себя хранителем архива. Я храню карты. Автор письма — Мирон.',
            'Борис называет себя хранителем архива. Я храню карты. Автор письма — Мирон.',
            'Я называю себя хранителем. Автор письма — Мирон. Автор записки — Борис.',
        ):
            self.assertFalse(conflicting_self_report_author(wrong,assessment,[{'text':other_voice}]),other_voice)

    def test_explicit_author_at_end_survives_source_context_preparation(self):
        quote = 'Я называю себя хранителем северного архива.'
        source = quote + '\n' + ('Промежуточные записи о коробках. ' * 90) + '\nАвтор письма — Мирон.'
        self.assertIn('Автор письма — Мирон',EvidenceAuditor.source_context(source,quote))


if __name__ == '__main__':
    unittest.main()
