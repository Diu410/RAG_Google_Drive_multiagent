"""A direct self-designation survives a failed biography without adopting examples."""
import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import support
from evidence_audit import documentary_statement_evidence
from source_grounding import documentary_overview
from verifier import Verifier, _quote_source_ids
from pydantic_models import VerificationClaim, VerificationResult


SOURCE = {'chunk_id': 'workspace:Письмо.md', 'metadata': {'source': 'Письмо.md'},
          'text': 'Жди, я уже иду за тобой, это всё я — Хранитель Северного архива Мирон Лисов и мне подвластно всё.'}
IDENTITY = 'В тексте Мирон называет себя «Хранитель Северного архива Мирон Лисов».'


class PersonIdentityRecoveryTests(unittest.TestCase):
    def test_literal_self_report_is_checked_without_model_rewriting(self):
        provider = SimpleNamespace(structured_generate=lambda **kw: self.fail('Exact self-report needs no rewrite'))
        checked = Verifier(provider).verify('Кто такой Мирон?', IDENTITY, [SOURCE],
                                            'WORKSPACE_QA', 'BALANCED', exact_statements=True)
        self.assertEqual(checked.status, 'SUPPORTED')
        self.assertEqual(checked.checked_answer, IDENTITY)
        self.assertEqual(len(checked.claims), 1)

    def test_wrong_name_example_and_conditional_title_do_not_prove_self_report(self):
        for text, report in [
            (SOURCE['text'], IDENTITY.replace('Мирон называет', 'Алексей называет')),
            ('Например — Я Хранитель Северного архива Мирон Лисов.', IDENTITY),
            ('Если я — Хранитель Северного архива Мирон Лисов, мне всё подвластно.', IDENTITY)]:
            with self.subTest(text=text, report=report):
                self.assertIsNone(documentary_statement_evidence(report, {'id': text}, _quote_source_ids))

    def test_workspace_retains_identity_after_bad_draft_without_checking_source_footer(self):
        env = support.UserEnvironment();self.addCleanup(env.close)
        provider = SimpleNamespace(generate=lambda *a, **kw: SimpleNamespace(text='Мирон управляет каждым человеком.'),
            structured_generate=lambda **kw: SimpleNamespace(text='{"status":"UNSUPPORTED","claims":[]}'))
        candidates = []
        original = Verifier.verify
        def verify(service, question, answer, *args, **kw):
            candidates.append(answer)
            if answer == 'Мирон управляет каждым человеком.':
                return VerificationResult(status='UNSUPPORTED', checked_answer=answer, claims=[
                    VerificationClaim(text=answer, status='UNSUPPORTED', basis='EXPLICIT',
                                      source_chunk_ids=[SOURCE['chunk_id']], evidence_quote=SOURCE['text'])])
            return original(service, question, answer, *args, **kw)
        with patch('model_runtime.get_llm_provider', return_value=provider), patch('settings.VERIFIER_ENABLED', True), \
                patch('verifier.VERIFIER_ENABLED', True), \
                patch.object(Verifier, 'verify', verify):
            result = asyncio.run(env.execute('Кто такой Мирон?', _workspace_matches=[{
                'path': 'Письмо.md', 'content': SOURCE['text'], 'score': 1}]))
        self.assertIn(IDENTITY, result.data['final'])
        self.assertNotIn('управляет', result.data['final'])
        self.assertEqual(result.data['verification'].status, 'SUPPORTED')
        self.assertEqual(result.data['verification'].checked_answer, result.data['final'])
        recovery = next(c for c in candidates if c.startswith('В тексте Мирон'))
        self.assertNotIn('Источники:', recovery)
        self.assertIn('Источники:', documentary_overview('Кто такой Мирон?', [SOURCE]))

    def test_literal_verification_does_not_depend_on_question_pattern(self):
        provider = SimpleNamespace(structured_generate=lambda **kw: self.fail('Literal proof needs no model'))
        for question in ['Кто такой Мирон?', 'Как автор письма представляется?']:
            with self.subTest(question=question):
                checked = Verifier(provider).verify(question, IDENTITY, [SOURCE],
                    'WORKSPACE_QA', 'BALANCED', exact_statements=True)
                self.assertEqual(checked.status, 'SUPPORTED')
                self.assertEqual(checked.checked_answer, IDENTITY)
