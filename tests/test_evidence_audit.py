"""The production verifier requires entailment as well as literal citations."""
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import support
from llm_contract import GenerationCancelled, StructuredOutputTruncated
from verifier import Verifier, verify_answer, _quote_source_ids
from evidence_audit import is_literal_document_statement


class EvidenceAuditTests(unittest.TestCase):
    def test_optimistic_audit_cannot_specialize_a_generic_definition(self):
        quote = 'Общество это лишь вопрос твоей харизмы и подачи идеи.'
        text = 'Согласно тексту, Общество Рассвета — это лишь вопрос твоей харизмы и подачи идеи.'
        self.sources = [{'chunk_id': 'workspace:Письмо.md',
                         'text': quote + ' Я участник Общества Рассвета.', 'metadata': {}}]
        relation = dict(claim_subject='Общество Рассвета', source_subject='Общество',
                        source_subject_kind='CONCEPT', same_subject=True,
                        same_predicate=True, same_modality=True)
        provider, _ = self.provider(text, quote, relation)
        service = Verifier(provider)
        result = service.verify('Что за Общество Рассвета?', text, self.sources, 'WORKSPACE_QA', 'BALANCED')
        self.assertEqual(result.status, 'UNSUPPORTED')
        self.assertEqual(service.audit_records[0]['guard_rejection'], 'generic_definition_specialization')
        self.assertTrue(service.audit_records[0]['same_subject'])
        self.assertEqual(service.audit_records[0]['relevance_basis'], 'model')

    def test_specific_definition_and_literal_generic_quote_still_pass(self):
        for text, quote, relation in [
            ('Согласно тексту, Общество Рассвета — это группа читателей.',
             'Общество Рассвета это группа читателей.',
             dict(claim_subject='Общество Рассвета', source_subject='Общество',
                  source_subject_kind='ORGANIZATION', same_subject=True,
                  same_predicate=True, same_modality=True)),
            ('В тексте сказано: «Общество это вопрос харизмы».',
             'Общество это вопрос харизмы.', None),
        ]:
            with self.subTest(text=text):
                self.sources = [{'chunk_id': 'workspace:Письмо.md', 'text': quote, 'metadata': {}}]
                provider, _ = self.provider(text, quote, relation)
                result = Verifier(provider).verify('Что за Общество Рассвета?', text, self.sources,
                                                  'WORKSPACE_QA', 'BALANCED')
                self.assertEqual(result.status, 'SUPPORTED')

    def test_generic_definition_guard_uses_quote_even_if_model_invents_source_subject(self):
        quote = 'Общество — это вопрос харизмы.'
        text = 'Общество Рассвета — это вопрос харизмы.'
        self.sources = [{'chunk_id': 'workspace:Письмо.md', 'text': quote, 'metadata': {}}]
        relation = dict(claim_subject='Общество Рассвета', source_subject='Общество Рассвета',
                        source_subject_kind='ORGANIZATION', same_subject=True,
                        same_predicate=True, same_modality=True)
        provider, _ = self.provider(text, quote, relation)
        service = Verifier(provider)
        result = service.verify('Что за Общество Рассвета?', text, self.sources, 'WORKSPACE_QA', 'BALANCED')
        self.assertEqual(result.status, 'UNSUPPORTED')
        self.assertEqual(service.audit_records[0]['guard_rejection'], 'generic_definition_specialization')

    def test_reversed_relations_are_rejected_by_the_same_semantic_contract(self):
        for quote, text in [
            ('Мирон поставляет детали компании Вектор.', 'Компания Вектор поставляет детали Мирону.'),
            ('Мирон руководит отделом логистики.', 'Отдел логистики руководит Мироном.'),
            ('Мирон получает оплату от компании Вектор.', 'Мирон оплачивает услуги компании Вектор.'),
        ]:
            with self.subTest(quote=quote):
                self.sources = [{'chunk_id': 'workspace:Письмо.md',
                    'text': 'Меня зовут Мирон. ' + quote, 'metadata': {}}]
                relation = dict(claim_subject='Мирон', source_subject='Мирон',
                    same_subject=True, same_predicate=False, same_modality=True,
                    claim_modality='TEXTUAL_STATEMENT', source_modality='SELF_DESCRIPTION')
                provider, _ = self.provider(text, quote, relation)
                service = Verifier(provider)
                result = service.verify('Кто такой Мирон?', text, self.sources, 'WORKSPACE_QA', 'BALANCED')
                self.assertEqual(result.status, 'UNSUPPORTED')
                self.assertIsNone(service.audit_records[0]['guard_rejection'])
                self.assertFalse(service.audit_records[0]['same_predicate'])

    def test_consistent_power_direction_still_requires_and_accepts_a_semantic_audit(self):
        self.sources = [{'chunk_id': 'workspace:Письмо.md',
            'text': 'Меня зовут Мирон. Мне подвластно всё.', 'metadata': {}}]
        text = 'Мирон утверждает, что ему подвластно всё.'
        relation = dict(claim_subject='Мирон', source_subject='Мирон',
            same_subject=True, same_predicate=True, same_modality=True,
            claim_modality='TEXTUAL_STATEMENT', source_modality='SELF_DESCRIPTION')
        provider, calls = self.provider(text, 'Мне подвластно всё.', relation)
        service = Verifier(provider)
        self.assertEqual(service.verify('Кто такой Мирон?', text, self.sources,
                                       'WORKSPACE_QA', 'BALANCED').status, 'SUPPORTED')
        self.assertEqual(len(calls), 2)
        self.assertIsNone(service.audit_records[0]['guard_rejection'])
        relation['same_predicate'] = False
        self.assertEqual(self.verify(provider, text).status, 'UNSUPPORTED')

    def test_audit_prompt_has_no_domain_vocabulary(self):
        from evidence_audit import EvidenceAuditor
        prompt = EvidenceAuditor.instructions().casefold()
        for word in ('жатв', 'алчност', 'архиепископ', 'подвласт', 'мирон', 'аварит'):
            self.assertNotIn(word, prompt)
        self.assertIn('объект', prompt)
        self.assertIn('направление', prompt)

    def test_example_scope_cannot_be_strengthened_even_by_optimistic_verdict(self):
        from evidence_audit import EvidenceAuditor, EvidenceRelation
        claim = SimpleNamespace(text='Роман работает пилотом.', evidence_quote='Я пилот Роман.')
        relation = EvidenceRelation(claim_subject='Роман', source_subject='Роман',
            claim_predicate='работает пилотом', source_predicate='работает пилотом',
            claim_modality='FACT', source_modality='EXAMPLE', source_subject_kind='PERSON',
            same_subject=True, same_predicate=True, same_modality=True)
        auditor = EvidenceAuditor(None, None)
        self.assertFalse(auditor.evaluate(claim, [{'text': 'Пример: Я пилот Роман.'}], relation))
        self.assertEqual(auditor.guard_rejection, 'example_scope_strengthening')

    def test_reported_words_are_not_rejected_as_unproven_real_world_power(self):
        self.sources = [{'chunk_id': 'workspace:Письмо.md',
            'text': 'Меня зовут Мирон. Мне подвластно всё.', 'metadata': {}}]
        relation = dict(claim_subject='Мирон', source_subject='Мирон',
            same_subject=True, same_predicate=True, same_modality=False,
            claim_modality='FACT', source_modality='SELF_DESCRIPTION')
        text = 'В тексте Мирон утверждает, что ему подвластно всё.'
        provider, _ = self.provider(text, 'Мне подвластно всё.', relation)
        service = Verifier(provider)
        self.assertEqual(service.verify('Кто такой Мирон?', text, self.sources,
                                       'WORKSPACE_QA', 'BALANCED').status, 'SUPPORTED')
        self.assertEqual(service.audit_records[0]['guard_adjustment'], 'reported_self_description_modality')
        self.assertFalse(service.audit_records[0]['same_modality'])

    def test_reporting_rule_cannot_promote_a_world_fact_or_an_example(self):
        quote = 'Мне подвластно всё.'
        relation = dict(claim_subject='Мирон', source_subject='Мирон',
            same_subject=True, same_predicate=True, same_modality=False,
            claim_modality='FACT', source_modality='SELF_DESCRIPTION')
        for source_text, text in [
            ('Меня зовут Мирон. ' + quote, 'Мирону подвластно всё.'),
            ('Проговаривай, например: Меня зовут Мирон. ' + quote,
             'В тексте Мирон утверждает, что ему подвластно всё.'),
        ]:
            with self.subTest(text=text):
                self.sources = [{'chunk_id': 'workspace:Письмо.md', 'text': source_text, 'metadata': {}}]
                provider, _ = self.provider(text, quote, relation)
                self.assertEqual(self.verify(provider, text).status, 'UNSUPPORTED')

    sources = [{'chunk_id': 'workspace:Архив/Персонажи/Письмо.md',
        'text': 'Меня зовут Мирон. Я хранитель архива. Я не человек. Ты являешься посетителем. Питай меня.',
        'metadata': {'source': 'Архив/Персонажи/Письмо.md'}, 'found_by': ['workspace']}]

    def provider(self, text='Посетитель не человек.', quote='Я не человек.', relation=None, error=None):
        calls = []
        relation = {'claim_subject': 'Посетитель', 'source_subject': 'Мирон',
                    'same_subject': False, 'same_predicate': True, 'same_modality': True} if relation is None else relation
        def generate(**kwargs):
            calls.append(kwargs)
            if 'claims' in kwargs['schema']['properties']:
                payload = {'status': 'SUPPORTED', 'claims': [{
                    'text': text, 'status': 'SUPPORTED', 'basis': 'EXPLICIT',
                    'source_chunk_ids': [self.sources[0]['chunk_id']], 'evidence_quote': quote,
                    'relevant_to_question': True}]}
            else:
                if error:
                    raise error
                payload = {'claim_predicate': text, 'source_predicate': quote,
                           'claim_modality': 'FACT', 'source_modality': 'FACT',
                           'relevant_to_question': True,
                           'source_subject_kind': 'PERSON', **relation}
            return SimpleNamespace(text=json.dumps(payload, ensure_ascii=False))
        return SimpleNamespace(structured_generate=generate), calls

    def verify(self, provider, answer='Посетитель не человек.'):
        return Verifier(provider).verify('Кто такой посетитель?', answer, self.sources, 'WORKSPACE_QA', 'BALANCED')

    def test_default_verifier_rejects_quote_attributed_to_a_different_subject(self):
        provider, calls = self.provider()
        result = self.verify(provider)
        self.assertEqual(result.status, 'UNSUPPORTED')
        self.assertEqual(len(calls), 2)
        self.assertTrue(result.claims[0].evidence_quote)

    def test_literal_overlap_does_not_establish_an_invented_predicate(self):
        relation = {'claim_subject': 'Мирон', 'source_subject': 'Мирон',
                    'same_subject': True, 'same_predicate': False, 'same_modality': True}
        provider, _ = self.provider('Мирон требует жертв.', 'Питай меня.', relation)
        self.assertEqual(self.verify(provider, 'Мирон требует жертв.').status, 'UNSUPPORTED')

    def test_fact_cannot_replace_a_wish_or_condition(self):
        relation = {'claim_subject': 'Мирон', 'source_subject': 'Мирон',
                    'same_subject': True, 'same_predicate': True, 'same_modality': False}
        provider, _ = self.provider(relation=relation)
        self.assertEqual(self.verify(provider).status, 'UNSUPPORTED')

    def test_matching_subject_predicate_and_modality_preserve_a_grounded_fact(self):
        relation = {'claim_subject': 'Мирон', 'source_subject': 'Мирон',
                    'same_subject': True, 'same_predicate': True, 'same_modality': True}
        provider, _ = self.provider('Мирон — хранитель архива.', 'Я хранитель архива.', relation)
        self.assertEqual(self.verify(provider, 'Мирон — хранитель архива.').status, 'SUPPORTED')

    def test_a_true_statement_about_another_topic_does_not_answer_the_question(self):
        relation = {'claim_subject': 'Мирон', 'source_subject': 'Мирон',
                    'same_subject': True, 'same_predicate': True, 'same_modality': True,
                    'relevant_to_question': False}
        provider, calls = self.provider('Мирон — хранитель архива.', 'Я хранитель архива.', relation)
        result = self.verify(provider, 'Мирон — хранитель архива.')
        self.assertEqual(result.claims[0].status, 'SUPPORTED')
        self.assertFalse(result.claims[0].relevant_to_question)
        self.assertEqual(result.status, 'INSUFFICIENT_INFORMATION')
        self.assertIn('"question": "Кто такой посетитель?"', calls[-1]['messages'][1]['content'])

    def test_optimistic_boolean_cannot_turn_intent_advice_condition_or_image_into_ability(self):
        for modality in ('INTENTION', 'ADVICE', 'CONDITION', 'METAPHOR'):
            with self.subTest(modality=modality):
                relation = {'claim_subject': 'Мирон', 'source_subject': 'Мирон',
                    'same_subject': True, 'same_predicate': True, 'same_modality': True,
                    'claim_modality': 'FACT', 'source_modality': modality}
                provider, _ = self.provider('Мирон обладает способностью собирать ресурсы.', 'Я хранитель архива.', relation)
                self.assertEqual(self.verify(provider).status, 'UNSUPPORTED')

    def test_qualified_self_description_is_not_rejected_as_a_physical_fact(self):
        relation = {'claim_subject': 'Мирон', 'source_subject': 'Мирон',
            'same_subject': True, 'same_predicate': True, 'same_modality': True,
            'claim_modality': 'TEXTUAL_STATEMENT', 'source_modality': 'SELF_DESCRIPTION'}
        provider, calls = self.provider('Он называет себя хранителем архива.', 'Я хранитель архива.', relation)
        result = self.verify(provider, 'Персонаж Мирон упоминается в письме. Он называет себя хранителем архива.')
        self.assertEqual(result.status, 'SUPPORTED')
        audit = calls[-1]
        payload = json.loads(audit['messages'][-1]['content'].split('\nSources JSON:', 1)[0])
        self.assertIn('Персонаж Мирон', payload['answer_context'])
        self.assertTrue({'claim_predicate', 'source_predicate', 'claim_modality', 'source_modality'} <= set(audit['schema']['required']))

    def test_source_context_preserves_both_short_quote_and_distant_identity_at_the_end(self):
        from evidence_audit import EvidenceAuditor
        text = 'Я хранитель архива. ' + 'Нейтральный фрагмент. ' * 500 + 'Это всё я - Мирон и мне принадлежит архив.'
        context = EvidenceAuditor.source_context(text, 'Я хранитель архива.')
        self.assertIn('Я хранитель архива.', context)
        self.assertIn('Это всё я - Мирон', context)
        self.assertLess(len(context), 600)

    def test_context_reduction_preserves_all_spans_of_an_ellipsis_quote(self):
        from evidence_audit import EvidenceAuditor
        text = 'Архив открыт в полночь. ' + 'Нейтральный фрагмент. ' * 200 + 'Архив закрывается утром.'
        quote = 'Архив открыт в полночь … Архив закрывается утром'
        context = EvidenceAuditor.source_context(text, quote)
        self.assertEqual(_quote_source_ids(quote, {'source': context}), ['source'])

    def test_optimistic_audit_cannot_assign_an_impersonal_definition_to_a_named_character(self):
        # The source names a character in a suggested affirmation, not as its author.
        sources = [{**self.sources[0], 'text': 'Обучение — тренировка ума. Например: Я хранитель Мирон.'}]
        provider, _ = self.provider('Мирон определяет обучение как тренировку ума.', 'Обучение — тренировка ума.',
            {'claim_subject': 'Мирон', 'source_subject': 'Мирон',
             'same_subject': True, 'same_predicate': True, 'same_modality': True})
        result = Verifier(provider).verify('Кто такой Мирон?', 'Мирон определяет обучение как тренировку ума.',
                                           sources, 'WORKSPACE_QA', 'BALANCED')
        self.assertEqual(result.status, 'UNSUPPORTED')

    def test_a_suggested_first_person_affirmation_is_not_the_characters_own_statement(self):
        from evidence_audit import EvidenceRelation, _speaker_binding_is_explicit
        relation = EvidenceRelation(claim_subject='Мирон', source_subject='Мирон',
            same_subject=True, same_predicate=True, same_modality=True)
        self.assertFalse(_speaker_binding_is_explicit(relation, 'Я хранитель архива.',
            [{'text': 'Повторяй ежедневно. Например: Меня зовут Мирон. Я хранитель архива.'}]))
        self.assertTrue(_speaker_binding_is_explicit(relation, 'Я хранитель архива.',
            [{'text': 'Меня зовут Мирон. Я хранитель архива.'}]))

    def test_explicit_attribution_of_an_impersonal_definition_remains_supported(self):
        from evidence_audit import EvidenceRelation, _speaker_binding_is_explicit
        relation = EvidenceRelation(claim_subject='Мирон', source_subject='Мирон',
            same_subject=True, same_predicate=True, same_modality=True)
        self.assertTrue(_speaker_binding_is_explicit(relation, 'Обучение — тренировка ума.',
            [{'text': 'Мирон считает: обучение — тренировка ума.'}]))

    def test_exact_document_quotation_needs_no_interpretation(self):
        answer = 'В тексте прямо сказано: «Я хранитель архива». '
        provider, calls = self.provider(answer, 'Я хранитель архива.')
        self.assertEqual(self.verify(provider, answer).status, 'SUPPORTED')
        self.assertEqual(len(calls), 0)

    def test_quotation_with_added_conclusion_must_still_be_audited(self):
        mapping = {self.sources[0]['chunk_id']: self.sources[0]['text']}
        self.assertFalse(is_literal_document_statement(
            'В тексте сказано: «Питай меня», значит, он требует жертв.', mapping, _quote_source_ids))
        self.assertFalse(is_literal_document_statement(
            'В тексте сказано: «Мирон требует жертв».', mapping, _quote_source_ids))

    def test_only_an_exact_topic_mention_in_an_explicit_designation_is_a_documentary_fact(self):
        statement = 'Северный архив упоминается в самоназвании говорящего в тексте — «Хранитель Северного архива Мирон».'
        source = {'id': 'Это всё я — Хранитель Северного архива Мирон и мне подвластно всё.'}
        self.assertTrue(is_literal_document_statement(statement, source, _quote_source_ids))
        self.assertFalse(is_literal_document_statement(statement + ' Поэтому он бессмертен.', source, _quote_source_ids))
        self.assertFalse(is_literal_document_statement(statement.replace('Северный архив упоминается', 'Южный архив упоминается'), source, _quote_source_ids))
        self.assertFalse(is_literal_document_statement(statement, {'id': 'Например: ' + source['id']}, _quote_source_ids))

    def test_an_exact_quotation_with_another_plain_documentary_wrapper_needs_no_interpretation(self):
        source = {'id': 'Северный архив хранит карты.'}
        self.assertTrue(is_literal_document_statement('В этом же тексте сказано: «Северный архив хранит карты».', source, _quote_source_ids))

    def test_missing_semantic_fields_cannot_silently_accept_a_claim(self):
        provider, _ = self.provider(relation={'same_subject': True})
        with patch('verifier.VERIFIER_ENABLED', True):
            _, result = verify_answer('Кто?', 'Посетитель не человек.', self.sources, 'WORKSPACE_QA',
                                      'BALANCED', verifier=Verifier(provider))
        self.assertEqual(result.status, 'UNAVAILABLE')

    def test_incomplete_semantic_output_is_not_a_negative_fact_or_success(self):
        provider, _ = self.provider(error=StructuredOutputTruncated('audit incomplete'))
        with patch('verifier.VERIFIER_ENABLED', True):
            _, result = verify_answer('Кто?', 'Посетитель не человек.', self.sources, 'WORKSPACE_QA',
                                      'BALANCED', verifier=Verifier(provider))
        self.assertEqual(result.status, 'UNAVAILABLE')

    def test_cancellation_in_semantic_audit_propagates(self):
        provider, _ = self.provider(error=GenerationCancelled())
        with self.assertRaises(GenerationCancelled):
            self.verify(provider)

    def test_claim_with_invented_quote_does_not_reach_the_semantic_audit(self):
        provider, calls = self.provider(quote='В архиве хранится тысяча сокровищ.')
        self.assertEqual(self.verify(provider).status, 'UNSUPPORTED')
        self.assertEqual(len(calls), 1)

    def test_original_literal_quote_recovers_the_actual_source_before_model_selection(self):
        answer = 'В тексте сказано: «Архив закрывается в полночь». '
        provider, calls = self.provider(answer, 'Я хранитель архива.')
        sources = self.sources + [{'chunk_id': 'workspace:Другой.md',
            'text': 'Архив закрывается в полночь.', 'metadata': {'source': 'Другой.md'}}]
        result = Verifier(provider).verify('Что в архиве?', answer, sources, 'WORKSPACE_QA', 'BALANCED')
        self.assertEqual(result.status, 'SUPPORTED')
        self.assertEqual(result.claims[0].source_chunk_ids, ['workspace:Другой.md'])
        self.assertEqual(len(calls), 0)

    def test_model_cannot_name_an_anonymous_addressee_after_a_nearby_concept(self):
        from evidence_audit import EvidenceRelation, _recipient_binding_is_explicit
        relation = EvidenceRelation(claim_subject='Терпение', source_subject='Терпение',
            same_subject=True, same_predicate=True, same_modality=True)
        sources = [{'text': 'Ты стремишься к цели. В этом выражается Терпение.'}]
        self.assertFalse(_recipient_binding_is_explicit(relation, 'Ты стремишься к цели.', sources))

    def test_second_person_verb_alone_does_not_establish_a_named_addressee(self):
        from evidence_audit import EvidenceRelation, _recipient_binding_is_explicit
        relation = EvidenceRelation(claim_subject='Терпение', source_subject='Терпение',
            same_subject=True, same_predicate=True, same_modality=True)
        self.assertFalse(_recipient_binding_is_explicit(relation, 'Постоянно работаешь над собой.',
            [{'text': 'Постоянно работаешь над собой. В этом выражается Терпение.'}]))

    def test_explicitly_named_recipient_and_unnamed_recipient_remain_valid(self):
        from evidence_audit import EvidenceRelation, _recipient_binding_is_explicit
        named = EvidenceRelation(claim_subject='Мирон', source_subject='Мирон',
            same_subject=True, same_predicate=True, same_modality=True)
        self.assertTrue(_recipient_binding_is_explicit(named, 'Ты стремишься к цели.',
            [{'text': 'Дорогой Мирон, ты стремишься к цели.'}]))
        unnamed = EvidenceRelation(claim_subject='Собеседник', source_subject='Неназванный адресат',
            same_subject=True, same_predicate=True, same_modality=True)
        self.assertTrue(_recipient_binding_is_explicit(unnamed, 'Ты стремишься к цели.',
            [{'text': 'Ты стремишься к цели.'}]))

    def test_particle_lish_is_not_a_second_person_verb(self):
        from evidence_audit import EvidenceRelation, _recipient_binding_is_explicit
        relation = EvidenceRelation(claim_subject='Мирон', source_subject='Мирон',
            same_subject=True, same_predicate=True, same_modality=True)
        self.assertTrue(_recipient_binding_is_explicit(relation, 'Он хранит лишь карты.',
            [{'text': 'Мирон — хранитель архива. Он хранит лишь карты.'}]))
