"""Contextual review preserves explanations without certifying unchecked text."""
import asyncio
from contextlib import closing
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import support
from answer_review import review_answer
from llm_contract import GenerationCancelled, GenerationDeadlineExceeded
from verifier import verify_knowledge_answer


SOURCE = {'chunk_id': 'letter', 'text':
    'Пожалуйста, подготовь карты до пятницы. Я храню архивные карты. '
    'Автор письма — хранитель Марк.', 'metadata': {'source': 'Письмо.md'}}
FACT = 'Автор письма хранит архивные карты.'
UNKNOWN = 'В найденных фрагментах имя адресата не указано.'
GUESS = 'Адресат письма является хранителем архива.'


def payload(packet, decisions=None, issues=None):
    decisions = decisions or {}
    quotes = {key: text for source in packet['sources'] for key, text in source['passages'].items()}
    assessments = {}
    for key in packet['assess_only']:
        text = packet['answer'][key]['text']
        verdict, quote, positive = decisions.get(text, ('SUPPORTED', 'Я храню архивные карты.', True))
        ids = [key for key, value in quotes.items() if value == quote] if quote else []
        code = {'SUPPORTED': 'S', 'INFERENCE': 'I', 'MISSING': 'M', 'UNSUPPORTED': 'U', 'CONFLICTING': 'C'}[verdict]
        assessments[key] = {'v': code, 'q': ids, 'p': positive}
    keyed_issues = [{'statement_id': key, 'reason': 'ATTRIBUTION',
                    'replacement': issues[value['text']]['replacement']}
                   for key, value in packet['answer'].items()
                   if issues and value['text'] in issues and key in assessments]
    return {'assessments': assessments, 'issues': keyed_issues}


class ReviewProvider:
    def __init__(self, replies=None, *, draft=None, counter=None, audits=None, finish_reason='stop'):
        self.replies = list(replies or [])
        self.calls = []
        self.draft = draft
        self.counter = counter
        self.generation_calls = []
        self.audit_calls = []
        self.audits = audits or {}
        self.finish_reason = finish_reason

    def count_prompt(self, messages, alias, thinking):
        if '\nSources JSON: ' in messages[-1]['content']:
            return 100
        return self.counter(messages) if self.counter else 100

    def generate(self, messages, **options):
        self.generation_calls.append((messages, options))
        return SimpleNamespace(text=self.draft, metrics=SimpleNamespace(finish_reason=self.finish_reason))

    def structured_generate(self, **options):
        if 'claim_subject' in options['schema'].get('properties', {}):
            self.audit_calls.append(options)
            packet = json.loads(options['messages'][-1]['content'].split('\nSources JSON: ')[0])
            assessment = self.audits.get(packet['claim'], {})
            if isinstance(assessment, BaseException):
                raise assessment
            value = {'claim_subject': 'автор', 'source_subject': 'автор',
                'claim_predicate': 'хранит карты', 'source_predicate': 'хранит карты',
                'claim_modality': 'TEXTUAL_STATEMENT', 'source_modality': 'FACT',
                'source_subject_kind': 'PERSON', 'relevant_to_question': True,
                'same_subject': True, 'same_predicate': True, 'same_modality': True, **assessment}
            return SimpleNamespace(text=json.dumps(value, ensure_ascii=False),
                metrics=SimpleNamespace(finish_reason='stop'))
        self.calls.append(options)
        packet = json.loads(options['messages'][-1]['content'])
        reply = self.replies.pop(0) if self.replies else payload
        if isinstance(reply, BaseException):
            raise reply
        value = reply(packet) if callable(reply) else reply
        return SimpleNamespace(text=json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else value,
                               metrics=SimpleNamespace(finish_reason='stop'))


class ContextualReviewTests(unittest.TestCase):
    def setUp(self):
        self.enabled = patch('settings.VERIFIER_ENABLED', True)
        self.enabled.start()
        self.addCleanup(self.enabled.stop)

    def review(self, answer, provider, question='Что известно из письма?', sources=None):
        return review_answer(question, answer, sources or [SOURCE], 'WORKSPACE_QA', 'FAST', client=provider)

    def test_short_heading_is_literal_context_without_bypassing_independent_audit(self):
        source = {'chunk_id': 'archive', 'text': '# Хранитель\nХранитель — смотритель архива. Его имя — Мирон.'}
        claim = 'Хранителя зовут Мирон.'
        def supported(packet):
            quotes = packet['sources'][0]['passages']
            self.assertTrue(all(len(' '.join(text.split())) >= 12 for text in quotes.values()))
            self.assertTrue(all(text in source['text'] for text in quotes.values()))
            return {'assessments': {key: {'v': 'S', 'p': True, 'q': list(quotes)}
                    for key in packet['assess_only']}, 'issues': []}
        provider = ReviewProvider([supported])
        answer, check = self.review(claim, provider, sources=[source])
        self.assertEqual(answer, claim)
        self.assertEqual(check.status, 'SUPPORTED')
        self.assertEqual(len(provider.audit_calls), 1)
        self.assertTrue(check.review_details['evidence_audits'])

    def test_short_heading_never_grants_support_when_independent_audit_rejects(self):
        source = {'chunk_id': 'archive', 'text': '# Хранитель\nХранитель хранит карты.'}
        claim = 'Хранитель хранит деньги.'
        def supported(packet):
            return {'assessments': {key: {'v': 'S', 'p': True,
                    'q': list(packet['sources'][0]['passages'])} for key in packet['assess_only']}, 'issues': []}
        provider = ReviewProvider([supported, supported], audits={claim: {'same_predicate': False}})
        _, check = self.review(claim, provider, sources=[source])
        self.assertEqual(check.status, 'UNSUPPORTED')
        self.assertEqual(len(provider.audit_calls), 1)

    def test_uncertain_sentence_does_not_erase_explanation_or_paragraphs(self):
        draft = FACT + '\n\n' + GUESS
        provider = ReviewProvider([lambda p: payload(p, {GUESS: ('UNSUPPORTED', None, True)},
            {GUESS: {'reason': 'Роль говорящего перенесена на адресата.', 'replacement': ''}})])
        answer, check = self.review(draft, provider)
        self.assertEqual(answer, draft)
        self.assertEqual(check.checked_answer, draft)
        self.assertEqual(check.status, 'PARTIALLY_SUPPORTED')
        self.assertEqual(check.unsupported_claims, [GUESS])
        self.assertIn('адресат', check.claims[1].note)
        self.assertEqual(check.review_mode, 'CONTEXTUAL')
        self.assertIsNone(check.confidence)
        self.assertEqual(len(provider.calls), 1)

    def test_known_fact_and_unknown_detail_both_remain(self):
        draft = UNKNOWN + '\n\n' + FACT
        provider = ReviewProvider([lambda p: payload(p, {UNKNOWN: ('MISSING', None, False)})])
        answer, check = self.review(draft, provider)
        self.assertEqual(answer, draft)
        self.assertEqual(check.claims[0].basis, 'UNANSWERED')
        self.assertEqual(check.claims[1].status, 'SUPPORTED')
        self.assertEqual(check.status, 'PARTIALLY_SUPPORTED')
        self.assertEqual(len(provider.calls), 1)

    def test_mixed_missing_sentence_cannot_hide_positive_fact(self):
        draft = 'Имя адресата не указано, но он хранит карты.'
        provider = ReviewProvider([lambda p: payload(p, {draft: ('MISSING', None, True)})])
        answer, check = self.review(draft, provider)
        self.assertEqual(answer, draft)
        self.assertEqual(check.claims[0].status, 'UNSUPPORTED')
        self.assertEqual(check.claims[0].basis, 'EXPLICIT')

    def test_qualified_interpretation_retains_its_scope(self):
        text = 'Возможно, автор рассчитывает получить карты до пятницы.'
        provider = ReviewProvider([lambda p: payload(p,
            {text: ('INFERENCE', 'Пожалуйста, подготовь карты до пятницы.', True)})])
        answer, check = self.review(text, provider)
        self.assertEqual(answer, text)
        self.assertEqual(check.status, 'PARTIALLY_SUPPORTED')
        self.assertEqual(check.claims[0].basis, 'INFERENCE')
        self.assertIn('интерпретации', check.warning)

    def test_inference_without_visible_qualification_is_not_supported(self):
        text = 'Автор рассчитывает получить карты до пятницы.'
        provider = ReviewProvider([lambda p: payload(p,
            {text: ('INFERENCE', 'Пожалуйста, подготовь карты до пятницы.', True)})])
        _, check = self.review(text, provider)
        self.assertEqual(check.status, 'UNSUPPORTED')

    def test_only_a_rechecked_local_correction_can_change_the_answer(self):
        draft = FACT + '\n\n' + GUESS
        provider = ReviewProvider([
            lambda p: payload(p, {GUESS: ('UNSUPPORTED', None, True)},
                {GUESS: {'reason': 'Адресат не назван.', 'replacement': UNKNOWN}}),
            lambda p: payload(p, {UNKNOWN: ('MISSING', None, False)})])
        answer, check = self.review(draft, provider)
        self.assertEqual(answer, FACT + '\n\n' + UNKNOWN)
        self.assertEqual(check.checked_answer, answer)
        self.assertEqual(check.claims[1].text, UNKNOWN)
        self.assertEqual(check.claims[1].original_text, GUESS)
        self.assertEqual(len(provider.calls), 2)
        second = json.loads(provider.calls[1]['messages'][-1]['content'])
        self.assertEqual(second['answer']['0']['text'], FACT)
        self.assertEqual(second['assess_only'], ['1'])

    def test_rejected_correction_keeps_original_and_does_not_retry(self):
        replacement = 'Адресат хранит архив.'
        provider = ReviewProvider([
            lambda p: payload(p, {GUESS: ('CONFLICTING', None, True)},
                {GUESS: {'reason': 'Участники перепутаны.', 'replacement': replacement}}),
            lambda p: payload(p, {replacement: ('UNSUPPORTED', None, True)},
                {replacement: {'reason': 'Снова нет основания.', 'replacement': FACT}})])
        answer, check = self.review(FACT + '\n' + GUESS, provider)
        self.assertEqual(answer, FACT + '\n' + GUESS)
        self.assertEqual(check.review_details['edits'], [])
        self.assertEqual(check.status, 'CONFLICTING')
        self.assertEqual(len(provider.calls), 2)

    def test_failed_correction_preserves_completed_review(self):
        provider = ReviewProvider([
            lambda p: payload(p, {GUESS: ('UNSUPPORTED', None, True)},
                {GUESS: {'reason': 'Нет основания.', 'replacement': UNKNOWN}}),
            GenerationDeadlineExceeded('deadline')])
        answer, check = self.review(FACT + '\n' + GUESS, provider)
        self.assertEqual(answer, FACT + '\n' + GUESS)
        self.assertEqual(check.claims[0].status, 'SUPPORTED')
        self.assertEqual(check.claims[1].status, 'UNSUPPORTED')
        self.assertIn('лимиту времени', check.warning)
        self.assertEqual(len(provider.calls), 2)

    def test_supported_sentence_cannot_be_rewritten_by_an_issue(self):
        provider = ReviewProvider([lambda p: payload(p, issues={FACT: {
            'reason': 'Измени стиль.', 'replacement': 'Короткая цитата.'}})])
        answer, check = self.review(FACT, provider)
        self.assertEqual(answer, FACT)
        self.assertEqual(check.review_details['edits'], [])
        self.assertEqual(len(provider.calls), 1)

    def test_timeout_retains_draft_without_certifying_it(self):
        provider = ReviewProvider([GenerationDeadlineExceeded('deadline')])
        answer, check = self.review(FACT + '\n\n' + GUESS, provider)
        self.assertEqual(answer, FACT + '\n\n' + GUESS)
        self.assertEqual(check.status, 'UNAVAILABLE')
        self.assertEqual(check.claims, [])
        self.assertIn('лимиту времени', check.warning)
        self.assertEqual(len(provider.calls), 1)

    def test_incomplete_or_unknown_evidence_never_certifies_a_draft(self):
        def omit(packet):
            value = payload(packet)
            value['assessments'].clear()
            return value
        def bad_quote(packet):
            value = payload(packet)
            next(iter(value['assessments'].values()))['q'] = ['invented']
            return value
        for response in (omit, bad_quote, '{"assessments":',
                         '{"assessments":{},"issues":{},"issues":{}}'):
            with self.subTest(response=response):
                provider = ReviewProvider([response])
                answer, check = self.review(FACT, provider)
                self.assertEqual(answer, FACT)
                self.assertEqual(check.status, 'UNAVAILABLE')
                self.assertEqual(check.claims, [])
                self.assertEqual(len(provider.calls), 1)

    def test_claim_without_evidence_cannot_be_supported(self):
        provider = ReviewProvider([lambda p: payload(p, {FACT: ('SUPPORTED', None, True)})])
        _, check = self.review(FACT, provider)
        self.assertEqual(check.status, 'UNSUPPORTED')

    def test_fact_after_source_footer_is_not_skipped(self):
        text = 'В файле Письмо.md адресат назван хранителем.'
        draft = FACT + '\n\nИсточники:\nПисьмо.md\n\n' + text
        provider = ReviewProvider([lambda p: payload(p, {text: ('UNSUPPORTED', None, True)})])
        answer, check = self.review(draft, provider)
        self.assertEqual(answer, draft)
        self.assertEqual(check.unsupported_claims, [text])
        self.assertEqual([c.text for c in check.claims], [FACT, text])

    def test_numbered_source_footer_does_not_create_false_claims(self):
        provider = ReviewProvider()
        draft = FACT + '\n\nИспользованные файлы:\n1. Письмо.md'
        answer, check = self.review(draft, provider)
        self.assertEqual(answer, draft)
        self.assertEqual([c.text for c in check.claims], [FACT])
        self.assertEqual(check.status, 'SUPPORTED')

    def test_source_label_is_presentation_rather_than_a_factual_claim(self):
        draft = FACT + '\n\n**Источники:**\n* Файл `Письмо.md`'
        _, check = self.review(draft, ReviewProvider())
        self.assertEqual([c.text for c in check.claims], [FACT])

    def test_context_overflow_never_drops_a_conflicting_source(self):
        second = {'chunk_id': 'other', 'text': 'Другой документ содержит дополнительные данные.'}
        provider = ReviewProvider(counter=lambda m: 5000 if len(json.loads(m[-1]['content'])['sources']) > 1 else 100)
        _, check = self.review(FACT, provider, sources=[SOURCE, second])
        self.assertEqual(check.status, 'UNAVAILABLE')
        self.assertFalse(provider.calls)

    def test_long_answer_is_batched_with_all_sources_retained(self):
        other = {'chunk_id': 'other', 'text': 'В другой версии автор не хранит карты.'}
        draft = '\n\n'.join(f'Автор хранит набор карт номер {i}.' for i in range(8))
        provider = ReviewProvider(counter=lambda m: 5000
            if len(json.loads(m[-1]['content']).get('assess_only', [])) > 2 else 100)
        answer, check = self.review(draft, provider, sources=[SOURCE, other])
        self.assertEqual(answer, draft)
        self.assertEqual(len(provider.calls), 4)
        self.assertEqual(len(check.claims), 8)
        for call in provider.calls:
            self.assertEqual(len(json.loads(call['messages'][-1]['content'])['sources']), 2)

    def test_unrelated_real_quote_is_rejected_by_independent_audit(self):
        text = 'Автор управляет обсерваторией.'
        provider = ReviewProvider(audits={text: {'same_predicate': False}})
        _, check = self.review(text, provider)
        self.assertEqual(check.status, 'UNSUPPORTED')
        self.assertEqual(len(provider.audit_calls), 1)
        self.assertFalse(check.review_details['evidence_audits'][0]['supported'])

    def test_initial_false_rejection_requires_audit_and_full_source_agreement(self):
        provider = ReviewProvider([
            lambda p: payload(p, {FACT: ('UNSUPPORTED', 'Я храню архивные карты.', True)}),
            payload])
        answer, check = self.review(FACT, provider)
        self.assertEqual(answer, FACT)
        self.assertEqual(check.status, 'SUPPORTED')
        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(len(provider.audit_calls), 1)

    def test_audit_rejection_can_trigger_one_checked_local_repair(self):
        wrong = 'Автор управляет обсерваторией.'
        provider = ReviewProvider([
            payload,
            lambda p: payload(p, {wrong: ('UNSUPPORTED', None, True)},
                {wrong: {'replacement': FACT}}), payload],
            audits={wrong: {'same_predicate': False}})
        answer, check = self.review(wrong, provider)
        self.assertEqual(answer, FACT)
        self.assertEqual(check.status, 'SUPPORTED')
        self.assertEqual(check.claims[0].original_text, wrong)
        self.assertEqual(len(provider.audit_calls), 2)

    def test_audit_failure_never_inherits_first_review_support(self):
        provider = ReviewProvider(audits={FACT: GenerationDeadlineExceeded('deadline')})
        _, check = self.review(FACT, provider)
        self.assertEqual(check.status, 'UNAVAILABLE')
        self.assertEqual(check.claims[0].status, 'UNAVAILABLE')

    def test_audit_timeout_preserves_completed_comparisons_in_the_same_group(self):
        other = 'Автор проверяет каталоги.'
        provider = ReviewProvider(audits={other: GenerationDeadlineExceeded('deadline')})
        _, check = self.review(FACT + '\n\n' + other, provider)
        self.assertEqual(check.status, 'UNAVAILABLE')
        self.assertEqual(check.claims[0].status, 'SUPPORTED')
        self.assertEqual(check.claims[1].status, 'UNAVAILABLE')

    def test_multiple_local_corrections_are_independently_checked(self):
        other_guess = 'Адресат руководит обсерваторией.'
        corrected = 'В найденных фрагментах должность адресата не указана.'
        provider = ReviewProvider([
            lambda p: payload(p, {GUESS: ('UNSUPPORTED', None, True),
                other_guess: ('UNSUPPORTED', None, True)},
                {GUESS: {'replacement': UNKNOWN}, other_guess: {'replacement': corrected}}),
            lambda p: payload(p, {UNKNOWN: ('MISSING', None, False), corrected: ('MISSING', None, False)})])
        answer, check = self.review(FACT + '\n\n' + GUESS + '\n\n' + other_guess, provider)
        self.assertEqual(answer, FACT + '\n\n' + UNKNOWN + '\n\n' + corrected)
        self.assertEqual(len(check.review_details['edits']), 2)

    def test_workspace_truncation_keeps_complete_text_and_original_draft(self):
        environment = support.UserEnvironment()
        self.addCleanup(environment.close)
        draft = FACT + '\n\nНезавершённая фраз'
        provider = ReviewProvider(draft=draft, finish_reason='length')
        with patch('settings.ANSWER_REVIEW_MODE', 'contextual'), \
             patch('model_runtime.get_llm_provider', return_value=provider):
            result = asyncio.run(environment.execute('Что известно о письме?', _workspace_matches=[{
                'path': 'Письмо.md', 'content': SOURCE['text'], 'score': 1}]))
        self.assertNotIn('Незавершённая', result.data['final'])
        self.assertIn(FACT, result.data['final'])
        self.assertEqual(result.data['draft'], draft)
        self.assertTrue(result.data['verification'].review_details['generation_truncated'])

    def test_impossible_context_is_unavailable_and_does_not_drop_the_answer(self):
        provider = ReviewProvider(counter=lambda m: 10000)
        answer, check = self.review(FACT, provider)
        self.assertEqual(answer, FACT)
        self.assertEqual(check.status, 'UNAVAILABLE')
        self.assertFalse(provider.calls)

    def test_cancellation_propagates(self):
        with self.assertRaises(GenerationCancelled):
            self.review(FACT, ReviewProvider([GenerationCancelled()]))

    def test_knowledge_routes_use_the_same_contextual_review(self):
        provider = ReviewProvider()
        with patch('settings.ANSWER_REVIEW_MODE', 'contextual'), \
             patch('model_runtime.get_llm_provider', return_value=provider):
            answer, check = verify_knowledge_answer('Что известно?', FACT, [SOURCE], 'RAG', 'FAST')
        self.assertEqual(answer, FACT)
        self.assertEqual(check.review_mode, 'CONTEXTUAL')
        self.assertEqual(len(provider.calls), 1)

    def test_workspace_overview_generates_full_draft_and_never_runs_fact_gate(self):
        environment = support.UserEnvironment()
        self.addCleanup(environment.close)
        draft = FACT + '\n\n' + UNKNOWN
        provider = ReviewProvider([lambda p: payload(p, {UNKNOWN: ('MISSING', None, False)})], draft=draft)
        with patch('settings.ANSWER_REVIEW_MODE', 'contextual'), \
             patch('model_runtime.get_llm_provider', return_value=provider), \
             patch('overview_synthesis.OverviewSynthesizer.compose', side_effect=AssertionError('Legacy fact gate')):
            result = asyncio.run(environment.execute('Что известно о хранителе?', _workspace_matches=[{
                'path': 'Письмо.md', 'content': SOURCE['text'], 'score': 1}]))
        self.assertEqual(result.data['final'], draft)
        self.assertEqual(result.data['verification'].review_mode, 'CONTEXTUAL')
        self.assertEqual(len(provider.generation_calls), 1)
        self.assertEqual(len(provider.calls), 1)
        with closing(environment.connection()) as db:
            actions = [r[0] for r in db.execute('select action_type from assistant_actions')]
        self.assertNotIn('SYNTHESIZE_OVERVIEW', actions)
        self.assertNotIn('VERIFY_DOCUMENTARY_OVERVIEW', actions)
        self.assertIn('FINALIZE_RESPONSE', actions)

    def test_contextual_overview_focuses_writer_but_keeps_all_review_sources(self):
        environment=support.UserEnvironment()
        self.addCleanup(environment.close)
        provider=ReviewProvider(draft=FACT)
        with patch('settings.ANSWER_REVIEW_MODE','contextual'), \
                patch('model_runtime.get_llm_provider',return_value=provider):
            asyncio.run(environment.execute('Кто такой Мирон?',_workspace_matches=[
                {'path':'Письмо.md','content':'В письме Мирон называет себя хранителем карт.','score':5},
                {'path':'Другое.md','content':'Совет читателю: воспринимать любую боль как прогресс.','score':1}]))
        writer=provider.generation_calls[0][0][-1]['content']
        self.assertIn('Мирон',writer)
        self.assertNotIn('любую боль',writer)
        review_packet=json.loads(provider.calls[0]['messages'][-1]['content'])
        self.assertEqual(len(review_packet['sources']),2)

    def test_source_followup_does_not_replay_old_assistant_assertions_to_writer(self):
        environment=support.UserEnvironment()
        self.addCleanup(environment.close)
        provider=ReviewProvider(draft=FACT)
        with patch('settings.ANSWER_REVIEW_MODE','contextual'), \
                patch('model_runtime.get_llm_provider',return_value=provider):
            asyncio.run(environment.execute('Чего он хранитель?',_workspace_matches=[
                {'path':'Письмо.md','content':SOURCE['text'],'score':5}],_depends_on_history=True,
                _retrieval_question='Чего Мирон хранитель?',_chat_history=[
                    {'role':'user','content':'Кто такой Мирон?'},
                    {'role':'assistant','content':'OLD_UNVERIFIED_ASSERTION: владелец Луны.'}]))
        messages=provider.generation_calls[0][0]
        self.assertNotIn('OLD_UNVERIFIED_ASSERTION',' '.join(m['content'] for m in messages))
        self.assertTrue(any(m['content']=='Кто такой Мирон?' for m in messages))

    def test_authored_description_does_not_use_suggested_affirmation_as_identity(self):
        environment=support.UserEnvironment()
        self.addCleanup(environment.close)
        example='Нужно проговаривать пример: «Я Мирон, я властелин Луны».'
        provider=ReviewProvider(draft=FACT)
        with patch('settings.ANSWER_REVIEW_MODE','contextual'), \
                patch('model_runtime.get_llm_provider',return_value=provider):
            asyncio.run(environment.execute('Кто такой Мирон? Ответь кратко.',_document_authoring=True,
                _authoring_request='Создай документ о Мироне',_workspace_matches=[
                    {'path':'Письмо.md','content':'В письме Мирон называет себя хранителем карт.','score':5},
                    {'path':'Приём.md','content':example,'score':1}]))
        writer=provider.generation_calls[0][0][-1]['content']
        self.assertNotIn('властелин Луны',writer)
        packet=json.loads(provider.calls[0]['messages'][-1]['content'])
        self.assertEqual(len(packet['sources']),2)

    def test_person_overview_does_not_inherit_biography_from_suggested_self_example(self):
        environment=support.UserEnvironment()
        self.addCleanup(environment.close)
        provider=ReviewProvider(draft=FACT)
        with patch('settings.ANSWER_REVIEW_MODE','contextual'), \
                patch('model_runtime.get_llm_provider',return_value=provider):
            asyncio.run(environment.execute('Кто такой Мирон?',_workspace_matches=[
                {'path':'Письмо.md','content':'В письме Мирон называет себя хранителем карт.','score':5},
                {'path':'Приём.md','content':'Нужно проговаривать: «Я Мирон, во мне одна ярость».','score':1}]))
        self.assertNotIn('одна ярость',provider.generation_calls[0][0][-1]['content'])
        self.assertEqual(len(json.loads(provider.calls[0]['messages'][-1]['content'])['sources']),2)

    def test_explicit_question_about_self_example_keeps_its_original_context(self):
        environment=support.UserEnvironment()
        self.addCleanup(environment.close)
        provider=ReviewProvider(draft=FACT)
        with patch('settings.ANSWER_REVIEW_MODE','contextual'), \
                patch('model_runtime.get_llm_provider',return_value=provider):
            asyncio.run(environment.execute('Кто такой Мирон в примере самовнушения?',_workspace_matches=[
                {'path':'Приём.md','content':'Нужно проговаривать: «Я Мирон, во мне одна ярость».','score':1}]))
        self.assertIn('Нужно проговаривать',provider.generation_calls[0][0][-1]['content'])

    def test_workspace_timeout_preserves_the_explanation_and_journals_actual_answer(self):
        environment = support.UserEnvironment()
        self.addCleanup(environment.close)
        draft = FACT + '\n\n' + GUESS
        provider = ReviewProvider([GenerationDeadlineExceeded('deadline')], draft=draft)
        with patch('settings.ANSWER_REVIEW_MODE', 'contextual'), \
             patch('model_runtime.get_llm_provider', return_value=provider):
            result = asyncio.run(environment.execute('Что известно о письме?', _workspace_matches=[{
                'path': 'Письмо.md', 'content': SOURCE['text'], 'score': 1}]))
        self.assertEqual(result.data['final'], draft)
        self.assertEqual(result.data['verification'].checked_answer, draft)
        self.assertEqual(result.data['verification'].status, 'UNAVAILABLE')
        self.assertEqual(len(provider.calls), 1)


if __name__ == '__main__':
    unittest.main()
