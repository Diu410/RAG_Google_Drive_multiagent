"""Keep a short introduction focused while preserving the original evidence."""
import unittest

import support
from source_grounding import documentary_overview, focused_source_text, overview_topic_terms


class OverviewFocusTests(unittest.TestCase):
    def test_detailed_overview_preserves_the_whole_source(self):
        text = 'Мирон — смотритель.\nДругие важные сведения.'
        self.assertEqual(focused_source_text('Подробно расскажи о Мироне.', text), text)

    def test_documentary_recovery_explains_the_direct_mention_without_a_biography(self):
        text = ('Северный архив хранит карты. Это всё я — Хранитель Северного архива Мирон '
                'и мне подвластно всё. Я соберу урожай.')
        source = {'chunk_id': 'workspace:Письмо.md', 'text': text, 'metadata': {'source': 'Письмо.md'}}
        answer = documentary_overview('Что такое Северный архив?', [source])
        self.assertTrue(answer.startswith('Северный архив упоминается в самоназвании'))
        self.assertIn('«Хранитель Северного архива Мирон»', answer)
        self.assertIn('«Северный архив хранит карты».', answer)
        self.assertNotIn('урожай', answer)
        self.assertNotIn('подвластно', answer)
        identity = documentary_overview('Кто такой Мирон?', [source])
        self.assertTrue(identity.startswith('В тексте Мирон называет себя'))

    def test_negative_self_designation_and_denied_or_hypothetical_speech_are_not_recovered(self):
        for text in ('Я — не Хранитель архива Мирон.',
                     'Он не говорил: Это всё я — Хранитель архива Мирон.',
                     'Если бы он ответил: Это всё я — Хранитель архива Мирон.'):
            source = {'chunk_id': 'workspace:Заметка.md', 'text': text}
            self.assertIsNone(documentary_overview('Кто такой Мирон?', [source]))

    def test_a_concept_inside_the_title_is_not_identified_as_its_speaker(self):
        source = {'chunk_id': 'workspace:Письмо.md',
                  'text': 'Это всё я — Архиепископ Культа Терпения Мирон Северный.'}
        answer = documentary_overview('Кто такой Терпение?', [source])
        self.assertIsNotNone(answer)
        self.assertNotIn('Терпение называет себя', answer)
        self.assertIn('упоминается в самоназвании', answer)

    def test_documentary_recovery_refuses_a_self_affirmation_example_or_competing_identities(self):
        source = {'chunk_id': 'workspace:Заметка.md', 'text': 'Например: это всё я — Хранитель архива Мирон.'}
        self.assertIsNone(documentary_overview('Кто такой Мирон?', [source]))
        a = {**source, 'text': 'Это всё я — Хранитель архива Мирон.'}
        b = {**source, 'text': 'Это всё я — Посетитель архива Мирон.'}
        self.assertIsNone(documentary_overview('Кто такой Мирон?', [a, b]))
        self.assertIsNone(documentary_overview('Кто такой Мирон и что он умеет?', [a]))

    def test_generic_advice_is_not_selected_as_an_organization_description(self):
        text = ('Волевое действие совершается без альтернатив.\n'
                'Например: Я участник Культа Терпения и я повторяю эти слова.\n'
                'Учёба развивает ум.')
        focused = focused_source_text('Что за культ Терпения?', text)
        self.assertIn('Например:', focused)
        self.assertIn('Культа Терпения', focused)
        self.assertNotIn('Волевое действие', focused)
        self.assertNotIn('Учёба', focused)
        self.assertIn('Учёба', text)

    def test_an_unrelated_game_setting_does_not_describe_an_organization(self):
        self.assertEqual(focused_source_text('Что за культ Терпения?', 'Мир после войны. Это идея для игры.'), '')

    def test_pronouns_and_single_fact_questions_do_not_trigger_topic_filtering(self):
        for question in ('Кто такой этот?', 'Что ты знаешь о нём?', 'Какой телефон у Мирона?'):
            self.assertEqual(overview_topic_terms(question), set())
            self.assertEqual(focused_source_text(question, 'Исходный текст.'), 'Исходный текст.')

    def test_long_paragraph_keeps_the_topic_and_nearby_negation_context(self):
        unrelated = 'Совсем другая тема. ' * 80
        text = unrelated + 'Это условный пример. Мирон — смотритель архива. Это не биография автора. ' + unrelated
        focused = focused_source_text('Кто такой Мирон?', text)
        self.assertIn('Это условный пример.', focused)
        self.assertIn('Мирон — смотритель архива.', focused)
        self.assertIn('Это не биография автора.', focused)
        self.assertLess(len(focused), 200)
