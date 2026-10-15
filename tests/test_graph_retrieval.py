"""Quality, fallback and provenance checks for query-directed graph retrieval."""
from contextlib import closing
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from support import UserEnvironment
from test_semantic_graph import EmbeddedStore
from graph_retrieval import graph_search, query_index
from hybrid_context_extractor import HybridRetriever, SearchResult
from index_transactions import invalidate_index
from semantic_graph import SemanticDocumentGraph
from workspace_vector_index import WorkspaceVectorIndex


class GraphRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.env = UserEnvironment()
        self.addCleanup(self.env.close)
        self.store = EmbeddedStore()
        with closing(self.env.connection()) as conn, conn:
            conn.execute('CREATE TABLE document_store(id INTEGER PRIMARY KEY,filename TEXT)')
            conn.executemany('INSERT INTO document_store VALUES(?,?)', [(i, f'{i}.txt') for i in range(1, 5)])
        self.store.rows = {
            'a': ('Проект Сапфир использует устройство Термометр-17.', {'document_id': 1}, [1, 0, 0, 0]),
            'b': ('Устройство Термометр-17 выпускает предприятие Северный прибор.', {'document_id': 2}, [0, 1, 0, 0]),
            'c': ('Предприятие Северный прибор располагается в городе Тверь.', {'document_id': 3}, [0, 0, 1, 0]),
            'd': ('Предприятие Западный прибор располагается в городе Омск.', {'document_id': 4}, [0, 0, 0, 1]),
        }
        self.graph = SemanticDocumentGraph(self.store, connection_factory=self.env.connection)

    def search(self, question, limit=3):
        return graph_search(self.store, question, limit, connection_factory=self.env.connection)

    def test_exact_reference_path_finds_two_hops_without_semantic_similarity(self):
        self.assertEqual(self.graph.map()['edges'], [])
        result = self.search('Город производителя устройства проекта Сапфир?')
        self.assertEqual([r.chunk_id for r in result], ['a', 'b', 'c'])
        self.assertEqual(result[-1].metadata['graph_path'], ['a', 'b', 'c'])
        self.assertTrue(all(r.text == self.store.rows[r.chunk_id][0] for r in result))

    def test_fast_path_skips_embedding_and_unrelated_sources(self):
        base = SimpleNamespace(invoke=lambda q: self.fail('An embedding request must be skipped'))
        retriever = HybridRetriever(base, self.store, 'hybrid', needs_history_aware_input=False)
        def scoped(store, question, limit):
            return self.search(question, limit)
        with patch('graph_retrieval.graph_search', side_effect=scoped):
            result = retriever.search('Кто выпускает устройство проекта Сапфир?')
        self.assertEqual([r.chunk_id for r in result], ['a', 'b'])
        self.assertTrue(retriever.retrieval_info['embedding_skipped'])

    def test_missing_detail_does_not_claim_sufficient_context(self):
        self.assertIsNone(self.search('Телефон производителя устройства проекта Сапфир?'))
        self.assertIsNone(self.search('Город производителя устройства проекта Неизвестный?'))
        self.assertIsNone(self.search('Код проекта Сапфир для версии 2?'))
        self.assertIsNone(self.search('Кто не выпускает устройство проекта Сапфир?'))
        self.assertIsNone(self.search('Код проекта Сапфир для параметра λ?'))

    def test_similar_name_prefixes_do_not_join_different_objects(self):
        self.store.rows['a'] = ('For project AtlasOne the access code is 1234.', {'document_id': 1}, [1, 0, 0, 0])
        self.store.rows['b'] = ('For project AtlasTwo the access code is 9876.', {'document_id': 2}, [0, 1, 0, 0])
        result = self.search('Access code of project AtlasOne?')
        self.assertEqual([r.chunk_id for r in result], ['a'])
        self.assertIsNone(self.search('Access code of project AtlasThree?'))
        self.store.rows['a'] = ('Project Sapphire uses AtlasOne. The access code is 1234.', {'document_id': 1}, [1, 0, 0, 0])
        invalidate_index()
        self.assertIsNone(self.search('Access code of AtlasThree for project Sapphire?'))

    def test_broad_questions_keep_ordinary_retrieval(self):
        for question in ('Расскажи о проекте Сапфир', 'Сравни проекты Сапфир и Рубин', 'Кто такой Сапфир?'):
            self.assertIsNone(self.search(question))
        self.assertIsNone(self.search('What is Сапфир?'))

    def test_conflicting_direct_passages_are_both_retained(self):
        self.store.rows['a'] = ('Для проекта Сапфир код доступа равен 1234.', {'document_id': 1}, [1, 0, 0, 0])
        self.store.rows['b'] = ('Для проекта Сапфир код доступа равен 9876.', {'document_id': 2}, [0, 1, 0, 0])
        result = self.search('Код доступа проекта Сапфир?')
        self.assertEqual({r.chunk_id for r in result}, {'a', 'b'})
        self.assertIsNone(self.search('Код доступа проекта Сапфир?', limit=1))

    def test_requested_year_does_not_pull_other_year(self):
        self.store.rows['a'] = ('Для проекта Сапфир в 2026 году код доступа равен 1234.', {'document_id': 1}, [1, 0, 0, 0])
        self.store.rows['b'] = ('Для проекта Сапфир в 2025 году код доступа равен 9876.', {'document_id': 2}, [0, 1, 0, 0])
        result = self.search('Код доступа проекта Сапфир в 2026 году?')
        self.assertEqual([r.chunk_id for r in result], ['a'])

    def test_metadata_or_text_change_after_cache_is_rejected(self):
        index = query_index(self.graph)
        text, _, vec = self.store.rows['b']
        self.store.rows['b'] = (text, {'document_id': 4}, vec)
        self.assertIsNone(index.search('Кто выпускает устройство проекта Сапфир?', 3))
        self.store.rows['b'] = ('Изменённый текст.', {'document_id': 2}, vec)
        self.assertIsNone(index.search('Кто выпускает устройство проекта Сапфир?', 3))

    def test_other_namespace_cannot_complete_a_path(self):
        text, meta, vec = self.store.rows['b']
        self.store.rows['b'] = (text, {**meta, 'workspace_id': 'other'}, vec)
        self.assertIsNone(self.search('Кто выпускает устройство проекта Сапфир?'))

    def test_cache_invalidation_tracks_vector_revision_and_document_deletion(self):
        self.assertIsNotNone(self.search('Кто выпускает устройство проекта Сапфир?'))
        with closing(self.env.connection()) as conn, conn:
            conn.execute('DELETE FROM document_store WHERE id=2')
        self.assertIsNone(self.search('Кто выпускает устройство проекта Сапфир?'))
        invalidate_index()
        self.assertIsNone(self.search('Кто выпускает устройство проекта Сапфир?'))

    def test_expansion_selects_a_missing_detail_not_an_extra_similar_fact(self):
        text, meta, _ = self.store.rows['a']
        seed = SearchResult(1, 'a', text, meta)
        self.assertEqual(self.graph.expand([seed], question='Проект Сапфир'), [])
        result = self.graph.expand([seed], question='Кто выпускает устройство проекта Сапфир?')
        self.assertEqual([r.chunk_id for r in result], ['b'])

    def test_workspace_lexical_anchor_uses_matching_chunk_instead_of_first(self):
        wid = self.env.workspace.id
        for filename, content in [('Anchor.txt', 'Приготовление еды.\n\nПроект Сапфир использует устройство Термометр-17.'),
                                  ('Device.txt', 'Устройство Термометр-17 выпускает предприятие Северный прибор.')]:
            (self.env.root / filename).write_text(content, encoding='utf-8')
        self.env.filesystem.sync_index(wid)
        records = {r.relative_path: r for r in self.env.repository.indexed_files(wid)}
        for cid, path, text in [('first', 'Anchor.txt', 'Приготовление еды.'),
                                 ('actual', 'Anchor.txt', 'Проект Сапфир использует устройство Термометр-17.'),
                                 ('target', 'Device.txt', 'Устройство Термометр-17 выпускает предприятие Северный прибор.')]:
            record = records[path]
            self.env.repository.update_vector_state(wid, path, content_hash=record.content_hash, status='INDEXED')
            self.store.rows[cid] = (text, {'workspace_id': wid, 'relative_path': path,
                'content_hash': record.content_hash, 'workspace_file_key': WorkspaceVectorIndex.file_key(wid, path)}, [1, 0, 0, 0])
        index = WorkspaceVectorIndex(self.env.filesystem, lambda: self.store)
        result = index.expand_context(wid, [{'path': 'Anchor.txt', 'content': 'Проект Сапфир использует устройство Термометр-17.'}],
                                      question='Кто выпускает устройство проекта Сапфир?')
        self.assertEqual([r['chunk_id'] for r in result], ['target'])
        (self.env.root / 'Anchor.txt').write_text('Изменился якорь.', encoding='utf-8')
        self.assertEqual(index.expand_context(wid, [{'path': 'Anchor.txt', 'content': 'Сапфир'}],
                                              question='Кто выпускает устройство проекта Сапфир?'), [])

    def test_workspace_fixed_budget_preserves_direct_excerpts(self):
        for filename in ('aaa.txt', 'bbb.txt', 'ccc.txt'):
            (self.env.root / filename).write_text('Гидросфера регулирует поток воды. ' * 100, encoding='utf-8')
        question = 'Как гидросфера регулирует поток воды?'
        index = SimpleNamespace(graph_context=lambda *args: None,
                                expand_context=lambda *args, **kwargs: [{'path': 'extra.txt', 'content': 'Соседний текст.' * 20}])
        self.env.skill.vector_indexer = index
        results = self.env.skill.relevant_context(question, self.env.workspace.id, max_characters=900)
        self.assertEqual(len(results), 3)
        self.assertEqual([len(r['content']) for r in results], [300, 300, 300])

    def test_failed_graph_retrieval_still_invokes_normal_search(self):
        from langchain_core.documents import Document
        invoked = []
        def normal(query):
            invoked.append(query)
            return [Document(id='a', page_content=self.store.rows['a'][0], metadata={'document_id': 1})]
        retriever = HybridRetriever(SimpleNamespace(invoke=normal), self.store, 'hybrid', needs_history_aware_input=False)
        with patch('graph_retrieval.graph_search', return_value=None):
            results = retriever.search('Телефон проекта Сапфир?')
        self.assertEqual(invoked, ['Телефон проекта Сапфир?'])
        self.assertTrue(results)
        self.assertFalse(retriever.retrieval_info['embedding_skipped'])
