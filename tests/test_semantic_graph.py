"""Automatic graph behavior with controlled embeddings and isolated user data."""
from contextlib import closing
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from support import UserEnvironment
from langchain_core.documents import Document
from hybrid_context_extractor import HybridRetriever, SearchResult
from index_transactions import invalidate_index
from semantic_graph import SemanticDocumentGraph, automatic_graph_sources
from workspace_vector_index import WorkspaceVectorIndex


class EmbeddedStore:
    def __init__(self):
        self.rows = {}
        self.calls = []

    def get(self, **kwargs):
        self.calls.append(kwargs)
        keys = kwargs.get('ids', list(self.rows))
        keys = [key for key in keys if key in self.rows and all(
            self.rows[key][1].get(field) == value for field, value in kwargs.get('where', {}).items())]
        if 'ids' not in kwargs:
            start = kwargs.get('offset', 0)
            keys = keys[start:start + kwargs.get('limit', len(keys))]
        result = {'ids': keys}
        if 'documents' in kwargs['include']:
            result['documents'] = [self.rows[key][0] for key in keys]
        if 'metadatas' in kwargs['include']:
            result['metadatas'] = [self.rows[key][1] for key in keys]
        if 'embeddings' in kwargs['include']:
            result['embeddings'] = np.asarray([self.rows[key][2] for key in keys])
        return result


class SemanticGraphTests(unittest.TestCase):
    def setUp(self):
        self.env = UserEnvironment()
        self.addCleanup(self.env.close)
        self.store = EmbeddedStore()
        with closing(self.env.connection()) as conn, conn:
            conn.execute('CREATE TABLE document_store(id INTEGER PRIMARY KEY, filename TEXT)')
            conn.execute("INSERT INTO document_store VALUES(1,'Резерв.txt'),(2,'Архив.md'),(3,'Рецепт.txt'),(4,'Пустой.txt')")
        self.store.rows = {
            'a': ('Копия хранится отдельно.', {'document_id': 1}, [1, 0, 0]),
            'b': ('Восстановление выполняется из архива.', {'document_id': 2}, [.98, .12, 0]),
            'c': ('Смешайте муку и воду.', {'document_id': 3}, [0, 0, 1]),
        }
        self.graph = SemanticDocumentGraph(self.store, connection_factory=self.env.connection)

    def seed(self, key='a'):
        text, metadata, _ = self.store.rows[key]
        return SearchResult(metadata.get('document_id'), key, text, metadata)

    def test_links_depend_on_embeddings_even_when_names_and_words_differ(self):
        result = self.graph.map()
        self.assertEqual({(row['source'], row['target']) for row in result['edges']}, {('1', '2')})
        self.assertEqual(len(result['nodes']), 4)
        self.assertEqual(result['coverage']['indexed_documents'], 3)
        self.assertFalse(next(row for row in result['nodes'] if row['id'] == '4')['indexed'])

    def test_link_explanation_reads_closest_original_passages_in_requested_order(self):
        self.store.rows['a2'] = ('Сохраните вторую копию.', {'document_id': 1}, [.8, .6, 0])
        self.store.rows['b2'] = ('Вторая копия позволяет восстановить архив.', {'document_id': 2}, [.8, .6, 0])
        result = self.graph.explain_link('2', '1')
        self.assertEqual([row['document_id'] for row in result['passages']], ['2', '1'])
        self.assertEqual([row['text'] for row in result['passages']],
                         [self.store.rows['b2'][0], self.store.rows['a2'][0]])
        self.assertAlmostEqual(result['passage_similarity'], 1, places=5)
        self.assertFalse(result['sampled'])
        self.assertEqual(self.store.calls[-1]['include'], ['documents', 'metadatas'])
        self.assertEqual(len(self.store.calls[-1]['ids']), 2)

    def test_link_explanation_rejects_missing_edge_or_changed_source_owner(self):
        with self.assertRaises(LookupError):
            self.graph.explain_link('1', '3')
        self.graph.snapshot()
        self.store.rows['b'] = ('Другой workspace.', {'document_id': 2, 'workspace_id': 'other'}, [.98, .12, 0])
        with self.assertRaises(LookupError):
            self.graph.explain_link('1', '2')

    def test_link_explanation_limits_large_documents_and_reports_sampling(self):
        for index in range(65):
            self.store.rows[f'long-{index}'] = ('Фрагмент архива ' + str(index), {'document_id': 2}, [.98, .12, 0])
        result = self.graph.explain_link('1', '2')
        self.assertTrue(result['sampled'])
        self.assertEqual(len(result['passages']), 2)
        self.assertEqual(len(self.store.calls[-1]['ids']), 2)

    def test_semantic_retrieval_keeps_storage_id_for_legacy_metadata(self):
        base = SimpleNamespace(invoke=lambda query: [Document(
            id='a', page_content=self.store.rows['a'][0], metadata={'document_id': 1})])
        retriever = HybridRetriever(base, vectorstore=self.store, mode='semantic',
                                    needs_history_aware_input=False)
        def expand(store, sources, **kwargs):
            return automatic_graph_sources(store, sources, connection_factory=self.env.connection, **kwargs)
        with patch('semantic_graph.automatic_graph_sources', side_effect=expand):
            result = retriever.search('Откуда восстановить данные?')
        self.assertEqual([item.chunk_id for item in result], ['a', 'b'])

    def test_http_map_routes_scope_and_bounds(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        import graph_routes
        self.workspace_graph()
        real_graph = SemanticDocumentGraph
        def scoped(store, workspace_id=None, **kwargs):
            return real_graph(store, workspace_id, connection_factory=self.env.connection, **kwargs)
        app = FastAPI()
        app.include_router(graph_routes.router)
        with patch('semantic_graph.SemanticDocumentGraph', side_effect=scoped), \
             patch('chroma_utils.vectorstore', self.store), \
             patch('chroma_utils.workspace_vectorstore', self.store), \
             patch('workspace.WorkspaceRepository', return_value=self.env.repository), \
             patch('workspace.FilesystemProvider', return_value=self.env.filesystem), TestClient(app) as client:
            library = client.get('/graph/map').json()
            folder = client.get('/graph/map', params={'workspace_id': self.env.workspace.id}).json()
            self.assertEqual({node['id'] for node in library['nodes']}, {'1', '2', '3', '4'})
            self.assertEqual(len(folder['edges']), 1)
            self.assertNotIn('Резерв.txt', {node['label'] for node in folder['nodes']})
            link = client.get('/graph/link', params={'source_id': '1', 'target_id': '2'})
            self.assertEqual(link.status_code, 200)
            self.assertEqual(link.json()['passages'][1]['text'], self.store.rows['b'][0])
            self.assertEqual(client.get('/graph/link', params={
                'source_id': '1', 'target_id': '2', 'workspace_id': self.env.workspace.id}).status_code, 409)
            self.assertEqual(client.get('/graph/link', params={'source_id': '', 'target_id': '2'}).status_code, 422)
            for params in ({'limit': 0}, {'limit': 501}, {'workspace_id': ''}):
                self.assertEqual(client.get('/graph/map', params=params).status_code, 422)

    def test_graph_does_not_add_similar_text_when_question_is_already_covered(self):
        retriever = HybridRetriever(None, vectorstore=self.store, mode='lexical')
        # Redirect storage while exercising the production retrieval path.
        def expand(store, sources, **kwargs):
            return automatic_graph_sources(store, sources, connection_factory=self.env.connection, **kwargs)
        with patch('semantic_graph.automatic_graph_sources', side_effect=expand):
            result = retriever.search('Копия')
        self.assertEqual([item.chunk_id for item in result], ['a'])

    def test_only_one_neighbor_and_no_duplicate_direct_sources(self):
        with closing(self.env.connection()) as conn, conn:
            conn.execute("INSERT INTO document_store VALUES(5,'Дополнение.txt')")
        self.store.rows['d'] = ('Другой близкий текст.', {'document_id': 5}, [.97, .1, 0])
        added = self.graph.expand([self.seed()])
        self.assertEqual(len(added), 1)
        self.assertEqual(self.graph.expand([self.seed(), self.seed('b'), self.seed('d')]), [])

    def test_map_cache_and_inventory_deletion(self):
        self.graph.map()
        calls = len(self.store.calls)
        self.graph.map()
        self.assertEqual(len(self.store.calls), calls)
        with closing(self.env.connection()) as conn, conn:
            conn.execute('DELETE FROM document_store WHERE id=2')
        self.assertEqual(self.graph.map()['edges'], [])
        self.assertEqual(self.graph.expand([self.seed()]), [])

    def test_vector_commit_invalidates_cached_links(self):
        self.assertTrue(self.graph.map()['edges'])
        self.store.rows['b'] = ('Другой предмет.', {'document_id': 2}, [0, 1, 0])
        invalidate_index()
        self.assertEqual(self.graph.map()['edges'], [])

    def test_cross_namespace_chunk_cannot_form_a_library_link(self):
        self.store.rows['b'] = ('Чужая папка.', {'document_id': 2, 'workspace_id': 'other'}, [1, 0, 0])
        self.assertEqual(self.graph.map()['edges'], [])

    def test_neighbor_requires_close_chunk_as_well_as_close_document(self):
        self.store.rows['a'] = ('Узкая тема.', {'document_id': 1}, [1, 0, 0])
        self.store.rows['a2'] = ('Другая тема.', {'document_id': 1}, [0, 1, 0])
        self.store.rows['b'] = ('Общий обзор.', {'document_id': 2}, [.7, .7, 0])
        self.assertTrue(self.graph.map()['edges'])
        self.assertEqual(self.graph.expand([self.seed()]), [])

    def test_fresh_metadata_cannot_change_neighbor_source_identity(self):
        self.graph.snapshot()
        self.store.rows['b'] = ('Текст другой папки.', {'document_id': 2, 'source_kind': 'workspace'}, [.98, .12, 0])
        self.assertEqual(self.graph.expand([self.seed()]), [])

    def test_missing_embeddings_keep_ordinary_hits_available(self):
        class Unavailable:
            def get(self, **kwargs):
                raise RuntimeError('Embedding store unavailable')
        self.assertEqual(automatic_graph_sources(Unavailable(), [self.seed()],
                         connection_factory=self.env.connection), [])

    def workspace_graph(self):
        workspace_id = self.env.workspace.id
        for name, vector in [('Защита.txt', [1, 0, 0]), ('Хранение.md', [.98, .12, 0])]:
            path = self.env.root / name
            path.write_text('Служебный текст про копии: ' + name, encoding='utf-8')
        self.env.filesystem.sync_index(workspace_id)
        for record in self.env.repository.indexed_files(workspace_id):
            if record.relative_path not in {'Защита.txt', 'Хранение.md'}:
                continue
            self.env.repository.update_vector_state(workspace_id, record.relative_path,
                content_hash=record.content_hash, status='INDEXED')
            meta = {'workspace_id': workspace_id, 'relative_path': record.relative_path,
                    'content_hash': record.content_hash,
                    'workspace_file_key': WorkspaceVectorIndex.file_key(workspace_id, record.relative_path)}
            vector = [1, 0, 0] if record.relative_path == 'Защита.txt' else [.98, .12, 0]
            self.store.rows[record.relative_path] = ((self.env.root / record.relative_path).read_text(encoding='utf-8'), meta, vector)
        return SemanticDocumentGraph(self.store, workspace_id, connection_factory=self.env.connection,
                                     filesystem=self.env.filesystem)

    def test_workspace_graph_works_without_successful_llm_relation_extraction(self):
        graph = self.workspace_graph()
        self.assertEqual(len(graph.map()['edges']), 1)
        index = WorkspaceVectorIndex(self.env.filesystem, lambda: self.store)
        added = index.expand_context(self.env.workspace.id,
                                    [{'path': 'Защита.txt', 'content': 'Текст', 'matched': []}])
        self.assertEqual([row['path'] for row in added], ['Хранение.md'])
        self.assertEqual(added[0]['found_by'], ['semantic_graph'])

    def test_modified_file_is_not_used_even_before_inventory_rescan(self):
        graph = self.workspace_graph()
        graph.snapshot()
        (self.env.root / 'Хранение.md').write_text('Изменилось.', encoding='utf-8')
        self.assertEqual(graph.expand([self.seed('Защита.txt')]), [])
        with self.assertRaises(LookupError):
            graph.explain_link('Защита.txt', 'Хранение.md')
        self.env.filesystem.sync_index(self.env.workspace.id)
        self.assertEqual(graph.map()['edges'], [])

    def test_another_workspace_cannot_supply_a_neighbor(self):
        graph = self.workspace_graph()
        _, meta, _ = self.store.rows['Хранение.md']
        self.store.rows['Хранение.md'] = ('Чужое.', {**meta, 'workspace_id': 'foreign'}, [1, 0, 0])
        self.assertEqual(graph.map()['edges'], [])

    def test_pagination_reads_embeddings_without_document_payloads(self):
        self.store.rows = {f'part-{i}': ('Текст', {'document_id': 1}, [1, 0, 0]) for i in range(501)}
        self.graph.map()
        pages = [call for call in self.store.calls if 'embeddings' in call['include']]
        self.assertEqual([call['offset'] for call in pages], [0, 500])
        self.assertTrue(all('documents' not in call['include'] for call in pages))
