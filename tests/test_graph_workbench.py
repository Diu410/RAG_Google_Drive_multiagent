"""Graph user scenarios with isolated SQLite, files and controlled inference."""
import asyncio
import sqlite3
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from support import UserEnvironment
from assistant_models import AssistantError
from graph_repository import GraphRepository
from graph_workbench import GraphWorkbench
from knowledge_skill import KnowledgeServices, KnowledgeSkill
from pydantic_models import GraphSelection, QueryInput
from query_router import QueryStrategy, route_by_rules
from workspace_graph import WorkspaceGraphRepository
from workspace_vector_index import WorkspaceVectorIndex


class Store:
    def __init__(self):
        self.chunks = {}

    def get(self, *, ids, include):
        keys = [key for key in ids if key in self.chunks]
        return {'ids': keys, 'documents': [self.chunks[key][0] for key in keys],
                'metadatas': [self.chunks[key][1] for key in keys]}


class GraphWorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.env = UserEnvironment()
        self.addCleanup(self.env.close)
        self.store = Store()
        with closing(self.env.connection()) as conn, conn:
            conn.execute('CREATE TABLE document_store (id INTEGER PRIMARY KEY,filename TEXT)')
            conn.execute("INSERT INTO document_store VALUES(1,'Команда.txt'),(2,'Технологии.txt')")
        self.library = GraphRepository(self.env.connection)
        self.anna = self.library.upsert_entity('Анна', 'PERSON')
        self.project = self.library.upsert_entity('Альфа', 'PROJECT')
        self.tech = self.library.upsert_entity('Python', 'TECHNOLOGY')
        self.other = self.library.upsert_entity('Бета', 'PROJECT')
        self.edge(self.anna, 'WORKS_ON', self.project, 1, 'team', 'Анна работает над Альфой.')
        self.edge(self.project, 'USES', self.tech, 2, 'tech', 'Альфа использует Python.')
        self.workbench = GraphWorkbench(connection_factory=self.env.connection, store=self.store)

    def edge(self, source, relation, target, doc, chunk, quote):
        self.library.add_relation(source, relation, target, doc, chunk, evidence=quote)
        self.store.chunks[chunk] = (quote, {'document_id': doc, 'source': 'fixture.txt'})

    def workspace_graph(self):
        path = Path(self.env.workspace.root_path) / 'Связи.txt'
        path.write_text('Анна работает над Альфой.', encoding='utf-8')
        fs = self.env.filesystem
        fs.sync_index(self.env.workspace.id)
        file_hash = fs._hash_file(path)
        self.env.repository.update_vector_state(self.env.workspace.id, path.name, content_hash=file_hash, status='INDEXED')
        self.env.repository.update_graph_state(self.env.workspace.id, path.name, content_hash=file_hash, status='INDEXED')
        repo = WorkspaceGraphRepository(self.env.connection)
        key = WorkspaceVectorIndex.file_key(self.env.workspace.id, path.name)
        repo.replace_workspace_file_graph(self.env.workspace.id, key,
            [{'name': 'Анна', 'type': 'PERSON'}, {'name': 'Альфа', 'type': 'PROJECT'}],
            [{'source': 'Анна', 'target': 'Альфа', 'relation': 'WORKS_ON', 'chunk_id': 'workspace-team',
              'evidence': 'Анна работает над Альфой.'}], path.name)
        self.store.chunks['workspace-team'] = ('Анна работает над Альфой.', {
            'workspace_id': self.env.workspace.id, 'workspace_file_key': key, 'relative_path': path.name,
            'content_hash': file_hash, 'source_kind': 'workspace'})
        return GraphWorkbench(self.env.workspace.id, connection_factory=self.env.connection,
                              filesystem=fs, store=self.store), path

    def test_overview_is_immediately_useful_and_includes_current_quotes(self):
        result = self.workbench.explore()
        self.assertEqual(len(result['edges']), 2)
        self.assertEqual({node['label'] for node in result['nodes']}, {'Анна', 'Альфа', 'Python'})
        self.assertEqual(result['scope']['workspace_id'], None)
        self.assertEqual(result['coverage']['indexed_documents'], 2)
        self.assertEqual({edge['provenance']['document_name'] for edge in result['edges']},
                         {'Команда.txt', 'Технологии.txt'})
        self.assertTrue(all(edge['provenance']['evidence'] in edge['provenance']['text'] for edge in result['edges']))

    def test_path_crosses_documents_and_preserves_direction_when_walked_backwards(self):
        result = self.workbench.connect(self.tech, self.anna)
        self.assertTrue(result['path_found'])
        self.assertEqual([edge['relation'] for edge in result['edges']], ['USES', 'WORKS_ON'])
        self.assertEqual(result['edges'][0]['source'], f'entity:{self.project}')
        self.assertFalse(self.workbench.connect(self.anna, self.tech, max_hops=1)['path_found'])

    def test_shared_document_mention_does_not_invent_semantic_connection(self):
        document = self.library.upsert_entity('Команда.txt', 'DOCUMENT')
        self.edge(document, 'MENTIONS', self.anna, 1, 'mention-a', 'Анна')
        self.edge(document, 'MENTIONS', self.other, 1, 'mention-b', 'Бета')
        self.assertFalse(self.workbench.connect(self.anna, self.other)['path_found'])

    def test_dangling_or_wrong_quote_cannot_be_used_for_a_path_or_chat(self):
        self.store.chunks['tech'] = ('Здесь нет сведений о технологии.', {'document_id': 2})
        self.assertFalse(self.workbench.connect(self.anna, self.tech)['path_found'])
        with self.assertRaises(AssistantError) as error:
            self.workbench.selection([self.relation_id('tech')])
        self.assertEqual(error.exception.status_code, 409)

    def relation_id(self, chunk):
        with closing(self.env.connection()) as conn:
            return conn.execute('SELECT id FROM knowledge_relations WHERE chunk_id=?', (chunk,)).fetchone()[0]

    def test_selection_uses_chunk_text_not_relation_interpretation(self):
        with closing(self.env.connection()) as conn, conn:
            conn.execute("UPDATE knowledge_relations SET relation_type='OWNS' WHERE chunk_id='team'")
        focus = self.workbench.selection([self.relation_id('team')])
        self.assertEqual(focus['sources'][0].text, 'Анна работает над Альфой.')
        self.assertNotIn('OWNS', focus['sources'][0].text)

    def test_wrong_document_namespace_is_rejected_even_with_matching_quote(self):
        self.store.chunks['team'][1]['document_id'] = 999
        with self.assertRaises(AssistantError):
            self.workbench.selection([self.relation_id('team')])

    def test_workspace_and_library_ids_and_sources_never_mix(self):
        workspace, _ = self.workspace_graph()
        result = workspace.explore()
        self.assertEqual(len(result['edges']), 1)
        self.assertEqual(result['scope']['workspace_id'], self.env.workspace.id)
        self.assertEqual(result['edges'][0]['provenance']['chunk_id'], 'workspace-team')
        self.assertNotIn('Python', [node['label'] for node in result['nodes']])
        self.assertEqual(len(self.workbench.explore()['edges']), 2)
        with self.assertRaises(AssistantError):
            workspace.connect(9999, 10000)

    def test_file_change_after_map_load_rejects_old_selection(self):
        workspace, path = self.workspace_graph()
        edge_id = int(workspace.explore()['edges'][0]['id'].split(':')[1])
        path.write_text('Анна больше не работает над проектом.', encoding='utf-8')
        with self.assertRaises(AssistantError) as error:
            workspace.selection([edge_id])
        self.assertEqual(error.exception.status_code, 409)
        refreshed = GraphWorkbench(self.env.workspace.id, connection_factory=self.env.connection,
                                  filesystem=self.env.filesystem, store=self.store)
        self.assertEqual(refreshed.explore()['edges'], [])
        self.assertEqual(refreshed.coverage()['indexed_documents'], 0)

    def test_workspace_chunk_from_another_folder_is_rejected(self):
        workspace, _ = self.workspace_graph()
        self.store.chunks['workspace-team'][1]['workspace_id'] = 'another-folder'
        self.assertEqual(workspace.explore()['edges'], [])

    def test_graph_question_uses_selected_sources_and_verifies_answer_without_rerouting(self):
        from langchain_core.runnables import RunnableLambda
        import langchain_utils
        generated_prompts = []
        def model(prompt):
            generated_prompts.append(str(prompt))
            return 'Анна работает над Альфой.'
        def result(document):
            return {'document_id': document.metadata['document_id'], 'chunk_id': document.metadata['chunk_id'],
                'text': document.page_content, 'metadata': document.metadata, 'score': 1,
                'found_by': ['graph_selection']}
        verifier = Mock(side_effect=lambda q, answer, *args: (answer, None))
        route = Mock(side_effect=AssertionError('Graph selection must bypass router'))
        services = KnowledgeServices(route=route, history=lambda *_: [], general_chat=None, multi_chain=None,
            rag_chain=langchain_utils.get_rag_chain, document_result=result, verify=verifier)
        self.env.core.skills._skills['knowledge'] = KnowledgeSkill(services)
        self.env.core.agent_planner = Mock(side_effect=AssertionError('No graph workflow planning'))
        with patch('graph_workbench.graph_workbench', return_value=self.workbench), \
             patch.object(langchain_utils, 'create_chat_model', return_value=RunnableLambda(model)), \
             patch.object(langchain_utils, 'current_retriever', None), \
             patch.object(langchain_utils, 'vectorstore', None), \
             patch('graph_rag.GraphService', side_effect=AssertionError('No second graph search')):
            response, _ = asyncio.run(self.env.core.process_input(QueryInput(question='Как связаны Анна и Альфа?',
                model='FAST', graph_selection=GraphSelection(relation_ids=[self.relation_id('team')]))))
        self.assertEqual(response.strategy, 'GRAPH_QUERY')
        self.assertEqual([source.chunk_id for source in response.results], ['team'])
        self.assertNotIn('Python', generated_prompts[0])
        self.assertEqual(verifier.call_count, 1)
        self.assertEqual(response.graph['retrieval']['selected_source_count'], 1)
        self.assertTrue(any(action.action_type == 'SELECT_GRAPH_SOURCES' for action in response.actions))

    def test_selection_scope_must_match_request(self):
        request = QueryInput(question='Объясни связь', workspace_id=self.env.workspace.id,
                             graph_selection=GraphSelection(relation_ids=[1]))
        with self.assertRaises(AssistantError) as error:
            asyncio.run(self.env.core.process_input(request))
        self.assertEqual(error.exception.status_code, 422)
        self.assertEqual(self.env.knowledge.calls, [])

    def test_workspace_graph_chat_keeps_scope_and_ignores_old_file_choice(self):
        from langchain_core.runnables import RunnableLambda
        from hybrid_context_extractor import _result
        import langchain_utils
        workspace, _ = self.workspace_graph()
        edge_id = int(workspace.explore()['edges'][0]['id'].split(':')[1])
        route = Mock(side_effect=AssertionError('No source rerouting'))
        verify = Mock(side_effect=lambda q, answer, *args: (answer, None))
        services = KnowledgeServices(route=route, history=None, general_chat=None, multi_chain=None,
            rag_chain=langchain_utils.get_rag_chain,
            document_result=lambda document: _result(document.page_content, document.metadata).as_dict(), verify=verify)
        self.env.core.skills._skills['knowledge'] = KnowledgeSkill(services)
        self.env.core.history_loader = Mock(side_effect=AssertionError('No history for a focused question'))
        session = self.env.assistant.ensure_session(None)
        self.env.assistant.save_pending_plan(session.id, {'kind': 'workspace_file_choice',
            'workspace_id': self.env.workspace.id, 'choices': ['old.txt'], 'original_question': 'Открой файл'})
        with patch('graph_workbench.graph_workbench', return_value=workspace), \
             patch.object(langchain_utils, 'create_chat_model', return_value=RunnableLambda(lambda _: 'Анна работает над Альфой.')), \
             patch.object(langchain_utils, 'current_retriever', None), \
             patch.object(langchain_utils, 'vectorstore', None):
            response, _ = asyncio.run(self.env.core.process_input(QueryInput(
                question='Как связаны Анна и Альфа?', session_id=session.id, model='FAST',
                workspace_id=self.env.workspace.id,
                graph_selection=GraphSelection(workspace_id=self.env.workspace.id, relation_ids=[edge_id]))))
        self.assertEqual([source.chunk_id for source in response.results], ['workspace-team'])
        self.assertEqual(response.results[0].metadata['workspace_id'], self.env.workspace.id)
        self.assertEqual(response.graph['scope']['workspace_id'], self.env.workspace.id)
        self.assertEqual(verify.call_count, 1)
        route.assert_not_called()
        self.assertEqual(self.env.opened, [])
        self.assertIsNone(self.env.assistant.pending_plan(session.id))

    def test_api_limits_and_workspace_scope_are_explicit(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        import graph_routes
        app = FastAPI()
        app.include_router(graph_routes.router)
        with TestClient(app) as client, patch.object(graph_routes, '_workbench', return_value=self.workbench) as factory:
            self.assertEqual(client.get('/graph/explore').status_code, 200)
            self.assertEqual(client.get('/graph/connect', params={
                'source_id': self.anna, 'target_id': self.tech, 'workspace_id': 'chosen-folder'}).status_code, 200)
            factory.assert_called_with('chosen-folder')
            for path in ('/graph/explore?depth=3', '/graph/explore?entity_ids=-1',
                         '/graph/explore?entity_ids=9223372036854775808',
                         '/graph/connect?source_id=1&target_id=2&max_hops=7',
                         '/graph/explore?' + '&'.join('entity_ids=1' for _ in range(21))):
                self.assertEqual(client.get(path).status_code, 422)

    def test_natural_relation_question_uses_graph_route(self):
        self.assertEqual(route_by_rules('Как связаны Анна и Альфа?').strategy, QueryStrategy.GRAPH_QUERY)

    def test_first_open_before_any_graph_was_extracted_returns_empty_map(self):
        from graph_workbench import graph_workbench
        database = self.env.directory / 'fresh-graph.db'
        def connection():
            conn = sqlite3.connect(database)
            conn.row_factory = sqlite3.Row
            return conn
        with closing(connection()) as conn, conn:
            conn.execute('CREATE TABLE document_store (id INTEGER PRIMARY KEY, filename TEXT)')
        with patch('chroma_utils.vectorstore', None), patch('chroma_utils.workspace_vectorstore', None):
            graph = graph_workbench(connection_factory=connection)
        self.assertEqual(graph.explore()['nodes'], [])
        self.assertEqual(graph.entities(), [])


if __name__ == '__main__':
    unittest.main()
