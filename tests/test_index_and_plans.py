"""Workspace index freshness, graph retries and untrusted planner arguments."""
import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from support import UserEnvironment
from test_sources_and_verification import MemoryStore
from workspace_vector_index import WorkspaceVectorIndex
from agent_planner import AgentPlan, AgentPlanStep
from assistant_skills import AgentExecutor
from assistant_models import AssistantError
from pydantic import ValidationError


class WorkspaceIndexTests(unittest.TestCase):
    def setUp(self):
        self.env = UserEnvironment()
        self.addCleanup(self.env.close)
        self.store = MemoryStore()
        self.calls = []
        self.graph_ok = False
        graph = SimpleNamespace(delete_missing=lambda *args: 0)

        def extract(chunks, key, path):
            self.calls.extend((chunk.metadata['chunk_id'], chunk.page_content) for chunk in chunks)
            return self.graph_ok

        self.index = WorkspaceVectorIndex(self.env.filesystem, lambda: self.store, graph,
            lambda _: SimpleNamespace(extract_document=extract, last_status='FAILED'))

    def test_graph_retry_uses_real_committed_chunk_ids(self):
        first = self.index.sync(self.env.workspace.id)
        self.assertGreater(first.graph_failed, 0)
        ids = set(self.store.rows)
        self.assertTrue(all(':r:' in key for key in ids))
        self.calls.clear()
        self.graph_ok = True
        second = self.index.sync(self.env.workspace.id)
        self.assertEqual(second.indexed, 0)
        self.assertGreater(second.graph_indexed, 0)
        self.assertTrue(all(key in self.store.rows for key, _ in self.calls))
        self.assertEqual(ids, set(self.store.rows))

    def test_semantic_map_indexing_includes_subfolders_without_llm_relations(self):
        nested = self.env.root / 'Вложенная папка'
        nested.mkdir()
        (nested / 'Заметка.md').write_text('Запись во вложенной папке.', encoding='utf-8')
        first = self.index.sync(self.env.workspace.id, extract_relations=False)
        self.assertGreater(first.indexed, 0)
        self.assertEqual(first.graph_failed, 0)
        self.assertEqual(self.calls, [])
        self.assertTrue(any(doc.metadata.get('relative_path') == 'Вложенная папка/Заметка.md'
                            for doc in self.store.rows.values()))
        before = set(self.store.rows)
        second = self.index.sync(self.env.workspace.id, extract_relations=False)
        self.assertEqual(second.indexed, 0)
        self.assertGreater(second.unchanged, 0)
        self.assertEqual(self.calls, [])
        self.assertEqual(before, set(self.store.rows))

    def test_index_status_excludes_stale_vectors_and_other_workspaces(self):
        self.index.sync(self.env.workspace.id, extract_relations=False)
        status = self.index.status(self.env.workspace.id)
        self.assertEqual(status['indexed_documents'], status['readable_documents'])
        (self.env.root / 'Без названия 4.md').write_text('Изменённая заметка.', encoding='utf-8')
        self.env.filesystem.sync_index(self.env.workspace.id)
        updated = self.index.status(self.env.workspace.id)
        self.assertEqual(updated['indexed_documents'], status['indexed_documents'] - 1)
        other = self.env.root.parent / 'Другая папка'
        other.mkdir()
        (other / 'Документ.txt').write_text('Отдельный документ.', encoding='utf-8')
        workspace = self.env.repository.select(str(other))
        self.env.filesystem.sync_index(workspace.id)
        self.assertEqual(self.index.status(workspace.id)['indexed_documents'], 0)
        self.assertEqual(self.index.status(workspace.id)['readable_documents'], 1)

    def test_unchanged_files_are_not_reembedded_or_reextracted(self):
        self.graph_ok = True
        self.index.sync(self.env.workspace.id)
        self.calls.clear()
        result = self.index.sync(self.env.workspace.id)
        self.assertEqual(result.indexed, 0)
        self.assertEqual(result.graph_indexed, 0)
        self.assertGreater(result.unchanged, 0)
        self.assertEqual(self.calls, [])

    def test_updated_file_replaces_old_chunks_and_graph_evidence(self):
        self.graph_ok = True
        self.index.sync(self.env.workspace.id)
        (self.env.root / 'Без названия 4.md').write_text('UPDATED_ONLY_985', encoding='utf-8')
        self.calls.clear()
        result = self.index.sync(self.env.workspace.id)
        self.assertEqual(result.indexed, 1)
        self.assertEqual(result.graph_indexed, 1)
        self.assertTrue(any(text == 'UPDATED_ONLY_985' for _, text in self.calls))
        self.assertFalse(any('NOTE_FOUR' in doc.page_content for doc in self.store.rows.values()))

    def test_deleted_file_cannot_be_returned_from_vectors(self):
        self.graph_ok = True
        self.index.sync(self.env.workspace.id)
        (self.env.root / 'Без названия 4.md').unlink()
        result = self.index.sync(self.env.workspace.id)
        self.assertGreater(result.deleted_chunks, 0)
        self.assertFalse(any(doc.metadata.get('relative_path') == 'Без названия 4.md'
                             for doc in self.store.rows.values()))

    def test_semantic_search_passes_workspace_filter_to_store(self):
        received = []
        store = SimpleNamespace(similarity_search=lambda question, **kw: received.append(kw) or [])
        index = WorkspaceVectorIndex(self.env.filesystem, lambda: store)
        self.assertEqual(index.search(self.env.workspace.id, 'хранитель'), [])
        self.assertEqual(received[0]['filter'], {'workspace_id': self.env.workspace.id})

    def test_embedding_outage_is_reported_and_does_not_mark_files_indexed(self):
        self.store.fail_add = True
        result = self.index.sync(self.env.workspace.id)
        self.assertGreater(result.failed, 0)
        self.assertEqual(self.store.rows, {})
        self.assertTrue(all(f.vector_status != 'INDEXED' for f in
                            self.env.repository.indexed_files(self.env.workspace.id) if f.readable and f.size))


class PlanValidationTests(unittest.TestCase):
    def test_duplicate_step_ids_are_rejected(self):
        with self.assertRaises(ValidationError):
            AgentPlan(objective='test', steps=[AgentPlanStep(id='same', skill='workspace'),
                                              AgentPlanStep(id='same', skill='workspace')])

    def test_forward_dependency_and_cycle_are_rejected(self):
        with self.assertRaises(ValidationError):
            AgentPlan(objective='test', steps=[AgentPlanStep(id='first', skill='workspace', depends_on=['last']),
                                              AgentPlanStep(id='last', skill='workspace', depends_on=['first'])])

    def test_unknown_dependency_is_rejected(self):
        with self.assertRaises(ValidationError):
            AgentPlan(objective='test', steps=[AgentPlanStep(id='first', skill='workspace', depends_on=['unknown'])])

    def test_reference_requires_declared_dependency_including_nested_values(self):
        for value in ('${first.data.path}', {'nested': ['${first.data.path}']}):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                AgentPlan(objective='test', steps=[AgentPlanStep(id='first', skill='workspace'),
                    AgentPlanStep(id='second', skill='workspace', arguments={'value': value})])

    def test_missing_result_field_is_explicit_error(self):
        with self.assertRaises(AssistantError):
            AgentExecutor._resolve('${first.data.path}', {'first': SimpleNamespace(data={})})

    def test_model_arguments_cannot_override_workspace_or_hidden_context(self):
        env = UserEnvironment()
        self.addCleanup(env.close)
        plan = AgentPlan(objective='Read', steps=[AgentPlanStep(id='read', skill='workspace',
            arguments={'question': 'Прочитай Без названия 4.md', 'workspace_id': 'other',
                       'session_id': 'other', '_workspace_matches': [{'content': 'INJECTED'}]})])
        session = env.assistant.ensure_session()
        from assistant_core import ExecutionContext
        task = env.assistant.create_task(session.id, 'Прочитай Без названия 4.md')
        result = asyncio.run(env.core.agent_executor.execute(plan, ExecutionContext(env.assistant, task),
            {'workspace_id': env.workspace.id, 'session_id': session.id, 'model': 'FAST'}))
        self.assertIn('NOTE_FOUR', result.data['final'])
        self.assertNotIn('INJECTED', result.data['final'])

    def test_typed_negated_open_has_no_open_step(self):
        from workspace_planner import WorkspacePlanner
        plan = WorkspacePlanner().plan('Найди последнюю версию отчёта и прочитай, но не открывай её')
        self.assertEqual([step.operation.value for step in plan.steps], ['FIND_LATEST_FILE', 'READ_FILE'])
