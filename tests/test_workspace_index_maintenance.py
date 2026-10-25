"""Index diagnostics, targeted retry and revision-safe commits on isolated files."""
from contextlib import closing
import os
import unittest
from unittest.mock import Mock, patch

from support import UserEnvironment, authorized_client
from test_sources_and_verification import MemoryStore
from workspace_vector_index import WorkspaceVectorIndex


class WorkspaceIndexMaintenanceTests(unittest.TestCase):
    def setUp(self):
        self.env = UserEnvironment()
        self.addCleanup(self.env.close)
        self.id = self.env.workspace.id
        self.store = MemoryStore()
        self.index = WorkspaceVectorIndex(self.env.filesystem, lambda: self.store)
        self.path = 'Без названия 4.md'

    def write(self, path, text):
        (self.env.root / path).write_text(text, encoding='utf-8')

    def sync(self, **kwargs):
        return self.index.sync(self.id, extract_relations=False, **kwargs)

    def scan(self):
        return self.env.filesystem.sync_index(self.id)

    def record(self, path):
        return next(r for r in self.env.repository.indexed_files(self.id) if r.relative_path == path)

    def rows(self, path):
        return {key: doc.page_content for key, doc in self.store.rows.items()
                if doc.metadata['relative_path'] == path}

    def test_status_distinguishes_pending_stale_failed_and_current(self):
        self.sync()
        self.write(self.path, 'Изменённый документ.')
        self.write('Новый.md', 'Новый документ.')
        self.write('Ошибка.md', 'Документ с ошибкой embeddings.')
        self.scan()
        self.env.repository.update_vector_state(self.id, 'Ошибка.md', content_hash=None, status='FAILED',
            error='PRIVATE_RUNTIME_DETAIL')
        status = self.index.status(self.id)
        self.assertEqual(status['changed_documents'], 1)
        self.assertEqual(status['pending_documents'], 1)
        self.assertEqual(status['failed_documents'], 1)
        self.assertEqual(status['readable_documents'], sum(status[key] for key in (
            'indexed_documents','changed_documents','pending_documents','failed_documents','unavailable_documents')))
        self.assertEqual({item['path']: item['state'] for item in status['issues']},
            {self.path:'STALE', 'Новый.md':'PENDING', 'Ошибка.md':'FAILED'})
        self.assertNotIn('PRIVATE_RUNTIME_DETAIL', str(status))

    def test_unstable_metadata_is_not_counted_as_indexed(self):
        self.sync()
        with closing(self.env.connection()) as conn, conn:
            conn.execute('UPDATE assistant_workspace_files SET content_hash=NULL WHERE workspace_id=? AND relative_path=?',
                         (self.id, self.path))
        status = self.index.status(self.id)
        self.assertEqual(status['unavailable_documents'], 1)
        self.assertIn({'path':self.path,'state':'UNAVAILABLE'}, status['issues'])

    def test_diagnostics_are_bounded_without_hiding_total_counts(self):
        for number in range(55):
            self.write(f'Document{number}.md', f'Content {number}')
        self.scan()
        status = self.index.status(self.id)
        self.assertGreater(status['pending_documents'], 50)
        self.assertEqual(len(status['issues']), 50)
        self.assertTrue(status['issues_truncated'])

    def test_update_only_reprocesses_changed_and_new_documents(self):
        self.sync()
        self.write(self.path, 'Новая редакция.')
        self.write('Новый.md', 'Новый документ.')
        with patch.object(self.store, 'add_documents', wraps=self.store.add_documents) as add:
            result = self.sync()
        self.assertEqual(result.indexed, 2)
        touched = {doc.metadata['relative_path'] for call in add.call_args_list for doc in call.args[0]}
        self.assertEqual(touched, {self.path,'Новый.md'})
        with patch.object(self.store, 'add_documents', wraps=self.store.add_documents) as add:
            self.assertEqual(self.sync().indexed, 0)
            add.assert_not_called()

    def test_retry_only_errors_preserves_other_pending_work_and_does_not_cleanup(self):
        self.sync()
        deleted_path = 'Без названия 9.md'
        self.assertTrue(self.rows(deleted_path))
        self.write(self.path, 'Редакция с неудачным прошлым запуском.')
        self.write('Новый.md', 'Ещё не индексировался.')
        self.write('Изменённый.md', 'Изменённый документ.')
        self.scan()
        self.env.repository.update_vector_state(self.id, self.path,
            content_hash=self.record(self.path).vector_hash, status='FAILED')
        (self.env.root / deleted_path).unlink()
        result = self.sync(mode='retry_failed')
        self.assertEqual(result.indexed, 1)
        self.assertGreater(result.skipped, 0)
        self.assertFalse(self.rows('Новый.md'))
        self.assertTrue(self.rows(deleted_path))
        self.assertEqual(result.deleted_chunks, 0)
        status = self.index.status(self.id)
        self.assertEqual(status['failed_documents'], 0)
        self.assertEqual(status['pending_documents'], 2)
        self.sync()
        self.assertFalse(self.rows(deleted_path))
        self.assertEqual(self.index.status(self.id)['pending_documents'], 0)

    def test_extraction_failure_is_reported_without_exposing_exception(self):
        original = self.env.filesystem.read_text_file
        def read(workspace_id, path):
            if path == self.path:
                raise ValueError('SECRET_ABSOLUTE_PATH_AND_CREDENTIALS')
            return original(workspace_id, path)
        with patch.object(self.env.filesystem, 'read_text_file', side_effect=read):
            result = self.sync()
        self.assertEqual(result.failed, 1)
        self.assertEqual(result.issues, [{'path':self.path,'code':'EXTRACTION_FAILED'}])
        self.assertNotIn('SECRET_ABSOLUTE', str(result))

    def test_embedding_failure_keeps_previous_revision_and_can_be_retried(self):
        self.sync()
        previous = self.rows(self.path)
        self.write(self.path, 'Новая редакция для векторного индекса.')
        self.store.fail_add = True
        result = self.sync()
        self.assertEqual(result.failed, 1)
        self.assertEqual(result.issues, [{'path':self.path,'code':'EMBEDDING_FAILED'}])
        self.assertEqual(self.rows(self.path), previous)
        self.assertEqual(self.index.status(self.id)['failed_documents'], 1)
        self.store.fail_add = False
        result = self.sync(mode='retry_failed')
        self.assertEqual(result.indexed, 1)
        self.assertEqual(self.index.status(self.id)['failed_documents'], 0)
        self.assertNotEqual(self.rows(self.path), previous)

    def test_change_while_reading_cannot_publish_mismatched_hash(self):
        original = self.env.filesystem.read_text_file
        def read(workspace_id, path):
            result = original(workspace_id, path)
            if path == self.path:
                self.write(path, 'Changed after extraction.')
            return result
        with patch.object(self.env.filesystem, 'read_text_file', side_effect=read):
            result = self.sync()
        self.assertEqual(result.failed, 1)
        self.assertEqual(result.issues, [{'path':self.path,'code':'SOURCE_CHANGED'}])
        self.assertFalse(self.rows(self.path))
        self.assertNotEqual(self.record(self.path).vector_status, 'INDEXED')

    def test_change_during_embedding_rolls_back_staging_and_preserves_old_chunks(self):
        self.sync()
        previous = self.rows(self.path)
        self.write(self.path, 'Second revision before embeddings.')
        add = self.store.add_documents
        def add_and_change(docs, ids):
            add(docs, ids)
            if any(doc.metadata['relative_path'] == self.path for doc in docs):
                path = self.env.root / self.path
                stat = path.stat()
                self.write(self.path, 'Third revision during embeddings.')
                os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        with patch.object(self.store, 'add_documents', side_effect=add_and_change):
            result = self.sync()
        self.assertEqual(result.failed, 1)
        self.assertEqual(result.issues, [{'path':self.path,'code':'SOURCE_CHANGED'}])
        self.assertEqual(self.rows(self.path), previous)
        with closing(self.env.connection()) as conn:
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM index_operations').fetchone()[0], 0)
        self.assertEqual(self.sync(mode='retry_failed').indexed, 1)
        self.assertEqual(set(self.rows(self.path).values()), {'Third revision during embeddings.'})

    def test_partial_scan_defers_cleanup_of_unobserved_vectors_and_graphs(self):
        self.sync()
        orphan = next(iter(self.store.rows.values())).model_copy(deep=True)
        orphan.metadata['workspace_file_key'] = 'not-observed-key'
        self.store.rows['orphan'] = orphan
        self.index.graph_repository = Mock()
        scan = self.scan()
        for partial in (scan.model_copy(update={'truncated':True}), scan.model_copy(update={'errors':['I/O failed']})):
            with self.subTest(partial=partial), patch.object(self.env.filesystem,'sync_index',return_value=partial):
                result = self.sync()
            self.assertTrue(result.cleanup_deferred)
            self.assertIn('orphan', self.store.rows)
        self.index.graph_repository.delete_missing.assert_not_called()
        self.index.graph_repository = None
        result = self.sync()
        self.assertFalse(result.cleanup_deferred)
        self.assertNotIn('orphan', self.store.rows)

    def test_unavailable_store_and_invalid_mode_do_not_modify_state(self):
        self.scan()
        self.index.vectorstore_getter = lambda: None
        self.assertTrue(self.sync().unavailable)
        self.assertEqual(self.index.status(self.id)['indexed_documents'], 0)
        with self.assertRaises(ValueError):
            self.sync(mode='invented-mode')


class WorkspaceIndexHttpTests(unittest.TestCase):
    def setUp(self):
        import main
        self.main = main
        self.env = UserEnvironment()
        self.addCleanup(self.env.close)
        self.store = MemoryStore()
        self.index = WorkspaceVectorIndex(self.env.filesystem, lambda: self.store)
        for name, value in [('filesystem_provider',self.env.filesystem),('workspace_vector_index',self.index)]:
            patcher = patch.object(main, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.client = authorized_client(main.app)
        self.addCleanup(self.client.close)
        self.base = f'/workspaces/{self.env.workspace.id}'

    def test_status_checks_disk_and_retry_mode_is_validated_and_authenticated(self):
        status = self.client.get(self.base+'/index-status')
        self.assertEqual(status.status_code, 200)
        self.assertGreater(status.json()['pending_documents'], 0)
        self.assertIn('checked_at',status.json())
        response = self.client.post(self.base+'/vector-sync?extract_relations=false&mode=retry_failed')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['indexed'], 0)
        self.assertGreater(response.json()['skipped'], 0)
        self.assertEqual(self.client.post(self.base+'/vector-sync?mode=invented').status_code, 422)
        from fastapi.testclient import TestClient
        client = TestClient(self.main.app, base_url='http://127.0.0.1')
        try:
            self.assertEqual(client.get(self.base+'/index-status').status_code, 401)
            self.assertEqual(client.post(self.base+'/vector-sync?mode=retry_failed').status_code, 401)
        finally:
            client.close()
