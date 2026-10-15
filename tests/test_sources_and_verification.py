"""Source replacement, failed writes, invented evidence and context limits."""
from contextlib import closing
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from support import UserEnvironment
from langchain_core.documents import Document
from index_transactions import IndexTransactions, IndexRecoveryError
from document_service import DocumentService
from verifier import Verifier, _quote_source_ids, calculate_confidence, LLMVerification, ClaimAssessment
from prompt_budget import fit_prompt
from llm_contract import ProviderError


class MemoryStore:
    """Failure injection at the persistence boundary, with real SQLite journal."""
    def __init__(self):
        self.rows = {}
        self.fail_add = False
        self.fail_delete = False
        self._collection = self

    def get(self, where=None, include=None):
        selected = [(key, doc) for key, doc in self.rows.items()
                    if all(doc.metadata.get(k) == v for k, v in (where or {}).items())]
        return {'ids': [key for key, _ in selected],
                'documents': [doc.page_content for _, doc in selected],
                'metadatas': [doc.metadata.copy() for _, doc in selected],
                'embeddings': [[1.0, 0.0] for _ in selected]}

    def add_documents(self, docs, ids):
        for key, doc in zip(ids, docs):
            self.rows[key] = doc.model_copy(deep=True)
        if self.fail_add:
            raise RuntimeError('embedding write failed after a partial batch')

    def delete(self, ids):
        if self.fail_delete:
            raise RuntimeError('disk cleanup failed')
        for key in ids:
            self.rows.pop(key, None)

    def upsert(self, ids, documents, metadatas, embeddings):
        for key, text, metadata in zip(ids, documents, metadatas):
            self.rows[key] = Document(page_content=text, metadata=metadata)


class SourceTransactionTests(unittest.TestCase):
    def setUp(self):
        self.env = UserEnvironment()
        self.addCleanup(self.env.close)
        self.store = MemoryStore()
        self.transactions = IndexTransactions(self.env.connection)
        self.old = Document(page_content='OLD_FACT', metadata={'file_id': 1})
        self.new = Document(page_content='NEW_FACT', metadata={'file_id': 1})
        self.store.rows['old'] = self.old

    def pending(self):
        with closing(self.env.connection()) as conn:
            return [dict(row) for row in conn.execute('SELECT * FROM index_operations')]

    def replace(self, commit=None):
        return self.transactions.replace(self.store, 'rag_collection', {'file_id': 1}, [self.new], ['new'], commit)

    def test_success_replaces_old_revision_and_finishes_journal(self):
        ids = self.replace()
        self.assertEqual(set(self.store.rows), set(ids))
        self.assertEqual(self.store.rows[ids[0]].page_content, 'NEW_FACT')
        self.assertEqual(self.store.rows[ids[0]].metadata['chunk_id'], ids[0])
        self.assertEqual(self.pending(), [])

    def test_embedding_failure_preserves_previous_answer_sources(self):
        self.store.fail_add = True
        with self.assertRaises(RuntimeError):
            self.replace()
        self.assertEqual(list(self.store.rows), ['old'])
        self.assertEqual(self.pending(), [])

    def test_database_commit_failure_discards_staged_vectors(self):
        def failing_commit(conn):
            conn.execute('CREATE TABLE should_rollback(value TEXT)')
            raise RuntimeError('database commit failed')
        with self.assertRaises(RuntimeError):
            self.replace(failing_commit)
        self.assertEqual(list(self.store.rows), ['old'])
        self.assertEqual(self.pending(), [])

    def test_committed_cleanup_failure_is_recovered_without_losing_new_fact(self):
        self.store.fail_delete = True
        with self.assertLogs(level='ERROR'):
            ids = self.replace()
        self.assertEqual(self.pending()[0]['state'], 'COMMITTED')
        self.store.fail_delete = False
        self.transactions.recover({'rag_collection': self.store})
        self.assertEqual(set(self.store.rows), set(ids))
        self.assertEqual(self.pending(), [])

    def test_failed_rollback_keeps_recoverable_journal(self):
        self.store.fail_add = self.store.fail_delete = True
        with self.assertRaises(IndexRecoveryError):
            self.replace()
        self.assertEqual(self.pending()[0]['state'], 'PREPARED')
        self.store.fail_add = self.store.fail_delete = False
        self.transactions.recover({'rag_collection': self.store})
        self.assertEqual(list(self.store.rows), ['old'])

    def test_recovery_does_not_touch_a_different_collection(self):
        self.transactions.begin('other_collection', 'replace', {'old_ids': ['old'], 'new_ids': ['new']})
        self.transactions.recover({'rag_collection': self.store})
        self.assertIn('old', self.store.rows)
        self.assertEqual(len(self.pending()), 1)

    def test_snapshot_restoration_needs_no_embedding_model(self):
        snapshot = self.transactions.snapshot(self.store, {'file_id': 1})
        self.store.rows.clear()
        self.transactions.restore(self.store, snapshot)
        self.assertEqual(self.store.rows['old'].page_content, 'OLD_FACT')


class DocumentLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.env = UserEnvironment()
        self.addCleanup(self.env.close)
        self.store = MemoryStore()
        with closing(self.env.connection()) as conn, conn:
            conn.executescript('''
                CREATE TABLE document_store(id INTEGER PRIMARY KEY, filename TEXT,
                    source_content BLOB, source_kind TEXT);
                CREATE TABLE document_sources(source_id TEXT PRIMARY KEY,id INTEGER REFERENCES document_store(id));
                CREATE TABLE application_state(key TEXT PRIMARY KEY,value TEXT);
            ''')
        self.index_success = True

        def insert(filename, source, kind):
            with closing(self.env.connection()) as conn, conn:
                return conn.execute('INSERT INTO document_store(filename,source_content,source_kind) VALUES(?,?,?)',
                                    (filename, source, kind)).lastrowid

        def delete(file_id):
            with closing(self.env.connection()) as conn, conn:
                conn.execute('DELETE FROM document_store WHERE id=?', (file_id,))

        def get(file_id):
            with closing(self.env.connection()) as conn:
                row = conn.execute('SELECT * FROM document_store WHERE id=?', (file_id,)).fetchone()
                return dict(row) if row else None

        def index(data, file_id, commit=None, source_name=None):
            if not self.index_success:
                return False
            IndexTransactions(self.env.connection).replace(self.store, 'rag_collection', {'file_id': file_id},
                [Document(page_content=str(data), metadata={'file_id': file_id, 'source': source_name})],
                [f'doc:{file_id}'], commit=commit)
            return True

        self.service = DocumentService(index, insert, delete, lambda: self.store,
                                       lambda _: True, self.env.connection, get)

    def test_drive_reimport_keeps_id_and_replaces_source_and_citations(self):
        first = self.service.upload('old.txt', b'OLD', 'file', 'OLD', 'drive:123')
        second = self.service.upload('new.txt', b'NEW', 'file', 'NEW', 'drive:123')
        self.assertEqual(first, second)
        self.assertEqual(self.service.source_getter(first)['source_content'], b'NEW')
        self.assertEqual(self.service.source_getter(first)['filename'], 'new.txt')
        self.assertEqual([d.metadata['source'] for d in self.store.rows.values()], ['new.txt'])
        with closing(self.env.connection()) as conn:
            self.assertEqual(conn.execute('SELECT value FROM application_state WHERE key=?',
                                         ('knowledge_epoch',)).fetchone()[0], '1')

    def test_failed_new_upload_does_not_create_phantom_document(self):
        self.index_success = False
        with self.assertRaises(ValueError):
            self.service.upload('empty.txt', b'', 'file', '', 'drive:empty')
        with closing(self.env.connection()) as conn:
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM document_store').fetchone()[0], 0)
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM document_sources').fetchone()[0], 0)

    def test_failed_reimport_preserves_original_bytes_and_vectors(self):
        file_id = self.service.upload('old.txt', b'OLD', 'file', 'OLD', 'drive:123')
        self.index_success = False
        with self.assertRaises(ValueError):
            self.service.upload('new.txt', b'NEW', 'file', 'NEW', 'drive:123')
        self.assertEqual(self.service.source_getter(file_id)['source_content'], b'OLD')
        self.assertTrue(all(d.page_content == 'OLD' for d in self.store.rows.values()))

    def test_reindex_missing_source_is_a_clear_error(self):
        with self.assertRaises(LookupError):
            self.service.reindex(100)

    def test_reindex_unavailable_original_preserves_existing_index(self):
        file_id = self.service.insert('legacy.txt', b'', 'file')
        with self.assertRaisesRegex(ValueError, 'Original source'):
            self.service.reindex(file_id)


class VerificationTests(unittest.TestCase):
    sources = [{'chunk_id': 'real:1', 'text': 'Хранителя северного архива зовут Мирон. Он хранит карты.',
                'document_id': 1, 'metadata': {}, 'score': 0.8, 'found_by': ['lexical']}]

    def verify(self, claims, status='SUPPORTED', strategy='WORKSPACE_QA'):
        payload = json.dumps({'status': status, 'claims': claims}, ensure_ascii=False)
        client = SimpleNamespace(structured_generate=lambda **_: SimpleNamespace(text=payload))
        # These tests isolate citation validation; entailment has its own staged
        # provider tests in test_evidence_audit.py and is enabled by default.
        return Verifier(client, audit_evidence=False).verify('Кто хранитель?', 'Мирон', self.sources, strategy, 'FAST')

    def claim(self, **changes):
        return dict(text='Хранителя зовут Мирон', status='SUPPORTED', source_chunk_ids=['real:1'],
                    basis='EXPLICIT', evidence_quote='Хранителя северного архива зовут Мирон.', **changes)

    def test_literal_evidence_is_supported(self):
        result = self.verify([self.claim()])
        self.assertEqual(result.status, 'SUPPORTED')
        self.assertEqual(result.source_chunk_ids, ['real:1'])

    def test_invented_quote_is_rejected_even_with_real_source_id(self):
        claim = self.claim()
        claim['evidence_quote'] = 'Хранитель Мирон работает банкиром.'
        self.assertEqual(self.verify([claim]).status, 'UNSUPPORTED')

    def test_filename_citation_is_recovered_only_from_literal_evidence(self):
        claim = self.claim()
        claim['source_chunk_ids'] = ['Хранитель.md']
        self.assertEqual(self.verify([claim]).source_chunk_ids, ['real:1'])

    def test_library_invented_reference_does_not_gain_support(self):
        claim = self.claim()
        claim['source_chunk_ids'] = ['invented:99']
        self.assertEqual(self.verify([claim], strategy='SIMPLE_FACT').status, 'UNSUPPORTED')

    def test_short_common_quote_is_not_proof(self):
        self.assertEqual(_quote_source_ids('Мирон', {'1': self.sources[0]['text']}), [])

    def test_ellipsis_must_preserve_every_quoted_span(self):
        self.assertEqual(_quote_source_ids('Хранителя северного архива…Он хранит карты.',
                                         {'1': self.sources[0]['text']}), ['1'])
        self.assertEqual(_quote_source_ids('Хранителя северного архива…Он работает банкиром.',
                                         {'1': self.sources[0]['text']}), [])

    def test_no_claims_cannot_be_declared_supported(self):
        self.assertEqual(self.verify([]).status, 'INSUFFICIENT_INFORMATION')

    def test_no_sources_cannot_be_declared_supported(self):
        client = SimpleNamespace(structured_generate=lambda **_: self.fail('Must not call model without sources'))
        result = Verifier(client).verify('Кто?', 'Мирон', [], 'WORKSPACE_QA', 'FAST')
        self.assertEqual(result.confidence, 0)

    def test_conflict_dominates_supported_claim(self):
        other = self.claim()
        other.update(text='Противоречие', status='CONFLICTING')
        self.assertEqual(self.verify([self.claim(), other]).status, 'CONFLICTING')

    def test_confidence_does_not_compare_incompatible_raw_scores(self):
        assessment = LLMVerification(status='SUPPORTED', claims=[ClaimAssessment(**self.claim())])
        a, _ = calculate_confidence(assessment, self.sources)
        b, _ = calculate_confidence(assessment, [{**self.sources[0], 'score': 999999}])
        self.assertEqual(a, b)
        self.assertTrue(0 <= a <= 1)


class PromptBudgetTests(unittest.TestCase):
    @staticmethod
    def count(messages):
        return sum(len(m['content']) for m in messages)

    def test_long_sources_are_trimmed_but_question_is_preserved(self):
        original = [{'role': 'system', 'content': 'RULE'},
                    {'role': 'user', 'content': 'QUESTION\nКонтекст:\n' + 'x' * 5000}]
        fitted, output = fit_prompt(original, 512, 128, self.count)
        self.assertTrue(fitted[-1]['content'].startswith('QUESTION\nКонтекст:\n'))
        self.assertLessEqual(self.count(fitted) + output + 32, 512)
        self.assertEqual(len(original[-1]['content']), 5019)

    def test_old_chat_is_discarded_before_current_source(self):
        fitted, _ = fit_prompt([{'role': 'system', 'content': 'RULE'},
            {'role': 'user', 'content': 'OLD' * 200}, {'role': 'assistant', 'content': 'STALE' * 200},
            {'role': 'user', 'content': 'CURRENT\nКонтекст:\nIMPORTANT'}], 512, 128, self.count)
        self.assertIn('IMPORTANT', fitted[-1]['content'])
        self.assertFalse(any('STALE' in m['content'] for m in fitted))

    def test_unfit_question_is_rejected_instead_of_silently_truncated(self):
        with self.assertRaises(ProviderError):
            fit_prompt([{'role': 'user', 'content': 'USER_QUESTION' * 500}], 512, 128, self.count)

    def test_multimodal_message_is_rejected(self):
        with self.assertRaises(ProviderError):
            fit_prompt([{'role': 'user', 'content': [{'image': 'x'}]}], 512, 128, self.count)
