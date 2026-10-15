"""Document lifecycle shared by HTTP uploads, Drive import and reindexing."""
from contextlib import closing
import json
from pathlib import Path
import tempfile

from db_utils import get_db_connection, get_document_source
from index_transactions import IndexTransactions, index_lock, mutation_lock

document_lock = mutation_lock


class DocumentService:
    def __init__(self, index, insert, delete_record, store_getter, delete_vectors,
                 connection_factory=get_db_connection, source_getter=get_document_source):
        self.index, self.insert, self.delete_record = index, insert, delete_record
        self.store_getter, self.delete_vectors = store_getter, delete_vectors
        self.connection_factory = connection_factory
        self.source_getter = source_getter

    def upload(self, filename, source, kind, input_data, source_id=None):
        with document_lock:
            file_id = None
            if source_id:
                with closing(self.connection_factory()) as conn:
                    row = conn.execute('SELECT id FROM document_sources WHERE source_id=?', (source_id,)).fetchone()
                    file_id = row['id'] if row else None
            created = file_id is None
            file_id = file_id or self.insert(filename, source, kind)
            source_committed = False

            def commit_source(conn):
                nonlocal source_committed
                previous = conn.execute(
                    'SELECT filename,source_content,source_kind FROM document_store WHERE id=?',
                    (file_id,),
                ).fetchone()
                if previous is None:
                    raise LookupError('Document disappeared during indexing')
                changed = not created and (previous['filename'] != filename or
                    previous['source_content'] != source or previous['source_kind'] != kind)
                conn.execute('UPDATE document_store SET filename=?,source_content=?,source_kind=? WHERE id=?',
                             (filename, source, kind, file_id))
                if source_id:
                    conn.execute('INSERT INTO document_sources(source_id,id) VALUES(?,?) '
                                 'ON CONFLICT(source_id) DO UPDATE SET id=excluded.id', (source_id, file_id))
                if changed:
                    # Answers based on the previous source must not remain
                    # hidden model context after a successful replacement.
                    conn.execute("INSERT INTO application_state(key,value) VALUES('knowledge_epoch','1') "
                                 "ON CONFLICT(key) DO UPDATE SET value=CAST(value AS INTEGER)+1")
                source_committed = True

            success = self.index(input_data, file_id, commit=commit_source, source_name=filename)
            if not success:
                if created:
                    self.delete_record(file_id)
                raise ValueError('Document is empty or could not be indexed')
            # Compatibility for injected index adapters that report success
            # without using the transactional callback. The production writer
            # commits source metadata with its index journal in one transaction.
            if not source_committed:
                with closing(self.connection_factory()) as conn, conn:
                    commit_source(conn)
            return file_id

    def reindex(self, file_id):
        with document_lock:
            source = self.source_getter(file_id)
            if not source:
                raise LookupError('Document not found')
            if not source.get('source_content'):
                raise ValueError('Original source is unavailable; upload the document again before reindexing')
            if source.get('source_kind') == 'documents_json':
                from langchain_core.documents import Document
                records = json.loads(bytes(source['source_content']).decode('utf-8'))
                data = [Document(page_content=item['page_content'], metadata=item.get('metadata') or {})
                        for item in records]
                success = self.index(data, file_id)
            else:
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / ('reindex' + Path(source['filename']).suffix.lower())
                    path.write_bytes(bytes(source['source_content']))
                    success = self.index(str(path), file_id)
            if not success:
                raise RuntimeError('Reindex failed; previous index and original source were preserved')

    def delete(self, file_id):
        from graph_repository import GraphRepository
        with document_lock, index_lock:
            if not self.source_getter(file_id):
                raise LookupError('Document not found')
            store = self.store_getter()
            if store is None:
                raise RuntimeError('Vector store unavailable')
            transactions = IndexTransactions(self.connection_factory)
            transactions.recover({'rag_collection': store})
            graph = GraphRepository(self.connection_factory)
            snapshot = transactions.snapshot(store, {'file_id': file_id})
            operation = transactions.begin('rag_collection', 'delete', {'snapshot': snapshot})
            try:
                if not self.delete_vectors(file_id):
                    raise RuntimeError('Vector cleanup failed')
                with closing(self.connection_factory()) as conn, conn:
                    conn.execute('BEGIN IMMEDIATE')
                    graph.delete_document(file_id, connection=conn)
                    conn.execute('DELETE FROM document_sources WHERE id=?', (file_id,))
                    conn.execute('DELETE FROM document_store WHERE id=?', (file_id,))
                    conn.execute("INSERT INTO application_state(key,value) VALUES('knowledge_epoch','1') "
                                 "ON CONFLICT(key) DO UPDATE SET value=CAST(value AS INTEGER)+1")
                    conn.execute("UPDATE index_operations SET state='COMMITTED' WHERE id=?", (operation,))
            except BaseException:
                transactions.restore(store, snapshot)
                transactions.finish(operation)
                raise
            transactions.finish(operation)


def create_document_schema():
    with closing(get_db_connection()) as conn, conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS document_sources (
            source_id TEXT PRIMARY KEY, id INTEGER NOT NULL REFERENCES document_store(id) ON DELETE CASCADE)''')
        # Backfill remote identities from sources preserved by older versions.
        for row in conn.execute("SELECT id,source_content FROM document_store WHERE source_kind='documents_json'"):
            try:
                records = json.loads(bytes(row['source_content']).decode('utf-8'))
                remote_id = records[0].get('metadata', {}).get('gdrive_id') if records else None
                if remote_id:
                    conn.execute('INSERT OR IGNORE INTO document_sources VALUES(?,?)',
                                 ('gdrive:' + str(remote_id), row['id']))
            except (ValueError, TypeError, IndexError, AttributeError):
                continue
