"""Recoverable mutations of the local vector stores.

New revisions are written before old chunks are removed. SQLite records the
commit decision, so startup can finish cleanup or discard an interrupted write.
The lock also gives in-process readers a consistent view during a replacement.
"""
from contextlib import closing
from functools import wraps
import json
import logging
import threading
import uuid

from db_utils import get_db_connection

index_lock = threading.RLock()
mutation_lock = threading.RLock()
_revision = 0


class IndexRecoveryError(RuntimeError):
    """Cleanup could not finish; the operation journal must retain the source."""


def index_revision():
    return _revision


def invalidate_index():
    global _revision
    _revision += 1


def consistent_read(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with index_lock:
            return function(*args, **kwargs)
    return wrapped


def serialized_mutation(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with mutation_lock:
            return function(*args, **kwargs)
    return wrapped


class IndexTransactions:
    def __init__(self, connection_factory=get_db_connection):
        self.connection_factory = connection_factory
        with closing(connection_factory()) as conn, conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS index_operations (
                id TEXT PRIMARY KEY, namespace TEXT NOT NULL, kind TEXT NOT NULL,
                state TEXT NOT NULL, payload TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)''')

    def begin(self, namespace, kind, payload):
        operation_id = str(uuid.uuid4())
        with closing(self.connection_factory()) as conn, conn:
            conn.execute('INSERT INTO index_operations(id,namespace,kind,state,payload) VALUES(?,?,?,?,?)',
                         (operation_id, namespace, kind, 'PREPARED', json.dumps(payload)))
        return operation_id

    def finish(self, operation_id):
        with closing(self.connection_factory()) as conn, conn:
            conn.execute('DELETE FROM index_operations WHERE id=?', (operation_id,))

    def replace(self, store, namespace, where, chunks, ids, commit=None, batch_size=50):
        with index_lock:
            old = store.get(where=where, include=['metadatas'])
            old_ids = list(old.get('ids') or [])
            revision = uuid.uuid4().hex
            new_ids = [f'{value}:r:{revision}' for value in ids]
            for chunk, chunk_id in zip(chunks, new_ids):
                chunk.metadata['chunk_id'] = chunk_id
            operation_id = self.begin(namespace, 'replace', {'old_ids': old_ids, 'new_ids': new_ids})
            committed = False
            try:
                for offset in range(0, len(chunks), batch_size):
                    store.add_documents(chunks[offset:offset + batch_size], ids=new_ids[offset:offset + batch_size])
                with closing(self.connection_factory()) as conn, conn:
                    if commit:
                        commit(conn)
                    conn.execute("UPDATE index_operations SET state='COMMITTED' WHERE id=?", (operation_id,))
                committed = True
                if old_ids:
                    store.delete(ids=old_ids)
                self.finish(operation_id)
            except BaseException as exc:
                if not committed:
                    # Delete only this attempt's revision; the old index stays.
                    try:
                        store.delete(ids=new_ids)
                        self.finish(operation_id)
                    except Exception as cleanup:
                        raise IndexRecoveryError('Index rollback awaits recovery; keep the original source') from cleanup
                    raise
                # A committed cleanup failure is recovered on the next startup
                # and before any subsequent read/mutation of this collection.
                logging.exception('Committed index revision awaits old chunk cleanup')
            finally:
                invalidate_index()
            return new_ids

    @staticmethod
    def snapshot(store, where):
        data = store.get(where=where, include=['documents', 'metadatas', 'embeddings'])
        snapshot = {}
        for field in ('ids', 'documents', 'metadatas', 'embeddings'):
            value = data.get(field)
            if hasattr(value, 'tolist'):
                value = value.tolist()
            snapshot[field] = value if value is not None else []
        return snapshot

    @staticmethod
    def restore(store, snapshot):
        if snapshot['ids']:
            # Restore stored vectors without depending on an embedding server.
            store._collection.upsert(**snapshot)

    def recover(self, stores):
        with index_lock:
            with closing(self.connection_factory()) as conn:
                rows = conn.execute('SELECT * FROM index_operations ORDER BY rowid').fetchall()
            for row in rows:
                store = stores.get(row['namespace'])
                if store is None:
                    continue
                payload = json.loads(row['payload'])
                if row['kind'] == 'replace':
                    ids = payload['old_ids'] if row['state'] == 'COMMITTED' else payload['new_ids']
                    if ids:
                        store.delete(ids=ids)
                elif row['kind'] == 'collection_switch':
                    if row['state'] != 'COMMITTED':
                        for family, name in payload['new_names'].items():
                            if not name.startswith(family + '_r_'):
                                raise ValueError('Invalid staging collection name')
                            existing = {item.name if hasattr(item, 'name') else item for item in store._client.list_collections()}
                            if name in existing:
                                store._client.delete_collection(name=name)
                elif row['state'] != 'COMMITTED':
                    self.restore(store, payload['snapshot'])
                self.finish(row['id'])
                invalidate_index()
