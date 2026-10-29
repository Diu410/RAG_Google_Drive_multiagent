"""Local revision-bound provenance, not a trust score or a filename heuristic."""
import json
from contextlib import closing


class ArtifactOrigins:
    def __init__(self, connection_factory):
        self.connection_factory = connection_factory
        with closing(connection_factory()) as conn,conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS workspace_artifact_origins (
                workspace_id TEXT NOT NULL, path TEXT NOT NULL, revision TEXT NOT NULL,
                kind TEXT NOT NULL, task_id TEXT, parents TEXT NOT NULL,
                PRIMARY KEY(workspace_id,path))''')

    def save(self, workspace_id, path, revision, kind, parents=(), task_id=None):
        with closing(self.connection_factory()) as conn,conn:
            conn.execute('''INSERT INTO workspace_artifact_origins VALUES(?,?,?,?,?,?)
                ON CONFLICT(workspace_id,path) DO UPDATE SET revision=excluded.revision,
                kind=excluded.kind,task_id=excluded.task_id,parents=excluded.parents''',
                (workspace_id,path,revision,kind,task_id,json.dumps(list(parents),ensure_ascii=False)))

    def get(self, workspace_id, path, revision):
        with closing(self.connection_factory()) as conn:
            row = conn.execute('SELECT * FROM workspace_artifact_origins WHERE workspace_id=? AND path=? AND revision=?',
                               (workspace_id,path,revision)).fetchone()
        if not row:return {'kind':'unknown'}
        return {'kind':row['kind'],'task_id':row['task_id'],'parents':json.loads(row['parents']),
                'independent_evidence':False}


def source_origins(filesystem, workspace_id, sources):
    """Annotate snapshots only; access/revision validation remains with retrieval."""
    try:
        store = ArtifactOrigins(filesystem.repository.connection_factory)
        for source in sources:
            meta = source.setdefault('metadata',{})
            meta['origin'] = store.get(workspace_id,meta.get('source',''),meta.get('revision',''))
    except Exception:
        # Metadata failure must not promote a source to primary evidence.
        for source in sources:source.setdefault('metadata',{})['origin'] = {'kind':'unknown'}
