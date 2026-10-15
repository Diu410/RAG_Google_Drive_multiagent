"""Additive, transactional SQLite migration and execution persistence."""
import json
import uuid
from contextlib import closing

from assistant_models import Action, Session, Task, TaskStatus, utc_now
from db_utils import get_db_connection


class AssistantRepository:
    def __init__(self, connection_factory=get_db_connection):
        self.connection_factory = connection_factory
        self.migrate()

    def migrate(self):
        # Separate statements keep DDL + backfill + migration marker atomic.
        with closing(self.connection_factory()) as conn, conn:
            conn.execute('BEGIN IMMEDIATE')
            conn.execute('CREATE TABLE IF NOT EXISTS assistant_migrations (version INTEGER PRIMARY KEY)')
            conn.execute('''CREATE TABLE IF NOT EXISTS assistant_responses (
                task_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, request_id TEXT,
                request_fingerprint TEXT NOT NULL, response TEXT NOT NULL,
                UNIQUE(session_id, request_id))''')
            if conn.execute('SELECT 1 FROM assistant_migrations WHERE version=1').fetchone():
                conn.execute('''CREATE TABLE IF NOT EXISTS assistant_pending_plans (
                    session_id TEXT PRIMARY KEY REFERENCES assistant_sessions(id) ON DELETE CASCADE,
                    state TEXT NOT NULL, updated_at TEXT NOT NULL)''')
                return
            conn.execute('''CREATE TABLE assistant_sessions (
                id TEXT PRIMARY KEY, chat_id TEXT UNIQUE, created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL, status TEXT NOT NULL, metadata TEXT NOT NULL)''')
            statuses = "('PENDING','RUNNING','COMPLETED','FAILED','CANCELLED')"
            conn.execute(f'''CREATE TABLE assistant_tasks (
                id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES assistant_sessions(id),
                user_input TEXT NOT NULL, task_type TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN {statuses}), created_at TEXT NOT NULL,
                started_at TEXT, completed_at TEXT, error TEXT, metadata TEXT NOT NULL)''')
            conn.execute(f'''CREATE TABLE assistant_actions (
                id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES assistant_tasks(id),
                action_type TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN {statuses}),
                input TEXT NOT NULL, output TEXT NOT NULL, started_at TEXT, completed_at TEXT, error TEXT)''')
            conn.execute('CREATE INDEX assistant_tasks_session ON assistant_tasks(session_id, created_at)')
            conn.execute('CREATE INDEX assistant_actions_task ON assistant_actions(task_id)')
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            now = utc_now()
            if 'workspace_chats' in tables:
                for row in conn.execute('SELECT id, data, updated_at FROM workspace_chats').fetchall():
                    try:
                        data = json.loads(row['data'])
                    except (ValueError, TypeError):
                        continue
                    sid = data.get('session_id') or str(uuid.uuid4())
                    conn.execute('''INSERT OR IGNORE INTO assistant_sessions
                        (id,chat_id,created_at,updated_at,status,metadata) VALUES(?,?,?,?,?,?)''',
                                 (sid, row['id'], row['updated_at'], row['updated_at'], 'ACTIVE', '{}'))
            if 'application_logs' in tables:
                conn.execute('''INSERT OR IGNORE INTO assistant_sessions
                    SELECT session_id, NULL, COALESCE(MIN(created_at), ?),
                    COALESCE(MAX(created_at), ?), 'ACTIVE', '{}' FROM application_logs
                    WHERE session_id IS NOT NULL AND session_id != '' GROUP BY session_id''', (now, now))
            conn.execute('INSERT INTO assistant_migrations VALUES(1)')
            conn.execute('''CREATE TABLE IF NOT EXISTS assistant_pending_plans (
                session_id TEXT PRIMARY KEY REFERENCES assistant_sessions(id) ON DELETE CASCADE,
                state TEXT NOT NULL, updated_at TEXT NOT NULL)''')

    def ensure_session(self, session_id=None, chat_id=None, metadata=None):
        with closing(self.connection_factory()) as conn, conn:
            conn.execute('BEGIN IMMEDIATE')
            if chat_id and conn.execute("SELECT 1 FROM sqlite_master WHERE name='deleted_chats'").fetchone():
                if conn.execute('SELECT 1 FROM deleted_chats WHERE id=?', (chat_id,)).fetchone():
                    raise ValueError('This chat was deleted; create a new chat')
            row = conn.execute('SELECT * FROM assistant_sessions WHERE chat_id=?', (chat_id,)).fetchone() if chat_id else None
            if row and session_id and row['id'] != session_id:
                raise ValueError('Chat is already associated with another session')
            if not row and session_id:
                row = conn.execute('SELECT * FROM assistant_sessions WHERE id=?', (session_id,)).fetchone()
            # Chats may be imported/created after the schema migration. Reuse
            # their historical model session instead of silently losing context.
            if not row and chat_id and not session_id and conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='workspace_chats'").fetchone():
                chat = conn.execute('SELECT data FROM workspace_chats WHERE id=?', (chat_id,)).fetchone()
                if chat:
                    session_id = json.loads(chat['data']).get('session_id')
                    if session_id:
                        row = conn.execute('SELECT * FROM assistant_sessions WHERE id=?', (session_id,)).fetchone()
            if row:
                if chat_id and row['chat_id'] and row['chat_id'] != chat_id:
                    raise ValueError('Session is already associated with another chat')
                conn.execute('UPDATE assistant_sessions SET updated_at=?, chat_id=COALESCE(chat_id,?) WHERE id=?',
                             (utc_now(), chat_id, row['id']))
                sid = row['id']
            else:
                sid = session_id or str(uuid.uuid4())
                now = utc_now()
                conn.execute('''INSERT INTO assistant_sessions
                    (id,chat_id,created_at,updated_at,status,metadata) VALUES(?,?,?,?,?,?)''',
                             (sid, chat_id, now, now, 'ACTIVE', json.dumps(metadata or {})))
            return self._decode(Session, conn.execute('SELECT * FROM assistant_sessions WHERE id=?', (sid,)).fetchone())

    @staticmethod
    def _decode(model, row):
        if row is None:
            raise LookupError(f'{model.__name__} not found')
        data = dict(row)
        for field in ('metadata', 'input', 'output'):
            if field in data:
                data[field] = json.loads(data[field])
        return model(**data)

    def _insert(self, table, record):
        data = record.model_dump(mode='json')
        for field in ('metadata', 'input', 'output'):
            if field in data:
                data[field] = json.dumps(data[field], ensure_ascii=False)
        with closing(self.connection_factory()) as conn, conn:
            conn.execute('PRAGMA foreign_keys=ON')
            conn.execute(f'INSERT INTO {table} ({",".join(data)}) VALUES ({",".join("?" for _ in data)})', tuple(data.values()))
        return record

    def create_task(self, session_id, user_input, metadata=None):
        return self._insert('assistant_tasks', Task(id=str(uuid.uuid4()), session_id=session_id,
            user_input=user_input, created_at=utc_now(), metadata=metadata or {}))

    def recover_interrupted_tasks(self):
        """A previous process cannot still own PENDING/RUNNING executions."""
        with closing(self.connection_factory()) as conn, conn:
            for table in ('assistant_actions', 'assistant_tasks'):
                conn.execute(f'''UPDATE {table} SET status='FAILED', completed_at=?,
                    error='Execution interrupted by backend restart'
                    WHERE status IN ('PENDING','RUNNING')''', (utc_now(),))

    def create_action(self, task_id, action_type, inputs=None):
        return self._insert('assistant_actions', Action(id=str(uuid.uuid4()), task_id=task_id,
            action_type=action_type, input=inputs or {}))

    def save_pending_plan(self, session_id, state):
        with closing(self.connection_factory()) as conn, conn:
            conn.execute('''INSERT INTO assistant_pending_plans(session_id,state,updated_at) VALUES(?,?,?)
                ON CONFLICT(session_id) DO UPDATE SET state=excluded.state, updated_at=excluded.updated_at''',
                         (session_id, json.dumps(state, ensure_ascii=False), utc_now()))

    def pending_plan(self, session_id):
        with closing(self.connection_factory()) as conn:
            row = conn.execute('SELECT state FROM assistant_pending_plans WHERE session_id=?',
                               (session_id,)).fetchone()
            return json.loads(row['state']) if row else None

    def clear_pending_plan(self, session_id):
        with closing(self.connection_factory()) as conn, conn:
            conn.execute('DELETE FROM assistant_pending_plans WHERE session_id=?', (session_id,))

    def get_task(self, task_id):
        with closing(self.connection_factory()) as conn:
            return self._decode(Task, conn.execute('SELECT * FROM assistant_tasks WHERE id=?', (task_id,)).fetchone())

    @staticmethod
    def _request_fingerprint(request):
        return json.dumps({'question': request.question, 'workspace_id': request.workspace_id,
            'graph_selection': request.graph_selection.model_dump(mode='json') if request.graph_selection else None},
            sort_keys=True, ensure_ascii=False)

    def saved_response(self, session_id, request):
        if not request.request_id:
            return None
        with closing(self.connection_factory()) as conn:
            row = conn.execute('SELECT r.* FROM assistant_responses r JOIN assistant_tasks t ON t.id=r.task_id '
                "WHERE r.session_id=? AND r.request_id=? AND t.status='COMPLETED'",
                (session_id, request.request_id)).fetchone()
        if not row:
            return None
        if row['request_fingerprint'] != self._request_fingerprint(request):
            raise ValueError('Request id was already used for another question or source')
        from pydantic_models import QueryResponse
        response = QueryResponse.model_validate_json(row['response'])
        response.task_status = TaskStatus.COMPLETED
        response.actions = self.actions(row['task_id'])
        return response

    def save_response(self, request, response, task):
        """Commit visible chat, inference history and recoverable result together."""
        from chat_store import ensure_chat_schema, append_response
        from db_utils import get_knowledge_epoch
        from llm_contract import current_scope, GenerationCancelled
        now = utc_now()
        with closing(self.connection_factory()) as conn, conn:
            conn.execute('BEGIN IMMEDIATE')
            ensure_chat_schema(conn)
            response.chat_version = append_response(conn, request, response)
            conn.execute('INSERT INTO application_logs(session_id,user_query,gpt_response,model,knowledge_epoch) '
                'VALUES(?,?,?,?,?)', (response.session_id, request.question, response.answer,
                                     request.model.value, get_knowledge_epoch(conn)))
            scope = current_scope.get()
            if scope and scope.cancelled.is_set():
                raise GenerationCancelled()
            changed = conn.execute("UPDATE assistant_tasks SET status='COMPLETED',completed_at=? WHERE id=? AND status='RUNNING'",
                                   (now, task.id))
            if changed.rowcount != 1:
                raise ValueError('Task changed while saving the response')
            conn.execute('UPDATE assistant_sessions SET updated_at=? WHERE id=?', (now, response.session_id))
            response.task_status = TaskStatus.COMPLETED
            conn.execute('INSERT INTO assistant_responses(task_id,session_id,request_id,request_fingerprint,response) '
                'VALUES(?,?,?,?,?)', (response.task_id, response.session_id, request.request_id,
                    self._request_fingerprint(request), response.model_dump_json()))
        task.status = TaskStatus.COMPLETED
        task.completed_at = now

    def actions(self, task_id):
        with closing(self.connection_factory()) as conn:
            return [self._decode(Action, row) for row in conn.execute(
                'SELECT * FROM assistant_actions WHERE task_id=? ORDER BY rowid', (task_id,))]

    def transition(self, record, status, *, error=None, output=None):
        table = 'assistant_tasks' if isinstance(record, Task) else 'assistant_actions'
        allowed = {TaskStatus.PENDING: {TaskStatus.RUNNING, TaskStatus.CANCELLED},
                   TaskStatus.RUNNING: {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}}
        if status not in allowed.get(record.status, set()):
            raise ValueError(f'Invalid transition: {record.status} -> {status}')
        changes = {'status': status.value}
        changes['started_at' if status == TaskStatus.RUNNING else 'completed_at'] = utc_now()
        if error:
            changes['error'] = error[:500]
        if output is not None and isinstance(record, Action):
            changes['output'] = json.dumps(output, ensure_ascii=False)
        with closing(self.connection_factory()) as conn, conn:
            cursor = conn.execute(f'UPDATE {table} SET {",".join(k+"=?" for k in changes)} WHERE id=? AND status=?',
                                 (*changes.values(), record.id, record.status.value))
            if cursor.rowcount != 1:
                raise ValueError('Execution state changed concurrently')
            if isinstance(record, Task):
                conn.execute('UPDATE assistant_sessions SET updated_at=? WHERE id=?', (utc_now(), record.session_id))
        for key, value in changes.items():
            setattr(record, key, status if key == 'status' else output if key == 'output' else value)
