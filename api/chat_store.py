"""Storage shared by chat routes and server-side answer persistence."""
import json
from assistant_models import utc_now


def ensure_chat_schema(conn):
    conn.execute('CREATE TABLE IF NOT EXISTS workspace_chats (id TEXT PRIMARY KEY, data TEXT NOT NULL, updated_at TEXT NOT NULL)')
    columns = {row[1] for row in conn.execute('PRAGMA table_info(workspace_chats)')}
    if 'version' not in columns:
        conn.execute('ALTER TABLE workspace_chats ADD COLUMN version INTEGER NOT NULL DEFAULT 0')
    conn.execute('CREATE TABLE IF NOT EXISTS deleted_chats (id TEXT PRIMARY KEY, deleted_at TEXT NOT NULL)')
    conn.execute('CREATE INDEX IF NOT EXISTS workspace_chats_updated ON workspace_chats(updated_at DESC, id DESC)')


def append_response(conn, request, response):
    if not request.chat_id:
        return None
    if conn.execute('SELECT 1 FROM deleted_chats WHERE id=?', (request.chat_id,)).fetchone():
        raise ValueError('This chat was deleted; create a new chat')
    row = conn.execute('SELECT data,version FROM workspace_chats WHERE id=?', (request.chat_id,)).fetchone()
    now = utc_now()
    chat = json.loads(row['data']) if row else {
        'id': request.chat_id, 'title': request.question[:65], 'messages': [], 'session_id': response.session_id,
    }
    messages = chat.setdefault('messages', [])
    request_id = request.request_id
    if request_id and any(message.get('reply_to') == request_id for message in messages):
        return row['version']
    matching_user = next((message for message in reversed(messages)
                          if message.get('role') == 'user' and (
                              message.get('id') == request_id if request_id else message.get('content') == request.question)), None)
    if not matching_user:
        messages.append({'role': 'user', 'content': request.question, 'timestamp': now, 'id': request_id})
    messages.append({'role': 'assistant', 'content': response.answer, 'timestamp': now,
        'reply_to': request_id, 'task_id': response.task_id,
        'context': response.context, 'strategy': response.strategy,
        'verification': response.verification.model_dump(mode='json') if response.verification else None,
        'graph': response.graph,
        'steps': {'draft': response.draft, 'edited': response.edited, 'enhanced': response.enhanced}})
    version = (row['version'] if row else 0) + 1
    chat.update(session_id=response.session_id, updated_at=now, version=version)
    conn.execute('INSERT INTO workspace_chats(id,data,updated_at,version) VALUES(?,?,?,?) '
                 'ON CONFLICT(id) DO UPDATE SET data=excluded.data,updated_at=excluded.updated_at,version=excluded.version',
                 (request.chat_id, json.dumps(chat, ensure_ascii=False), now, version))
    return version
