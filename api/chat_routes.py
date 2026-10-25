"""Persistent workspace conversations, independent of model execution logs."""
import json
import os
import base64
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, model_validator
from chat_store import ensure_chat_schema
from db_utils import get_db_connection
from settings import API_DIR

router = APIRouter(prefix="/chats", tags=["chats"])


class Chat(BaseModel):
    id: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=120)
    messages: list[dict] = Field(default_factory=list)
    session_id: str | None = None
    updated_at: str
    version: int | None = Field(default=None, ge=0)

    @model_validator(mode='after')
    def bounded_messages(self):
        if len(self.messages) > 10000 or sum(len(json.dumps(m, ensure_ascii=False)) for m in self.messages) > 7_000_000:
            raise ValueError('Chat is too large')
        return self


def connection():
    conn = get_db_connection()
    conn.create_function('casefold', 1, lambda value: str(value or '').casefold(), deterministic=True)
    ensure_chat_schema(conn)
    conn.execute('CREATE TABLE IF NOT EXISTS workspace_migrations (name TEXT PRIMARY KEY)')
    conn.execute('CREATE TABLE IF NOT EXISTS deleted_chats (id TEXT PRIMARY KEY, deleted_at TEXT NOT NULL)')
    if not os.getenv('RAG_DB_PATH') and not conn.execute("SELECT 1 FROM workspace_migrations WHERE name='streamlit-json'").fetchone():
        archive = API_DIR.parent / 'app' / 'chat_history.json'
        if archive.is_file():
            # Leave the original archive untouched. Never reimport deleted chats.
            for raw in json.loads(archive.read_text(encoding='utf-8')):
                chat = Chat(id=raw['id'], title=raw['title'], messages=raw.get('messages', []),
                            session_id=raw.get('session_id'), updated_at=raw.get('updated_at') or datetime.now(timezone.utc).isoformat())
                conn.execute('INSERT OR IGNORE INTO workspace_chats(id,data,updated_at) VALUES (?, ?, ?)', (chat.id, chat.model_dump_json(), chat.updated_at))
        conn.execute("INSERT INTO workspace_migrations VALUES ('streamlit-json')")
    conn.commit()
    return conn


@router.get("")
def list_chats(limit: int = Query(default=50, ge=1, le=100), offset: int = Query(default=0, ge=0)):
    conn = connection()
    try:
        return [{**json.loads(row['data']), 'version': row['version']} for row in conn.execute(
            'SELECT data,version FROM workspace_chats ORDER BY updated_at DESC, id DESC LIMIT ? OFFSET ?', (limit, offset))]
    finally:
        conn.close()


@router.get('/page')
def chat_page(limit: int = Query(default=50, ge=1, le=100), cursor: str | None = Query(default=None, max_length=1024),
              search: str = Query(default='', max_length=120)):
    conn = connection()
    try:
        where, arguments = [], []
        if search:
            where.append("instr(casefold(json_extract(data,'$.title')),casefold(?)) > 0")
            arguments.append(search)
        if cursor:
            try:
                updated_at, identifier = json.loads(base64.urlsafe_b64decode(cursor.encode()))
                if not isinstance(updated_at, str) or not isinstance(identifier, str):
                    raise ValueError()
            except (ValueError, TypeError, UnicodeError):
                raise HTTPException(422, 'Invalid chat cursor')
            where.append('(updated_at < ? OR (updated_at = ? AND id < ?))')
            arguments.extend((updated_at, updated_at, identifier))
        condition = ' WHERE ' + ' AND '.join(where) if where else ''
        rows = conn.execute("SELECT id,updated_at,version,json_extract(data,'$.title') AS title,"
            "json_extract(data,'$.session_id') AS session_id,json_array_length(data,'$.messages') AS message_count "
            'FROM workspace_chats' + condition + ' ORDER BY updated_at DESC,id DESC LIMIT ?',
            (*arguments, limit + 1)).fetchall()
        more, rows = len(rows) > limit, rows[:limit]
        items = [{**dict(row), 'messages': [], 'messages_loaded': False} for row in rows]
        next_cursor = (base64.urlsafe_b64encode(json.dumps([rows[-1]['updated_at'], rows[-1]['id']]).encode()).decode()
                       if more else None)
        return {'items': items, 'next_cursor': next_cursor}
    finally:
        conn.close()


@router.get('/{chat_id}')
def get_chat(chat_id: str):
    conn = connection()
    try:
        row = conn.execute('SELECT data,version FROM workspace_chats WHERE id=?', (chat_id,)).fetchone()
        if not row:
            raise HTTPException(404, 'Chat not found')
        return {**json.loads(row['data']), 'version': row['version'], 'messages_loaded': True}
    finally:
        conn.close()


@router.put('/{chat_id}')
def save_chat(chat_id: str, chat: Chat):
    if chat_id != chat.id:
        raise HTTPException(422, 'Chat id mismatch')
    conn = connection()
    try:
        conn.execute('BEGIN IMMEDIATE')
        if conn.execute('SELECT 1 FROM deleted_chats WHERE id=?', (chat_id,)).fetchone():
            raise HTTPException(409, 'This chat was deleted; create a new chat')
        previous = conn.execute('SELECT version FROM workspace_chats WHERE id=?', (chat_id,)).fetchone()
        if previous and chat.version is not None and chat.version != previous['version']:
            raise HTTPException(409, 'Chat changed. Reload it before saving.')
        chat.version = (previous['version'] if previous else 0) + 1
        conn.execute('INSERT INTO workspace_chats(id,data,updated_at,version) VALUES (?, ?, ?, ?) '
                     'ON CONFLICT(id) DO UPDATE SET data=excluded.data, updated_at=excluded.updated_at,version=excluded.version',
                     (chat.id, chat.model_dump_json(), chat.updated_at, chat.version))
        conn.commit()
        return chat
    finally:
        conn.close()


@router.delete('/{chat_id}')
def delete_chat(chat_id: str):
    conn = connection()
    try:
        conn.execute('BEGIN IMMEDIATE')
        row = conn.execute('SELECT data FROM workspace_chats WHERE id=?', (chat_id,)).fetchone()
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        sessions = set()
        if 'assistant_sessions' in tables:
            sessions.update(row[0] for row in conn.execute('SELECT id FROM assistant_sessions WHERE chat_id=?', (chat_id,)))
        if row:
            historical_id = json.loads(row['data']).get('session_id')
            if historical_id:
                owner = conn.execute('SELECT chat_id FROM assistant_sessions WHERE id=?', (historical_id,)).fetchone() if 'assistant_sessions' in tables else None
                if not owner or not owner[0] or owner[0] == chat_id:
                    sessions.add(historical_id)
        for session_id in sessions:
            if 'assistant_responses' in tables:
                conn.execute('DELETE FROM assistant_responses WHERE session_id=?', (session_id,))
            if 'assistant_tasks' in tables:
                if conn.execute("SELECT 1 FROM assistant_tasks WHERE session_id=? AND status IN ('PENDING','RUNNING')",
                                (session_id,)).fetchone():
                    raise HTTPException(409, 'Stop the active task before deleting this chat')
                conn.execute('DELETE FROM assistant_actions WHERE task_id IN (SELECT id FROM assistant_tasks WHERE session_id=?)', (session_id,))
                conn.execute('DELETE FROM assistant_tasks WHERE session_id=?', (session_id,))
            if 'assistant_pending_plans' in tables:
                conn.execute('DELETE FROM assistant_pending_plans WHERE session_id=?', (session_id,))
            for table in ('assistant_context_checkpoints','assistant_context_origins'):
                if table in tables:conn.execute(f'DELETE FROM {table} WHERE session_id=?',(session_id,))
            if 'assistant_history_scopes' in tables:
                conn.execute('DELETE FROM assistant_history_scopes WHERE log_id IN (SELECT id FROM application_logs WHERE session_id=?)',(session_id,))
            conn.execute('DELETE FROM application_logs WHERE session_id=?', (session_id,))
            if 'assistant_sessions' in tables:
                conn.execute('DELETE FROM assistant_sessions WHERE id=?', (session_id,))
        conn.execute('DELETE FROM workspace_chats WHERE id=?', (chat_id,))
        conn.execute('INSERT OR IGNORE INTO deleted_chats VALUES(?,?)',
                     (chat_id, datetime.now(timezone.utc).isoformat()))
        conn.commit()
        return {'deleted': chat_id}
    finally:
        conn.close()
