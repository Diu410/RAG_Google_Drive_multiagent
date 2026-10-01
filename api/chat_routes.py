"""Persistent workspace conversations, independent of model execution logs."""
import json
import os
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from db_utils import get_db_connection
from settings import API_DIR

router = APIRouter(prefix="/chats", tags=["chats"])


class Chat(BaseModel):
    id: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=120)
    messages: list[dict] = Field(default_factory=list)
    session_id: str | None = None
    updated_at: str


def connection():
    conn = get_db_connection()
    conn.execute('CREATE TABLE IF NOT EXISTS workspace_chats (id TEXT PRIMARY KEY, data TEXT NOT NULL, updated_at TEXT NOT NULL)')
    conn.execute('CREATE TABLE IF NOT EXISTS workspace_migrations (name TEXT PRIMARY KEY)')
    conn.execute('CREATE TABLE IF NOT EXISTS deleted_chats (id TEXT PRIMARY KEY, deleted_at TEXT NOT NULL)')
    if not os.getenv('RAG_DB_PATH') and not conn.execute("SELECT 1 FROM workspace_migrations WHERE name='streamlit-json'").fetchone():
        archive = API_DIR.parent / 'app' / 'chat_history.json'
        if archive.is_file():
            # Leave the original archive untouched. Never reimport deleted chats.
            for raw in json.loads(archive.read_text(encoding='utf-8')):
                chat = Chat(id=raw['id'], title=raw['title'], messages=raw.get('messages', []),
                            session_id=raw.get('session_id'), updated_at=raw.get('updated_at') or datetime.now(timezone.utc).isoformat())
                conn.execute('INSERT OR IGNORE INTO workspace_chats VALUES (?, ?, ?)', (chat.id, chat.model_dump_json(), chat.updated_at))
        conn.execute("INSERT INTO workspace_migrations VALUES ('streamlit-json')")
    conn.commit()
    return conn


@router.get("")
def list_chats():
    conn = connection()
    try:
        return [json.loads(row['data']) for row in conn.execute('SELECT data FROM workspace_chats ORDER BY updated_at DESC, id')]
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
        conn.execute('INSERT INTO workspace_chats VALUES (?, ?, ?) ON CONFLICT(id) DO UPDATE SET data=excluded.data, updated_at=excluded.updated_at',
                     (chat.id, chat.model_dump_json(), chat.updated_at))
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
            if 'assistant_tasks' in tables:
                if conn.execute("SELECT 1 FROM assistant_tasks WHERE session_id=? AND status IN ('PENDING','RUNNING')",
                                (session_id,)).fetchone():
                    raise HTTPException(409, 'Stop the active task before deleting this chat')
                conn.execute('DELETE FROM assistant_actions WHERE task_id IN (SELECT id FROM assistant_tasks WHERE session_id=?)', (session_id,))
                conn.execute('DELETE FROM assistant_tasks WHERE session_id=?', (session_id,))
            if 'assistant_pending_plans' in tables:
                conn.execute('DELETE FROM assistant_pending_plans WHERE session_id=?', (session_id,))
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
