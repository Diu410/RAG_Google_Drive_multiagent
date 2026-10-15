import sqlite3
from datetime import datetime
from contextlib import closing

"""Тут мы сохраняем нашу историю чата. Чтобы позже её анализировать и использовать для контекста.
   Так же тут мы сохраняем таблицу со всеми документами."""

from settings import DB_PATH
DB_NAME = DB_PATH


def get_db_connection():
    conn = sqlite3.connect(DB_NAME, timeout=30) #Подключаемся к файлу с БД.
    conn.row_factory = sqlite3.Row #Настройка, которая позволяет обращаться к столбцам по именам, а не по индексам.
    conn.execute('PRAGMA foreign_keys=ON')
    conn.execute('PRAGMA busy_timeout=30000')
    return conn


def create_application_logs():
    #Тут мы создаём таблицу где будет хранится история чата.
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS application_logs
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      session_id TEXT,
                      user_query TEXT,
                      gpt_response TEXT,
                      model TEXT,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.close()

def insert_application_logs(session_id, user_query, gpt_response, model):
    conn = get_db_connection()
    conn.execute('''INSERT INTO application_logs
                    (session_id, user_query, gpt_response, model, knowledge_epoch)
                    VALUES (?, ?, ?, ?, ?)''',
                 (session_id, user_query, gpt_response, model, get_knowledge_epoch(conn)))
    conn.commit() #Сохраняем изменения в БД.
    conn.close()


def get_chat_history(session_id, limit=3):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''SELECT user_query, gpt_response FROM (
                      SELECT id,user_query,gpt_response FROM application_logs
                      WHERE session_id = ? AND knowledge_epoch = ? ORDER BY id DESC LIMIT ?)
                      ORDER BY id''', (session_id, get_knowledge_epoch(conn), max(1, min(limit, 100))))
    messages = []
    for row in cursor.fetchall():
        messages.extend([
            {"role": "human", "content": row['user_query']},
            {"role": "ai", "content": row['gpt_response']}
        ]) #Создаём два словаря и записываем их в список.
    conn.close()
    return messages

def create_document_store():
    # Тут мы создаём таблицу где будут хранится информация про файлы.
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS document_store
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      filename TEXT,
                      upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    columns = {row['name'] for row in conn.execute('PRAGMA table_info(document_store)').fetchall()}
    if 'source_content' not in columns:
        conn.execute('ALTER TABLE document_store ADD COLUMN source_content BLOB')
    if 'source_kind' not in columns:
        conn.execute('ALTER TABLE document_store ADD COLUMN source_kind TEXT')
    conn.commit()
    columns = {row['name'] for row in conn.execute('PRAGMA table_info(application_logs)').fetchall()}
    if 'knowledge_epoch' not in columns:
        conn.execute('ALTER TABLE application_logs ADD COLUMN knowledge_epoch INTEGER NOT NULL DEFAULT 0')
    conn.execute('''CREATE TABLE IF NOT EXISTS application_state
                    (key TEXT PRIMARY KEY, value TEXT NOT NULL)''')
    conn.commit()
    conn.close()


def get_knowledge_epoch(conn=None):
    owns_connection = conn is None
    conn = conn or get_db_connection()
    try:
        row = conn.execute("SELECT value FROM application_state WHERE key='knowledge_epoch'").fetchone()
        return int(row['value']) if row else 0
    finally:
        if owns_connection:
            conn.close()


def bump_knowledge_epoch():
    with closing(get_db_connection()) as conn, conn:
        conn.execute('BEGIN IMMEDIATE')
        epoch = get_knowledge_epoch(conn) + 1
        conn.execute("INSERT INTO application_state(key,value) VALUES('knowledge_epoch',?) "
                     "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (str(epoch),))
        return epoch


def initialize_knowledge_epoch():
    """Invalidate legacy inference memory when all indexed documents are gone."""
    with closing(get_db_connection()) as conn, conn:
        if conn.execute("SELECT 1 FROM application_state WHERE key='knowledge_epoch'").fetchone():
            return
        has_history = conn.execute('SELECT 1 FROM application_logs LIMIT 1').fetchone() is not None
        has_documents = conn.execute('SELECT 1 FROM document_store LIMIT 1').fetchone() is not None
        conn.execute("INSERT INTO application_state(key,value) VALUES('knowledge_epoch',?)",
                     ('1' if has_history and not has_documents else '0',))

def insert_document_record(filename, source_content=None, source_kind=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO document_store (filename, source_content, source_kind) VALUES (?, ?, ?)',
                   (filename, source_content, source_kind))
    file_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return file_id #Возращаем айди что позже использовать его для связи с индексированными частями


def delete_document_record(file_id):
    conn = get_db_connection()
    cursor = conn.execute('DELETE FROM document_store WHERE id = ?', (file_id,))
    conn.commit()
    conn.close()
    return cursor.rowcount > 0


def get_all_documents():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, filename, upload_timestamp, source_content IS NOT NULL AS can_reindex '
                   'FROM document_store ORDER BY upload_timestamp DESC')
    documents = cursor.fetchall()
    conn.close()
    return [dict(doc) for doc in documents]


def get_document(file_id):
    conn = get_db_connection()
    try:
        row = conn.execute('SELECT id, filename, upload_timestamp FROM document_store WHERE id = ?',
                           (file_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_document_source(file_id):
    conn = get_db_connection()
    try:
        row = conn.execute('SELECT id, filename, source_content, source_kind FROM document_store WHERE id = ?',
                           (file_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


create_application_logs() #Автоматически создаём таблицу для истории чата
create_document_store() # Автоматически создаём таблицу для документов
initialize_knowledge_epoch()
with closing(get_db_connection()) as _conn, _conn:
    _conn.execute('CREATE INDEX IF NOT EXISTS application_logs_session_epoch ON application_logs(session_id,knowledge_epoch,id)')
