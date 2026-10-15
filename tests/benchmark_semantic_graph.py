"""Opt-in measurement with the installed embedding GGUF and synthetic documents.

Uses its own server, SQLite and Chroma under .user-evaluation; does not import
the application, read personal documents, or start a chat model.
"""
import json
import os
from pathlib import Path
import socket
import sqlite3
import sys
import time


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / '.user-evaluation' / ('semantic-graph-' + time.strftime('%Y%m%d-%H%M%S'))
    output.mkdir(parents=True, exist_ok=False)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    for key, value in {
        'KONTUR_DATA_DIR': str(output), 'RAG_DB_PATH': str(output / 'fixture.db'),
        'CHROMA_PATH': str(output / 'chroma'), 'LLAMA_AUTOSTART': 'true',
        'LLAMA_SERVER_EXECUTABLE': str(root / 'runtime/llama/llama-server.exe'),
        'EMBEDDING_MODEL_PATH': str(root / 'api/models/nomic-embed-text.gguf'),
        'EMBEDDING_SERVER_URL': f'http://127.0.0.1:{port}', 'EMBEDDING_BACKEND': 'llama',
        'ANONYMIZED_TELEMETRY': 'False',
    }.items():
        os.environ[key] = value
    sys.path.insert(0, str(root / 'api'))
    import numpy as np
    import chromadb
    from embedding_provider import LocalEmbeddings
    from hybrid_context_extractor import SearchResult
    from semantic_graph import SemanticDocumentGraph, SIMILARITY_THRESHOLD

    texts = [
        'Резервное копирование базы данных выполняется каждый день. Для защиты от потери данных резервные копии сохраняют отдельно от основного сервера.',
        'Восстановление базы данных из архива. При сбое основного сервера данные возвращают из сохранённого снимка на отдельном хранилище. Архив обновляется ежедневно.',
        'Для приготовления блинов смешайте муку, яйца и молоко. Жарьте тесто на горячей сковороде до золотистой корочки.',
        'Database backups are created every day. Store backup copies separately from the main server to protect against data loss.',
        'Restore the database from a saved snapshot after the primary server fails. An independent storage keeps the recovery archive and updates it daily.',
        'Mix flour, eggs and milk to make pancakes. Cook the batter in a hot frying pan until golden brown.',
    ]
    provider, client = LocalEmbeddings(), None
    try:
        start = time.perf_counter()
        provider.manager.ensure_ready()
        startup = time.perf_counter() - start
        vectors = provider.embed_documents(texts)
        matrix = np.asarray(vectors)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        similarities = matrix @ matrix.T
        def connection():
            conn = sqlite3.connect(output / 'fixture.db')
            conn.row_factory = sqlite3.Row
            return conn
        with connection() as conn:
            conn.execute('CREATE TABLE IF NOT EXISTS document_store(id INTEGER PRIMARY KEY,filename TEXT)')
            conn.executemany('INSERT INTO document_store(id,filename) VALUES(?,?)',
                             [(i + 1, f'Синтетический документ {i + 1}.txt') for i in range(len(texts))])
        client = chromadb.PersistentClient(path=str(output / 'chroma'))
        store = client.create_collection('semantic_measurement', embedding_function=None)
        store.add(ids=[str(i + 1) for i in range(len(texts))], documents=texts,
                  metadatas=[{'document_id': i + 1} for i in range(len(texts))], embeddings=vectors)
        graph = SemanticDocumentGraph(store, connection_factory=connection)
        start = time.perf_counter()
        payload = graph.map()
        build_ms = (time.perf_counter() - start) * 1000
        added = graph.expand([SearchResult(1, '1', texts[0], {'document_id': 1})])
        result = {'threshold': SIMILARITY_THRESHOLD, 'startup_seconds': startup, 'graph_build_ms': build_ms,
                  'pair_similarities': {'ru_related': float(similarities[0, 1]),
                                        'ru_unrelated': float(similarities[0, 2]),
                                        'en_related': float(similarities[3, 4]),
                                        'en_unrelated': float(similarities[3, 5])},
                  'map': payload, 'added_documents': [item.document_id for item in added]}
        (output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({key: value for key, value in result.items() if key != 'map'}, ensure_ascii=False), flush=True)
    finally:
        provider.manager.stop()
        if client is not None:
            client._system.stop()


if __name__ == '__main__':
    main()
