"""Real Nomic / optional chat+review ablation in isolated stores and servers.

Run: .venv/Scripts/python.exe tests/benchmark_graph_retrieval.py --answers --repeats 2
Private outputs are ignored. No production application lifespan is invoked.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics
import sys
import time


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--answers', action='store_true')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--fixture', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = root / '.user-evaluation' / time.strftime('graph-retrieval-%Y%m%d-%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    for key, value in {
        'KONTUR_DATA_DIR': str(out), 'RAG_DB_PATH': str(out / 'fixture.db'),
        'CHROMA_PATH': str(out / 'chroma'), 'LLAMA_AUTOSTART': 'true',
        'LLAMA_SERVER_EXECUTABLE': str(root / 'runtime/llama/llama-server.exe'),
        'LLAMA_MODEL_PATH': os.getenv('EVALUATION_MODEL_PATH', str(root / 'api/models/chat.gguf')),
        'EMBEDDING_MODEL_PATH': str(root / 'api/models/nomic-embed-text.gguf'),
        'LLAMA_SERVER_URL': f'http://127.0.0.1:{free_port()}',
        'EMBEDDING_SERVER_URL': f'http://127.0.0.1:{free_port()}',
        'EMBEDDING_BACKEND': 'llama', 'ANONYMIZED_TELEMETRY': 'False',
        'VERIFIER_ENABLED': 'true', 'ANSWER_REVIEW_MODE': 'contextual',
    }.items():
        os.environ[key] = value
    sys.path.insert(0, str(root / 'api'))
    from embedding_provider import LocalEmbeddings
    from langchain_chroma import Chroma
    from hybrid_context_extractor import HybridRetriever
    from db_utils import get_db_connection
    from graph_retrieval import query_index
    from semantic_graph import SemanticDocumentGraph
    documents, chunks, questions = [], [], []
    def add(name, text):
        did = len(documents) + 1
        cid = f'chunk-{did}'
        documents.append({'id': did, 'name': name})
        chunks.append({'id': cid, 'document_id': did, 'text': text})
        return cid
    filler = ('\n\nОбщие правила ведения журнала: записывайте время проверки, номер записи и результат наблюдения. '
              'Сохраняйте исходный порядок записей. Проверяйте полноту заполнения служебных полей. '
              'Информация о техническом обслуживании относится к порядку оформления документации. ') * 3
    a = add('sapphire-project.txt', 'Проект Сапфир использует устройство Термометр-17. Код проекта Сапфир — 6241.' + filler)
    b = add('thermometer-maker.txt', 'Устройство Термометр-17 выпускает предприятие Северный прибор.' + filler)
    c = add('northern-company.txt', 'Предприятие Северный прибор располагается в городе Тверь.' + filler)
    add('western-company.txt', 'Предприятие Западный прибор располагается в городе Омск. Код проекта Рубин — 9876.' + filler)
    e = add('cedar-access.txt', 'Employees of project Cedar have archive access for 14 days. The access code for project Cedar is 7318.' + filler)
    add('maple-access.txt', 'Employees of project Maple have archive access for 60 days. The access code for project Maple is 9999.' + filler)
    add('recipe.txt', 'Приготовление блинов требует муку, яйца и молоко.' + filler)
    add('journal.txt', 'Общие правила заполнения журнала проверок.' + filler)
    questions = [
        {'id': 1, 'text': 'Какой код проекта Сапфир?', 'gold': [a], 'required': ['6241'], 'forbidden': ['9876'], 'answer': True},
        {'id': 2, 'text': 'Кто выпускает устройство проекта Сапфир?', 'gold': [a, b], 'required': ['Северный прибор'], 'forbidden': ['Западный'], 'answer': True},
        {'id': 3, 'text': 'В каком городе находится производитель устройства проекта Сапфир?', 'gold': [a, b, c], 'required': ['Твер'], 'forbidden': ['Омск'], 'answer': True},
        {'id': 4, 'text': 'What is the access code for project Cedar?', 'gold': [e], 'required': ['7318'], 'forbidden': ['9999'], 'answer': True},
        {'id': 5, 'text': 'Телефон производителя устройства проекта Сапфир?', 'gold': [a, b], 'required': [], 'forbidden': [], 'answer': False},
        {'id': 6, 'text': 'Расскажи о проекте Сапфир и его устройстве.', 'gold': [a, b], 'required': [], 'forbidden': [], 'answer': False},
    ]
    if args.fixture:
        fixture = json.loads(args.fixture.read_text(encoding='utf-8'))
        documents, chunks, questions = fixture['documents'], fixture['chunks'], fixture['questions']
    fixture = {'documents': documents, 'chunks': chunks, 'questions': questions}
    (out / 'fixture.json').write_text(json.dumps(fixture, ensure_ascii=False, indent=2), encoding='utf-8')
    protected = [root / 'api' / name for name in ('rag_app.db', 'chat-model-selection.json')]
    digests = {p: hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None for p in protected}
    embeddings, store, provider = LocalEmbeddings(), None, None
    rows, answer_rows, calls = [], [], []
    try:
        started = time.perf_counter()
        embeddings.manager.ensure_ready()
        store = Chroma(collection_name='graph_retrieval_benchmark', embedding_function=embeddings,
                       persist_directory=str(out / 'chroma'))
        vectors = embeddings.embed_documents([c['text'] for c in chunks])
        store._collection.add(ids=[c['id'] for c in chunks], embeddings=vectors,
            documents=[c['text'] for c in chunks], metadatas=[{'document_id': c['document_id'],
                'chunk_id': c['id'], 'source': documents[c['document_id'] - 1]['name']} for c in chunks])
        with get_db_connection() as conn:
            conn.executemany('INSERT INTO document_store(id,filename) VALUES(?,?)', [(d['id'], d['name']) for d in documents])
        indexing_seconds = time.perf_counter() - started
        base = store.as_retriever(search_type='mmr', search_kwargs={'k': 6, 'fetch_k': 16, 'lambda_mult': .7})
        retrievers = {enabled: HybridRetriever(base, store, 'hybrid', needs_history_aware_input=False,
                         lexical_weight=1.25, graph_enabled=enabled) for enabled in (False, True)}
        started = time.perf_counter()
        query_index(SemanticDocumentGraph(store))
        graph_build_ms = (time.perf_counter() - started) * 1000
        # Warm each retrieval path. Actual query embeddings are not cached.
        for retriever in retrievers.values():
            retriever.search(questions[0]['text'])
        for repeat in range(args.repeats):
            for q in questions:
                for enabled in ((False, True) if repeat % 2 == 0 else (True, False)):
                    retriever = retrievers[enabled]
                    started = time.perf_counter()
                    sources = retriever.search(q['text'])
                    ids = [s.chunk_id for s in sources]
                    row = {'question_id': q['id'], 'graph': enabled, 'repeat': repeat,
                           'complete': set(q['gold']) <= set(ids), 'ids': ids,
                           'recall': len(set(q['gold']) & set(ids)) / len(q['gold']),
                           'seconds': time.perf_counter() - started,
                           **retriever.retrieval_info}
                    rows.append(row)
        print(json.dumps({'output': str(out), 'documents': len(documents), 'questions': len(questions),
                          'graph_build_ms': graph_build_ms}), flush=True)
        if args.answers:
            from model_runtime import get_llm_provider
            from answer_review import review_answer
            provider = get_llm_provider()
            provider.manager.ensure_ready()
            original_stream = provider.stream
            def observed_stream(messages, **options):
                started = time.perf_counter()
                row = {'schema': bool(options.get('schema'))}
                try:
                    for delta in original_stream(messages, **options):
                        if delta.metrics:
                            row.update(asdict(delta.metrics))
                        yield delta
                finally:
                    row['seconds'] = time.perf_counter() - started
                    calls.append(row)
            provider.stream = observed_stream
            provider.generate([{'role': 'user', 'content': 'Ответь одним словом: готов.'}],
                              profile='FAST', thinking=False, temperature=0, max_tokens=32)
            from langchain_utils import qa_prompt
            for repeat in range(args.repeats):
                for q in (q for q in questions if q.get('answer')):
                    for enabled in ((False, True) if repeat % 2 == 0 else (True, False)):
                        start_calls, started = len(calls), time.perf_counter()
                        sources = retrievers[enabled].search(q['text'])
                        retrieval_seconds = time.perf_counter() - started
                        context = '\n\n'.join(f'[Источник: {s.metadata["source"]}; chunk: {s.chunk_id}]\n{s.text}' for s in sources)
                        messages = [{'role': 'user' if m.type == 'human' else m.type, 'content': m.content}
                                    for m in qa_prompt.format_messages(context=context, chat_history=[], input=q['text'])]
                        generation_started = time.perf_counter()
                        answer = provider.generate(messages, profile='FAST', thinking=False, temperature=0, max_tokens=384).text
                        generation_seconds = time.perf_counter() - generation_started
                        review_started = time.perf_counter()
                        final, review = review_answer(q['text'], answer, [s.as_dict() for s in sources],
                                                     'DOCUMENT_QA', 'FAST', client=provider)
                        review_seconds = time.perf_counter() - review_started
                        checks = {w: w.casefold() in final.casefold() for w in q['required']}
                        checks.update({'absent:' + w: w.casefold() not in final.casefold() for w in q['forbidden']})
                        checks['review_supported'] = review is not None and review.status == 'SUPPORTED'
                        row = {'question_id': q['id'], 'repeat': repeat, 'graph': enabled,
                               'seconds': time.perf_counter() - started, 'retrieval_seconds': retrieval_seconds,
                               'generation_seconds': generation_seconds, 'review_seconds': review_seconds,
                               'answer': final, 'draft': answer, 'review': review.model_dump(mode='json') if review else None,
                               'checks': checks, 'passed': all(checks.values()), 'source_ids': [s.chunk_id for s in sources],
                               'calls': calls[start_calls:]}
                        answer_rows.append(row)
                        (out / 'answers.json').write_text(json.dumps(answer_rows, ensure_ascii=False, indent=2), encoding='utf-8')
                        print(f'Answer {q["id"]} graph={enabled} repeat={repeat}: {row["seconds"]:.2f}s {"PASS" if row["passed"] else "FAIL"}', flush=True)
        groups = {}
        for enabled in (False, True):
            selected = [r for r in rows if r['graph'] == enabled]
            ans = [r for r in answer_rows if r['graph'] == enabled]
            groups[str(enabled)] = {'samples': len(selected), 'complete': sum(r['complete'] for r in selected),
                'mean_recall': statistics.mean(r['recall'] for r in selected),
                'retrieval_median_ms': statistics.median(r['seconds'] for r in selected) * 1000,
                'mean_context_characters': statistics.mean(r['context_characters'] for r in selected),
                'fast_paths': sum(r['embedding_skipped'] for r in selected),
                'answers_passed': sum(r['passed'] for r in ans), 'answer_samples': len(ans),
                'answer_median_seconds': statistics.median(r['seconds'] for r in ans) if ans else None}
        unchanged = all((hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None) == d for p, d in digests.items())
        result = {'groups': groups, 'rows': rows, 'answers': answer_rows,
                  'indexing_seconds': indexing_seconds, 'graph_build_ms': graph_build_ms,
                  'model_configuration': asdict(provider.manager.profile) if provider is not None else None,
                  'embedding_configuration': asdict(embeddings.manager.profile),
                  'protected_files_unchanged': unchanged,
                  'source_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in
                                    ('api/graph_retrieval.py', 'api/semantic_graph.py', 'api/hybrid_context_extractor.py',
                                     'api/workspace.py', 'api/workspace_vector_index.py', 'api/answer_review.py')}}
        (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
        print(json.dumps({'groups': groups, 'protected_files_unchanged': unchanged}), flush=True)
        # The baseline can fail a question that the graph recovers; preserve its
        # failed checks as comparison data. Every graph answer must pass.
        if not unchanged or any(not r['passed'] for r in answer_rows if r['graph']):
            return 1
        return 0
    finally:
        embeddings.manager.stop()
        if provider is not None:
            provider.manager.stop()
            provider.close()
        if store is not None:
            store._client._system.stop()


if __name__ == '__main__':
    sys.exit(main())
