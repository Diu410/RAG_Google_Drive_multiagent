"""Opt-in real GGUF latency/quality benchmark; all data stays in .user-evaluation.

Run with the existing venv. Startup/indexing/warmup are excluded from samples.
--phase audit avoids indexing and compares identical independent evidence checks.
"""
import argparse
import asyncio
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import statistics
import sys
import time


async def benchmark(args):
    if args.model:
        os.environ['EVALUATION_MODEL_PATH'] = str(Path(args.model).resolve())
    # The existing harness configures isolated DB, Chroma and free server ports
    # before importing the application. No application lifespan/migration runs.
    import evaluate_user_queries as harness
    import settings
    from evidence_audit import EvidenceAuditor
    from model_profiles import get_profile
    from pydantic_models import QueryInput, VerificationClaim
    from llm_contract import current_action
    from verifier import Verifier
    main, run = harness.main, harness.RUN
    code_hashes = {name: hashlib.sha256((harness.ROOT / 'api' / name).read_bytes()).hexdigest()
                   for name in ('assistant_core.py', 'agent_planner.py', 'workspace.py', 'response_budget.py',
                                'evidence_audit.py', 'verifier.py', 'llama_process.py', 'llama_provider.py',
                                'graph_extraction.py', 'answer_editing.py')}
    logging.getLogger().setLevel(logging.ERROR)
    provider, embeddings = harness.get_llm_provider(), harness.get_embedding_provider()
    calls, records = [], []
    original_stream = provider.stream
    original_batch = settings.EVIDENCE_AUDIT_BATCH_SIZE
    original_compact = settings.EVIDENCE_AUDIT_COMPACT
    if args.audit_format in ('legacy', 'compact'):
        settings.EVIDENCE_AUDIT_COMPACT = args.audit_format == 'compact'
    protected = [harness.ROOT / 'api' / name for name in ('rag_app.db', 'chat-model-selection.json')]
    digests = {path: hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
               for path in protected}

    def observed_stream(messages, **options):
        started = time.perf_counter()
        schema = options.get('schema') or {}
        props = schema.get('properties', {})
        stage = schema.get('title') or ','.join(props) or 'text_generation'
        row = {'stage': stage, 'profile': options.get('profile'), 'thinking': options.get('thinking'),
               'max_tokens': options.get('max_tokens'), 'action_id': current_action.get(),
               'input_characters': sum(len(message.get('content', '')) for message in messages),
               'text_characters': 0, 'reasoning_characters': 0}
        try:
            for delta in original_stream(messages, **options):
                row['text_characters'] += len(delta.text)
                row['reasoning_characters'] += len(delta.reasoning)
                if delta.metrics:
                    row.update(asdict(delta.metrics))
                yield delta
        except BaseException as error:
            row['error'] = type(error).__name__
            raise
        finally:
            row['wall_seconds'] = time.perf_counter() - started
            calls.append(row)

    provider.stream = observed_stream

    async def measure(label, operation, batch, profile, repeat):
        settings.EVIDENCE_AUDIT_BATCH_SIZE = batch
        start_call, started = len(calls), time.perf_counter()
        row = {'id': label, 'batch_size': batch, 'profile': profile, 'repeat': repeat,
               'audit_format': 'compact' if settings.EVIDENCE_AUDIT_COMPACT else 'legacy'}
        try:
            row.update(await operation())
        except Exception as error:
            row.update(passed=False, error=f'{type(error).__name__}: {error}')
        row.update(seconds=time.perf_counter() - started, calls=calls[start_call:])
        actions = (row.get('response') or {}).get('actions', [])
        row['action_times'] = [{'action': action['action_type'], 'status': action['status'],
            'seconds': (datetime.fromisoformat(action['completed_at']) -
                        datetime.fromisoformat(action['started_at'])).total_seconds()}
            for action in actions if action.get('started_at') and action.get('completed_at')]
        if actions:
            row['outside_actions_seconds'] = max(0, row['seconds'] - sum(a['seconds'] for a in row['action_times']))
        action_names = {action['id']: action['action_type'] for action in actions}
        for call in row['calls']:
            call['action'] = action_names.get(call['action_id'])
        records.append(row)
        (run / 'benchmark.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"{label} {profile} batch={batch} repeat={repeat}: "
              f"{'PASS' if row['passed'] else 'FAIL'} {row['seconds']:.2f}s, {len(row['calls'])} LLM calls", flush=True)
        if row['action_times']:
            print('  ' + ', '.join(f"{a['action']}={a['seconds']:.2f}s" for a in row['action_times']), flush=True)

    try:
        print(f'Output: {run}', flush=True)
        started = time.perf_counter()
        await asyncio.to_thread(provider.manager.ensure_ready)
        startup = time.perf_counter() - started
        await asyncio.to_thread(provider.generate, [{'role': 'user', 'content': 'Ответь одним словом: готов.'}],
                                profile='FAST', max_tokens=32, temperature=0, thinking=False)
        print(f'Model ready in {startup:.2f}s; warmup excluded', flush=True)
        source = {'chunk_id': 'benchmark:standard', 'text': 'Рост Мирона составляет 184 сантиметра. '
                  'Допустимая нагрузка Мирона составляет 27 килограммов.'}
        question = 'Какой рост и допустимая нагрузка у Мирона?'
        valid = [VerificationClaim(text=text, status='SUPPORTED', basis='EXPLICIT',
                    source_chunk_ids=[source['chunk_id']], evidence_quote=text) for text in
                 ('Рост Мирона составляет 184 сантиметра.', 'Допустимая нагрузка Мирона составляет 27 килограммов.')]
        invalid = valid[1].model_copy(update={'text': 'Допустимая нагрузка Мирона составляет 99 килограммов.'})

        if args.phase in ('all', 'audit'):
            for repeat in range(1, args.repeats + 1):
                for label, claims, expected in [('audit-valid', valid, [True, True]),
                                                 ('audit-mixed', [valid[0], invalid], [True, False])]:
                    for batch in ((1, 2) if repeat % 2 else (2, 1)):
                        async def audit(claims=claims, expected=expected):
                            outcomes = await asyncio.to_thread(
                                EvidenceAuditor(provider, Verifier(provider).bounded_sources).supports_many,
                                claims, [source], 'FAST', question=question, think=False)
                            actual = [supported for supported, _ in outcomes]
                            return {'passed': actual == expected, 'actual': actual, 'expected': expected,
                                    'assessments': [auditor.last_assessment.model_dump() for _, auditor in outcomes]}
                        formats = (('legacy', 'compact') if repeat % 2 else ('compact', 'legacy')) if args.audit_format == 'both' else (args.audit_format,)
                        for wire_format in formats:
                            if wire_format:
                                settings.EVIDENCE_AUDIT_COMPACT = wire_format == 'compact'
                            await measure(label, audit, batch, 'FAST', repeat)

        if args.phase in ('all', 'overview'):
            from overview_facts import CheckedFact, FactFirstOverview
            service = Verifier(provider)
            outcomes = await asyncio.to_thread(EvidenceAuditor(provider, service.bounded_sources).supports_many,
                valid, [source], 'FAST', question=question, think=False)
            assert all(supported for supported, _ in outcomes), 'Render benchmark needs independently checked facts'
            facts = []
            for index, (claim, (_, auditor)) in enumerate(zip(valid, outcomes)):
                relation = auditor.last_assessment
                facts.append(CheckedFact(str(index), claim.text, source['chunk_id'], 'standard.txt',
                    claim.evidence_quote, relation.claim_subject, relation.claim_predicate,
                    relation.claim_modality, relation.source_modality, 'standard.txt', 'EXPLICIT', claim))
            render_sources = [{**source, 'metadata': {'source': 'standard.txt'}, 'score': 1, 'found_by': ['lexical']}]
            for repeat in range(1, args.repeats + 1):
                for profile in (('FAST', 'BALANCED') if repeat % 2 else ('BALANCED', 'FAST')):
                    async def render(profile=profile):
                        overview = FactFirstOverview(service)
                        answer, verification = await asyncio.to_thread(overview.render, question, facts, render_sources, profile)
                        return {'passed': all(word in answer for word in ('184', '27')) and verification.status == 'SUPPORTED',
                                'answer': answer, 'verification': verification.model_dump(mode='json'), 'report': overview.report}
                    await measure('overview-render', render, original_batch, profile, repeat)

        if args.phase in ('all', 'requests'):
            await asyncio.to_thread(embeddings.manager.ensure_ready)
            root = harness.create_workspace(run / 'fixture')
            before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            workspace = main.workspace_repository.select(str(root))
            main.filesystem_provider.opener = lambda path: (_ for _ in ()).throw(AssertionError('Unexpected open'))
            # Separate index cost from query latency. Workspace graph is not used
            # by these single-document questions, so do not extract unrelated edges.
            main.workspace_vector_index.graph_extractor_factory = None
            main.workspace_vector_index.graph_repository = None
            started = time.perf_counter()
            sync = await asyncio.to_thread(main.workspace_vector_index.sync, workspace.id)
            print(f'Index: {time.perf_counter()-started:.2f}s, {sync}', flush=True)
            cases = [case for case in harness.CASES if case[0] in
                     (args.case or ('fact', 'physical-fact', 'source-injection'))]
            for repeat in range(1, args.repeats + 1):
                for label, question, required, forbidden, _ in cases:
                    for batch in ([args.batch_size] if args.batch_size else ((1, 2) if repeat % 2 else (2, 1))):
                        async def request(question=question, required=required, forbidden=forbidden, label=label):
                            response, _ = await main.assistant_core.process_input(QueryInput(
                                question=question, model=args.profile, workspace_id=workspace.id,
                                use_multi_agent=args.use_multi_agent))
                            checks = {f'contains:{word}': word.casefold() in response.answer.casefold() for word in required}
                            checks.update({f'excludes:{word}': word.casefold() not in response.answer.casefold() for word in forbidden})
                            checks['completed'] = response.task_status.value == 'COMPLETED'
                            check = response.verification
                            checks['review_completed'] = bool(check and check.status != 'UNAVAILABLE')
                            if label in ('fact', 'physical-fact'):
                                checks['facts_supported'] = bool(check and check.status == 'SUPPORTED')
                            if label in ('unavailable-fact', 'mixed-known-missing'):
                                checks['admits_missing_fact'] = any(word in response.answer.casefold() for word in
                                    ('не указан', 'нет информац', 'недостаточно', 'не содерж', 'не найден', 'не предостав'))
                            return {'passed': all(checks.values()), 'checks': checks,
                                    'response': response.model_dump(mode='json')}
                        await measure(label, request, batch, args.profile, repeat)
            assert before == {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
        if args.phase in ('all', 'graph'):
            from chroma_utils import vectorstore, retriever
            from hybrid_context_extractor import HybridRetriever
            from graph_rag import GraphAugmentedRetriever
            from graph_service import GraphService
            await asyncio.to_thread(embeddings.manager.ensure_ready)
            index_started, first_call = time.perf_counter(), len(calls)
            for name, text in [('orion.txt', 'Проект Орион курирует Мирон.'),
                               ('archive.txt', 'Мирон хранит карты северного архива.')]:
                path = run / name
                path.write_text(text, encoding='utf-8')
                await asyncio.to_thread(main.document_service.upload, name, path.read_bytes(), 'file', str(path))
            (run / 'graph-index-cost.json').write_text(json.dumps({
                'seconds': time.perf_counter()-index_started, 'calls': calls[first_call:]}, indent=2), encoding='utf-8')
            print(f'Graph document ingestion: {time.perf_counter()-index_started:.2f}s', flush=True)
            question = 'Проект Орион'
            hybrid = HybridRetriever(retriever, vectorstore=vectorstore, mode='hybrid',
                                     needs_history_aware_input=False, result_k=1)
            augmented = GraphAugmentedRetriever(hybrid, vectorstore, question, ['Проект Орион'], 2)
            for repeat in range(1, args.repeats + 1):
                async def graph_retrieve():
                    docs = await asyncio.to_thread(augmented, {'input': question})
                    graph = augmented.selected_graph()
                    sources = {doc.metadata['chunk_id']: doc.page_content for doc in docs}
                    rows = (graph or {}).get('relations', [])
                    quotes_valid = all(not row.get('evidence') or row['evidence'] in sources.get(row['chunk_id'], '')
                                       for row in rows)
                    def curator_direction(row):
                        source_name, target_name = row['source_name'].casefold(), row['target_name'].casefold()
                        relation = row['relation_type']
                        return (source_name == 'мирон' and 'орион' in target_name
                                and relation in {'CURATES', 'CURATOR_OF', 'MANAGES', 'OVERSEES'}
                            or 'орион' in source_name and target_name == 'мирон'
                                and relation in {'CURATED_BY', 'IS_CURATED_BY', 'HAS_CURATOR'})
                    direction_valid = any(curator_direction(row) for row in rows)
                    entity_ids = [row['id'] for row in (graph or {}).get('entities', [])]
                    started = time.perf_counter()
                    navigation = await asyncio.to_thread(GraphService().entities_subgraph,
                                                         entity_ids, 1, 20, vectorstore) if entity_ids else None
                    return {'passed': bool(rows) and quotes_valid and direction_valid
                                and augmented.retrieval_info['graph_added_source_count'] == 1,
                            'checks': {'literal_quotes': quotes_valid, 'curator_direction': direction_valid}, 'graph': graph,
                            'navigation_seconds': time.perf_counter()-started,
                            'navigation': navigation, 'source_paths': [doc.metadata.get('source') for doc in docs]}
                await measure('graph-retrieval', graph_retrieve, original_batch, 'FAST', repeat)
        groups = defaultdict(list)
        for row in records:
            groups[f"{row['id']}:{row['profile']}:batch{row['batch_size']}:{row['audit_format']}"].append(row)
        summary = {'model': str(get_profile().model_path), 'startup_seconds': startup,
                   'code_hashes': code_hashes,
                   'configuration': asdict(get_profile(args.profile)), 'repeats': args.repeats,
                   'request_profile': args.profile, 'use_multi_agent': args.use_multi_agent,
                   'passed': sum(r['passed'] for r in records), 'total': len(records),
                   'groups': {key: {'median_seconds': statistics.median(r['seconds'] for r in rows),
                                    'samples': len(rows), 'passed': sum(r['passed'] for r in rows),
                                    'llm_calls': [len(r['calls']) for r in rows]}
                              for key, rows in groups.items()}}
        summary['protected_files_unchanged'] = all(
            (hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None) == digest
            for path, digest in digests.items())
        (run / 'benchmark-summary.json').write_text(json.dumps(summary, default=str, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"RESULT {summary['passed']}/{summary['total']}; protected files unchanged: "
              f"{summary['protected_files_unchanged']}; {run}", flush=True)
        return summary['passed'] == summary['total'] and summary['protected_files_unchanged']
    finally:
        settings.EVIDENCE_AUDIT_BATCH_SIZE = original_batch
        settings.EVIDENCE_AUDIT_COMPACT = original_compact
        provider.stream = original_stream
        try:
            provider.close()
        finally:
            try:
                provider.manager.stop()
            finally:
                embeddings.manager.stop()
                from chroma_utils import vectorstore, workspace_vectorstore
                for system in {store._client._system for store in (vectorstore, workspace_vectorstore)}:
                    system.stop()


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', help='Absolute GGUF path; working configuration is unchanged')
    parser.add_argument('--phase', choices=('all', 'audit', 'requests', 'graph', 'overview'), default='all')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--profile', choices=('FAST', 'BALANCED', 'REASONING'), default='FAST',
                        help='Inference profile for request samples')
    parser.add_argument('--use-multi-agent', action='store_true',
                        help='Include the automatic editing used by the ordinary chat UI')
    parser.add_argument('--batch-size', type=int, choices=(1, 2),
                        help='Measure one audit batch size instead of the alternating comparison')
    parser.add_argument('--audit-format', choices=('legacy', 'compact', 'both'),
                        help='Select transport; both interleaves formats in the audit phase')
    parser.add_argument('--case', action='append', choices=('fact', 'physical-fact', 'source-injection',
                        'unavailable-fact', 'mixed-known-missing'), help='Select request scenarios')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    if args.audit_format == 'both' and args.phase != 'audit':
        parser.error('--audit-format both requires --phase audit')
    sys.exit(0 if asyncio.run(benchmark(args)) else 1)
