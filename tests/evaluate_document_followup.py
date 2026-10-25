"""Opt-in replay of file actions from a private chat, on isolated copies.

Read the original SQLite in read-only mode; reuse its completed creation
journal. No inference servers or desktop applications are started or stopped.
Private paths, responses and source copies stay in ignored .user-evaluation/.
"""
import argparse
import asyncio
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


async def evaluate(chat_name):
    run = ROOT / '.user-evaluation' / time.strftime('document-followup-%Y%m%d-%H%M%S')
    run.mkdir(parents=True)
    database = ROOT / 'api/rag_app.db'
    protected = {p: digest(p) for p in (database, ROOT / 'api/chat-model-selection.json')}
    for key, value in {
        'KONTUR_DATA_DIR': str(run), 'RAG_DB_PATH': str(run / 'evaluation.db'),
        'CHROMA_PATH': str(run / 'chroma'), 'LLAMA_AUTOSTART': 'false',
        'VERIFIER_ENABLED': 'false', 'ANONYMIZED_TELEMETRY': 'False',
    }.items():
        os.environ[key] = value
    sys.path.insert(0, str(ROOT / 'api'))
    from assistant_repository import AssistantRepository
    from assistant_models import TaskStatus
    from pydantic_models import QueryInput
    from workspace_intent import asks_last_opened_file, references_created_file

    def readonly():
        conn = sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    # Do not invoke the migration constructor on the original database.
    original_repository = object.__new__(AssistantRepository)
    original_repository.connection_factory = readonly
    with closing(readonly()) as conn:
        chats = [json.loads(row[0]) for row in conn.execute('SELECT data FROM workspace_chats')]
        chat = next(c for c in chats if c.get('title', '').casefold() == chat_name.casefold())
        session = conn.execute('SELECT s.*,w.root_path FROM assistant_sessions s '
            'JOIN assistant_workspaces w ON w.id=s.workspace_id WHERE s.id=?', (chat['session_id'],)).fetchone()
        created = original_repository.last_file_action(session['id'], session['workspace_id'], created=True)
        opened = original_repository.last_file_action(session['id'], session['workspace_id'])
        if not created or not opened:
            raise ValueError('Chat needs a completed creation and a completed file opening')
        historical = [dict(row) for row in conn.execute('SELECT * FROM assistant_actions WHERE task_id=? '
            "AND status='COMPLETED' AND action_type IN ('CREATE_DOCUMENT','CONVERT_DOCUMENT','APPLY_DOCUMENT_CHANGE') "
            'ORDER BY rowid', (created['task_id'],))]
    source_root = Path(session['root_path']).resolve()
    copied = run / 'documents'
    copied.mkdir()
    originals = {}
    for relative in {created['path'], opened['path']}:
        source = source_root / relative
        if not source.resolve().is_relative_to(source_root) or source.is_symlink():
            raise ValueError('Unsafe source path')
        originals[source] = digest(source)
        target = copied / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    (run / 'original-chat.json').write_text(json.dumps(chat, ensure_ascii=False, indent=2), encoding='utf-8')

    import main
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    code_hashes = {p.name: digest(p) for p in (ROOT / 'api').glob('*.py')}
    workspace = main.workspace_repository.select(str(copied))
    repository = main.assistant_repository
    isolated_session = repository.ensure_session()
    seed_task = repository.create_task(isolated_session.id, 'Replayed completed creation journal')
    repository.transition(seed_task, TaskStatus.RUNNING)
    for row in historical:
        inputs = json.loads(row['input'])
        inputs['workspace_id'] = workspace.id
        action = repository.create_action(seed_task.id, row['action_type'], inputs)
        repository.transition(action, TaskStatus.RUNNING)
        repository.transition(action, TaskStatus.COMPLETED, output=json.loads(row['output']))
    repository.transition(seed_task, TaskStatus.COMPLETED)
    captured, rows, inference_attempts = [], [], []
    main.filesystem_provider.opener = lambda path: captured.append(Path(path).relative_to(copied).as_posix())
    questions = [m['content'] for m in chat['messages'] if m['role'] == 'user']
    pronoun = next(q for q in questions if q.strip().casefold() == 'открой его')
    explicit = next(q for q in questions if Path(created['path']).name.casefold() in q.casefold()
                    and 'открой' in q.casefold())
    status = next(q for q in questions if asks_last_opened_file(q))
    corrective = next(q for q in reversed(questions) if references_created_file(q) and 'открыть' in q.casefold())

    def no_inference(*args, **kwargs):
        inference_attempts.append(True)
        raise AssertionError('File action unexpectedly requested model inference')

    async def ask(case, question, expected_path, *, status_only=False):
        before_open = len(captured)
        started = time.perf_counter()
        response, _ = await main.assistant_core.process_input(QueryInput(
            question=question, model='FAST', workspace_id=workspace.id, session_id=isolated_session.id))
        actions = [a.action_type for a in response.actions]
        checks = {
            'completed': response.task_status.value == 'COMPLETED',
            'strategy': response.strategy == ('WORKSPACE_ACTION_STATUS' if status_only else 'WORKSPACE_OPEN'),
            'path_reported': expected_path in response.answer,
            'effect': len(captured) == before_open if status_only else captured[before_open:] == [expected_path],
            'no_read_or_write': not set(actions) & {'READ_FILE', 'CREATE_DOCUMENT', 'APPLY_DOCUMENT_CHANGE', 'CONVERT_DOCUMENT'},
            'no_inference': not inference_attempts,
        }
        rows.append({'id': case, 'question': question, 'seconds': time.perf_counter() - started,
                     'checks': checks, 'passed': all(checks.values()), 'response': response.model_dump(mode='json')})
        print(case, 'PASS' if rows[-1]['passed'] else 'FAIL', round(rows[-1]['seconds'], 3), 's', flush=True)

    try:
        with patch('model_runtime.get_llm_provider', side_effect=no_inference), \
             patch('assistant_core.get_llm_provider', side_effect=no_inference), \
             patch.object(main.assistant_core.conversation_resolver, 'resolve', side_effect=no_inference), \
             patch.object(main.assistant_core.agent_planner, 'plan', side_effect=no_inference):
            main.workspace_repository.bind_session(isolated_session.id, workspace.id,
                session['workspace_path'], created['path'])
            await ask('open-it-after-create', pronoun, created['path'])
            main.workspace_repository.bind_session(isolated_session.id, workspace.id,
                session['workspace_path'], opened['path'])
            await ask('explicit-name-over-stale-directory', explicit, created['path'])
            # Replay the actual previous system-open result without opening a window.
            task = repository.create_task(isolated_session.id, 'Replayed previous open journal')
            repository.transition(task, TaskStatus.RUNNING)
            action = repository.create_action(task.id, 'OPEN_FILE', {'workspace_id': workspace.id, 'path': opened['path']})
            repository.transition(action, TaskStatus.RUNNING)
            repository.transition(action, TaskStatus.COMPLETED, output={'data': {'path': opened['path']}})
            repository.transition(task, TaskStatus.COMPLETED)
            main.workspace_repository.bind_session(isolated_session.id, workspace.id,
                session['workspace_path'], opened['path'])
            await ask('report-actual-previous-open', status, opened['path'], status_only=True)
            await ask('recover-created-from-old-journal', corrective, created['path'])
    finally:
        summary = {'passed': sum(r['passed'] for r in rows), 'total': len(rows),
            'inference_attempts': len(inference_attempts), 'captured_opens': captured,
            'original_database_and_model_selection_unchanged': all(digest(p) == h for p, h in protected.items()),
            'original_sources_unchanged': all(digest(p) == h for p, h in originals.items()),
            'copies_unchanged': all(digest(copied / p.relative_to(source_root)) == h for p, h in originals.items()),
            'code_hashes': code_hashes, 'code_unchanged_during_run': code_hashes == {
                p.name: digest(p) for p in (ROOT / 'api').glob('*.py')}}
        (run / 'document-followup.json').write_text(json.dumps({'summary': summary, 'cases': rows},
            ensure_ascii=False, indent=2), encoding='utf-8')
        from chroma_utils import vectorstore, workspace_vectorstore
        for system in {s._client._system for s in (vectorstore, workspace_vectorstore)}:
            system.stop()
    print('RESULT', summary['passed'], '/', summary['total'], 'inference:', summary['inference_attempts'], flush=True)
    print('Output:', run, flush=True)
    return summary['total'] == 4 and summary['passed'] == 4 and all(summary[k] for k in (
        'original_database_and_model_selection_unchanged', 'original_sources_unchanged', 'copies_unchanged',
        'code_unchanged_during_run'))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chat', required=True)
    args = parser.parse_args()
    sys.exit(0 if asyncio.run(evaluate(args.chat)) else 1)
