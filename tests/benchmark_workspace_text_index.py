"""Offline lexical latency/coverage ablation; synthetic documents, isolated SQLite.

No model/server is started; data paths are isolated before application imports.
Both modes use a fresh reader for each query; extraction counts expose cache reuse.
"""
import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import statistics
import sys
import time
from unittest.mock import patch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--documents', type=int, default=600)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if not 510 <= args.documents <= 2000 or args.repeats < 2:
        parser.error('Use 510..2000 documents and at least 2 repeats')
    root = Path(__file__).resolve().parents[1]
    run = root / '.user-evaluation' / time.strftime('workspace-text-index-%Y%m%d-%H%M%S')
    run.mkdir(parents=True, exist_ok=False)
    protected = {path: digest(path) for path in (root/'api/rag_app.db', root/'api/chat-model-selection.json')}
    os.environ.update(KONTUR_DATA_DIR=str(run), RAG_DB_PATH=str(run/'fixture.db'),
                      CHROMA_PATH=str(run/'unused-chroma'), WORKSPACE_TEXT_INDEX_ENABLED='true')
    sys.path.insert(0, str(root/'api'))
    from workspace import FilesystemProvider, WorkspaceRepository, WorkspaceSkill
    def connection():
        conn = sqlite3.connect(run/'fixture.db', timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA foreign_keys=ON')
        return conn
    repository = WorkspaceRepository(connection)
    sources = run/'documents'
    sources.mkdir()
    for number in range(args.documents):
        text = (f'Объект ITEM-{number:04d} имеет метку CODE-{number:04d}.\n' +
                'Нейтральные наблюдения о корпусе и кабельных каналах.\n' * 60)
        (sources/f'doc{number:04d}.md').write_text(text, encoding='utf-8')
    hashes = {path.name:digest(path) for path in sources.iterdir()}
    workspace = repository.select(str(sources))
    filesystem = FilesystemProvider(repository)
    filesystem.sync_index(workspace.id)
    filesystem.text_index.ensure_schema()
    cold_rows = []
    for repeat in range(args.repeats):
        for enabled in ((False, True) if repeat % 2 == 0 else (True, False)):
            # These rows belong only to this script's newly created fixture.
            with closing(connection()) as conn, conn:
                conn.execute('DELETE FROM workspace_text_documents WHERE workspace_id=?', (workspace.id,))
            fs = FilesystemProvider(repository, text_index=None if enabled else False)
            with patch.object(fs.document_reader, '_decode_plain', wraps=fs.document_reader._decode_plain) as extract:
                started = time.perf_counter()
                cold = WorkspaceSkill(fs).relevant_context('Что записано об ITEM-0003?', workspace.id, semantic=False)
                cold_rows.append({'enabled':enabled, 'seconds':time.perf_counter()-started,
                                  'extractions':extract.call_count, 'repeat':repeat+1,
                                  'correct':bool(cold and cold[0]['path']=='doc0003.md')})
    started = time.perf_counter()
    population = filesystem.sync_text_index(workspace.id)
    build_seconds = time.perf_counter()-started
    rows = []
    early, late = 3, args.documents-1
    for repeat in range(args.repeats):
        for enabled in ((False, True) if repeat % 2 == 0 else (True, False)):
            for number in (early, late):
                fs = FilesystemProvider(repository, text_index=None if enabled else False)
                with patch.object(fs.document_reader, '_decode_plain', wraps=fs.document_reader._decode_plain) as extract:
                    started = time.perf_counter()
                    found = WorkspaceSkill(fs).relevant_context(f'Что записано об ITEM-{number:04d}?',
                                                               workspace.id, semantic=False)
                    seconds = time.perf_counter()-started
                expected_path = f'doc{number:04d}.md'
                correct = bool(found and {r['path'] for r in found} == {expected_path} and
                               f'CODE-{number:04d}' in found[0]['content'])
                rows.append({'enabled':enabled, 'case':'early' if number == early else 'late',
                             'repeat':repeat+1, 'seconds':seconds, 'extractions':extract.call_count,
                             'correct':correct, 'coverage':found[0]['coverage'] if found else None})
    groups = {}
    for enabled in (False, True):
        for case in ('early','late'):
            group = [r for r in rows if r['enabled'] == enabled and r['case'] == case]
            groups[f'{"index" if enabled else "scan"}:{case}'] = {
                'median_seconds':statistics.median(r['seconds'] for r in group),
                'correct':sum(r['correct'] for r in group), 'total':len(group),
                'extractions':[r['extractions'] for r in group]}
    summary = {'documents':args.documents,'population':population,'build_seconds':build_seconds,'groups':groups,
               'cold_queries':cold_rows,
               'cold_medians':{str(enabled):statistics.median(r['seconds'] for r in cold_rows if r['enabled']==enabled)
                               for enabled in (False,True)},
               'protected_files_unchanged':all(digest(p)==h for p,h in protected.items()),
               'sources_unchanged':all(digest(sources/name)==value for name,value in hashes.items()),
               'code_hashes':{name:digest(root/'api'/name) for name in (
                   'workspace.py','workspace_document_reader.py','workspace_text_index.py')}}
    (run/'results.json').write_text(json.dumps({'summary':summary,'rows':rows}, ensure_ascii=False, indent=2),encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(run)
    passed = (all(r['correct'] for r in rows if r['enabled'] or r['case']=='early') and
              all(r['extractions']==0 for r in rows if r['enabled']) and
              summary['protected_files_unchanged'] and summary['sources_unchanged'])
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
