"""Resumable desktop migration. Existing user stores are never overwritten."""
from contextlib import ExitStack, closing
import json
import os
from pathlib import Path
import shutil
import sqlite3
import uuid


def _write_state(path, state):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, ensure_ascii=False), encoding='utf-8')
    os.replace(temporary, path)


def _backup_database(source, destination):
    with closing(sqlite3.connect(source, timeout=10)) as reader, closing(sqlite3.connect(destination)) as writer:
        with writer:
            reader.backup(writer)
            if writer.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
                raise RuntimeError('Database snapshot validation failed')


def _copy_item(source, destination):
    if source.is_symlink() or source.is_junction():
        raise ValueError('Migration does not follow filesystem links')
    if source.is_dir():
        destination.mkdir(parents=True, exist_ok=True)
        for child in source.iterdir():
            if child.name.endswith(('-wal', '-shm', '-journal')):
                continue
            _copy_item(child, destination / child.name)
    elif source.suffix in {'.db', '.sqlite', '.sqlite3'}:
        destination.parent.mkdir(parents=True, exist_ok=True)
        _backup_database(source, destination)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        before = source.stat()
        shutil.copy2(source, destination)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or destination.stat().st_size != before.st_size:
            raise RuntimeError('Source changed during migration; close the old application and retry')


def initialise_data(target: Path, legacy: Path | None) -> None:
    target = target.resolve()
    target.mkdir(parents=True, exist_ok=True)
    if legacy is None or not legacy.is_dir():
        return
    legacy = legacy.resolve()
    if legacy == target:
        return
    marker = target / '.legacy-migration.json'
    state = json.loads(marker.read_text(encoding='utf-8')) if marker.exists() else None
    if state and state['state'] == 'complete':
        return
    if state and state['source'] != str(legacy):
        raise RuntimeError('An unfinished migration belongs to another source directory')
    if not state:
        # An installation with its own database is authoritative. Never combine
        # its rows with vectors copied from a different installation.
        existing_store = (target / 'rag_app.db').exists()
        if not existing_store and (target / 'chroma_db').exists():
            raise RuntimeError('Existing Chroma without its document database requires manual recovery; files were preserved')
        names = [name for name in ('.env', 'credentials.json', 'token.json')
                 if (legacy / name).is_file() and not (target / name).exists()]
        if not existing_store:
            names += [name for name in ('rag_app.db', 'chroma_db') if (legacy / name).exists()]
        if (legacy / 'models').is_dir():
            names += ['models/' + path.name for path in (legacy / 'models').iterdir()
                      if path.is_file() and not (target / 'models' / path.name).exists()]
        state = {'state': 'copying', 'source': str(legacy),
                 'stage': '.migration-stage-' + uuid.uuid4().hex,
                 'components': {name: 'pending' for name in names}}
        _write_state(marker, state)
    stage = target / state['stage']
    if stage.parent != target or not stage.name.startswith('.migration-stage-'):
        raise RuntimeError('Invalid migration staging directory')
    if state['state'] == 'copying':
        # Only our verified staging directory is disposable; live stores stay untouched.
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir()
        with ExitStack() as stack:
            for name in ('rag_app.db', 'chroma_db/chroma.sqlite3'):
                source = legacy / name
                if source.is_file():
                    lock = sqlite3.connect(source, timeout=10)
                    stack.callback(lock.close)
                    lock.execute('BEGIN IMMEDIATE')
                    tables = {row[0] for row in lock.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                    if name == 'rag_app.db':
                        if 'index_operations' in tables and lock.execute('SELECT 1 FROM index_operations LIMIT 1').fetchone():
                            raise RuntimeError('Recover pending indexing in the old application before migration')
                        if 'assistant_tasks' in tables and lock.execute(
                                "SELECT 1 FROM assistant_tasks WHERE status IN ('PENDING','RUNNING') LIMIT 1").fetchone():
                            raise RuntimeError('Close or recover active tasks in the old application before migration')
            for name in state['components']:
                _copy_item(legacy / name, stage / name)
        state['state'] = 'promoting'
        _write_state(marker, state)
    for name, status in list(state['components'].items()):
        source, destination = stage / name, target / name
        if status == 'done':
            continue
        if status == 'moving' and not source.exists() and destination.exists():
            state['components'][name] = 'done'
            _write_state(marker, state)
            continue
        if destination.exists():
            raise RuntimeError('Migration target changed; existing files were preserved')
        state['components'][name] = 'moving'
        _write_state(marker, state)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, destination)
        state['components'][name] = 'done'
        _write_state(marker, state)
    state['state'] = 'complete'
    _write_state(marker, state)
    # Empty staging parents contain no user data.
    shutil.rmtree(stage)
