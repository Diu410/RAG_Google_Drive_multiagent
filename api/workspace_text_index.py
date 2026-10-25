"""Disposable, revision-bound extracted documents and lexical passage postings.

The metadata table remains authoritative. A cached extraction is never a grant
to read a path, and FTS results are hints, not evidence verification.
"""
from contextlib import closing
from dataclasses import asdict
import json
import re
import threading

from source_grounding import factual_source_text
from text_normalization import lexical_key
from workspace_document_reader import ExtractedDocument, SourceBlock

FORMAT_VERSION = 1
_WORDS = re.compile(r'[\w-]+', re.UNICODE)
_WIDTH, _STEP = 2400, 2160


def passage_rows(path, document):
    """Keep literal offsets; normalize full words before assigning windows."""
    text = document.text
    starts = list(range(0, max(1, len(text)), _STEP))
    terms = [set() for _ in starts]
    offset = 0
    for line in text.splitlines(keepends=True):
        if factual_source_text(line):
            for match in _WORDS.finditer(line):
                term = lexical_key(match[0])
                if len(match[0]) < 3:
                    continue
                position = offset + match.start()
                index = min(position // _STEP, len(starts) - 1)
                terms[index].add(term)
                if index and position < starts[index - 1] + _WIDTH:
                    terms[index - 1].add(term)
        offset += len(line)
    path_terms = ' '.join(sorted({lexical_key(word) for word in _WORDS.findall(path)}))
    return [(start, min(len(text), start + _WIDTH), ' '.join(sorted(words)), path_terms)
            for start, words in zip(starts, terms)]


class WorkspaceTextIndex:
    def __init__(self, connection_factory, max_bytes=64 * 1024 * 1024):
        self.connection_factory = connection_factory
        self.max_bytes = max_bytes
        self._ready = False
        self._lock = threading.Lock()

    def ensure_schema(self):
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            with closing(self.connection_factory()) as conn, conn:
                conn.execute('BEGIN IMMEDIATE')
                conn.execute('''CREATE TABLE IF NOT EXISTS workspace_text_documents (
                    id INTEGER PRIMARY KEY, workspace_id TEXT NOT NULL, relative_path TEXT NOT NULL,
                    revision TEXT NOT NULL, format_version INTEGER NOT NULL, text TEXT NOT NULL,
                    blocks TEXT NOT NULL, plain_text INTEGER NOT NULL, byte_count INTEGER NOT NULL,
                    UNIQUE(workspace_id,relative_path),
                    FOREIGN KEY(workspace_id,relative_path) REFERENCES assistant_workspace_files(workspace_id,relative_path)
                    ON DELETE CASCADE)''')
                conn.execute('''CREATE TABLE IF NOT EXISTS workspace_text_passages (
                    id INTEGER PRIMARY KEY, document_id INTEGER NOT NULL REFERENCES workspace_text_documents(id)
                    ON DELETE CASCADE, start INTEGER NOT NULL, end INTEGER NOT NULL,
                    terms TEXT NOT NULL, path_terms TEXT NOT NULL)''')
                conn.execute('CREATE INDEX IF NOT EXISTS workspace_text_passages_doc ON workspace_text_passages(document_id)')
                conn.execute('''CREATE VIRTUAL TABLE IF NOT EXISTS workspace_text_fts USING fts5(
                    terms, path_terms, content='workspace_text_passages', content_rowid='id',
                    tokenize="unicode61 remove_diacritics 0 tokenchars '-_'")''')
                conn.execute('''CREATE TRIGGER IF NOT EXISTS workspace_text_insert AFTER INSERT ON workspace_text_passages
                    BEGIN INSERT INTO workspace_text_fts(rowid,terms,path_terms) VALUES(new.id,new.terms,new.path_terms); END''')
                conn.execute('''CREATE TRIGGER IF NOT EXISTS workspace_text_delete AFTER DELETE ON workspace_text_passages
                    BEGIN INSERT INTO workspace_text_fts(workspace_text_fts,rowid,terms,path_terms)
                    VALUES('delete',old.id,old.terms,old.path_terms); END''')
                conn.execute('''CREATE TABLE IF NOT EXISTS workspace_text_failures (
                    workspace_id TEXT NOT NULL, relative_path TEXT NOT NULL, revision TEXT NOT NULL,
                    PRIMARY KEY(workspace_id,relative_path),
                    FOREIGN KEY(workspace_id,relative_path) REFERENCES assistant_workspace_files(workspace_id,relative_path)
                    ON DELETE CASCADE)''')
            self._ready = True

    @staticmethod
    def prune(conn, workspace_id):
        # Run inside the existing metadata transaction, without forcing a new
        # schema on old/disabled installations. Incomplete scans keep file rows.
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='workspace_text_documents'").fetchone():
            return
        conn.execute('''DELETE FROM workspace_text_documents AS d WHERE workspace_id=? AND NOT EXISTS (
            SELECT 1 FROM assistant_workspace_files f WHERE f.workspace_id=d.workspace_id
            AND f.relative_path=d.relative_path AND f.readable=1 AND f.revision=d.revision)''', (workspace_id,))
        conn.execute('''DELETE FROM workspace_text_failures AS d WHERE workspace_id=? AND NOT EXISTS (
            SELECT 1 FROM assistant_workspace_files f WHERE f.workspace_id=d.workspace_id
            AND f.relative_path=d.relative_path AND f.readable=1 AND f.revision=d.revision)''', (workspace_id,))

    def get(self, workspace_id, path, revision):
        self.ensure_schema()
        with closing(self.connection_factory()) as conn:
            row = conn.execute('''SELECT d.* FROM workspace_text_documents d JOIN assistant_workspace_files f
                ON f.workspace_id=d.workspace_id AND f.relative_path=d.relative_path
                WHERE d.workspace_id=? AND d.relative_path=? AND d.revision=? AND d.format_version=?
                AND f.readable=1 AND f.revision=d.revision''', (workspace_id,path,revision,FORMAT_VERSION)).fetchone()
        if row is None:
            return None
        blocks = tuple(SourceBlock(**block) for block in json.loads(row['blocks']))
        if any(not 0 <= block.start <= block.end <= len(row['text']) for block in blocks):
            return None
        return ExtractedDocument(row['revision'],row['text'],blocks,bool(row['plain_text']))

    def put(self, workspace_id, path, document, validate):
        self.ensure_schema()
        size = len(document.text.encode('utf-8'))
        if size > self.max_bytes:
            return False
        passages = passage_rows(path,document)
        with closing(self.connection_factory()) as conn, conn:
            conn.execute('BEGIN IMMEDIATE')
            row = conn.execute('''SELECT revision,readable FROM assistant_workspace_files
                WHERE workspace_id=? AND relative_path=?''', (workspace_id,path)).fetchone()
            if not row or not row['readable'] or row['revision'] != document.revision:
                return False
            old = conn.execute('SELECT revision,format_version,byte_count FROM workspace_text_documents '
                               'WHERE workspace_id=? AND relative_path=?', (workspace_id,path)).fetchone()
            if old and old['revision'] == document.revision and old['format_version'] == FORMAT_VERSION:
                validate()
                return True
            used = conn.execute('SELECT COALESCE(SUM(byte_count),0) FROM workspace_text_documents WHERE workspace_id=?',
                                (workspace_id,)).fetchone()[0]
            if used - (old['byte_count'] if old else 0) + size > self.max_bytes:
                return False
            validate()
            conn.execute('DELETE FROM workspace_text_documents WHERE workspace_id=? AND relative_path=?', (workspace_id,path))
            cursor = conn.execute('''INSERT INTO workspace_text_documents
                (workspace_id,relative_path,revision,format_version,text,blocks,plain_text,byte_count)
                VALUES(?,?,?,?,?,?,?,?)''', (workspace_id,path,document.revision,FORMAT_VERSION,document.text,
                json.dumps([asdict(block) for block in document.blocks],ensure_ascii=False),int(document.plain_text),size))
            conn.executemany('INSERT INTO workspace_text_passages(document_id,start,end,terms,path_terms) VALUES(?,?,?,?,?)',
                             [(cursor.lastrowid,*passage) for passage in passages])
            conn.execute('DELETE FROM workspace_text_failures WHERE workspace_id=? AND relative_path=?', (workspace_id,path))
            validate()
        return True

    def revisions(self, workspace_id):
        self.ensure_schema()
        with closing(self.connection_factory()) as conn:
            return {row['relative_path']:row['revision'] for row in conn.execute('''SELECT d.relative_path,d.revision
                FROM workspace_text_documents d JOIN assistant_workspace_files f
                ON f.workspace_id=d.workspace_id AND f.relative_path=d.relative_path
                WHERE d.workspace_id=? AND d.format_version=? AND f.readable=1 AND f.revision=d.revision''',
                (workspace_id,FORMAT_VERSION))}

    def failures(self, workspace_id):
        self.ensure_schema()
        with closing(self.connection_factory()) as conn:
            return {row['relative_path'] for row in conn.execute('SELECT relative_path FROM workspace_text_failures WHERE workspace_id=?',
                                                                (workspace_id,))}

    def mark_failed(self, workspace_id, path, revision):
        self.ensure_schema()
        with closing(self.connection_factory()) as conn, conn:
            conn.execute('''INSERT INTO workspace_text_failures(workspace_id,relative_path,revision)
                SELECT workspace_id,relative_path,revision FROM assistant_workspace_files
                WHERE workspace_id=? AND relative_path=? AND revision=? AND readable=1
                ON CONFLICT(workspace_id,relative_path) DO UPDATE SET revision=excluded.revision''', (workspace_id,path,revision))

    def search(self, workspace_id, terms, allowed):
        self.ensure_schema()
        if not terms or not allowed:
            return []
        # Never splice a user's FTS expression or SQL into MATCH. Normalized
        # literal tokens use the same exact-key semantics as the previous scan.
        expression = ' OR '.join('"'+term.replace('"','""')+'"' for term in sorted(terms))
        candidates = {}
        with closing(self.connection_factory()) as conn:
            conn.execute('CREATE TEMP TABLE allowed_text_paths(path TEXT PRIMARY KEY, revision TEXT NOT NULL)')
            conn.executemany('INSERT INTO allowed_text_paths VALUES(?,?)', allowed.items())
            rows = conn.execute('''SELECT d.relative_path,d.revision,p.terms,p.path_terms,p.start,p.end
                FROM workspace_text_fts JOIN workspace_text_passages p ON p.id=workspace_text_fts.rowid
                JOIN workspace_text_documents d ON d.id=p.document_id
                JOIN allowed_text_paths a ON a.path=d.relative_path AND a.revision=d.revision
                JOIN assistant_workspace_files f ON f.workspace_id=d.workspace_id AND f.relative_path=d.relative_path
                WHERE workspace_text_fts MATCH ? AND d.workspace_id=? AND d.format_version=?
                AND f.readable=1 AND f.revision=d.revision''', (expression,workspace_id,FORMAT_VERSION))
            for row in rows:
                path = row['relative_path']
                if allowed.get(path) != row['revision']:
                    continue
                candidate = candidates.setdefault(path, {'path':path,'content':'','matched':set(),'path_matched':set()})
                candidate['matched'].update(terms & set(row['terms'].split()))
                candidate['path_matched'].update(terms & set(row['path_terms'].split()))
        output = []
        for candidate in candidates.values():
            candidate['score'] = len(candidate['matched']) + 3 * len(candidate.pop('path_matched'))
            candidate['matched'] = sorted(candidate['matched'])
            if candidate['score']:
                output.append(candidate)
        return sorted(output,key=lambda item:(-item['score'],item['path'].casefold()))
