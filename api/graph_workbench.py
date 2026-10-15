"""Scoped graph navigation and authoritative source selection for chat.

Edges are navigation hints, never evidence of a fact by themselves. Only their
current source chunks, with matching quotes and namespace, reach generation.
"""
from collections import deque
from contextlib import closing
from pathlib import Path

from assistant_models import AssistantError
from db_utils import get_db_connection
from entity_normalization import normalize_entity_name
from index_transactions import consistent_read
from pydantic_models import SearchResult


class GraphWorkbench:
    def __init__(self, workspace_id=None, *, connection_factory=get_db_connection,
                 store=None, filesystem=None):
        self.workspace_id = workspace_id
        self.connection_factory = connection_factory
        self.store = store
        self.filesystem = filesystem
        self.label = 'Общая библиотека'
        if workspace_id:
            if filesystem is None:
                raise AssistantError('Граф папки недоступен.', 503)
            try:
                self.label = filesystem.repository.get(workspace_id).name
            except LookupError as exc:
                raise AssistantError('Папка больше не доступна.', 404) from exc
            # Refresh metadata, without embeddings or model calls.
            filesystem.sync_index(workspace_id)

    @property
    def scope(self):
        return {'workspace_id': self.workspace_id, 'label': self.label}

    def _query(self, condition='1=1', values=(), limit=51):
        if self.workspace_id:
            sql = """SELECT r.*, s.display_name source_name, s.entity_type source_type,
                t.display_name target_name, t.entity_type target_type,
                f.content_hash current_hash FROM workspace_graph_relations r
                JOIN workspace_graph_entities s ON s.id=r.source_entity_id AND s.workspace_id=r.workspace_id
                JOIN workspace_graph_entities t ON t.id=r.target_entity_id AND t.workspace_id=r.workspace_id
                JOIN assistant_workspace_files f ON f.workspace_id=r.workspace_id AND f.relative_path=r.document_name
                WHERE r.workspace_id=? AND f.readable=1 AND f.content_hash=f.vector_hash
                AND f.content_hash=f.graph_hash AND f.vector_status='INDEXED' AND f.graph_status='INDEXED'"""
            args = [self.workspace_id]
        else:
            sql = """SELECT r.*, s.display_name source_name, s.entity_type source_type,
                t.display_name target_name, t.entity_type target_type, d.filename document_name
                FROM knowledge_relations r JOIN knowledge_entities s ON s.id=r.source_entity_id
                JOIN knowledge_entities t ON t.id=r.target_entity_id
                JOIN document_store d ON d.id=r.document_id WHERE 1=1"""
            args = []
        sql += " AND r.relation_type!='MENTIONS' AND trim(r.evidence)!='' AND (" + condition + ')'
        with closing(self.connection_factory()) as conn:
            return [dict(row) for row in conn.execute(
                sql + ' ORDER BY r.confidence DESC,r.id LIMIT ?', (*args, *values, limit))]

    def _entities(self, ids=None, query='', limit=20):
        table = 'workspace_graph_entities' if self.workspace_id else 'knowledge_entities'
        relation_table = 'workspace_graph_relations' if self.workspace_id else 'knowledge_relations'
        conditions, args = [], []
        if self.workspace_id:
            conditions.append('e.workspace_id=?')
            args.append(self.workspace_id)
        if ids is not None:
            if not ids:
                return []
            conditions.append('e.id IN (' + ','.join('?' for _ in ids) + ')')
            args.extend(ids)
        if query:
            value = normalize_entity_name(query).replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
            conditions.append("e.canonical_name LIKE ? ESCAPE '\\'")
            args.append('%' + value + '%')
        if ids is None:
            conditions.append("e.entity_type!='DOCUMENT'")
        relation_scope = 'AND r.workspace_id=e.workspace_id' if self.workspace_id else ''
        with closing(self.connection_factory()) as conn:
            rows = conn.execute(f"""SELECT e.*, (SELECT count(*) FROM {relation_table} r
                WHERE (r.source_entity_id=e.id OR r.target_entity_id=e.id)
                AND r.relation_type!='MENTIONS' AND trim(r.evidence)!='' {relation_scope}) degree
                FROM {table} e WHERE {' AND '.join(conditions)} ORDER BY degree DESC,e.display_name,e.id LIMIT ?""",
                (*args, limit)).fetchall()
        return [dict(row) for row in rows]

    def entities(self, query='', limit=20):
        return [{'id': row['id'], 'label': row['display_name'], 'type': row['entity_type'],
                 'connections': row['degree']} for row in self._entities(query=query, limit=limit)]

    def coverage(self):
        """Index coverage is separate from the bounded, source-checked map."""
        with closing(self.connection_factory()) as conn:
            if self.workspace_id:
                row = conn.execute("""SELECT count(*) documents,
                    coalesce(sum(graph_status='INDEXED' AND vector_status='INDEXED'
                        AND content_hash=graph_hash AND content_hash=vector_hash),0) indexed_documents,
                    coalesce(sum(graph_status='FAILED' OR vector_status='FAILED'),0) failed_documents
                    FROM assistant_workspace_files WHERE workspace_id=? AND readable=1""",
                    (self.workspace_id,)).fetchone()
            else:
                row = conn.execute("""SELECT count(*) documents,
                    coalesce(sum(EXISTS(SELECT 1 FROM knowledge_relations r WHERE r.document_id=d.id
                        AND r.relation_type!='MENTIONS' AND trim(r.evidence)!='')),0) indexed_documents,
                    0 failed_documents FROM document_store d""").fetchone()
        return {**dict(row), 'source_index_available': self.store is not None}

    def _sources(self, rows, *, check_files=False):
        if self.store is None or not rows:
            return [], []
        ids = list(dict.fromkeys(row['chunk_id'] for row in rows))
        stored = self.store.get(ids=ids, include=['documents', 'metadatas'])
        chunks = {key: (text, meta or {}) for key, text, meta in zip(
            stored.get('ids') or [], stored.get('documents') or [], stored.get('metadatas') or [])}
        valid, results, checked_files = [], {}, {}
        for row in rows:
            text, meta = chunks.get(row['chunk_id'], ('', {}))
            if not text or not ' '.join(row['evidence'].casefold().split()) in ' '.join(text.casefold().split()):
                continue
            if self.workspace_id:
                from workspace_vector_index import WorkspaceVectorIndex
                if (meta.get('workspace_id') != self.workspace_id
                        or meta.get('relative_path') != row['document_name']
                        or meta.get('workspace_file_key') != row['file_key']
                        or row['file_key'] != WorkspaceVectorIndex.file_key(self.workspace_id, row['document_name'])
                        or meta.get('content_hash') != row['current_hash']):
                    continue
                if check_files:
                    path = row['document_name']
                    if path not in checked_files:
                        try:
                            workspace, _, _ = self.filesystem.read_text_file(self.workspace_id, path)
                            resolved = self.filesystem._resolve(Path(workspace.root_path).resolve(strict=True), path)
                            checked_files[path] = self.filesystem._hash_file(resolved) == row['current_hash']
                        except (OSError, ValueError, LookupError):
                            checked_files[path] = False
                    if not checked_files[path]:
                        continue
                document_id = f'workspace:{self.workspace_id}:{row["document_name"]}'
            else:
                document_id = row['document_id']
                if (meta.get('workspace_id') or meta.get('source_kind') == 'workspace'
                        or str(meta.get('document_id', meta.get('file_id'))) != str(document_id)):
                    continue
            metadata = {**meta, 'chunk_id': row['chunk_id'], 'document_id': document_id,
                        'source': row['document_name']}
            results[row['chunk_id']] = SearchResult(document_id=document_id, chunk_id=row['chunk_id'],
                text=text, metadata=metadata, score=1.0, found_by=['graph_selection'])
            valid.append({**row, 'document_id': document_id, '_metadata': metadata, '_text': text})
        return valid, list(results.values())

    def _payload(self, rows, centers=(), *, truncated=False, **extra):
        nodes = {}
        edges = []
        for row in rows:
            for side in ('source', 'target'):
                identifier = row[f'{side}_entity_id']
                nodes[identifier] = {'id': f'entity:{identifier}', 'entity_id': identifier,
                    'label': row[f'{side}_name'], 'type': row[f'{side}_type'], 'metadata': {}}
            meta = row.get('_metadata', {})
            edges.append({'id': f'relation:{row["id"]}', 'source': f'entity:{row["source_entity_id"]}',
                'target': f'entity:{row["target_entity_id"]}', 'relation': row['relation_type'],
                'provenance': {'document_id': row['document_id'], 'document_name': row['document_name'],
                    'chunk_id': row['chunk_id'], 'evidence': row['evidence'], 'text': row.get('_text', '')[:4000],
                    'page': meta.get('page'), 'section': meta.get('section')}})
        for row in self._entities(ids=list(centers), limit=20):
            nodes.setdefault(row['id'], {'id': f'entity:{row["id"]}', 'entity_id': row['id'],
                'label': row['display_name'], 'type': row['entity_type'], 'metadata': {}})
        return {'nodes': list(nodes.values()), 'edges': edges,
                'center_ids': [f'entity:{key}' for key in centers], 'scope': self.scope,
                'truncated': truncated, 'limit': 50, 'coverage': self.coverage(), **extra}

    @consistent_read
    def explore(self, entity_ids=(), depth=1, document_id=None):
        if entity_ids and len(self._entities(ids=entity_ids, limit=20)) != len(set(entity_ids)):
            raise AssistantError('Объект не найден в выбранном источнике.', 404)
        rows, seen, frontier, truncated = [], set(), set(entity_ids), False
        for _ in range(depth if entity_ids else 1):
            condition, values = '1=1', ()
            if frontier:
                marks = ','.join('?' for _ in frontier)
                condition = f'r.source_entity_id IN ({marks}) OR r.target_entity_id IN ({marks})'
                values = (*frontier, *frontier)
            elif document_id is not None:
                condition = 'r.document_name=?' if self.workspace_id else 'r.document_id=?'
                values = (document_id,)
            candidates = self._query(condition, values, 201)
            valid, _ = self._sources(candidates)
            truncated |= len(candidates) == 201
            next_frontier = set()
            for row in valid:
                if row['id'] not in seen:
                    rows.append(row)
                    seen.add(row['id'])
                    next_frontier.update((row['source_entity_id'], row['target_entity_id']))
            frontier = next_frontier - frontier
            if len(rows) >= 50 or not frontier:
                break
        return self._payload(rows[:50], entity_ids, truncated=truncated or len(rows) > 50,
                             unavailable_sources=self.store is None)

    @consistent_read
    def connect(self, source_id, target_id, max_hops=4):
        if len(self._entities(ids=list({source_id, target_id}), limit=2)) != len({source_id, target_id}):
            raise AssistantError('Объекты не найдены в выбранном источнике.', 404)
        queue, visited, budget, truncated = deque([(source_id, [])]), {source_id}, 0, False
        while queue and budget < 2000:
            current, path = queue.popleft()
            if current == target_id:
                return self._payload(path, (source_id, target_id), truncated=truncated, path_found=True)
            if len(path) >= max_hops:
                continue
            candidates = self._query('r.source_entity_id=? OR r.target_entity_id=?', (current, current), 101)
            budget += len(candidates)
            truncated |= len(candidates) == 101
            valid, _ = self._sources(candidates)
            for row in valid:
                following = row['target_entity_id'] if row['source_entity_id'] == current else row['source_entity_id']
                if following not in visited:
                    visited.add(following)
                    queue.append((following, path + [row]))
        return self._payload([], (source_id, target_id), truncated=truncated or bool(queue), path_found=False)

    @consistent_read
    def selection(self, relation_ids):
        marks = ','.join('?' for _ in relation_ids)
        rows = self._query(f'r.id IN ({marks})', relation_ids, 20)
        valid, sources = self._sources(rows, check_files=True)
        if len(valid) != len(set(relation_ids)):
            raise AssistantError('Связь или её источник изменились. Обновите граф и выберите связь снова.', 409)
        if sum(len(source.text) for source in sources) > 120000:
            raise AssistantError('Выберите меньше связей для одного вопроса.', 422)
        entities = {row[f'{side}_entity_id']: {'id': row[f'{side}_entity_id'],
            'display_name': row[f'{side}_name'], 'entity_type': row[f'{side}_type']}
            for row in valid for side in ('source', 'target')}
        graph = {'scope': self.scope, 'entities': list(entities.values()),
                 'relations': [{key: value for key, value in row.items() if not key.startswith('_')} for row in valid],
                 'documents': list({row['document_id']: {'id': row['document_id'], 'name': row['document_name']}
                                    for row in valid}.values()),
                 'retrieval': {'selected_source_count': len(sources), 'graph_added_source_count': len(sources)}}
        return {'sources': sources, 'graph': graph}


def graph_workbench(workspace_id=None, *, filesystem=None, connection_factory=get_db_connection):
    if workspace_id is None:
        from graph_repository import GraphRepository
        GraphRepository(connection_factory)
    from chroma_utils import vectorstore, workspace_vectorstore
    if workspace_id and filesystem is None:
        from workspace import FilesystemProvider, WorkspaceRepository
        filesystem = FilesystemProvider(WorkspaceRepository(connection_factory))
    return GraphWorkbench(workspace_id, connection_factory=connection_factory,
        store=workspace_vectorstore if workspace_id else vectorstore, filesystem=filesystem)
