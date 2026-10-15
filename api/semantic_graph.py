"""Automatic document links from committed embeddings, shared by map and retrieval.

Links express similarity, never facts. Building this graph does not run a model
or write a second copy of the user's corpus. The bounded cache is invalidated
by vector commits and by changes in the authoritative document/file inventory.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import closing
import hashlib
import logging
from pathlib import Path

import numpy as np

from db_utils import get_db_connection
from index_transactions import consistent_read, index_revision
from settings import SEMANTIC_GRAPH_SIMILARITY, SEMANTIC_GRAPH_NEIGHBORS

logger = logging.getLogger(__name__)
SIMILARITY_THRESHOLD = SEMANTIC_GRAPH_SIMILARITY
NEIGHBORS = SEMANTIC_GRAPH_NEIGHBORS
MAX_DOCUMENTS = 1000
MAX_CHUNKS = 20000
EXPLANATION_CHUNKS = 64
_cache = OrderedDict()


class SemanticDocumentGraph:
    def __init__(self, store, workspace_id=None, *, connection_factory=get_db_connection,
                 filesystem=None):
        self.store = store
        self.workspace_id = workspace_id
        self.connection_factory = connection_factory
        self.filesystem = filesystem

    def inventory(self):
        with closing(self.connection_factory()) as conn:
            if self.workspace_id:
                rows = conn.execute("""SELECT relative_path,content_hash,vector_hash,vector_status
                    FROM assistant_workspace_files WHERE workspace_id=? AND readable=1
                    ORDER BY relative_path""", (self.workspace_id,)).fetchall()
                return {row['relative_path']: {
                    'label': row['relative_path'], 'hash': row['content_hash'],
                    'indexed': bool(row['content_hash'] and row['content_hash'] == row['vector_hash']
                                    and row['vector_status'] == 'INDEXED'),
                } for row in rows}
            return {str(row['id']): {'label': row['filename'] or f'Документ {row["id"]}', 'hash': None, 'indexed': True}
                    for row in conn.execute('SELECT id,filename FROM document_store ORDER BY id')}

    def document_key(self, meta, inventory):
        if self.workspace_id:
            key = meta.get('relative_path')
            record = inventory.get(key)
            expected_key = hashlib.sha256(f'{self.workspace_id}:{key}'.encode()).hexdigest()
            if (meta.get('workspace_id') != self.workspace_id or not record or not record['indexed']
                    or meta.get('content_hash') != record['hash']
                    or meta.get('workspace_file_key') != expected_key):
                return None
            return key
        if meta.get('workspace_id') or meta.get('source_kind') == 'workspace':
            return None
        key = str(meta.get('document_id', meta.get('file_id')))
        return key if key in inventory else None

    @consistent_read
    def snapshot(self):
        inventory = self.inventory()
        signature = (index_revision(), tuple((key, row['label'], row['hash'], row['indexed'])
                                            for key, row in inventory.items()))
        cache_key = (id(self.store), self.workspace_id, self.connection_factory)
        cached = _cache.get(cache_key)
        if cached and cached[0] is self.store and cached[1] == signature:
            _cache.move_to_end(cache_key)
            return cached[2]
        chunks, groups, dimension = {}, {}, None
        truncated = len(inventory) > MAX_DOCUMENTS
        inventory = dict(list(inventory.items())[:MAX_DOCUMENTS])
        if self.store is not None:
            offset = 0
            while offset < MAX_CHUNKS:
                args = {'include': ['embeddings', 'metadatas'], 'limit': 500, 'offset': offset}
                if self.workspace_id:
                    args['where'] = {'workspace_id': self.workspace_id}
                page = self.store.get(**args)
                ids = page.get('ids') or []
                embeddings = page.get('embeddings')
                if not ids or embeddings is None:
                    break
                for chunk_id, meta, embedding in zip(ids, page.get('metadatas') or [], embeddings):
                    meta = meta or {}
                    key = self.document_key(meta, inventory)
                    if key is None or embedding is None:
                        continue
                    vector = np.asarray(embedding, dtype=np.float32)
                    if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
                        continue
                    if dimension is None:
                        dimension = vector.size
                    norm = float(np.linalg.norm(vector))
                    if vector.size != dimension or norm < 1e-8:
                        continue
                    vector = vector / norm
                    chunks[chunk_id] = (key, vector, meta)
                    groups.setdefault(key, []).append(chunk_id)
                offset += len(ids)
                if len(ids) < 500:
                    break
            truncated |= offset >= MAX_CHUNKS
        keys, centroids = [], []
        for key, ids in groups.items():
            centroid = np.mean([chunks[chunk_id][1] for chunk_id in ids], axis=0)
            norm = float(np.linalg.norm(centroid))
            if norm > 1e-8:
                keys.append(key)
                centroids.append(centroid / norm)
        edges = {}
        if centroids:
            matrix = np.asarray(centroids, dtype=np.float32)
            similarities = matrix @ matrix.T
            np.fill_diagonal(similarities, -1)
            for index, key in enumerate(keys):
                for other in np.argsort(-similarities[index], kind='stable')[:NEIGHBORS]:
                    score = float(similarities[index, other])
                    if score >= SIMILARITY_THRESHOLD:
                        pair = tuple(sorted((key, keys[other])))
                        edges[pair] = min(1.0, score)
        snapshot = {'inventory': inventory, 'chunks': chunks, 'groups': groups,
                    'edges': edges, 'truncated': truncated}
        _cache[cache_key] = (self.store, signature, snapshot)
        _cache.move_to_end(cache_key)
        while len(_cache) > 4:
            _cache.popitem(last=False)
        return snapshot

    @consistent_read
    def map(self, limit=200):
        snapshot = self.snapshot()
        degrees = {key: 0 for key in snapshot['inventory']}
        for source, target in snapshot['edges']:
            degrees[source] += 1
            degrees[target] += 1
        keys = sorted(degrees, key=lambda key: (-degrees[key], snapshot['inventory'][key]['label'].casefold()))[:limit]
        selected = set(keys)
        return {
            'nodes': [{'id': key, 'label': snapshot['inventory'][key]['label'],
                       'connections': degrees[key], 'indexed': key in snapshot['groups']} for key in keys],
            'edges': [{'id': f'{source}\x1f{target}', 'source': source, 'target': target,
                       'similarity': score} for (source, target), score in snapshot['edges'].items()
                      if source in selected and target in selected],
            'coverage': {'documents': len(self.inventory()), 'indexed_documents': len(snapshot['groups'])},
            'truncated': snapshot['truncated'] or len(degrees) > limit,
        }

    def _file_is_current(self, key, inventory):
        if not self.workspace_id:
            return True
        if self.filesystem is None:
            return False
        try:
            workspace, _, _ = self.filesystem.read_text_file(self.workspace_id, key)
            path = self.filesystem._resolve(Path(workspace.root_path).resolve(strict=True), key)
            return self.filesystem._hash_file(path) == inventory[key]['hash']
        except (OSError, ValueError, LookupError):
            return False

    @consistent_read
    def explain_link(self, source, target):
        """Show similarity and original passages, without inventing a relation type."""
        snapshot = self.snapshot()
        pair = tuple(sorted((source, target)))
        if pair not in snapshot['edges'] or self.store is None:
            raise LookupError('Связь больше не доступна. Обновите карту.')
        chunks, groups, inventory = snapshot['chunks'], snapshot['groups'], snapshot['inventory']
        left, right = groups[source], groups[target]

        def candidates(ids, other_ids):
            # Rank against the other document, then compare a bounded chunk pair matrix.
            # Short documents are compared in full; large documents report sampling.
            if len(ids) <= EXPLANATION_CHUNKS:
                return ids
            center = np.mean([chunks[key][1] for key in other_ids], axis=0)
            vectors = np.asarray([chunks[key][1] for key in ids], dtype=np.float32)
            order = np.argsort(-(vectors @ center), kind='stable')[:EXPLANATION_CHUNKS]
            return [ids[index] for index in order]

        left_ids, right_ids = candidates(left, right), candidates(right, left)
        matrix = (np.asarray([chunks[key][1] for key in left_ids], dtype=np.float32)
                  @ np.asarray([chunks[key][1] for key in right_ids], dtype=np.float32).T)
        row, column = np.unravel_index(np.argmax(matrix), matrix.shape)
        ids = [left_ids[row], right_ids[column]]
        stored = self.store.get(ids=ids, include=['documents', 'metadatas'])
        records = {key: (text, meta or {}) for key, text, meta in zip(
            stored.get('ids') or [], stored.get('documents') or [], stored.get('metadatas') or [])}
        passages = []
        for key, owner in zip(ids, (source, target)):
            text, meta = records.get(key, ('', {}))
            if (not text or self.document_key(meta, inventory) != owner
                    or not self._file_is_current(owner, inventory)):
                raise LookupError('Источник связи изменился. Переиндексируйте документы.')
            passages.append({'document_id': owner, 'label': inventory[owner]['label'],
                             'chunk_id': key, 'text': text[:1800], 'truncated': len(text) > 1800})
        return {'similarity': snapshot['edges'][pair], 'threshold': SIMILARITY_THRESHOLD,
                'passage_similarity': min(1.0, float(matrix[row, column])), 'passages': passages,
                'sampled': len(left_ids) < len(left) or len(right_ids) < len(right)}

    @consistent_read
    def expand(self, seeds, limit=1, *, question=None):
        """Select a query-directed supplement; retain cosine-only compatibility.

        With a question, require a new lexical match and a valid source path.
        Legacy callers without a question keep the bounded similarity expansion.
        """
        from hybrid_context_extractor import SearchResult
        if not seeds or self.store is None:
            return []
        if question is not None:
            from graph_retrieval import query_index, query_terms, covered_terms, _BROAD, _QUALIFIED, _FORMULA
            if _BROAD.search(question) or _QUALIFIED.search(question) or _FORMULA.search(question):
                return []
            wanted = query_terms(question)
            covered = set.union(*(covered_terms(wanted, query_terms(seed.text)) for seed in seeds))
            if wanted <= covered:
                return []
            return query_index(self).expand(question, seeds, limit)
        snapshot = self.snapshot()
        chunks, inventory = snapshot['chunks'], snapshot['inventory']
        seed_keys = {seed.metadata.get('relative_path') if self.workspace_id
                     else self.document_key(seed.metadata, inventory) for seed in seeds}
        candidates = {}
        for seed in seeds[:3]:
            anchor = chunks.get(seed.chunk_id)
            # Workspace lexical reads have file identity rather than chunk identity.
            if anchor is None and self.workspace_id:
                key = seed.metadata.get('relative_path')
                ids = snapshot['groups'].get(key, [])
                if ids:
                    anchor = chunks[ids[0]]
            if anchor is None:
                continue
            source, vector, _ = anchor
            if not self._file_is_current(source, inventory):
                continue
            for (left, right), weight in snapshot['edges'].items():
                target = right if left == source else left if right == source else None
                if target is None or target in seed_keys:
                    continue
                for chunk_id in snapshot['groups'].get(target, []):
                    score = float(vector @ chunks[chunk_id][1])
                    if score >= SIMILARITY_THRESHOLD and score > candidates.get(chunk_id, (0,))[0]:
                        candidates[chunk_id] = (score, source, weight)
        if not candidates:
            return []
        ranked = sorted(candidates, key=lambda key: (-candidates[key][0], key))[:8]
        stored = self.store.get(ids=ranked, include=['documents', 'metadatas'])
        sources = {key: (text, meta or {}) for key, text, meta in zip(
            stored.get('ids') or [], stored.get('documents') or [], stored.get('metadatas') or [])}
        output, used = [], set()
        for chunk_id in ranked:
            text, meta = sources.get(chunk_id, ('', {}))
            target = self.document_key(meta, inventory)
            if not text or target is None or target in used or not self._file_is_current(target, inventory):
                continue
            score, anchor, weight = candidates[chunk_id]
            document_id = f'workspace:{self.workspace_id}:{target}' if self.workspace_id else int(target)
            metadata = {**meta, 'chunk_id': chunk_id, 'document_id': document_id,
                        'source': inventory[target]['label'], 'graph_anchor': inventory[anchor]['label'],
                        'graph_similarity': weight}
            output.append(SearchResult(document_id, chunk_id, text, metadata, score, ['semantic_graph']))
            used.add(target)
            if len(output) >= limit:
                break
        return output


def automatic_graph_sources(store, seeds, *, question=None, workspace_id=None,
                            connection_factory=get_db_connection, filesystem=None):
    """Graph unavailability must not interrupt ordinary source retrieval."""
    try:
        return SemanticDocumentGraph(store, workspace_id, connection_factory=connection_factory,
                                     filesystem=filesystem).expand(seeds, question=question)
    except Exception:
        logger.debug('Semantic graph unavailable; keeping direct sources', exc_info=True)
        return []
