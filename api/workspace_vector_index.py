"""Incremental Chroma namespace for local workspace files."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from langchain_core.documents import Document
from index_transactions import IndexTransactions, consistent_read, invalidate_index, serialized_mutation, index_lock
from file_policy import is_secret_file
from workspace_source_scope import WorkspaceSourceScope
from file_revision import file_revision


class SourceChangedDuringIndex(RuntimeError):
    pass


def index_file_state(item) -> str:
    """Mutually exclusive states; the recorded metadata scan owns freshness."""
    if not item.content_hash:
        return 'UNAVAILABLE'
    if item.vector_status == 'INDEXED' and item.vector_hash == item.content_hash:
        return 'INDEXED'
    if item.vector_status == 'FAILED':
        return 'FAILED'
    return 'STALE' if item.vector_hash else 'PENDING'


@dataclass
class WorkspaceVectorSync:
    indexed: int = 0
    unchanged: int = 0
    failed: int = 0
    deleted_chunks: int = 0
    graph_indexed: int = 0
    graph_failed: int = 0
    deleted_graph_files: int = 0
    unavailable: bool = False
    skipped: int = 0
    scan_truncated: bool = False
    scan_errors: int = 0
    cleanup_deferred: bool = False
    issues: list[dict] = field(default_factory=list)
    issues_truncated: bool = False
    text_index: dict = field(default_factory=dict)


class WorkspaceVectorIndex:
    def __init__(self, filesystem, vectorstore_getter=None, graph_repository=None,
                 graph_extractor_factory=None):
        self.filesystem = filesystem
        self.repository = filesystem.repository
        self.vectorstore_getter = vectorstore_getter or self._default_vectorstore
        self.graph_repository = graph_repository
        self.graph_extractor_factory = graph_extractor_factory

    @staticmethod
    def _default_vectorstore():
        from chroma_utils import workspace_vectorstore
        return workspace_vectorstore

    @staticmethod
    def file_key(workspace_id: str, relative_path: str) -> str:
        return hashlib.sha256(f"{workspace_id}:{relative_path}".encode("utf-8")).hexdigest()

    @consistent_read
    def status(self, workspace_id: str) -> dict:
        self.repository.get(workspace_id)
        records = [item for item in self.repository.indexed_files(workspace_id) if item.readable]
        states = [(item, index_file_state(item)) for item in records]
        issues = [{'path': item.relative_path, 'state': state} for item, state in states if state != 'INDEXED']
        return {
            'readable_documents': len(records),
            'indexed_documents': sum(state == 'INDEXED' for _, state in states),
            'pending_documents': sum(state == 'PENDING' for _, state in states),
            'changed_documents': sum(state == 'STALE' for _, state in states),
            'failed_documents': sum(state == 'FAILED' for _, state in states),
            'unavailable_documents': sum(state == 'UNAVAILABLE' for _, state in states),
            'issues': issues[:50], 'issues_truncated': len(issues) > 50,
            'text_index': self.filesystem.text_index_status(workspace_id, records),
        }

    @serialized_mutation
    def sync(self, workspace_id: str, *, extract_relations: bool = True,
             mode: Literal['update', 'retry_failed'] = 'update') -> WorkspaceVectorSync:
        if mode not in {'update', 'retry_failed'}:
            raise ValueError('Unknown workspace indexing mode')
        scan = self.filesystem.sync_index(workspace_id)
        text_result = self.filesystem.sync_text_index(workspace_id, mode)
        store = self.vectorstore_getter()
        result = WorkspaceVectorSync(scan_truncated=scan.truncated, scan_errors=len(scan.errors),
            cleanup_deferred=bool(scan.truncated or scan.errors), text_index=text_result)
        if store is None:
            result.unavailable = True
            return result
        transactions = IndexTransactions(self.repository.connection_factory)
        transactions.recover({'workspace_collection': store})
        records = self.repository.indexed_files(workspace_id)
        root = Path(self.repository.get(workspace_id).root_path)
        active_keys = {self.file_key(workspace_id, item.relative_path) for item in records if item.readable}
        stored = store.get(where={"workspace_id": workspace_id}, include=["metadatas"])
        stale_ids = [storage_id for storage_id, metadata in zip(
            stored.get("ids") or [], stored.get("metadatas") or []
        ) if (metadata or {}).get("workspace_file_key") not in active_keys]
        if stale_ids and not result.cleanup_deferred and mode == 'update':
            with index_lock:
                store.delete(ids=stale_ids)
                invalidate_index()
            result.deleted_chunks = len(stale_ids)
        if self.graph_repository is not None and not result.cleanup_deferred and mode == 'update':
            result.deleted_graph_files = self.graph_repository.delete_missing(workspace_id, active_keys)

        from formula_processing import split_structured_documents
        for item in records:
            if not item.readable or not item.content_hash:
                continue
            if mode == 'retry_failed' and item.vector_status != 'FAILED' and not (
                    extract_relations and item.graph_status == 'FAILED'):
                result.skipped += 1
                continue
            vector_current = item.vector_status == "INDEXED" and item.vector_hash == item.content_hash
            graph_current = (not extract_relations or self.graph_repository is None or
                             (item.graph_status == "INDEXED" and item.graph_hash == item.content_hash))
            if vector_current and graph_current:
                result.unchanged += 1
                continue
            key = self.file_key(workspace_id, item.relative_path)
            def check_source():
                path = self.filesystem._resolve(root, item.relative_path)
                if file_revision(path) != item.revision:
                    raise SourceChangedDuringIndex('Source changed during indexing; retry required')
            stage = 'EXTRACTION_FAILED'
            try:
                check_source()
                _, _, content = self.filesystem.read_text_file(workspace_id, item.relative_path)
                check_source()
                chunks = split_structured_documents([Document(page_content=content, metadata={
                    "workspace_id": workspace_id, "workspace_file_key": key,
                    "relative_path": item.relative_path, "source": item.relative_path,
                    "source_kind": "workspace", "content_hash": item.content_hash,
                    "revision": item.revision,
                })])
                chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
                ids = [f"workspace:{key}:{index}:{hashlib.sha256(chunk.page_content.encode('utf-8')).hexdigest()[:16]}"
                       for index, chunk in enumerate(chunks)]
                for chunk_id, chunk in zip(ids, chunks):
                    chunk.metadata.update({
                        "chunk_id": chunk_id,
                        "document_id": f"workspace:{workspace_id}:{item.relative_path}",
                    })
                if not vector_current:
                    stage = 'EMBEDDING_FAILED'
                    def commit(conn):
                        check_source()
                        updated = conn.execute('''UPDATE assistant_workspace_files SET vector_hash=?,
                            vector_status='INDEXED', vector_error=NULL
                            WHERE workspace_id=? AND relative_path=? AND content_hash=? AND revision=?''',
                            (item.content_hash, workspace_id, item.relative_path, item.content_hash, item.revision))
                        if updated.rowcount != 1:
                            raise SourceChangedDuringIndex('Source metadata changed during indexing; retry required')
                    transactions.replace(
                        store, 'workspace_collection', {"workspace_file_key": key}, chunks, ids, commit=commit
                    )
                    result.indexed += 1
                if extract_relations and self.graph_repository is not None and not graph_current:
                    try:
                        if vector_current:
                            # A graph retry must use the committed revision,
                            # including its storage IDs, rather than newly split
                            # chunks whose IDs do not exist in Chroma.
                            stored_chunks = store.get(
                                where={"workspace_file_key": key},
                                include=["documents", "metadatas"],
                            )
                            chunks = [Document(page_content=content, metadata={
                                **(metadata or {}), "chunk_id": chunk_id,
                            }) for chunk_id, content, metadata in zip(
                                stored_chunks.get("ids") or [],
                                stored_chunks.get("documents") or [],
                                stored_chunks.get("metadatas") or [],
                            )]
                            if not chunks:
                                raise RuntimeError('Committed vector chunks are unavailable; retry required')
                        if self.graph_extractor_factory:
                            extractor = self.graph_extractor_factory(workspace_id)
                        else:
                            from graph_extraction import GraphExtractor
                            from workspace_graph import WorkspaceGraphAdapter
                            extractor = GraphExtractor(repository=WorkspaceGraphAdapter(
                                self.graph_repository, workspace_id
                            ))
                        success = extractor.extract_document(chunks, key, item.relative_path)
                        self.repository.update_graph_state(
                            workspace_id, item.relative_path,
                            content_hash=item.content_hash if success else None,
                            status="INDEXED" if success else getattr(extractor, 'last_status', 'FAILED'),
                            error=None if success else "Graph extraction failed",
                        )
                        if success:
                            result.graph_indexed += 1
                        else:
                            result.graph_failed += 1
                    except Exception as graph_exc:
                        self.repository.update_graph_state(
                            workspace_id, item.relative_path, content_hash=None,
                            status="FAILED", error=str(graph_exc),
                        )
                        result.graph_failed += 1
            except Exception as exc:
                self.repository.update_vector_state(
                    workspace_id, item.relative_path, content_hash=item.vector_hash, status="FAILED", error=str(exc)
                )
                result.failed += 1
                if len(result.issues) < 50:
                    result.issues.append({'path': item.relative_path,
                        'code': 'SOURCE_CHANGED' if isinstance(exc, SourceChangedDuringIndex) else stage})
                else:
                    result.issues_truncated = True
        return result

    @consistent_read
    def search(self, workspace_id: str, question: str, limit: int = 3,
               source_scope: WorkspaceSourceScope | None = None,
               score_threshold: float | None = None) -> list[dict]:
        store = self.vectorstore_getter()
        if store is None:
            return []
        if score_threshold is not None and not any(
                item.readable and item.content_hash and item.vector_status == 'INDEXED'
                and item.vector_hash == item.content_hash
                and (source_scope is None or source_scope.contains(item.relative_path))
                for item in self.repository.indexed_files(workspace_id)):
            return []
        IndexTransactions(self.repository.connection_factory).recover({'workspace_collection': store})
        where = {"workspace_id": workspace_id}
        if source_scope is not None and source_scope.kind != 'workspace':
            paths = [item.relative_path for item in self.repository.indexed_files(workspace_id)
                     if item.readable and source_scope.contains(item.relative_path)
                     and item.content_hash and item.vector_status == 'INDEXED'
                     and item.vector_hash == item.content_hash]
            if not paths:
                return []
            # Filter before vector selection. Post-filtering top-k alone lets
            # unrelated folders crowd all selected sources out of the result.
            where = {'$and': [where, {'relative_path': {'$in': paths}}]}
        documents = ([doc for doc, _ in store.similarity_search_with_relevance_scores(
            question, k=limit, filter=where, score_threshold=score_threshold)] if score_threshold is not None else
            store.similarity_search(question, k=limit, filter=where))
        output = [{
            "path": document.metadata.get("relative_path") or document.metadata.get("source", ""),
            "content": document.page_content, "score": 1.0, "matched": [],
            "chunk_id": document.metadata.get('chunk_id', ''),
            "found_by": ["workspace_semantic"],
        } for document in documents if document.page_content.strip()
            and not is_secret_file(document.metadata.get('relative_path') or document.metadata.get('source', ''))]
        unique = []
        seen = set()
        for item in output:
            signature = (item["path"], item["content"])
            if signature not in seen:
                unique.append(item); seen.add(signature)
            if len(unique) >= limit:
                break
        return unique

    @consistent_read
    def graph_context(self, workspace_id: str, question: str, limit: int = 3):
        from graph_retrieval import graph_search
        from settings import GRAPH_RETRIEVAL_ENABLED
        if not GRAPH_RETRIEVAL_ENABLED:
            return None
        store = self.vectorstore_getter()
        if store is None:
            return None
        IndexTransactions(self.repository.connection_factory).recover({'workspace_collection': store})
        results = graph_search(store, question, limit, workspace_id=workspace_id,
                               connection_factory=self.repository.connection_factory, filesystem=self.filesystem)
        if results is None:
            return None
        return [{'path': r.metadata['relative_path'], 'content': r.text, 'chunk_id': r.chunk_id,
                 'score': r.score, 'matched': [], 'found_by': r.found_by,
                 'graph_anchor': r.metadata['graph_anchor'], 'graph_path': r.metadata['graph_path'],
                 'graph_fast_path': True} for r in results]

    @consistent_read
    def expand_context(self, workspace_id: str, matches: list[dict], question=None) -> list[dict]:
        from hybrid_context_extractor import SearchResult
        from semantic_graph import automatic_graph_sources
        from settings import GRAPH_RETRIEVAL_ENABLED
        if question is not None and not GRAPH_RETRIEVAL_ENABLED:
            return []
        seeds = [SearchResult(f'workspace:{workspace_id}:{item["path"]}',
                              item.get('chunk_id', ''), item['content'],
                              {'relative_path': item['path'], 'workspace_id': workspace_id})
                 for item in matches]
        additions = automatic_graph_sources(self.vectorstore_getter(), seeds, question=question,
            workspace_id=workspace_id, connection_factory=self.repository.connection_factory,
            filesystem=self.filesystem)
        return [{'path': item.metadata['relative_path'], 'content': item.text,
                 'chunk_id': item.chunk_id, 'score': item.score, 'matched': [],
                 'found_by': item.found_by, 'graph_anchor': item.metadata['graph_anchor']}
                for item in additions]
