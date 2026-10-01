"""Incremental Chroma namespace for local workspace files."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from langchain_core.documents import Document
from index_transactions import IndexTransactions, consistent_read, invalidate_index, serialized_mutation, index_lock


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

    @serialized_mutation
    def sync(self, workspace_id: str) -> WorkspaceVectorSync:
        self.filesystem.sync_index(workspace_id)
        store = self.vectorstore_getter()
        result = WorkspaceVectorSync()
        if store is None:
            result.unavailable = True
            return result
        transactions = IndexTransactions(self.repository.connection_factory)
        transactions.recover({'workspace_collection': store})
        records = self.repository.indexed_files(workspace_id)
        active_keys = {self.file_key(workspace_id, item.relative_path) for item in records if item.readable}
        stored = store.get(where={"workspace_id": workspace_id}, include=["metadatas"])
        stale_ids = [storage_id for storage_id, metadata in zip(
            stored.get("ids") or [], stored.get("metadatas") or []
        ) if (metadata or {}).get("workspace_file_key") not in active_keys]
        if stale_ids:
            with index_lock:
                store.delete(ids=stale_ids)
                invalidate_index()
            result.deleted_chunks = len(stale_ids)
        if self.graph_repository is not None:
            result.deleted_graph_files = self.graph_repository.delete_missing(workspace_id, active_keys)

        from formula_processing import split_structured_documents
        for item in records:
            if not item.readable or not item.content_hash:
                continue
            vector_current = item.vector_status == "INDEXED" and item.vector_hash == item.content_hash
            graph_current = (self.graph_repository is None or
                             (item.graph_status == "INDEXED" and item.graph_hash == item.content_hash))
            if vector_current and graph_current:
                result.unchanged += 1
                continue
            key = self.file_key(workspace_id, item.relative_path)
            try:
                _, _, content = self.filesystem.read_text_file(workspace_id, item.relative_path)
                chunks = split_structured_documents([Document(page_content=content, metadata={
                    "workspace_id": workspace_id, "workspace_file_key": key,
                    "relative_path": item.relative_path, "source": item.relative_path,
                    "source_kind": "workspace", "content_hash": item.content_hash,
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
                    def commit(conn):
                        updated = conn.execute('''UPDATE assistant_workspace_files SET vector_hash=?,
                            vector_status='INDEXED', vector_error=NULL
                            WHERE workspace_id=? AND relative_path=? AND content_hash=?''',
                            (item.content_hash, workspace_id, item.relative_path, item.content_hash))
                        if updated.rowcount != 1:
                            raise RuntimeError('File changed during vector synchronization; retry required')
                    transactions.replace(
                        store, 'workspace_collection', {"workspace_file_key": key}, chunks, ids, commit=commit
                    )
                    result.indexed += 1
                if self.graph_repository is not None and not graph_current:
                    try:
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
                    workspace_id, item.relative_path, content_hash=None, status="FAILED", error=str(exc)
                )
                result.failed += 1
        return result

    @consistent_read
    def search(self, workspace_id: str, question: str, limit: int = 3) -> list[dict]:
        store = self.vectorstore_getter()
        if store is None:
            return []
        IndexTransactions(self.repository.connection_factory).recover({'workspace_collection': store})
        documents = store.similarity_search(question, k=limit, filter={"workspace_id": workspace_id})
        output = [{
            "path": document.metadata.get("relative_path") or document.metadata.get("source", ""),
            "content": document.page_content, "score": 1.0, "matched": [],
            "found_by": ["workspace_semantic"],
        } for document in documents if document.page_content.strip()]
        if self.graph_repository is not None:
            relations = self.graph_repository.search(workspace_id, question, limit=limit * 2)
            chunk_ids = list(dict.fromkeys(row["chunk_id"] for row in relations if row.get("chunk_id")))
            if chunk_ids:
                stored = store.get(ids=chunk_ids, include=["documents", "metadatas"])
                graph_items = []
                for index, chunk_id in enumerate(stored.get("ids") or []):
                    metadata = (stored.get("metadatas") or [])[index] or {}
                    content = (stored.get("documents") or [])[index] or ""
                    graph_items.append({
                        "path": metadata.get("relative_path") or metadata.get("source", ""),
                        "content": content, "score": 2.0, "matched": [],
                        "found_by": ["workspace_graph"],
                    })
                output = graph_items + output
        unique = []
        seen = set()
        for item in output:
            signature = (item["path"], item["content"])
            if signature not in seen:
                unique.append(item); seen.add(signature)
            if len(unique) >= limit:
                break
        return unique
