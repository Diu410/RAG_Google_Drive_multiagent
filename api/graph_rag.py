"""GraphRAG retrieval adapter: graph edges select source chunks; Chroma remains truth."""
from __future__ import annotations

import logging

from langchain_core.documents import Document

from graph_service import GraphSearchResult, GraphService
from hybrid_context_extractor import SearchResult, reciprocal_rank_fusion
from settings import MAX_GRAPH_RESULTS
from index_transactions import consistent_read

logger = logging.getLogger(__name__)


def graph_supporting_results(graph: GraphSearchResult, vectorstore) -> list[SearchResult]:
    if not graph.chunk_ids or vectorstore is None:
        return []
    stored = vectorstore.get(ids=graph.chunk_ids, include=["documents", "metadatas"])
    results = []
    for index, chunk_id in enumerate(stored.get("ids") or []):
        text = (stored.get("documents") or [])[index]
        metadata = ((stored.get("metadatas") or [])[index] or {}).copy()
        metadata.update({"chunk_id": chunk_id, "document_id": metadata.get("document_id", metadata.get("file_id"))})
        results.append(SearchResult(metadata.get("document_id"), chunk_id, text, metadata,
                                    score=1.0, found_by=["graph"]))
    return results


class GraphAugmentedRetriever:
    def __init__(self, hybrid_retriever, vectorstore, query: str, entity_hints: list[str], depth: int):
        self.hybrid_retriever = hybrid_retriever
        self.vectorstore = vectorstore
        self.query = query
        self.entity_hints = entity_hints
        self.depth = depth
        self.graph = GraphSearchResult()
        self.last_results = []

    @consistent_read
    def __call__(self, value):
        question = value.get("input", self.query) if isinstance(value, dict) else str(value)
        history = value.get("chat_history", []) if isinstance(value, dict) else []
        hybrid = self.hybrid_retriever.search(question, history)
        try:
            self.graph = GraphService().search(question, self.entity_hints, self.depth)
            graph_results = graph_supporting_results(self.graph, self.vectorstore)
            self.last_results = reciprocal_rank_fusion(
                [("graph", graph_results, 1.5), ("hybrid", hybrid, 1.0)],
                k=min(MAX_GRAPH_RESULTS, max(self.hybrid_retriever.result_k, len(graph_results))),
            ) if graph_results else hybrid
        except Exception:
            logger.exception("Graph search failed; continuing with Hybrid Search")
            self.graph, self.last_results = GraphSearchResult(), hybrid
        return [item.as_document() for item in self.last_results]

    def with_config(self, **kwargs):
        return self
