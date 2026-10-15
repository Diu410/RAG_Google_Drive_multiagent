"""GraphRAG retrieval adapter: graph edges select source chunks; Chroma remains truth."""
from __future__ import annotations

import logging

from langchain_core.documents import Document

from graph_service import GraphSearchResult, GraphService
from hybrid_context_extractor import SearchResult
from index_transactions import consistent_read

logger = logging.getLogger(__name__)


class GraphFocusRetriever:
    """Use server-validated selected chunks without widening their source scope."""
    def __init__(self, focus):
        self.focus = focus

    def __call__(self, value):
        return [Document(page_content=source.text,
                         metadata={**source.metadata, 'score': source.score, 'found_by': 'graph_selection'})
                for source in self.focus['sources']]

    def selected_graph(self):
        return self.focus['graph']


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
        self.retrieval_info = {}

    @consistent_read
    def __call__(self, value):
        question = value.get("input", self.query) if isinstance(value, dict) else str(value)
        history = value.get("chat_history", []) if isinstance(value, dict) else []
        hybrid = self.hybrid_retriever.search(question, history)
        hybrid_ids = {item.chunk_id for item in hybrid}
        try:
            self.graph = GraphService().search(question, self.entity_hints, self.depth)
            graph_results = graph_supporting_results(self.graph, self.vectorstore)
            # Extracted relations are another retrieval hint. They must never
            # displace a direct hit or trigger an unbounded context expansion.
            budget = 0 if any('semantic_graph' in item.found_by for item in hybrid) else 1
            self.last_results = hybrid + [item for item in graph_results
                                          if item.chunk_id not in hybrid_ids][:budget]
        except Exception:
            logger.exception("Graph search failed; continuing with Hybrid Search")
            self.graph, self.last_results = GraphSearchResult(), hybrid
        selected_ids = {item.chunk_id for item in self.last_results}
        self.retrieval_info = {
            'selected_source_count': len(selected_ids),
            'graph_added_source_count': len(selected_ids - hybrid_ids),
        }
        return [item.as_document() for item in self.last_results]

    def selected_graph(self):
        """Only expose relations whose actual source chunks reached generation."""
        selected = {item.chunk_id for item in self.last_results}
        relations = [row for row in self.graph.relations if row.get('chunk_id') in selected]
        if not relations:
            return None
        from graph_service import GraphSearchResult
        entity_ids = {row[key] for row in relations for key in ('source_entity_id', 'target_entity_id')}
        graph = GraphSearchResult(
            entities=[row for row in self.graph.entities if row['id'] in entity_ids],
            relations=relations,
            document_ids=list(dict.fromkeys(int(row['document_id']) for row in relations)),
            chunk_ids=list(dict.fromkeys(row['chunk_id'] for row in relations)),
        )
        return {**graph.as_dict(), 'retrieval': self.retrieval_info}

    def with_config(self, **kwargs):
        return self
