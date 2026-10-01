"""Bounded graph search and query-to-entity linking."""
from __future__ import annotations

from dataclasses import dataclass, field

from entity_normalization import normalize_entity_name
from graph_repository import GraphRepository
from settings import MAX_GRAPH_DEPTH, MAX_GRAPH_RESULTS
from db_utils import get_document


@dataclass
class GraphSearchResult:
    entities: list[dict] = field(default_factory=list)
    relations: list[dict] = field(default_factory=list)
    document_ids: list[int] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        names = {int(row["document_id"]): row.get("document_name") for row in self.relations}
        return {"entities": self.entities, "relations": self.relations,
                "documents": [{"id": value, "name": names.get(value)} for value in self.document_ids]}


class GraphService:
    def __init__(self, repository: GraphRepository | None = None):
        self.repository = repository or GraphRepository()

    def find_entity(self, name: str, limit: int = MAX_GRAPH_RESULTS) -> list[dict]:
        return self.repository.find_entities(name, min(limit, MAX_GRAPH_RESULTS))

    def get_relations(self, entity_id: int, limit: int = MAX_GRAPH_RESULTS) -> list[dict]:
        return self.repository.relations_for_entities([entity_id], min(limit, MAX_GRAPH_RESULTS))

    def get_neighbors(self, entity_id: int, limit: int = MAX_GRAPH_RESULTS) -> list[dict]:
        return self.traverse([entity_id], 1, limit).entities

    def link_query_entities(self, query: str, hints: list[str] | None = None) -> list[dict]:
        selected, seen = [], set()
        for hint in hints or []:
            for entity in self.find_entity(hint):
                if entity["id"] not in seen:
                    selected.append(entity); seen.add(entity["id"])
        normalized_query = normalize_entity_name(query)
        for entity in self.repository.all_entities():
            name = entity["canonical_name"]
            if len(name) >= 2 and name in normalized_query and entity["id"] not in seen:
                selected.append(entity); seen.add(entity["id"])
                if len(selected) >= MAX_GRAPH_RESULTS:
                    break
        return selected

    def traverse(self, entity_ids: list[int], depth: int = 1,
                 limit: int = MAX_GRAPH_RESULTS) -> GraphSearchResult:
        depth = max(1, min(depth, MAX_GRAPH_DEPTH))
        limit = max(1, min(limit, MAX_GRAPH_RESULTS))
        visited, frontier, relations, relation_ids = set(entity_ids), set(entity_ids), [], set()
        for _ in range(depth):
            if not frontier or len(relations) >= limit:
                break
            layer = self.repository.relations_for_entities(sorted(frontier), limit - len(relations))
            for row in layer:
                if row["id"] not in relation_ids:
                    relations.append(row)
                    relation_ids.add(row["id"])
            adjacent = {int(row[key]) for row in layer for key in ("source_entity_id", "target_entity_id")}
            frontier = adjacent - visited
            visited |= adjacent
        entities_by_id = {}
        for row in relations:
            entities_by_id[row["source_entity_id"]] = {"id": row["source_entity_id"], "display_name": row["source_name"], "entity_type": row["source_type"]}
            entities_by_id[row["target_entity_id"]] = {"id": row["target_entity_id"], "display_name": row["target_name"], "entity_type": row["target_type"]}
        return GraphSearchResult(
            entities=list(entities_by_id.values())[:limit], relations=relations[:limit],
            document_ids=list(dict.fromkeys(int(row["document_id"]) for row in relations))[:limit],
            chunk_ids=list(dict.fromkeys(row["chunk_id"] for row in relations if row["chunk_id"]))[:limit],
        )

    def search(self, query: str, hints: list[str] | None = None, depth: int = 1) -> GraphSearchResult:
        linked = self.link_query_entities(query, hints)
        if not linked:
            return GraphSearchResult()
        result = self.traverse([item["id"] for item in linked], depth)
        if not result.entities:
            result.entities = linked
        return result

    def related_documents(self, entity_id: int) -> list[int]:
        return self.traverse([entity_id], 1).document_ids

    def documents_for_entities(self, entity_ids: list[int]) -> list[int]:
        if not entity_ids:
            return []
        sets = [set(self.related_documents(entity_id)) for entity_id in entity_ids]
        return sorted(set.intersection(*sets)) if sets and all(sets) else []

    @staticmethod
    def _chunk_metadata(chunk_ids: list[str], vectorstore) -> dict[str, dict]:
        if not chunk_ids or vectorstore is None:
            return {}
        try:
            stored = vectorstore.get(ids=chunk_ids, include=["metadatas", "documents"])
            output = {}
            for index, chunk_id in enumerate(stored.get("ids") or []):
                metadata = dict((stored.get("metadatas") or [])[index] or {})
                text = (stored.get("documents") or [""])[index] or ""
                output[str(chunk_id)] = {
                    "page": metadata.get("page"), "section": metadata.get("section"),
                    "source_position": metadata.get("source_position"), "block_type": metadata.get("block_type"),
                    "text": text[:2000],
                }
            return output
        except Exception:
            return {}

    def _visualization_payload(self, result: GraphSearchResult, center_ids: list[int],
                               vectorstore=None, limit: int = MAX_GRAPH_RESULTS) -> dict:
        relations, included_ids = [], set(center_ids)
        omitted = False
        for row in result.relations:
            endpoints = {int(row["source_entity_id"]), int(row["target_entity_id"])}
            if len(included_ids | endpoints) > limit or len(relations) >= limit:
                omitted = True
                continue
            included_ids |= endpoints
            relations.append(row)
        chunk_metadata = self._chunk_metadata(
            list(dict.fromkeys(str(row.get("chunk_id", "")) for row in relations if row.get("chunk_id"))),
            vectorstore,
        )
        nodes: dict[str, dict] = {}
        for entity in result.entities:
            if int(entity["id"]) not in included_ids:
                continue
            node_id = f"entity:{entity['id']}"
            metadata = {"center": int(entity["id"]) in center_ids}
            if entity["entity_type"] == "DOCUMENT":
                matching = next((row for row in relations if row.get("document_name") == entity["display_name"]), None)
                if matching:
                    metadata["document_id"] = int(matching["document_id"])
            nodes[node_id] = {
                "id": node_id, "entity_id": int(entity["id"]),
                "label": entity["display_name"], "type": entity["entity_type"],
                "metadata": metadata,
            }
        edges = []
        document_nodes: dict[int, str] = {}
        for row in relations:
            document_id = int(row["document_id"])
            document_node_id = next((node_id for node_id, node in nodes.items()
                                     if node["type"] == "DOCUMENT" and node["metadata"].get("document_id") == document_id),
                                    f"document:{document_id}")
            document_nodes[document_id] = document_node_id
            if document_node_id in nodes or len(nodes) < limit:
                nodes.setdefault(document_node_id, {
                    "id": document_node_id, "document_id": document_id,
                    "label": row.get("document_name") or f"Документ {document_id}", "type": "DOCUMENT",
                    "metadata": {"provenance_only": True},
                })
            provenance = {
                "document_id": document_id, "document_name": row.get("document_name"),
                "chunk_id": row.get("chunk_id"), "evidence": row.get("evidence") or "",
                **chunk_metadata.get(str(row.get("chunk_id", "")), {}),
            }
            edges.append({
                "id": f"relation:{row['id']}", "source": f"entity:{row['source_entity_id']}",
                "target": f"entity:{row['target_entity_id']}", "relation": row["relation_type"],
                "provenance": provenance,
            })
        # Keep provenance documents visible without fabricating semantic edges.
        return {"nodes": list(nodes.values()), "edges": edges, "center_ids": [f"entity:{i}" for i in center_ids],
                "truncated": omitted or len(result.relations) >= limit, "limit": limit,
                "documents": list(document_nodes.values())}

    def entity_subgraph(self, entity_id: int, depth: int = 1, limit: int = MAX_GRAPH_RESULTS,
                        vectorstore=None) -> dict | None:
        entity = self.repository.get_entity(entity_id)
        if entity is None:
            return None
        result = self.traverse([entity_id], depth, limit)
        if not any(item["id"] == entity_id for item in result.entities):
            result.entities.insert(0, entity)
        return self._visualization_payload(result, [entity_id], vectorstore, min(limit, MAX_GRAPH_RESULTS))

    def document_subgraph(self, document_id: int, limit: int = MAX_GRAPH_RESULTS,
                          vectorstore=None) -> dict | None:
        document = get_document(document_id)
        if document is None:
            return None
        relations = self.repository.relations_for_document(document_id, min(limit, MAX_GRAPH_RESULTS))
        entities = {}
        for row in relations:
            entities[row["source_entity_id"]] = {"id": row["source_entity_id"], "display_name": row["source_name"], "entity_type": row["source_type"]}
            entities[row["target_entity_id"]] = {"id": row["target_entity_id"], "display_name": row["target_name"], "entity_type": row["target_type"]}
        result = GraphSearchResult(entities=list(entities.values()), relations=relations,
                                   document_ids=[document_id],
                                   chunk_ids=list(dict.fromkeys(row["chunk_id"] for row in relations)))
        payload = self._visualization_payload(result, [], vectorstore, min(limit, MAX_GRAPH_RESULTS))
        payload["center_ids"] = [next((node["id"] for node in payload["nodes"]
                                       if node.get("metadata", {}).get("document_id") == document_id),
                                      f"document:{document_id}")]
        return payload
