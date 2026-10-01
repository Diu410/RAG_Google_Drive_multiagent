"""Batched, best-effort knowledge graph extraction via provider structured output."""
from __future__ import annotations

import logging
import json
from typing import Callable

from pydantic import BaseModel, Field, ValidationError

from graph_repository import GraphRepository
from model_runtime import get_llm_provider
from settings import (DEFAULT_CHAT_MODEL, GRAPH_BATCH_CHUNKS,
                      GRAPH_BATCH_MAX_CHARS, GRAPH_LLM_NUM_CTX, GRAPH_LLM_NUM_PREDICT)

logger = logging.getLogger(__name__)


class ExtractedEntity(BaseModel):
    name: str = Field(min_length=1, max_length=300)
    type: str = Field(default="OTHER", max_length=80)


class ExtractedRelation(BaseModel):
    source: str = Field(min_length=1, max_length=300)
    relation: str = Field(min_length=1, max_length=120)
    target: str = Field(min_length=1, max_length=300)
    evidence: str = Field(default="", max_length=2000)
    confidence: float = Field(default=1.0, ge=0, le=1)
    chunk_id: str | None = None


class GraphExtractionResult(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list, max_length=100)
    relations: list[ExtractedRelation] = Field(default_factory=list, max_length=150)


# Keep the grammar deliberately small. Pydantic's generated schema contains large
# maxLength/maxItems repetitions and nullable anyOf branches; some inference server
# builds reject that grammar before the model is invoked. Strict bounds are still
# enforced by GraphExtractionResult when the response is parsed below.
GRAPH_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "type": {"type": "string"}},
                "required": ["name", "type"],
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"}, "relation": {"type": "string"},
                    "target": {"type": "string"}, "evidence": {"type": "string"},
                    "confidence": {"type": "number"}, "chunk_id": {"type": "string"},
                },
                "required": ["source", "relation", "target", "evidence", "confidence", "chunk_id"],
            },
        },
    },
    "required": ["entities", "relations"],
}


class GraphExtractor:
    def __init__(self, repository: GraphRepository | None = None,
                 client_factory: Callable = get_llm_provider, model: str = DEFAULT_CHAT_MODEL):
        self.repository = repository or GraphRepository()
        self.client_factory = client_factory
        self.model = model

    def extract_document(self, chunks, document_id: int, document_name: str) -> bool:
        entities, relations = [], []
        self.last_status = "FAILED"
        self.successful_batches = self.failed_batches = 0
        try:
            batches = list(self._batches(chunks))
            logger.info("Graph extraction started for document %s: %s chunks in %s batches",
                        document_id, len(chunks), len(batches))
            for batch_number, batch in enumerate(batches, 1):
                try:
                    extracted = self._extract_batch_adaptive(batch)
                except Exception as exc:
                    self.failed_batches += 1
                    logger.warning("Graph extraction skipped malformed/failed batch for document %s: %s", document_id, exc)
                    # A rejected grammar is request-wide, not batch-specific. Avoid
                    # issuing the same doomed request for every batch of a large PDF.
                    if "failed to parse grammar" in str(exc).casefold():
                        logger.error("Graph extraction stopped for document %s because the provider rejected the output schema", document_id)
                        break
                    continue
                self.successful_batches += 1
                for source_batch, result in extracted:
                    entities.extend(item.model_dump() for item in result.entities)
                    known_types = {item.name.casefold(): item.type.upper() for item in result.entities}
                    default_chunk = source_batch[0].metadata["chunk_id"]
                    batch_ids = {chunk.metadata["chunk_id"] for chunk in source_batch}
                    for item in result.relations:
                        data = item.model_dump()
                        data["chunk_id"] = data["chunk_id"] if data["chunk_id"] in batch_ids else default_chunk
                        data["source_type"] = known_types.get(item.source.casefold())
                        data["target_type"] = known_types.get(item.target.casefold())
                        relations.append(data)
                if batch_number == 1 or batch_number % 5 == 0 or batch_number == len(batches):
                    logger.info("Graph extraction progress for document %s: %s/%s batches",
                                document_id, batch_number, len(batches))
            if self.failed_batches:
                # Keep the last complete graph. A partial extraction remains
                # retryable and must never advance the file's graph hash.
                self.last_status = "PARTIAL" if self.successful_batches else "FAILED"
                return False
            self.repository.replace_document_graph(document_id, entities, relations, document_name)
            self.last_status = "COMPLETED"
            logger.info("Graph extraction completed for document %s: %s entities, %s relations",
                        document_id, len(entities), len(relations))
            return True
        except Exception:
            logger.exception("Graph persistence failed for document %s; vector index remains available", document_id)
            return False

    @staticmethod
    def _batches(chunks):
        batch, size = [], 0
        for chunk in chunks:
            length = len(chunk.page_content)
            if batch and (len(batch) >= GRAPH_BATCH_CHUNKS or size + length > GRAPH_BATCH_MAX_CHARS):
                yield batch
                batch, size = [], 0
            batch.append(chunk)
            size += length
        if batch:
            yield batch

    def _extract_batch_adaptive(self, batch, split_depth: int = 0):
        """Retry only an over-dense/malformed batch as smaller neighboring groups."""
        try:
            return [(batch, self._extract_batch(batch))]
        except ValueError as exc:
            if len(batch) <= 2 or split_depth >= 3:
                raise
            middle = len(batch) // 2
            logger.info("Retrying dense graph batch as %s + %s chunks after invalid JSON",
                        middle, len(batch) - middle)
            output = []
            for part in (batch[:middle], batch[middle:]):
                try:
                    output.extend(self._extract_batch_adaptive(part, split_depth + 1))
                except ValueError as part_exc:
                    logger.warning("Graph extraction skipped irrecoverable sub-batch: %s", part_exc)
            if not output:
                raise exc
            return output

    def _extract_batch(self, batch) -> GraphExtractionResult:
        content = "\n\n".join(f"[CHUNK_ID={c.metadata['chunk_id']}]\n{c.page_content}" for c in batch)
        response = self.client_factory().structured_generate(
            profile=self.model,
            messages=[{"role": "system", "content": (
                "Extract a compact knowledge graph from the supplied document chunks. Preserve the exact CHUNK_ID "
                "for every relation. Entity types include PERSON, ORGANIZATION, PROJECT, DOCUMENT, DATE, TECHNOLOGY, "
                "PRODUCT, LOCATION, CONCEPT, OTHER. Relations may include WORKS_ON, BELONGS_TO, USES, SIGNED_BY, "
                "REPLACES, DEPENDS_ON, RELATED_TO, MENTIONS, but use another concise uppercase type when needed. "
                "Preserve formulas and Greek/special symbols verbatim. A named variable such as coefficient λ may be "
                "a CONCEPT, but never create entities for isolated operators such as +, =, ≤ or arrows. "
                "Only include relations supported by explicit evidence. Keep the graph compact: return at most 20 "
                "important entities and 24 important relations for this batch, omit generic/repeated mentions, and "
                "limit each evidence excerpt to 120 characters. Return only the requested JSON."
            )}, {"role": "user", "content": content}],
            schema=GRAPH_OUTPUT_SCHEMA,
            thinking=False, temperature=0, max_tokens=GRAPH_LLM_NUM_PREDICT,
        )
        raw = response.text
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise TypeError("root must be an object")
            payload["entities"] = list(payload.get("entities") or [])[:100]
            payload["relations"] = list(payload.get("relations") or [])[:150]
            for item in payload["entities"]:
                if isinstance(item, dict):
                    item["name"] = str(item.get("name") or "")[:300]
                    item["type"] = str(item.get("type") or "OTHER")[:80]
            for item in payload["relations"]:
                if not isinstance(item, dict):
                    continue
                for key, limit in (("source", 300), ("relation", 120), ("target", 300),
                                   ("evidence", 2000), ("chunk_id", 300)):
                    item[key] = str(item.get(key) or "")[:limit]
                try:
                    confidence = float(item.get("confidence", 1.0))
                except (TypeError, ValueError):
                    confidence = 1.0
                if 1 < confidence <= 100:
                    confidence /= 100
                item["confidence"] = max(0.0, min(1.0, confidence))
            return GraphExtractionResult.model_validate(payload)
        except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
            raise ValueError(f"Graph extractor returned invalid structured JSON: {exc}") from exc
