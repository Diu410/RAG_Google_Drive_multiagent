"""Semantic, lexical and hybrid retrieval over the existing Chroma collection."""
from __future__ import annotations

import hashlib
import logging
import math
import re
import unicodedata
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Union

from langchain_core.documents import Document
from settings import FAST_CONTEXT_CHUNKS, FAST_RETRIEVAL_K
from text_normalization import lexical_key
from index_transactions import consistent_read, index_revision, IndexTransactions

logger = logging.getLogger(__name__)
RRF_K = 60
_MATH_CHARS = "∑∫√±×÷≤≥≠≈∞∂∇∈∉⊂⊃⊆⊇∪∩→←↔=+−*/^%°"
_TOKEN_RE = re.compile(
    rf"[^\W_]+(?:[-./][^\W_]+)*|[{re.escape(_MATH_CHARS)}]|[₀-₉₊₋₌₍₎⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾]",
    re.UNICODE,
)
_FORMULA_MARKER_RE = re.compile(rf"[{re.escape(_MATH_CHARS)}₀-₉₊₋₌₍₎⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾]")


def _tokenize(text: str) -> List[str]:
    """Tokenize Unicode words, numbers and technical compound identifiers."""
    tokens: List[str] = []
    normalized = unicodedata.normalize("NFC", text).casefold()
    for match in _TOKEN_RE.finditer(normalized):
        compound = match.group(0)
        tokens.append(compound)
        lexical = lexical_key(compound)
        if lexical != compound:
            tokens.append(lexical)
        parts = re.findall(r"[^\W_]+", compound, re.UNICODE)
        if len(parts) > 1:
            tokens.extend(parts)
            tokens.extend(stem for part in parts if (stem := lexical_key(part)) != part)
    for fragment in re.findall(r"[^\s,;:!?()\[\]{}]+", normalized):
        if len(fragment) > 1 and _FORMULA_MARKER_RE.search(fragment):
            tokens.append(fragment)
    return tokens


def _formula_exact_boost(query: str, text: str) -> float:
    query = unicodedata.normalize("NFC", query).casefold().strip()
    if len(query) < 2 or not _FORMULA_MARKER_RE.search(query):
        return 0.0
    haystack = unicodedata.normalize("NFC", text).casefold()
    if query in haystack:
        return 4.0
    compact = re.sub(r"\s+", "", query)
    compact_text = re.sub(r"\s+", "", haystack)
    return 2.5 if len(compact) >= 2 and compact in compact_text else 0.0


def _phrase_exact_boost(query: str, text: str) -> float:
    """Prefer an exact multi-token local phrase over partial matches in short chunks."""
    normalized_query = " ".join(unicodedata.normalize("NFC", query).casefold().split())
    if len(_tokenize(normalized_query)) < 2:
        return 0.0
    normalized_text = " ".join(unicodedata.normalize("NFC", text).casefold().split())
    # The corpus can contain many very short title-like chunks. A substantial
    # boost is intentional: all query words in their original order are much
    # stronger evidence than several high-IDF partial matches in a short chunk.
    return 12.0 if normalized_query in normalized_text else 0.0


def _bm25_scores(query: str, chunks: Sequence[str], k1: float = 1.5, b: float = 0.75) -> List[float]:
    if not chunks:
        return []
    query_tokens = _tokenize(query)
    tokenized_chunks = [_tokenize(chunk) for chunk in chunks]
    if not query_tokens:
        return [0.0] * len(chunks)
    avg_len = sum(map(len, tokenized_chunks)) / len(tokenized_chunks) or 1.0
    document_frequency = {
        token: sum(token in set(tokens) for tokens in tokenized_chunks)
        for token in set(query_tokens)
    }
    scores: List[float] = []
    for tokens in tokenized_chunks:
        frequencies = Counter(tokens)
        score = 0.0
        for token in query_tokens:
            frequency = frequencies[token]
            if not frequency:
                continue
            df = document_frequency[token]
            idf = math.log(1 + (len(chunks) - df + 0.5) / (df + 0.5))
            score += idf * frequency * (k1 + 1) / (
                frequency + k1 * (1 - b + b * len(tokens) / avg_len)
            )
        source_text = chunks[len(scores)]
        scores.append(score + _formula_exact_boost(query, source_text)
                      + _phrase_exact_boost(query, source_text))
    return scores


def _fallback_chunk_id(document_id: Any, text: str, metadata: Dict[str, Any]) -> str:
    identity = f"{document_id}|{metadata.get('source', '')}|{metadata.get('page', '')}|{text}"
    return "legacy:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()


@dataclass
class SearchResult:
    document_id: Any
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    score: float = 0.0
    found_by: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {"document_id": self.document_id, "chunk_id": self.chunk_id, "text": self.text,
                "metadata": self.metadata, "score": self.score, "found_by": self.found_by}

    def as_document(self) -> Document:
        metadata = dict(self.metadata)
        metadata.update({"document_id": self.document_id, "chunk_id": self.chunk_id,
                         "score": self.score, "found_by": ",".join(self.found_by)})
        metadata.setdefault("source", metadata.get("document_name") or f"document-{self.document_id}")
        return Document(page_content=self.text, metadata=metadata)


def _result(text: str, metadata: Dict[str, Any], chunk_id: str | None = None) -> SearchResult:
    clean_metadata = dict(metadata or {})
    document_id = clean_metadata.get("document_id", clean_metadata.get("file_id"))
    stable_id = str(chunk_id or clean_metadata.get("chunk_id") or _fallback_chunk_id(document_id, text, clean_metadata))
    clean_metadata.setdefault("chunk_id", stable_id)
    if document_id is not None:
        clean_metadata.setdefault("document_id", document_id)
    return SearchResult(document_id, stable_id, text, clean_metadata)


def _collection_records(vectorstore: Any, page_size: int = 1000) -> List[SearchResult]:
    """Read the lexical corpus from Chroma; no second index or store is maintained."""
    records: List[SearchResult] = []
    offset = 0
    while True:
        batch = vectorstore.get(include=["documents", "metadatas"], limit=page_size, offset=offset)
        ids = batch.get("ids") or []
        documents = batch.get("documents") or []
        metadatas = batch.get("metadatas") or []
        for index, storage_id in enumerate(ids):
            text = documents[index] if index < len(documents) else ""
            metadata = metadatas[index] if index < len(metadatas) else {}
            if text and text.strip():
                records.append(_result(text, metadata or {}, str(storage_id)))
        if len(ids) < page_size:
            break
        offset += len(ids)
    return records


class LexicalCorpus:
    """Tokenize once per index revision, score only matching postings per query."""
    def __init__(self, records):
        self.records = records
        self.lengths = []
        self.postings = {}
        for index, item in enumerate(records):
            tokens = _tokenize(item.text)
            self.lengths.append(len(tokens))
            for token, count in Counter(tokens).items():
                self.postings.setdefault(token, {})[index] = count
        self.average_length = sum(self.lengths) / max(1, len(records)) or 1.0

    def search(self, query, limit):
        scores = Counter()
        for token in _tokenize(query):
            posting = self.postings.get(token, {})
            idf = math.log(1 + (len(self.records) - len(posting) + .5) / (len(posting) + .5))
            for index, frequency in posting.items():
                scores[index] += idf * frequency * 2.5 / (
                    frequency + 1.5 * (.25 + .75 * self.lengths[index] / self.average_length))
        for index in scores:
            text = self.records[index].text
            scores[index] += _formula_exact_boost(query, text) + _phrase_exact_boost(query, text)
        ranking = sorted(scores, key=lambda index: (-scores[index], self.records[index].chunk_id))
        return [self.records[index] for index in ranking[:limit] if scores[index] > 0]


_lexical_cache = OrderedDict()


@consistent_read
def cached_corpus(store):
    # Retain at most four namespaces, rather than one cache per chat/retriever.
    revision = index_revision()
    cached = _lexical_cache.get(store)
    if cached is None or cached[0] != revision:
        cached = (revision, LexicalCorpus(_collection_records(store)))
        _lexical_cache[store] = cached
    _lexical_cache.move_to_end(store)
    while len(_lexical_cache) > 4:
        _lexical_cache.popitem(last=False)
    return cached[1]


@consistent_read
def lexical_search(query: str, vectorstore: Any, k: int = FAST_RETRIEVAL_K) -> List[SearchResult]:
    return cached_corpus(vectorstore).search(query, k)


def reciprocal_rank_fusion(ranked_results: Iterable[tuple],
                           k: int = FAST_CONTEXT_CHUNKS, rrf_k: int = RRF_K) -> List[SearchResult]:
    fused: Dict[str, SearchResult] = {}
    for ranked in ranked_results:
        search_type, ranking = ranked[:2]
        weight = ranked[2] if len(ranked) > 2 else 1.0
        seen: set[str] = set()
        for rank, item in enumerate(ranking, start=1):
            if item.chunk_id in seen:
                continue
            seen.add(item.chunk_id)
            existing = fused.get(item.chunk_id)
            if existing is None:
                existing = SearchResult(item.document_id, item.chunk_id, item.text, dict(item.metadata))
                fused[item.chunk_id] = existing
            existing.score += weight / (rrf_k + rank)
            if search_type not in existing.found_by:
                existing.found_by.append(search_type)
    return sorted(fused.values(), key=lambda item: (-item.score, item.chunk_id))[:k]


class HybridRetriever:
    """LangChain-compatible retriever supporting semantic, lexical and hybrid modes."""

    def __init__(self, base_retriever: Any, vectorstore: Any = None, mode: str = "semantic",
                 needs_history_aware_input: bool = True, result_k: int = FAST_CONTEXT_CHUNKS,
                 lexical_weight: float = 1.0, query_variants: Sequence[str] | None = None):
        if mode not in {"semantic", "lexical", "hybrid"}:
            raise ValueError(f"Unsupported search mode: {mode}")
        # A direct vector retriever must retrieve enough candidates to satisfy
        # the caller's requested limit, without mutating the shared retriever.
        search_kwargs = getattr(base_retriever, 'search_kwargs', None)
        if not needs_history_aware_input and isinstance(search_kwargs, dict):
            search_kwargs = dict(search_kwargs)
            search_kwargs['k'] = max(search_kwargs.get('k', FAST_RETRIEVAL_K), result_k)
            if 'fetch_k' in search_kwargs:
                search_kwargs['fetch_k'] = max(search_kwargs['fetch_k'], search_kwargs['k'] * 2)
            base_retriever = base_retriever.model_copy(update={'search_kwargs': search_kwargs})
        self.base_retriever = base_retriever
        self.vectorstore = vectorstore
        self.mode = mode
        self.needs_history_aware_input = needs_history_aware_input
        self.result_k = result_k
        self.lexical_weight = lexical_weight
        self.query_variants = [value.strip() for value in (query_variants or []) if value.strip()]
        self.last_context = ""
        self.last_results: List[SearchResult] = []

    def _semantic(self, query: str, chat_history: List[Any]) -> List[SearchResult]:
        if self.base_retriever is None:
            return []
        invocation: Any = {"input": query, "chat_history": chat_history} if self.needs_history_aware_input else query
        documents = self.base_retriever.invoke(invocation)
        return [_result(doc.page_content, doc.metadata) for doc in documents if doc.page_content.strip()]

    @staticmethod
    def _align_legacy_ids(semantic: List[SearchResult], corpus: Sequence[SearchResult]) -> None:
        by_signature = {(item.text, str(item.document_id)): item.chunk_id for item in corpus}
        for item in semantic:
            storage_id = by_signature.get((item.text, str(item.document_id)))
            if storage_id and item.chunk_id.startswith("legacy:"):
                item.chunk_id = storage_id
                item.metadata["chunk_id"] = storage_id

    def _search_once(self, query: str, chat_history: List[Any]) -> List[SearchResult]:
        semantic = self._semantic(query, chat_history) if self.mode != "lexical" else []
        lexical: List[SearchResult] = []
        if self.mode != "semantic":
            if self.vectorstore is None:
                raise RuntimeError("Lexical search requires a Chroma vectorstore")
            corpus = cached_corpus(self.vectorstore)
            self._align_legacy_ids(semantic, corpus.records)
            lexical = corpus.search(query, max(FAST_RETRIEVAL_K, self.result_k))
        rankings = []
        if semantic:
            rankings.append(("semantic", semantic))
        if lexical:
            rankings.append(("lexical", lexical, self.lexical_weight))
        return reciprocal_rank_fusion(rankings, self.result_k)

    @consistent_read
    def search(self, query: str, chat_history: List[Any] | None = None) -> List[SearchResult]:
        if self.vectorstore is not None and hasattr(self.vectorstore, '_collection'):
            IndexTransactions().recover({'rag_collection': self.vectorstore})
        chat_history = chat_history or []
        if len(self.query_variants) > 1:
            rankings = [self._search_once(f"{query}\nОбъект сравнения: {entity}", chat_history)
                        for entity in self.query_variants]
            selected: List[SearchResult] = []
            seen: set[str] = set()
            for rank in range(max((len(items) for items in rankings), default=0)):
                for items in rankings:
                    if rank < len(items) and items[rank].chunk_id not in seen:
                        selected.append(items[rank])
                        seen.add(items[rank].chunk_id)
                        if len(selected) >= self.result_k:
                            break
                if len(selected) >= self.result_k:
                    break
            self.last_results = selected
        else:
            self.last_results = self._search_once(query, chat_history)
        self.last_context = "\n\n---\n\n".join(item.text for item in self.last_results)
        logger.debug("Search mode=%s variants=%s results=%s", self.mode, len(self.query_variants), len(self.last_results))
        return self.last_results

    def __call__(self, query: Union[str, Dict[str, Any]], chat_history: List[Any] | None = None) -> List[Document]:
        if isinstance(query, dict):
            chat_history = query.get("chat_history", chat_history)
            query = query.get("input", "")
        return [item.as_document() for item in self.search(query, chat_history)]

    def similarity_search(self, query: str, **kwargs: Any) -> List[Document]:
        return self(query, kwargs.get("chat_history", []))

    def with_config(self, **kwargs: Any) -> "HybridRetriever":
        return self


def hybrid_extract_context(query: str, chunks: List[str], max_chunks: int = 8) -> List[str]:
    """Compatibility helper retained for callers that pass an in-memory corpus."""
    unique = list(dict.fromkeys(chunk.strip() for chunk in chunks if chunk.strip()))
    ranked = sorted(zip(unique, _bm25_scores(query, unique)), key=lambda pair: pair[1], reverse=True)
    return [chunk for chunk, _ in ranked[:max_chunks]]
