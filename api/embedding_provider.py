"""Independent embedding backend using the same Nomic GGUF as the existing index."""
import os
import hashlib
import json
import threading
from dataclasses import replace
from functools import lru_cache
from typing import Protocol
import httpx
from langchain_core.embeddings import Embeddings
from llama_process import LlamaServerProcess
from model_profiles import BASE_PROFILE, configured_path
from settings import DATA_DIR
from llm_contract import ProviderError


class EmbeddingProvider(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class LocalEmbeddings(Embeddings):
    def __init__(self):
        self.backend = os.getenv('EMBEDDING_BACKEND', 'llama')
        if self.backend not in {'llama', 'ollama'}:
            raise ValueError('Unknown EMBEDDING_BACKEND')
        self.url = os.getenv('EMBEDDING_SERVER_URL', 'http://127.0.0.1:8082').rstrip('/')
        self.alias = os.getenv('EMBEDDING_MODEL_ALIAS', 'nomic-embed-text')
        self.manager = None
        self.batch_size = max(1, int(os.getenv('EMBEDDING_BATCH_SIZE', '16')))
        self._collections = []
        self._identity = None
        self._identity_stat = None
        self._lock = threading.RLock()
        if self.backend == 'llama':
            profile = replace(BASE_PROFILE, name='EMBEDDING', alias=self.alias,
                model_path=configured_path('EMBEDDING_MODEL_PATH', DATA_DIR / 'models/nomic-embed-text.gguf'),
                context_size=int(os.getenv('EMBEDDING_CONTEXT_SIZE', '4096')),
                gpu_layers=os.getenv('EMBEDDING_GPU_LAYERS', '0'), batch_size=4096)
            self.manager = LlamaServerProcess(self.url, profile, embedding=True)

    def bind_collection(self, collection):
        self._collections.append(collection)

    def fingerprint(self):
        if self.manager:
            path = self.manager.profile.model_path
            stat = path.stat() if path.is_file() else None
            current = (str(path), stat.st_size, stat.st_mtime_ns) if stat else None
            if current != self._identity_stat:
                self._identity = None
                self._identity_stat = current
        if self._identity is None:
            model_id = os.getenv('EMBEDDING_MODEL_ID', '')
            if self.manager:
                path = self.manager.profile.model_path
                if not path.is_file():
                    raise ProviderError(f'Embedding model is missing: {path}')
                digest = hashlib.sha256()
                with path.open('rb') as stream:
                    while block := stream.read(8 * 1024 * 1024):
                        digest.update(block)
                model_id = digest.hexdigest()
            elif not model_id:
                raise ProviderError('Set EMBEDDING_MODEL_ID to an immutable model digest for the external backend')
            self._identity = hashlib.sha256(json.dumps({
                'model': model_id, 'dimensions': int(os.getenv('EMBEDDING_DIMENSIONS', '768')),
                'preprocessing': 'raw-text-v1',
            }, sort_keys=True).encode()).hexdigest()
        return self._identity

    def _check_collections(self):
        if not self._collections:
            return
        fingerprint = self.fingerprint()
        for collection in self._collections:
            metadata = dict(collection.metadata or {})
            existing = metadata.get('embedding_fingerprint')
            if existing and existing != fingerprint:
                raise ProviderError('Embedding model changed. Rebuild the index in a new collection before searching it.')
            if not existing:
                if collection.count() and os.getenv('EMBEDDING_LEGACY_ADOPT', '').lower() != 'true':
                    raise ProviderError('Legacy index has no embedding fingerprint. Rebuild it, or explicitly set '
                                        'EMBEDDING_LEGACY_ADOPT=true after verifying its embedding model.')
                collection.modify(metadata={**metadata, 'embedding_fingerprint': fingerprint})

    def embed_documents(self, texts):
        if not texts:
            return []
        with self._lock:
            self._check_collections()
        vectors = []
        with httpx.Client(timeout=120, trust_env=False) as client:
            for offset in range(0, len(texts), self.batch_size):
                batch = texts[offset:offset + self.batch_size]
                vectors.extend(self._embed_batch(client, batch))
        if len(vectors) != len(texts):
            raise ProviderError('Embedding server returned an unexpected number of vectors')
        if any(len(v) != int(os.getenv('EMBEDDING_DIMENSIONS', '768')) for v in vectors):
            raise ProviderError('Размерность embeddings не совпадает с индексом. Проверьте модель; не смешивайте embedding spaces.')
        return vectors

    def _embed_batch(self, client, batch):
        # Preserve raw text, with no task prefixes. A count limit alone cannot
        # predict the token size of a batch, so shrink rejected batches safely.
        if self.backend == 'ollama':
            response = client.post(self.url + '/api/embed', json={'model': self.alias, 'input': batch})
        else:
            response = client.post(self.url + '/v1/embeddings', json={
                'model': self.alias, 'input': batch, 'encoding_format': 'float'})
        if (len(batch) > 1 and response.status_code in {400, 413}
                and any(word in response.text.lower() for word in ('context', 'token', 'batch', 'too large', 'too long'))):
            middle = len(batch) // 2
            return self._embed_batch(client, batch[:middle]) + self._embed_batch(client, batch[middle:])
        response.raise_for_status()
        if self.backend == 'ollama':
            vectors = response.json()['embeddings']
        else:
            rows = sorted(response.json()['data'], key=lambda row: row['index'])
            if [row['index'] for row in rows] != list(range(len(batch))):
                raise ProviderError('Embedding server returned incomplete or duplicate batch results')
            vectors = [row['embedding'] for row in rows]
        if len(vectors) != len(batch):
            raise ProviderError('Embedding server returned an unexpected number of vectors')
        return vectors

    def embed_query(self, text):
        return self.embed_documents([text])[0]


@lru_cache(maxsize=1)
def get_embedding_provider():
    return LocalEmbeddings()
