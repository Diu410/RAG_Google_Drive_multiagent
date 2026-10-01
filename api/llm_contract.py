"""Provider-neutral generation contracts and cancellation scopes."""
import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Event
from typing import Iterator, Protocol


class ProviderError(RuntimeError):
    pass


class GenerationCancelled(asyncio.CancelledError):
    pass


@dataclass
class GenerationScope:
    task_id: str
    cancelled: Event = field(default_factory=Event)


current_scope = ContextVar('generation_scope', default=None)
current_action = ContextVar('generation_action', default=None)
# Per-request observer used by the live transport. Structured service calls
# (routing/planning/verification) deliberately never publish their JSON.
current_token_sink = ContextVar('generation_token_sink', default=None)


@contextmanager
def generation_scope(task_id):
    scope = GenerationScope(task_id)
    token = current_scope.set(scope)
    try:
        yield scope
    finally:
        current_scope.reset(token)


@dataclass
class GenerationMetrics:
    llm_request_ms: float = 0
    time_to_first_token_ms: float | None = None
    generation_total_ms: float = 0
    tokens_per_second: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass
class GenerationResult:
    text: str
    generation_id: str
    metrics: GenerationMetrics


@dataclass
class TokenDelta:
    generation_id: str
    text: str = ''
    reasoning: str = ''
    done: bool = False
    metrics: GenerationMetrics | None = None


class LLMProvider(Protocol):
    def generate(self, messages: list[dict], **options) -> GenerationResult: ...
    def stream(self, messages: list[dict], **options) -> Iterator[TokenDelta]: ...
    def structured_generate(self, messages: list[dict], schema: dict, **options) -> GenerationResult: ...
    def health(self) -> dict: ...
    def cancel(self, generation_id: str) -> bool: ...
    def get_model_info(self) -> dict: ...
