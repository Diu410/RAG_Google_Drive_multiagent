"""Provider-neutral generation contracts and cancellation scopes."""
import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Event
import time
from typing import Iterator, Protocol


class ProviderError(RuntimeError):
    pass


class StructuredOutputTruncated(ProviderError):
    """The server exhausted its output budget before completing the assessment."""


class GenerationDeadlineExceeded(ProviderError):
    """Elapsed wall time, including a stream that keeps receiving tokens."""


class GenerationCancelled(asyncio.CancelledError):
    pass


@dataclass
class GenerationScope:
    task_id: str
    cancelled: Event = field(default_factory=Event)
    deadline: float | None = None
    started_at: float = field(default_factory=time.perf_counter)
    task_kind: str = 'STANDARD'

    def remaining_seconds(self):
        return None if self.deadline is None else max(0, self.deadline - time.perf_counter())

    def check_deadline(self):
        if self.deadline is not None and time.perf_counter() >= self.deadline:
            raise GenerationDeadlineExceeded('Достигнут лимит времени ответа. Непроверенные сведения не подтверждены.')


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
    preparation_ms: float = 0
    request_total_ms: float = 0
    llm_request_ms: float = 0
    time_to_first_token_ms: float | None = None
    generation_total_ms: float = 0
    tokens_per_second: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None


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


def completed_generation_text(text, finish_reason):
    """Keep complete sentences when a text generation reaches its token limit.

    This is presentation handling, not evidence verification. Callers must
    retain the original draft and expose the incomplete generation separately.
    """
    if finish_reason != 'length':
        return text, False
    import re
    boundaries = list(re.finditer(r'[.!?](?:[»”"\']*)(?=\s|$)', text))
    return (text[:boundaries[-1].end()].rstrip() if boundaries else ''), True
