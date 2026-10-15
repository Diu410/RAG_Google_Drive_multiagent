"""OpenAI-compatible SSE provider. Async socket ownership makes cancellation real.

The dedicated I/O loop lets existing synchronous chains consume deltas without
blocking the FastAPI loop. Cancelling the I/O task closes its HTTP stream even
when a synchronous consumer is blocked waiting for the next token.
"""
from __future__ import annotations
import asyncio
import json
import logging
import queue
import threading
import time
import uuid
from contextlib import contextmanager, asynccontextmanager
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
import httpx

from llm_contract import (GenerationCancelled, GenerationDeadlineExceeded, GenerationMetrics, GenerationResult,
                          ProviderError, StructuredOutputTruncated, TokenDelta, current_scope, current_action,
                          current_token_sink)
from model_profiles import get_profile, SERVER_URL, REQUEST_TIMEOUT
from prompt_budget import fit_prompt

logger = logging.getLogger(__name__)


@dataclass
class _Generation:
    id: str
    scope: object
    action_id: str | None
    queue: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=16))
    cancelled: threading.Event = field(default_factory=threading.Event)
    future: object = None
    request_started: float = 0
    deadline: float = 0
    timed_out: bool = False


class LlamaCppProvider:
    supports_batch_evidence = True
    supports_evidence_ids = True
    def __init__(self, url=SERVER_URL, *, transport=None, manager=None):
        self.url = url.rstrip('/')
        self.transport, self.manager = transport, manager
        self._active = {}
        self._lock = threading.Lock()
        self._loop = None
        self._thread = None
        self._token_counts = OrderedDict()
        self._tokenizer_available = None
        self._inflight = 0
        self._switching = False
        self._client = httpx.Client(transport=transport, timeout=10, trust_env=False)
        self._async_client = None

    @contextmanager
    def deployment_change(self):
        with self._lock:
            if self._inflight or self._switching:
                raise ProviderError('Дождитесь завершения текущего ответа перед сменой модели.')
            self._switching = True
        try:
            yield
        finally:
            with self._lock:
                self._token_counts.clear()
                self._tokenizer_available = None
                self._switching = False

    @contextmanager
    def _inference_use(self):
        with self._lock:
            if self._switching:
                raise ProviderError('Модель переключается. Дождитесь её загрузки.')
            self._inflight += 1
        try:
            yield
        finally:
            with self._lock:
                self._inflight -= 1

    def count_prompt(self, messages, alias, thinking):
        import hashlib
        scope = current_scope.get()
        if scope:
            scope.check_deadline()
        body = {'model': alias, 'messages': messages,
                'chat_template_kwargs': {'enable_thinking': thinking}}
        key = hashlib.sha256(json.dumps(body, ensure_ascii=False).encode()).hexdigest()
        with self._lock:
            cached = self._token_counts.get(key)
        if cached is not None:
            return cached
        count = None
        # Loading/restarting the server can temporarily fail either endpoint.
        # A byte estimate is deliberately conservative, not a tokenizer result:
        # never memoize it or disable future probes after one failed request.
        try:
            response = self._client.post(self.url + '/apply-template', json=body)
            response.raise_for_status()
            prompt = response.json()['prompt']
            response = self._client.post(self.url + '/tokenize', json={
                'content': prompt, 'add_special': True, 'parse_special': True})
            response.raise_for_status()
            tokens = response.json()['tokens']
            if not isinstance(tokens, list):
                raise ValueError('Invalid tokenizer response')
            count = len(tokens)
            self._tokenizer_available = True
        except (httpx.HTTPError, ValueError, KeyError):
            if self._tokenizer_available is not False:
                logger.warning('Server tokenizer unavailable; using a conservative UTF-8 byte budget')
            self._tokenizer_available = False
        if count is None:
            # Byte-level tokenizers cannot use more text tokens than UTF-8
            # bytes. Reserve template overhead when the endpoint is absent.
            return 128 + sum(len(item.get('content', '').encode('utf-8')) + 16 for item in messages)
        with self._lock:
            self._token_counts[key] = count
            while len(self._token_counts) > 256:
                self._token_counts.popitem(last=False)
        return count

    def _io_loop(self):
        with self._lock:
            if self._loop is None:
                self._loop = asyncio.new_event_loop()
                self._thread = threading.Thread(target=self._loop.run_forever, daemon=True, name='llm-http')
                self._thread.start()
            return self._loop

    def health(self):
        if self.manager:
            return self.manager.status()
        try:
            response = self._client.get(self.url + '/health', timeout=2)
            return {'ready': response.status_code == 200, 'server_ready': True,
                    'model_loaded': response.status_code == 200}
        except httpx.HTTPError:
            return {'ready': False, 'server_ready': False, 'model_loaded': False}

    def get_model_info(self):
        try:
            response = self._client.get(self.url + '/v1/models', timeout=2)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as exc:
            raise ProviderError(f'Не удалось получить информацию о модели: {self.url}') from exc

    @asynccontextmanager
    async def _session(self):
        # This client is created, used and closed on the dedicated I/O loop.
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(transport=self.transport,
                timeout=REQUEST_TIMEOUT, trust_env=False)
        yield self._async_client

    def cancel(self, generation_id):
        with self._lock:
            matches = [g for g in self._active.values() if generation_id in {
                g.id, g.action_id, g.scope.task_id if g.scope else None}]
            for g in matches:
                g.cancelled.set()
                if g.scope:
                    g.scope.cancelled.set()
                if g.future:
                    g.future.cancel()
        return bool(matches)

    def _check_cancel(self, generation):
        if generation.cancelled.is_set() or (generation.scope and generation.scope.cancelled.is_set()):
            raise GenerationCancelled()
        if generation.scope:
            try:
                generation.scope.check_deadline()
            except GenerationDeadlineExceeded:
                generation.timed_out = True
                raise
        if generation.deadline and time.perf_counter() >= generation.deadline:
            generation.timed_out = True
            raise GenerationDeadlineExceeded('Генерация превысила лимит времени. Неполный результат не считается проверенным.')

    async def _put(self, generation, value):
        while True:
            self._check_cancel(generation)
            try:
                generation.queue.put_nowait(value)
                return
            except queue.Full:
                await asyncio.sleep(0.01)

    async def _receive(self, generation, payload):
        started = time.perf_counter()
        metrics = GenerationMetrics(preparation_ms=(started - generation.request_started) * 1000)
        completed = False
        try:
            self._check_cancel(generation)
            async with self._session() as client:
                async with client.stream('POST', self.url + '/v1/chat/completions', json=payload) as response:
                    metrics.llm_request_ms = (time.perf_counter() - started) * 1000
                    if response.status_code != 200:
                        # Do not propagate server bodies: they can echo user prompts.
                        raise ProviderError(f'Сервер модели отклонил запрос (HTTP {response.status_code}). '
                                            'Проверьте контекст, JSON schema и журнал llama-server.')
                    async for line in response.aiter_lines():
                        self._check_cancel(generation)
                        if not line.startswith('data:'):
                            continue
                        raw = line[5:].strip()
                        if raw == '[DONE]':
                            completed = True
                            break
                        data = json.loads(raw)
                        if data.get('error'):
                            raise ProviderError('Ошибка генерации в потоке llama-server; проверьте журнал сервера')
                        usage = data.get('usage') or {}
                        if usage:
                            metrics.prompt_tokens = usage.get('prompt_tokens')
                            metrics.completion_tokens = usage.get('completion_tokens')
                        timings = data.get('timings') or {}
                        if timings.get('predicted_per_second'):
                            metrics.tokens_per_second = timings['predicted_per_second']
                        for choice in data.get('choices', []):
                            if choice.get('finish_reason'):
                                metrics.finish_reason = choice['finish_reason']
                            delta = choice.get('delta') or {}
                            text, reasoning = delta.get('content') or '', delta.get('reasoning_content') or ''
                            if text or reasoning:
                                if metrics.time_to_first_token_ms is None:
                                    metrics.time_to_first_token_ms = (time.perf_counter() - started) * 1000
                                await self._put(generation, TokenDelta(generation.id, text, reasoning))
                    if not completed:
                        raise ProviderError('Поток модели оборвался до завершения ответа')
            metrics.generation_total_ms = (time.perf_counter() - started) * 1000
            metrics.request_total_ms = (time.perf_counter() - generation.request_started) * 1000
            if metrics.tokens_per_second is None and metrics.completion_tokens:
                seconds = max((metrics.generation_total_ms - (metrics.time_to_first_token_ms or 0)) / 1000, .001)
                metrics.tokens_per_second = metrics.completion_tokens / seconds
            await self._put(generation, TokenDelta(generation.id, done=True, metrics=metrics))
        except asyncio.CancelledError:
            generation.cancelled.set()
            raise
        except Exception as exc:
            error = exc if isinstance(exc, ProviderError) else ProviderError(
                f'Связь с сервером модели прервана: {self.url} ({type(exc).__name__})')
            await self._put(generation, error)
        finally:
            metrics.generation_total_ms = (time.perf_counter() - started) * 1000
            metrics.request_total_ms = (time.perf_counter() - generation.request_started) * 1000
            if generation.timed_out:
                metrics.finish_reason = 'timeout'
            logger.info('llm_metrics %s', json.dumps({'generation_id': generation.id,
                'task_id': generation.scope.task_id if generation.scope else None,
                'action_id': generation.action_id, 'cancelled': generation.cancelled.is_set(), **asdict(metrics)}))

    def stream(self, messages, **options):
        with self._inference_use():
            yield from self._stream(messages, **options)

    def _stream(self, messages, *, profile=None, temperature=None, max_tokens=None,
               thinking=None, schema=None, generation_id=None, stop=None, reasoning_budget_tokens=None):
        request_started = time.perf_counter()
        if reasoning_budget_tokens is not None and (
                type(reasoning_budget_tokens) is not int or reasoning_budget_tokens < 0):
            raise ProviderError('Бюджет рассуждения должен быть неотрицательным целым числом.')
        selected = get_profile(profile)
        scope = current_scope.get()
        if scope and scope.cancelled.is_set():
            raise GenerationCancelled()
        if scope:
            scope.check_deadline()
        health = self.health()
        if not health['ready']:
            raise ProviderError(health.get('error') or f'Модель ещё не готова: {self.url}. '
                                f'Ожидаемый GGUF: {selected.model_path}')
        if self.manager and (selected.model_path, selected.alias, selected.context_size, selected.gpu_layers) != (
                self.manager.profile.model_path, self.manager.profile.alias,
                self.manager.profile.context_size, self.manager.profile.gpu_layers):
            raise ProviderError('Этот профиль требует перезапуска сервера с соответствующей моделью и параметрами')
        enabled_thinking = selected.thinking if thinking is None else thinking
        # Strict JSON needs the sampler's thought delimiters, even with a zero
        # budget. Ordinary text keeps the model's non-thinking prompt template.
        template_thinking = enabled_thinking or schema is not None
        messages, max_tokens = fit_prompt(
            messages, selected.context_size, max_tokens or selected.max_tokens,
            lambda values: self.count_prompt(values, selected.alias, template_thinking),
        )
        if scope and scope.cancelled.is_set():
            raise GenerationCancelled()
        if scope:
            scope.check_deadline()
        if time.perf_counter() - request_started >= selected.generation_seconds:
            raise GenerationDeadlineExceeded('Подготовка генерации превысила лимит времени.')
        payload = {'model': selected.alias, 'messages': messages, 'stream': True,
                   'stream_options': {'include_usage': True},
                   'temperature': selected.temperature if temperature is None else temperature,
                   'max_tokens': max_tokens,
                   'chat_template_kwargs': {'enable_thinking': template_thinking}}
        if enabled_thinking and reasoning_budget_tokens is None and max_tokens >= 128:
            reasoning_budget_tokens = max(0, min(256, max_tokens // 3, max_tokens - 128))
        if reasoning_budget_tokens is not None:
            if not enabled_thinking or reasoning_budget_tokens + 128 > max_tokens:
                raise ProviderError('Бюджет должен оставлять место для итогового ответа при включённом рассуждении.')
            payload['reasoning_budget_tokens'] = reasoning_budget_tokens
        if not enabled_thinking:
            # A template flag alone can still allow spontaneous thought channels.
            # For strict JSON the enabled template plus zero budget closes them.
            payload['reasoning_budget_tokens'] = 0
        if stop:
            payload['stop'] = stop
        if schema is not None:
            payload['response_format'] = {'type': 'json_schema', 'json_schema': {'name': 'result', 'schema': schema}}
        generation = _Generation(generation_id or str(uuid.uuid4()), scope, current_action.get())
        generation.request_started = request_started
        generation.deadline = request_started + selected.generation_seconds
        loop = self._io_loop()
        with self._lock:
            if generation.id in self._active:
                raise ValueError('Generation ID is already active')
            self._active[generation.id] = generation
            generation.future = asyncio.run_coroutine_threadsafe(self._receive(generation, payload), loop)
        try:
            while True:
                self._check_cancel(generation)
                try:
                    item = generation.queue.get(timeout=.05)
                except queue.Empty:
                    continue
                self._check_cancel(generation)
                if isinstance(item, Exception):
                    raise item
                sink = current_token_sink.get()
                if sink and schema is None and item.text:
                    sink(item.text)
                yield item
                if item.done:
                    break
        finally:
            generation.future.cancel()
            with self._lock:
                self._active.pop(generation.id, None)

    def generate(self, messages, **options):
        text, result = [], None
        for delta in self.stream(messages, **options):
            text.append(delta.text)
            result = delta
        return GenerationResult(''.join(text), result.generation_id, result.metrics)

    def structured_generate(self, messages, schema, **options):
        result = self.generate(messages, schema=schema, **options)
        if result.metrics.finish_reason == 'length':
            raise StructuredOutputTruncated('Структурированный ответ модели оборвался на лимите токенов.')
        # Existing consumers additionally validate with their own Pydantic models.
        json.loads(result.text)
        return result

    def close(self):
        with self._lock:
            ids = list(self._active)
        for generation_id in ids:
            self.cancel(generation_id)
        if self._loop:
            async def finish():
                tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                if self._async_client is not None:
                    await self._async_client.aclose()
                    self._async_client = None
                await self._loop.shutdown_asyncgens()
            asyncio.run_coroutine_threadsafe(finish(), self._loop).result(timeout=5)
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)
            self._loop.close()
            self._loop = None
        self._client.close()
