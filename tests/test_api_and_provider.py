"""HTTP user flows and streaming failures without external model calls."""
import asyncio
import json
import threading
import time
import unittest
from types import SimpleNamespace
from contextlib import ExitStack
from unittest.mock import patch

from support import UserEnvironment
from pydantic_models import QueryInput
import httpx
from support import authorized_client as TestClient
import main
from llama_provider import LlamaCppProvider
from llm_contract import ProviderError, GenerationCancelled, GenerationScope, current_scope, current_token_sink


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.env = UserEnvironment()
        self.addCleanup(self.env.close)
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        for name, value in [('assistant_core', self.env.core), ('assistant_repository', self.env.assistant),
                            ('workspace_repository', self.env.repository), ('filesystem_provider', self.env.filesystem)]:
            self.patches.enter_context(patch.object(main, name, value))
        # No lifespan context: offline API tests never start local model processes.
        self.client = TestClient(main.app)
        self.addCleanup(self.client.close)

    def test_input_returns_read_result_timing_and_task_status(self):
        result = self.client.post('/assistant/input', json={'question': 'Прочитай Без названия 4.md',
                                  'workspace_id': self.env.workspace.id})
        self.assertEqual(result.status_code, 200)
        self.assertIn('NOTE_FOUR', result.json()['answer'])
        self.assertGreaterEqual(float(result.headers['X-Process-Time-Ms']), 0)
        task = self.client.get('/assistant/tasks/' + result.json()['task_id'])
        self.assertEqual(task.status_code, 200)
        self.assertEqual(task.json()['task']['status'], 'COMPLETED')

    def test_legacy_query_alias_has_same_behavior(self):
        result = self.client.post('/query', json={'question': 'Прочитай Без названия 11.md',
                                  'workspace_id': self.env.workspace.id})
        self.assertEqual(result.status_code, 200)
        self.assertIn('4817', result.json()['answer'])

    def test_bad_inputs_are_422_not_internal_error(self):
        for body in ({'question': ''}, {'question': ' '}, {'question': 'x', 'model': 'unknown'},
                     {'question': 'x' * 32001}, {}):
            with self.subTest(body=str(body)[:80]):
                self.assertEqual(self.client.post('/assistant/input', json=body).status_code, 422)

    def test_workspace_file_read_and_traversal(self):
        url = f'/workspaces/{self.env.workspace.id}/file'
        result = self.client.get(url, params={'path': 'Без названия 4.md'})
        self.assertEqual(result.status_code, 200)
        self.assertIn('NOTE_FOUR', result.json()['content'])
        escape = self.client.get(url, params={'path': '../private.txt'})
        self.assertIn(escape.status_code, (400, 403, 404))
        self.assertNotEqual(escape.status_code, 500)

    def test_missing_task_is_404(self):
        self.assertEqual(self.client.get('/assistant/tasks/not-present').status_code, 404)

    def test_workspace_index_button_uses_vectors_without_llm_extraction(self):
        from workspace_vector_index import WorkspaceVectorIndex
        from test_sources_and_verification import MemoryStore
        indexer = WorkspaceVectorIndex(self.env.filesystem, lambda: MemoryStore())
        with patch.object(main, 'workspace_vector_index', indexer), patch.object(indexer, 'sync') as sync:
            sync.return_value = SimpleNamespace(indexed=2, unchanged=1, failed=0, unavailable=False)
            response = self.client.post(f'/workspaces/{self.env.workspace.id}/vector-sync?extract_relations=false')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['indexed'], 2)
            sync.assert_called_once_with(self.env.workspace.id, extract_relations=False)
            sync.reset_mock()
            response = self.client.post(f'/workspaces/{self.env.workspace.id}/vector-sync')
            self.assertEqual(response.status_code, 200)
            sync.assert_called_once_with(self.env.workspace.id)
            status = self.client.get(f'/workspaces/{self.env.workspace.id}/index-status')
            self.assertEqual(status.status_code, 200)
            self.assertEqual(status.json()['indexed_documents'], 0)
            self.assertGreater(status.json()['readable_documents'], 0)
            self.assertEqual(self.client.get('/workspaces/missing/index-status').status_code, 404)

    def test_live_final_matches_streamed_tokens(self):
        result = self.client.post('/assistant/live', json={'question': 'Прочитай Без названия 4.md',
                                  'workspace_id': self.env.workspace.id})
        self.assertEqual(result.status_code, 200)
        self.assertIn('application/x-ndjson', result.headers['content-type'])
        packets = [json.loads(line) for line in result.text.splitlines() if line.strip()]
        self.assertTrue(any(p['type'] == 'task' for p in packets))
        final = next(p for p in packets if p['type'] == 'result')
        tokens = ''.join(p['text'] for p in packets if p['type'] == 'token')
        self.assertIn('NOTE_FOUR', tokens)
        self.assertEqual(tokens, final['response']['answer'])

    def test_live_disconnect_unblocks_thread_with_full_token_queue(self):
        stopped, full = threading.Event(), threading.Event()
        cancelled = []

        async def process_input(request, *, source, on_task_started):
            on_task_started(SimpleNamespace(id='disconnect-test'))
            publish = current_token_sink.get()
            def flood():
                try:
                    for index in range(2000):
                        if index >= 255:
                            full.set()
                        publish('x')
                finally:
                    stopped.set()
            await asyncio.to_thread(flood)
            self.fail('Disconnected stream must not finish as a successful response')

        async def scenario():
            with patch.object(main, 'assistant_core', SimpleNamespace(
                    process_input=process_input, cancel_task=cancelled.append)):
                response = await main.live_query_endpoint(QueryInput(question='hello'))
                iterator = response.body_iterator
                try:
                    self.assertEqual(json.loads(await anext(iterator))['type'], 'stage')
                    self.assertTrue(await asyncio.to_thread(full.wait, 3))
                finally:
                    await asyncio.wait_for(iterator.aclose(), 3)
                self.assertTrue(await asyncio.to_thread(stopped.wait, 3), 'Token publisher stayed blocked after disconnect')
                self.assertIn('disconnect-test', cancelled)
        asyncio.run(scenario())


class ProviderTests(unittest.TestCase):
    def test_tokenizer_recovers_after_loading_without_caching_byte_estimates(self):
        ready = False
        requests = []

        def handle(request):
            requests.append(request.url.path)
            if not ready:
                return httpx.Response(503, json={'error': 'loading'})
            if request.url.path == '/apply-template':
                return httpx.Response(200, json={'prompt': 'actual template'})
            return httpx.Response(200, json={'tokens': [1, 2, 3]})

        provider = LlamaCppProvider(transport=httpx.MockTransport(handle))
        self.addCleanup(provider.close)
        messages = [{'role': 'user', 'content': 'Что сказано в документах?'}]
        self.assertGreater(provider.count_prompt(messages, 'assistant', True), 3)
        ready = True
        self.assertEqual(provider.count_prompt(messages, 'assistant', True), 3)
        self.assertEqual(provider.count_prompt(
            [{'role': 'user', 'content': 'Кто такой Мирон?'}], 'assistant', True), 3)
        count = len(requests)
        self.assertEqual(provider.count_prompt(messages, 'assistant', True), 3)
        self.assertEqual(len(requests), count, 'Only exact tokenizer counts may be cached')

    def test_tokenizer_retries_after_connection_and_tokenization_errors(self):
        for failure in ('connection', 'tokenize_503', 'invalid_tokens'):
            with self.subTest(failure=failure):
                ready = False

                def handle(request):
                    if not ready and failure == 'connection':
                        raise httpx.ConnectError('Server restarting', request=request)
                    if request.url.path == '/apply-template':
                        return httpx.Response(200, json={'prompt': 'template'})
                    if not ready:
                        return (httpx.Response(503) if failure == 'tokenize_503'
                                else httpx.Response(200, json={'tokens': 'not a list'}))
                    return httpx.Response(200, json={'tokens': [1, 2]})

                provider = LlamaCppProvider(transport=httpx.MockTransport(handle))
                self.addCleanup(provider.close)
                messages = [{'role': 'user', 'content': 'Проверь источник.'}]
                self.assertGreater(provider.count_prompt(messages, 'assistant', False), 2)
                ready = True
                self.assertEqual(provider.count_prompt(messages, 'assistant', False), 2)

    def test_reasoning_budget_reaches_server_and_reserves_answer_tokens(self):
        requests = []
        def handle(request):
            if request.url.path == '/health':
                return httpx.Response(200, json={'status': 'ok'})
            if request.url.path == '/apply-template':
                return httpx.Response(200, json={'prompt': 'small'})
            if request.url.path == '/tokenize':
                return httpx.Response(200, json={'tokens': [1, 2, 3]})
            requests.append(json.loads(request.content))
            return httpx.Response(200, content=self.event('Answer') + 'data: [DONE]\n\n')
        provider = LlamaCppProvider(transport=httpx.MockTransport(handle))
        self.addCleanup(provider.close)
        provider.generate([{'role': 'user', 'content': 'hi'}], thinking=True,
                          reasoning_budget_tokens=256, max_tokens=600)
        self.assertEqual(requests[0]['reasoning_budget_tokens'], 256)
        self.assertEqual(requests[0]['max_tokens'], 600)
        for value in (-1, True, '256'):
            with self.subTest(value=value), self.assertRaises(ProviderError):
                provider.generate([{'role': 'user', 'content': 'hi'}], reasoning_budget_tokens=value)
        for thinking, budget in [(False, 256), (True, 500)]:
            with self.subTest(thinking=thinking), self.assertRaises(ProviderError):
                provider.generate([{'role': 'user', 'content': 'hi'}], thinking=thinking,
                                  reasoning_budget_tokens=budget, max_tokens=600)
        self.assertEqual(len(requests), 1)

    def test_disabled_thinking_returns_json_even_when_template_flag_is_soft(self):
        schema = {'type': 'object', 'properties': {'supported': {'type': 'boolean'}},
                  'required': ['supported'], 'additionalProperties': False}
        templates, requests = [], []

        def handle(request):
            if request.url.path == '/health':
                return httpx.Response(200, json={'status': 'ok'})
            body = json.loads(request.content)
            if request.url.path == '/apply-template':
                templates.append(body['chat_template_kwargs'])
                return httpx.Response(200, json={'prompt': 'small'})
            if request.url.path == '/tokenize':
                return httpx.Response(200, json={'tokens': [1, 2, 3]})
            requests.append(body)
            if 'response_format' not in body:
                # A forced thought-ending template can leak subsequent analysis
                # into ordinary text when no JSON grammar constrains the decoder.
                text = ('PRIVATE_ANALYSIS' if body['chat_template_kwargs']['enable_thinking'] else 'Answer')
                stream = self.event(text) + 'data: [DONE]\n\n'
                return httpx.Response(200, content=stream)
            # A runtime may allow spontaneous thoughts with a disabled template,
            # and may only recognise their closing tags with the template enabled.
            bounded = (body['chat_template_kwargs'].get('enable_thinking') is True
                       and body.get('reasoning_budget_tokens') == 0)
            if bounded:
                stream = self.event('{"supported":false}')
                reason = 'stop'
            else:
                stream = self.event(reasoning='Unbounded thoughts consume the whole budget.')
                reason = 'length'
            stream += 'data: ' + json.dumps({'choices': [{'delta': {}, 'finish_reason': reason}]}) + '\n\n'
            return httpx.Response(200, content=stream + 'data: [DONE]\n\n')

        provider = LlamaCppProvider(transport=httpx.MockTransport(handle))
        self.addCleanup(provider.close)
        for profile, options in [('FAST', {}), ('REASONING', {'thinking': False})]:
            with self.subTest(profile=profile):
                result = provider.structured_generate([{'role': 'user', 'content': 'Check the source.'}],
                    schema, profile=profile, max_tokens=256, **options)
                self.assertEqual(json.loads(result.text), {'supported': False})
                self.assertEqual(result.metrics.finish_reason, 'stop')
        self.assertTrue(templates)
        self.assertTrue(all(t == requests[0]['chat_template_kwargs'] for t in templates))
        plain = provider.generate([{'role': 'user', 'content': 'Give the answer.'}], thinking=False)
        self.assertEqual(plain.text, 'Answer')

    def provider(self, body, status=200):
        def handle(request):
            if request.url.path == '/health':
                return httpx.Response(200, json={'status': 'ok'})
            if request.url.path == '/apply-template':
                return httpx.Response(200, json={'prompt': 'small'})
            if request.url.path == '/tokenize':
                return httpx.Response(200, json={'tokens': [1, 2, 3]})
            return httpx.Response(status, content=body, headers={'Content-Type': 'text/event-stream'})
        provider = LlamaCppProvider(transport=httpx.MockTransport(handle))
        self.addCleanup(provider.close)
        return provider

    @staticmethod
    def event(content=None, reasoning=None, usage=None):
        data = {'choices': [{'delta': {'content': content, 'reasoning_content': reasoning}}]}
        if usage:
            data['usage'] = usage
        return 'data: ' + json.dumps(data) + '\n\n'

    def test_stream_combines_content_and_preserves_usage(self):
        provider = self.provider(self.event('Добрый ') + self.event('день') +
                                 self.event(usage={'prompt_tokens': 3, 'completion_tokens': 2}) + 'data: [DONE]\n\n')
        result = provider.generate([{'role': 'user', 'content': 'hi'}], profile='FAST')
        self.assertEqual(result.text, 'Добрый день')
        self.assertEqual(result.metrics.completion_tokens, 2)
        self.assertGreaterEqual(result.metrics.preparation_ms, 0)
        self.assertGreaterEqual(result.metrics.request_total_ms, result.metrics.generation_total_ms)
        self.assertGreaterEqual(result.metrics.request_total_ms, result.metrics.preparation_ms)
        self.assertEqual(provider._active, {})

    def test_reasoning_is_not_mixed_into_answer(self):
        provider = self.provider(self.event(reasoning='PRIVATE_THOUGHT') + self.event('Answer') + 'data: [DONE]\n\n')
        self.assertEqual(provider.generate([{'role': 'user', 'content': 'hi'}]).text, 'Answer')

    def test_output_limit_is_preserved_and_incomplete_structured_result_is_rejected(self):
        from llm_contract import StructuredOutputTruncated
        ending = 'data: ' + json.dumps({'choices': [{'delta': {}, 'finish_reason': 'length'}]}) + '\n\n'
        provider = self.provider(self.event('{"ok":true}') + ending + 'data: [DONE]\n\n')
        result = provider.generate([{'role': 'user', 'content': 'hi'}])
        self.assertEqual(result.metrics.finish_reason, 'length')
        # Even syntactically valid JSON does not prove generation was complete.
        with self.assertRaises(StructuredOutputTruncated):
            provider.structured_generate([{'role': 'user', 'content': 'hi'}], schema={'type': 'object'})
        self.assertEqual(provider._active, {})

    def test_truncated_stream_is_error_instead_of_success(self):
        provider = self.provider(self.event('Incomplete'))
        with self.assertRaisesRegex(ProviderError, 'оборвался'):
            provider.generate([{'role': 'user', 'content': 'hi'}])
        self.assertEqual(provider._active, {})

    def test_invalid_json_packet_is_safe_provider_error(self):
        provider = self.provider('data: {INVALID_PRIVATE_TEXT}\n\n')
        with self.assertRaises(ProviderError) as error:
            provider.generate([{'role': 'user', 'content': 'hi'}])
        self.assertNotIn('INVALID_PRIVATE_TEXT', str(error.exception))

    def test_http_error_does_not_echo_prompt_or_server_body(self):
        provider = self.provider('PRIVATE_DOCUMENT_CONTENT', status=500)
        with self.assertRaises(ProviderError) as error:
            provider.generate([{'role': 'user', 'content': 'hi'}])
        self.assertNotIn('PRIVATE_DOCUMENT_CONTENT', str(error.exception))

    def test_structured_generation_does_not_publish_json_to_live_user(self):
        provider = self.provider(self.event('{"ok":true}') + 'data: [DONE]\n\n')
        emitted = []
        token = current_token_sink.set(emitted.append)
        try:
            response = provider.structured_generate([{'role': 'user', 'content': 'hi'}], schema={'type': 'object'})
        finally:
            current_token_sink.reset(token)
        self.assertEqual(json.loads(response.text), {'ok': True})
        self.assertEqual(emitted, [])

    def test_cancelled_scope_makes_no_generation_request(self):
        provider = self.provider('data: [DONE]\n\n')
        scope = GenerationScope('cancelled-task')
        scope.cancelled.set()
        token = current_scope.set(scope)
        try:
            with self.assertRaises(GenerationCancelled):
                provider.generate([{'role': 'user', 'content': 'hi'}])
        finally:
            current_scope.reset(token)
        self.assertEqual(provider._active, {})

    def test_cancellation_closes_stalled_socket_and_releases_consumer(self):
        entered, closed = threading.Event(), threading.Event()

        class Stalled(httpx.AsyncByteStream):
            async def __aiter__(self):
                entered.set()
                await asyncio.sleep(30)
                yield b'data: [DONE]\n\n'

            async def aclose(self):
                closed.set()

        async def handle(request):
            return httpx.Response(200, stream=Stalled())

        provider = LlamaCppProvider(transport=httpx.MockTransport(handle))
        self.addCleanup(provider.close)
        self.addCleanup(patch.stopall)
        patch.object(provider, 'health', return_value={'ready': True}).start()
        patch.object(provider, 'count_prompt', return_value=3).start()
        errors = []
        def consume():
            try:
                provider.generate([{'role': 'user', 'content': 'hi'}], generation_id='stalled')
            except BaseException as error:
                errors.append(error)
        thread = threading.Thread(target=consume, daemon=True)
        thread.start()
        self.assertTrue(entered.wait(3))
        self.assertTrue(provider.cancel('stalled'))
        thread.join(3)
        self.assertFalse(thread.is_alive(), 'Cancellation left the user request blocked')
        self.assertTrue(closed.wait(3))
        self.assertIsInstance(errors[0], GenerationCancelled)
        self.assertEqual(provider._active, {})
