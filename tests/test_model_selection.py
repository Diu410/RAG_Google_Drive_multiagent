"""Installed-model selection, persistence and protection of ongoing inference."""
import json
import os
import runpy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from support import UserEnvironment
from assistant_models import AssistantError
from llm_contract import ProviderError
from llama_provider import LlamaCppProvider
from llama_process import LlamaServerProcess
import model_profiles
import chat_model_selection as selection


class ModelSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'models').mkdir()
        self.four = self.root / 'models/chat.gguf'
        self.nine = self.root / 'models/Qwen3.5-9B-Q4_K_M.gguf'
        self.four.write_bytes(b'FOUR')
        self.nine.write_bytes(b'NINE')
        self.saved = self.root / 'chat-model-selection.json'
        self.profiles = dict(model_profiles.PROFILES)
        self.addCleanup(self.restore_profiles)
        for name, profile in self.profiles.items():
            model_profiles.PROFILES[name] = replace(profile, model_path=self.four)
        for patcher in (patch.object(model_profiles, 'DATA_DIR', self.root),
                        patch.object(model_profiles, 'BASE_PROFILE', replace(model_profiles.BASE_PROFILE, model_path=self.four)),
                        patch.object(selection, 'DATA_DIR', self.root)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.manager = SimpleNamespace(profile=model_profiles.get_profile(), error=None,
                                       prepare_replacement=Mock(), start_background=Mock())
        self.provider = LlamaCppProvider(manager=self.manager)

    def restore_profiles(self):
        model_profiles.PROFILES.clear()
        model_profiles.PROFILES.update(self.profiles)

    def test_both_models_are_listed_and_four_is_selected(self):
        result = selection.selection_status()
        self.assertEqual(result['selected_model_id'], 'qwen4b')
        self.assertEqual({m['id'] for m in result['chat_models'] if m['available']}, {'qwen4b', 'qwen9b'})

    def test_switch_retains_both_files_and_inference_modes(self):
        before = {name: (p.max_tokens, p.thinking) for name, p in model_profiles.PROFILES.items()}
        result = selection.switch_chat_model('qwen9b', self.provider)
        self.assertEqual(result['selected_model_id'], 'qwen9b')
        self.assertEqual(self.four.read_bytes(), b'FOUR')
        self.assertEqual(self.nine.read_bytes(), b'NINE')
        self.assertTrue(all(p.model_path == self.nine for p in model_profiles.PROFILES.values()))
        self.assertEqual(before, {name: (p.max_tokens, p.thinking) for name, p in model_profiles.PROFILES.items()})
        self.assertEqual(json.loads(self.saved.read_text())['model_id'], 'qwen9b')
        selection.switch_chat_model('qwen4b', self.provider)
        self.assertEqual(self.manager.profile.model_path, self.four)

    def test_saved_choice_is_restored_at_startup(self):
        self.saved.write_text('{"model_id":"qwen9b"}')
        import settings
        with patch.object(settings, 'DATA_DIR', self.root), patch.dict(os.environ, {'LLAMA_MODEL_PATH': str(self.four)}):
            namespace = runpy.run_path(str(Path(model_profiles.__file__)))
        self.assertEqual(namespace['get_profile']().model_path, self.nine)

    def test_missing_model_is_disabled_and_never_stops_current_model(self):
        self.nine.unlink()
        self.assertFalse(selection.selection_status()['chat_models'][1]['available'])
        with self.assertRaises(AssistantError) as error:
            selection.switch_chat_model('qwen9b', self.provider)
        self.assertEqual(error.exception.status_code, 404)
        self.manager.prepare_replacement.assert_not_called()

    def test_alternative_models_switch_and_restore_without_changing_modes(self):
        modes = {name: (p.max_tokens, p.thinking, p.request_seconds)
                 for name, p in model_profiles.PROFILES.items()}
        for model_id, filename in [('qwen9b-q3', 'Qwen3.5-9B-Q3_K_M.gguf'),
                                   ('gemma4e4b', 'gemma-4-E4B_q4_0-it.gguf')]:
            with self.subTest(model_id=model_id):
                candidate = self.root / 'models' / filename
                candidate.write_bytes(b'ALTERNATIVE')
                result = selection.switch_chat_model(model_id, self.provider)
                self.assertEqual(result['selected_model_id'], model_id)
                self.assertTrue(next(m for m in result['chat_models'] if m['id'] == model_id)['available'])
                self.assertEqual(modes, {name: (p.max_tokens, p.thinking, p.request_seconds)
                                        for name, p in model_profiles.PROFILES.items()})
                import settings
                with patch.object(settings, 'DATA_DIR', self.root), \
                     patch.dict(os.environ, {'LLAMA_MODEL_PATH': str(self.four)}):
                    namespace = runpy.run_path(str(Path(model_profiles.__file__)))
                self.assertEqual(namespace['get_profile']().model_path, candidate)
                self.assertEqual(self.four.read_bytes(), b'FOUR')
                self.assertEqual(self.nine.read_bytes(), b'NINE')

    def test_arbitrary_path_is_not_a_model_id(self):
        with self.assertRaises(AssistantError) as error:
            selection.switch_chat_model('../../other.gguf', self.provider)
        self.assertEqual(error.exception.status_code, 422)
        self.manager.prepare_replacement.assert_not_called()

    def test_active_inference_blocks_switch_without_cancellation(self):
        with self.provider._inference_use():
            with self.assertRaises(AssistantError) as error:
                selection.switch_chat_model('qwen9b', self.provider)
            self.assertEqual(error.exception.status_code, 409)
        self.manager.prepare_replacement.assert_not_called()
        self.assertFalse(self.saved.exists())

    def test_switch_blocks_new_inference_and_clears_old_tokenizer_cache(self):
        self.provider._token_counts['old'] = 42
        with self.provider.deployment_change():
            with self.assertRaises(ProviderError), self.provider._inference_use():
                pass
        self.assertFalse(self.provider._token_counts)
        self.assertFalse(self.provider._switching)

    def test_reselecting_current_model_does_not_restart(self):
        selection.switch_chat_model('qwen4b', self.provider)
        self.manager.prepare_replacement.assert_not_called()

    def test_failed_persistence_restores_old_deployment(self):
        with patch.object(selection.os, 'replace', side_effect=OSError('disk failure')):
            with self.assertRaises(OSError):
                selection.switch_chat_model('qwen9b', self.provider)
        self.assertEqual(model_profiles.get_profile().model_path, self.four)
        self.assertEqual(self.manager.profile.model_path, self.four)
        self.manager.start_background.assert_called_once()
        self.assertFalse(self.saved.exists())
        self.assertFalse(self.saved.with_suffix('.json.tmp').exists())

    def test_start_failure_restores_previous_saved_choice(self):
        self.saved.write_text('{"model_id":"qwen4b"}')
        self.manager.start_background.side_effect = [RuntimeError('thread failed'), None]
        with self.assertRaises(RuntimeError):
            selection.switch_chat_model('qwen9b', self.provider)
        self.assertEqual(json.loads(self.saved.read_text())['model_id'], 'qwen4b')
        self.assertEqual(self.manager.profile.model_path, self.four)

    def test_external_server_is_not_stopped(self):
        manager = LlamaServerProcess('http://127.0.0.1:9999', self.manager.profile)
        with patch.object(manager, 'health', return_value={'server_ready': True}), patch.object(manager, 'stop') as stop:
            with self.assertRaises(ProviderError):
                manager.prepare_replacement()
            stop.assert_not_called()

    def test_health_reuses_transport_but_checks_alias_and_readiness_each_time(self):
        import httpx
        manager = LlamaServerProcess('http://127.0.0.1:9999', self.manager.profile)
        self.addCleanup(manager.stop)
        client = Mock()
        client.get.side_effect = [
            httpx.Response(200),
            httpx.Response(200, json={'data': [{'id': self.manager.profile.alias}]},
                           request=httpx.Request('GET', 'http://localhost/v1/models')),
            httpx.Response(503),
            httpx.Response(200),
            httpx.Response(200, json={'data': [{'id': 'wrong-model'}]},
                           request=httpx.Request('GET', 'http://localhost/v1/models')),
        ]
        with patch('llama_process.httpx.Client', return_value=client) as factory:
            self.assertTrue(manager.health()['ready'])
            self.assertFalse(manager.health()['ready'])
            self.assertIn('alias', manager.health()['error'])
            factory.assert_called_once()
            self.assertEqual(client.get.call_count, 5)
        manager.stop()
        client.close.assert_called_once()
        self.assertIsNone(manager._health_client)
        # A subsequent startup/status poll can create a fresh transport.
        replacement = Mock()
        replacement.get.return_value = httpx.Response(503)
        with patch('llama_process.httpx.Client', return_value=replacement) as factory:
            self.assertFalse(manager.health()['ready'])
            factory.assert_called_once()

    def test_exited_child_does_not_grant_ownership_of_a_replacement_external_server(self):
        manager = LlamaServerProcess('http://127.0.0.1:9999', self.manager.profile)
        manager.child = SimpleNamespace(poll=lambda: 0)
        with patch.object(manager, 'health', return_value={'server_ready': True}), patch.object(manager, 'stop') as stop:
            with self.assertRaises(ProviderError):
                manager.prepare_replacement()
            stop.assert_not_called()

    def test_api_rejects_switch_during_an_assistant_task(self):
        from support import authorized_client as TestClient
        import main
        environment = UserEnvironment()
        self.addCleanup(environment.close)
        environment.core.active_tasks['task'] = object()
        client = TestClient(main.app)
        self.addCleanup(client.close)
        with patch.object(main, 'assistant_core', environment.core), patch.object(main, 'get_llm_provider') as provider:
            response = client.post('/api/chat-model', json={'model_id': 'qwen9b'})
        self.assertEqual(response.status_code, 409)
        provider.assert_not_called()

    def test_api_switch_returns_selected_choice(self):
        from support import authorized_client as TestClient
        import main
        environment = UserEnvironment()
        self.addCleanup(environment.close)
        client = TestClient(main.app)
        self.addCleanup(client.close)
        with patch.object(main, 'assistant_core', environment.core), patch.object(main, 'get_llm_provider', return_value=self.provider):
            response = client.post('/api/chat-model', json={'model_id': 'qwen9b'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['selected_model_id'], 'qwen9b')


if __name__ == '__main__':
    unittest.main()
