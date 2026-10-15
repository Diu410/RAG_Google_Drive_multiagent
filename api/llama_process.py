"""Own only the child we spawn. Existing inference servers are never terminated."""
import atexit
import subprocess
import threading
import time
from urllib.parse import urlparse
from pathlib import Path
import httpx
from model_profiles import SERVER_EXECUTABLE, AUTOSTART, STARTUP_TIMEOUT
from llm_contract import ProviderError
from settings import DATA_DIR


class LlamaServerProcess:
    def __init__(self, url, profile, *, embedding=False, executable=SERVER_EXECUTABLE,
                 autostart=AUTOSTART, timeout=STARTUP_TIMEOUT):
        self.url, self.profile, self.embedding = url, profile, embedding
        self.executable, self.autostart, self.timeout = Path(executable), autostart, timeout
        self.child = None
        self.log_path = DATA_DIR / ('embedding-server.log' if embedding else 'llama-server.log')
        self.state = 'stopped'
        self.error = None
        self._lock = threading.Lock()
        self._health_lock = threading.Lock()
        self._health_client = None
        self._stop = threading.Event()
        self._thread = None
        atexit.register(self.stop)

    def health(self):
        try:
            # Reuse the HTTP transport without caching readiness.
            # Serialize close with probes, including UI polling during a switch.
            with self._health_lock:
                if self._health_client is None:
                    self._health_client = httpx.Client(timeout=2, trust_env=False)
                client = self._health_client
                response = client.get(self.url + '/health')
                if response.status_code == 200:
                    models = client.get(self.url + '/v1/models')
                    models.raise_for_status()
                    ids = {m['id'] for m in models.json().get('data', [])}
                    if self.profile.alias not in ids:
                        return {'ready': False, 'server_ready': True, 'model_loaded': False,
                                'error': f'Сервер не содержит модель с alias {self.profile.alias}'}
                    return {'ready': True, 'server_ready': True, 'model_loaded': True}
                return {'ready': False, 'server_ready': True, 'model_loaded': False}
        except (httpx.HTTPError, ValueError, KeyError):
            return {'ready': False, 'server_ready': False, 'model_loaded': False}

    def status(self):
        health = self.health()
        if not health['ready'] and self.child and self.child.poll() is not None:
            self.state = 'error'
            self.error = f'Model process exited ({self.child.returncode}). Log: {self.log_path}'
        return {**health, 'state': 'ready' if health['ready'] else self.state,
                'process_started': bool(self.child and self.child.poll() is None),
                'owned': self.child is not None, 'error': health.get('error') or self.error,
                'model_path': str(self.profile.model_path), 'log_path': str(self.log_path)}

    def start_background(self):
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._start, daemon=True, name='llama-startup')
            self._thread.start()

    def _start(self):
        try:
            self.ensure_ready()
        except (ProviderError, OSError) as exc:
            self.stop()
            self.error, self.state = str(exc), 'error'

    def command(self):
        parsed = urlparse(self.url)
        if parsed.hostname not in {'127.0.0.1', 'localhost', '::1'} or parsed.scheme != 'http':
            raise ProviderError('Автозапуск разрешён только для локального HTTP llama-server')
        p = self.profile
        return [str(self.executable), '--model', str(p.model_path), '--alias', p.alias,
                '--host', parsed.hostname, '--port', str(parsed.port or 80),
                '--ctx-size', str(p.context_size), '--n-gpu-layers', p.gpu_layers,
                '--threads', str(p.threads), '--batch-size', str(p.batch_size),
                '--parallel', '1', '--no-webui',
                *(['--embedding', '--pooling', 'mean', '--ubatch-size', str(p.batch_size)]
                  if self.embedding else ['--jinja'])]

    def ensure_ready(self):
        with self._lock:
            if self._stop.is_set():
                raise ProviderError('Model startup was cancelled')
            probe = self.health()
            if probe['ready']:
                self.state, self.error = 'ready', None
                return
            if probe.get('error'):
                raise ProviderError(probe['error'])
            if not probe['server_ready'] and not (self.child and self.child.poll() is None):
                if not self.autostart:
                    raise ProviderError(f'Сервер модели недоступен: {self.url}')
                if not self.profile.model_path.is_file():
                    raise ProviderError(f'GGUF модель не найдена: {self.profile.model_path}')
                if not self.executable.is_file():
                    raise ProviderError(f'llama-server не найден: {self.executable}')
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open('ab') as log:
                    self.child = subprocess.Popen(self.command(), stdout=log, stderr=log,
                        stdin=subprocess.DEVNULL, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            self.state = 'loading'
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline and not self._stop.is_set():
            if self.child and self.child.poll() is not None:
                raise ProviderError(f'Модель не загрузилась (код {self.child.returncode}). Журнал: {self.log_path}')
            probe = self.health()
            if probe['ready']:
                self.state, self.error = 'ready', None
                return
            if probe.get('error'):
                raise ProviderError(probe['error'])
            self._stop.wait(0.25)
        raise ProviderError(f'Истекло время запуска модели. Журнал: {self.log_path}')

    def stop(self):
        self._stop.set()
        with self._lock:
            child, self.child = self.child, None
            if child and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
            self.state = 'stopped'
        with self._health_lock:
            if self._health_client is not None:
                self._health_client.close()
                self._health_client = None

    def prepare_replacement(self):
        if not (self.child and self.child.poll() is None) and self.health()['server_ready']:
            raise ProviderError('Этот сервер запущен вне приложения. Переключение доступно для управляемой локальной модели.')
        if not self.autostart:
            raise ProviderError('Включите локальный автозапуск для переключения моделей.')
        self.stop()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                raise ProviderError('Предыдущая загрузка ещё завершается. Повторите переключение позже.')
