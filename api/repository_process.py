"""Project checks with fixed argv, bounded output and owned process lifetime.

Running tests executes project code with the user's account permissions. This
is an explicit repository-task capability, not an operating-system sandbox.
No arbitrary shell commands, installation, Git writes or background services.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

from llm_contract import current_scope, GenerationCancelled, GenerationDeadlineExceeded


class ProjectChecks:
    def __init__(self, filesystem, timeout=45, output_limit=16000):
        self.filesystem = filesystem
        self.timeout = timeout
        self.output_limit = output_limit

    def root(self, workspace_id):
        return Path(self.filesystem.repository.get(workspace_id).root_path).resolve(strict=True)

    def discover(self, workspace_id):
        root = self.root(workspace_id)
        python = next((p for p in (root/'.venv/Scripts/python.exe', root/'.venv/bin/python')
                       if p.is_file() and not self.filesystem._is_link(p)), None)
        python = str(python) if python else (shutil.which('python') if getattr(sys, 'frozen', False) else sys.executable)
        checks = {}
        tests = root/'tests'
        if python and tests.is_dir() and not self.filesystem._is_link(tests):
            uses_pytest = False
            for test in list(tests.glob('test_*.py'))[:40]:
                try:
                    _, _, content = self.filesystem.read_text_file(workspace_id, test.relative_to(root).as_posix())
                    import re
                    uses_pytest = uses_pytest or bool(re.search(
                        r'(?m)^\s*(?:import pytest\b|from pytest\b)|^def test_\w*\(', content))
                except (OSError, ValueError):
                    continue
            command = ['-m', 'pytest', '-q', 'tests'] if uses_pytest else ['-m', 'unittest', 'discover', '-s', 'tests', '-v']
            checks['python-tests'] = {'title': 'Python pytest' if uses_pytest else 'Python unittest', 'argv': [python, *command]}
        # Read package metadata through the same credential/path policy as chat.
        if (root/'package.json').is_file():
            _, _, content = self.filesystem.read_text_file(workspace_id, 'package.json')
            scripts = json.loads(content).get('scripts', {})
            node, npm = shutil.which('node'), shutil.which('npm')
            cli = Path(npm).parent/'node_modules/npm/bin/npm-cli.js' if npm else None
            if node and cli and cli.is_file():
                for name in ('test', 'build', 'lint', 'typecheck'):
                    if isinstance(scripts.get(name), str) and scripts[name].strip():
                        checks['npm-'+name] = {'title': 'npm run '+name, 'argv': [node, str(cli), 'run', name]}
        return checks

    @staticmethod
    def _stop(process):
        if process.poll() is not None:
            return
        if os.name == 'nt':
            # Only this Popen child's process tree; never a global name/port kill.
            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW, check=False)
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)

    def run(self, workspace_id, check_id):
        checks = self.discover(workspace_id)
        if check_id not in checks:
            raise ValueError('Выберите ID существующей проверки из каталога проекта.')
        return self._run(self.root(workspace_id), checks[check_id]['argv'], check_id)

    def _run(self, root, argv, check_id):
        scope = current_scope.get()
        if scope:
            scope.check_deadline()
            if scope.cancelled.is_set():
                raise GenerationCancelled()
        started = time.perf_counter()
        env = dict(os.environ, CI='true', PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1',
                   GIT_TERMINAL_PROMPT='0', NO_COLOR='1')
        # Tool credentials are not automatically inherited by repository tests.
        for key in list(env):
            if any(word in key.upper() for word in ('TOKEN', 'SECRET', 'PASSWORD', 'API_KEY')):
                env.pop(key)
        isolated = tempfile.TemporaryDirectory(prefix='fortuna-project-check-')
        env.update(KONTUR_DATA_DIR=isolated.name, RAG_DB_PATH=str(Path(isolated.name)/'check.db'),
                   CHROMA_PATH=str(Path(isolated.name)/'chroma'), LLAMA_AUTOSTART='false',
                   PYTHONDONTWRITEBYTECODE='1', PYTHONPYCACHEPREFIX=str(Path(isolated.name)/'pycache'))
        process = subprocess.Popen(argv, cwd=root, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
            start_new_session=os.name != 'nt')
        tail = bytearray()
        total = 0

        def consume():
            nonlocal total
            while chunk := process.stdout.read(4096):
                total += len(chunk)
                tail.extend(chunk)
                del tail[:-self.output_limit]

        reader = threading.Thread(target=consume, daemon=True)
        reader.start()
        timed_out = False
        try:
            while process.poll() is None:
                if scope:
                    if scope.cancelled.is_set():
                        raise GenerationCancelled()
                    scope.check_deadline()
                if time.perf_counter()-started >= self.timeout:
                    timed_out = True
                    break
                time.sleep(.05)
        finally:
            self._stop(process)
            reader.join(timeout=2)
            process.stdout.close()
            isolated.cleanup()
        output = bytes(tail).decode('utf-8', errors='replace')
        empty_tests = 'Ran 0 tests' in output or 'no tests ran' in output
        return {'check': check_id, 'argv': argv, 'exit_code': process.returncode,
            'passed': process.returncode == 0 and not timed_out and not empty_tests,
            'timed_out': timed_out, 'seconds': round(time.perf_counter()-started, 3),
            'output': output, 'truncated': total > self.output_limit}
