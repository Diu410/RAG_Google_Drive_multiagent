"""Isolation and controlled dependencies for offline user-behaviour tests."""
import asyncio
import atexit
from contextlib import ExitStack
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'api'))
BOOTSTRAP = tempfile.TemporaryDirectory(prefix='kontur-unit-')

def close_bootstrap_clients():
    module = sys.modules.get('chroma_utils')
    clients = set()
    for name in ('vectorstore', 'workspace_vectorstore'):
        client = getattr(getattr(module, name, None), '_client', None)
        if client is not None and id(client._system) not in clients:
            clients.add(id(client._system))
            client._system.stop()

atexit.register(close_bootstrap_clients)
for key, value in {
    'RAG_DB_PATH': str(Path(BOOTSTRAP.name) / 'bootstrap.db'),
    'CHROMA_PATH': str(Path(BOOTSTRAP.name) / 'chroma'),
    'KONTUR_DATA_DIR': BOOTSTRAP.name,
    'LLAMA_AUTOSTART': 'false',
    'VERIFIER_ENABLED': 'false',
    # Existing suites exercise the explicit strict verifier and its legacy fallbacks.
    # Contextual review integration is covered separately with its production default.
    'ANSWER_REVIEW_MODE': 'strict',
    'ANONYMIZED_TELEMETRY': 'False',
}.items():
    os.environ[key] = value

from assistant_core import AssistantCore, ExecutionContext
from assistant_models import TaskStatus, ToolResult
from assistant_repository import AssistantRepository
from pydantic_models import QueryInput
from workspace import FilesystemProvider, WorkspaceRepository, WorkspaceSkill
from fixtures import create_workspace


class NoModel:
    """Fail loudly if an offline test accidentally reaches real inference."""
    def generate(self, *args, **kwargs):
        raise AssertionError('Unexpected real model generation in offline test')

    def structured_generate(self, *args, **kwargs):
        raise RuntimeError('Offline model unavailable')

    def cancel(self, *args):
        return True


def authorized_client(app, **kwargs):
    from fastapi.testclient import TestClient
    from local_access import API_TOKEN
    headers = {'X-Fortuna-Token': API_TOKEN, **kwargs.pop('headers', {})}
    return TestClient(app, base_url='http://127.0.0.1', headers=headers, **kwargs)


class KnowledgeStub:
    name = 'knowledge'
    description = 'Controlled knowledge fixture'
    input_schema = {}
    permissions = ()

    def __init__(self):
        self.calls = []

    async def execute(self, inputs, execution):
        self.calls.append(inputs)
        return ToolResult(success=True, tool=self.name, summary='Fixture',
                          data={'final': 'CONTROLLED_KNOWLEDGE', 'context': '', 'strategy': 'GENERAL_CHAT'})


class UserEnvironment:
    def __init__(self):
        self.temp = tempfile.TemporaryDirectory(prefix='kontur-user-test-')
        self.directory = Path(self.temp.name)
        self.database = self.directory / 'test.db'

        def connection():
            conn = sqlite3.connect(self.database, timeout=10)
            conn.row_factory = sqlite3.Row
            conn.execute('PRAGMA foreign_keys=ON')
            return conn

        self.connection = connection
        self.assistant = AssistantRepository(connection)
        self.repository = WorkspaceRepository(connection)
        self.opened = []
        self.filesystem = FilesystemProvider(self.repository, opener=self.opened.append)
        self.root = create_workspace(self.directory / 'Рабочая папка')
        self.workspace = self.repository.select(str(self.root))
        self.skill = WorkspaceSkill(self.filesystem)
        self.history = {}
        self.knowledge = KnowledgeStub()

        def save_history(session_id, question, answer, model):
            self.history.setdefault(session_id, []).extend([
                {'role': 'human', 'content': question}, {'role': 'ai', 'content': answer},
            ])

        self.core = AssistantCore(self.assistant, self.knowledge, save_history,
                                  workspace_skill=self.skill,
                                  history_loader=lambda session: self.history.get(session, []))
        self.patches = ExitStack()
        model = NoModel()
        self.patches.enter_context(patch('model_runtime.get_llm_provider', return_value=model))
        self.patches.enter_context(patch('assistant_core.get_llm_provider', return_value=model))

    async def ask(self, question, session_id=None, workspace_id=None):
        return (await self.core.process_input(QueryInput(
            question=question, model='FAST', session_id=session_id,
            workspace_id=workspace_id or self.workspace.id,
        )))[0]

    async def execute(self, question, operation='execute', **extra):
        session = self.assistant.ensure_session()
        task = self.assistant.create_task(session.id, question)
        self.assistant.transition(task, TaskStatus.RUNNING)
        execution = ExecutionContext(self.assistant, task)
        inputs = {'question': question, 'workspace_id': self.workspace.id, 'session_id': session.id,
                  'model': 'FAST', **extra}
        return await self.skill.execute_agent_step(operation, inputs, execution)

    def close(self):
        self.patches.close()
        self.temp.cleanup()
