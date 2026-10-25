"""Isolated real API for browser tests; no model startup, migration or private data."""
import argparse
import json
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, required=True)
parser.add_argument('--run', type=Path, required=True)
parser.add_argument('--repository-agent', action='store_true')
parser.add_argument('--workspace-sources', action='store_true')
parser.add_argument('--workspace-index', action='store_true')
args = parser.parse_args()
run = args.run.resolve()
run.mkdir(parents=True, exist_ok=True)
for key, value in {
    'KONTUR_DATA_DIR': str(run), 'RAG_DB_PATH': str(run / 'fixture.db'),
    'CHROMA_PATH': str(run / 'chroma'), 'LLAMA_AUTOSTART': 'false',
    'VERIFIER_ENABLED': 'false', 'ANONYMIZED_TELEMETRY': 'False',
}.items():
    os.environ[key] = value
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'api'))
import main
import uvicorn

root = run / 'workspace'
root.mkdir(exist_ok=True)
(root / 'Заметки.md').write_text('Отчёт Фортуны\nКод: 4817\nНагрузка: 27 кг.\n', encoding='utf-8')
(root / 'lib.rs').write_text('pub const ARCHIVE_CODE: u32 = 4817;\n', encoding='utf-8')
(root / '.gitignore').write_text('ignored/\n', encoding='utf-8')
(root / 'ignored').mkdir(exist_ok=True)
(root / 'ignored/скрыто.md').write_text('HIDDEN_FIXTURE', encoding='utf-8')
(root / 'История прогона.md').write_text('MEMORY_FILLER\n'+('Дополнительное объяснение для проверки длинного диалога. '*80),encoding='utf-8')
main.workspace_repository.select(str(root))
opened = []


def capture_open(path):
    opened.append(Path(path).relative_to(root).as_posix())
    (run / 'opened.json').write_text(json.dumps(opened, ensure_ascii=False), encoding='utf-8')


main.filesystem_provider.opener = capture_open
if args.workspace_index:
    (root / 'Ошибка индекса.md').write_text('Документ для проверки временного отказа embeddings.', encoding='utf-8')

    class FixtureVectorStore:
        """Controlled embeddings transport; API, SQLite and files remain real."""
        def __init__(self):
            self.rows = {}
            self.fail_once = True

        def get(self, where=None, include=None):
            rows = [(key, doc) for key, doc in self.rows.items()
                    if all(doc.metadata.get(k) == v for k, v in (where or {}).items())]
            return {'ids':[key for key, _ in rows], 'documents':[doc.page_content for _, doc in rows],
                    'metadatas':[doc.metadata.copy() for _, doc in rows], 'embeddings':[[1.0,0.0] for _ in rows]}

        def add_documents(self, docs, ids):
            for key, doc in zip(ids, docs):
                self.rows[key] = doc.model_copy(deep=True)
            if self.fail_once and any(doc.metadata.get('relative_path') == 'Ошибка индекса.md' for doc in docs):
                self.fail_once = False
                raise RuntimeError('Controlled one-time embedding outage')

        def delete(self, ids):
            for key in ids:
                self.rows.pop(key, None)

        def similarity_search(self, *args, **kwargs):
            return []

    index_store = FixtureVectorStore()
    main.workspace_vector_index.vectorstore_getter = lambda: index_store
if args.workspace_sources:
    from types import SimpleNamespace
    import model_runtime
    import assistant_core
    class SourceFixtureProvider:
        def generate(self, *args, **kwargs):
            return SimpleNamespace(text='Код: 4817.', metrics=SimpleNamespace(finish_reason='stop'))

        def structured_generate(self, *args, **kwargs):
            raise RuntimeError('Controlled source fixture has no structured model')

        def cancel(self, *args):
            pass

    provider = SourceFixtureProvider()
    model_runtime.get_llm_provider = lambda: provider
    assistant_core.get_llm_provider = lambda: provider
if main.assistant_core.conversation_memory:
    from types import SimpleNamespace
    # Browser checks never call a real model, even if a user's server is running.
    main.assistant_core.conversation_memory.selector=lambda entries,model:[p['id'] for p in entries[-3:]]
    main.assistant_core.conversation_memory.client_factory=lambda:SimpleNamespace(
        count_prompt=lambda messages,*args:40+len(json.dumps(messages,ensure_ascii=False))//3)
# Controlled author/reviewer for browser transport and UI checks. Real model
# quality is evaluated separately by evaluate_practical_summary.py.
from directory_summary import DirectorySummary
from pydantic_models import VerificationClaim
folder=root/'Документы прогона'
folder.mkdir()
for index in range(6):
    (folder/f'Тема {index}.md').write_text(f'В заметке описан отдельный проект {index} и его собственная задача.',encoding='utf-8')


def summary_draft(instance,source,index,text,model,client,findings=None):
    return [VerificationClaim(text=text,status='SUPPORTED',basis='EXPLICIT',
        evidence_quote=text,source_chunk_ids=[source['path']])],[]


def summary_audit(instance,claims,results,model,client):
    return [{'claim':c.model_dump(mode='json')} for c in claims]


DirectorySummary.draft=summary_draft
DirectorySummary.audit=summary_audit
from llm_contract import GenerationDeadlineExceeded
workspace_execute=main.assistant_core.workspace_skill.execute


async def execute_with_deadline_fixture(inputs,execution):
    if inputs.get('question')=='Прочитай Лимит времени.md':
        raise GenerationDeadlineExceeded('controlled deadline fixture')
    return await workspace_execute(inputs,execution)


main.assistant_core.workspace_skill.execute=execute_with_deadline_fixture
if args.repository_agent:
    from repository_agent import RepositoryStep
    (root / 'calc.py').write_text('def add(a, b):\n    return a - b\n', encoding='utf-8')
    (root / 'tests').mkdir()
    (root / 'tests/test_calc.py').write_text('import unittest\nfrom calc import add\n'
        'class CalcTests(unittest.TestCase):\n'
        '    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n', encoding='utf-8')

    def step(tool, **values):
        return RepositoryStep(**{'tool':tool,'path':'','text':'','replacement':'','check':'','note':'',**values})

    steps = iter([step('read',path='calc.py'),step('check',check='python-tests'),
        step('edit',path='calc.py',text='return a - b',replacement='return a + b'),
        step('finish')])
    main.assistant_core.repository_agent_skill.decide = lambda state, profile: next(steps)
# Main routes and real access middleware, filesystem and SQLite, with all model
# startup disabled. Uvicorn owns only this test server, on the supplied port.
uvicorn.run(main.app, host='127.0.0.1', port=args.port, lifespan='off', log_level='warning')
