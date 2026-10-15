"""Review regressions: local boundaries, resource budgets and recoverable results."""
import asyncio
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

from support import UserEnvironment, authorized_client
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
import main
import chat_routes
import data_migration
from local_access import API_TOKEN, LocalAccessMiddleware
from extraction_limits import ExtractionLimitError
from pydantic_models import QueryInput, VerificationResult


class LocalAccessTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app, base_url='http://127.0.0.1')
        self.addCleanup(self.client.close)

    def test_sensitive_routes_require_token_even_without_origin(self):
        for method, path in [('GET', '/chats'), ('GET', '/list-docs'), ('GET', '/workspaces/active'),
                             ('POST', '/assistant/input'), ('POST', '/upload-doc'), ('DELETE', '/chats/example')]:
            with self.subTest(path=path):
                self.assertEqual(self.client.request(method, path).status_code, 401)

    def test_foreign_origin_cannot_mutate_with_or_without_token(self):
        with patch.object(main, 'document_service') as service:
            for token in ('', API_TOKEN):
                response = self.client.post('/upload-doc', files={'file': ('test.txt', b'FIXTURE')},
                    headers={'Origin': 'https://foreign.invalid', 'X-Fortuna-Token': token})
                self.assertEqual(response.status_code, 403)
            service.upload.assert_not_called()

    def test_foreign_host_is_rejected_even_with_token(self):
        self.assertEqual(self.client.get('/chats', headers={
            'Host': 'foreign.invalid', 'X-Fortuna-Token': API_TOKEN}).status_code, 400)

    def test_bootstrap_requires_custom_header_and_origin(self):
        self.assertEqual(self.client.get('/access-token').status_code, 403)
        self.assertEqual(self.client.get('/access-token', headers={
            'X-Fortuna-Client': 'web', 'Origin': 'https://foreign.invalid'}).status_code, 403)
        response = self.client.get('/access-token', headers={'X-Fortuna-Client': 'web'})
        self.assertEqual(response.json()['token'], API_TOKEN)
        self.assertEqual(response.headers['cache-control'], 'no-store')

    def test_desktop_does_not_offer_http_token_bootstrap(self):
        app = FastAPI()
        app.add_middleware(LocalAccessMiddleware, desktop=True)
        with TestClient(app, base_url='http://127.0.0.1') as client:
            self.assertEqual(client.get('/access-token', headers={'X-Fortuna-Client': 'web'}).status_code, 403)

    def test_preflight_allows_app_token_header(self):
        response = self.client.options('/assistant/input', headers={'Origin': 'http://tauri.localhost',
            'Access-Control-Request-Method': 'POST', 'Access-Control-Request-Headers': 'X-Fortuna-Token,Content-Type'})
        self.assertEqual(response.status_code, 200)

    def test_urlencoded_never_reaches_upload_form_parser(self):
        with patch('python_multipart.QuerystringParser.write', side_effect=AssertionError('parser reached')):
            response = self.client.post('/upload-doc', content=b'a;' * 1024,
                headers={'X-Fortuna-Token': API_TOKEN, 'Content-Type': 'application/x-www-form-urlencoded'})
        self.assertEqual(response.status_code, 415)

    def test_request_limit_covers_content_length_and_streamed_body(self):
        app = FastAPI()
        app.add_middleware(LocalAccessMiddleware)
        @app.post('/echo')
        async def echo(request: Request):
            return {'length': len(await request.body())}
        with patch.dict('os.environ', {'MAX_JSON_REQUEST_BYTES': '8'}), \
                TestClient(app, base_url='http://127.0.0.1', headers={'X-Fortuna-Token': API_TOKEN}) as client:
            self.assertEqual(client.post('/echo', content=b'012345678').status_code, 413)
            self.assertEqual(client.post('/echo', content=iter([b'01234', b'56789'])).status_code, 413)


class FileAndExtractionTests(unittest.TestCase):
    def setUp(self):
        self.env = UserEnvironment()
        self.addCleanup(self.env.close)

    def test_open_rejects_executables_scripts_shortcuts_and_unknown_types(self):
        for extension in ('.exe', '.cmd', '.bat', '.ps1', '.py', '.lnk', '.url', '.hta', '.reg', '.vbs'):
            path = self.env.root / ('open-review' + extension)
            path.write_text('FIXTURE', encoding='utf-8')
            with self.subTest(extension=extension), self.assertRaises(PermissionError):
                self.env.filesystem.open_file(self.env.workspace.id, path.name)
        self.assertFalse(self.env.opened)
        self.env.filesystem.open_file(self.env.workspace.id, 'Без названия 4.md')
        self.assertEqual(len(self.env.opened), 1)

    def test_credentials_cannot_be_read_or_indexed(self):
        from workspace_vector_index import WorkspaceVectorIndex
        from test_sources_and_verification import MemoryStore
        for name in ('token.json', 'credentials.json', 'client_secret_demo.json', 'service-account-test.json'):
            (self.env.root / name).write_text('{"fixture":"SECRET_FIXTURE"}', encoding='utf-8')
            with self.assertRaises(PermissionError):
                self.env.filesystem.read_text_file(self.env.workspace.id, name)
        self.env.filesystem.sync_index(self.env.workspace.id)
        records = self.env.repository.indexed_files(self.env.workspace.id)
        self.assertFalse(any(item.readable for item in records if item.name == 'token.json'))
        store = MemoryStore()
        WorkspaceVectorIndex(self.env.filesystem, lambda: store).sync(self.env.workspace.id, extract_relations=False)
        self.assertFalse(any('SECRET_FIXTURE' in doc.page_content for doc in store.rows.values()))

    def test_docx_expansion_limits_do_not_fall_back_to_unbounded_loader(self):
        import formula_processing
        path = self.env.root / 'oversized.docx'
        with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('word/document.xml', b'A' * 1024)
        with patch.object(formula_processing, 'MAX_DOCX_XML_BYTES', 100), \
                patch('langchain_community.document_loaders.Docx2txtLoader') as fallback:
            with self.assertRaises(ExtractionLimitError):
                formula_processing.load_docx_with_formulas(str(path))
            fallback.assert_not_called()

    def test_credentials_with_arbitrary_filename_are_excluded_by_content(self):
        path = self.env.root / 'custom-identity.json'
        path.write_text(json.dumps({'type': 'service_account', 'private_key': 'PRIVATE_FIXTURE_MATERIAL_ONLY'}), encoding='utf-8')
        with self.assertRaises(PermissionError):
            self.env.filesystem.read_text_file(self.env.workspace.id, path.name)
        self.env.filesystem.sync_index(self.env.workspace.id)
        record = next(item for item in self.env.repository.indexed_files(self.env.workspace.id) if item.name == path.name)
        self.assertFalse(record.readable)

    def test_pdf_coverage_failure_does_not_fall_back_to_partial_text(self):
        from chroma_utils import load_pdf
        with patch('chroma_utils.load_pdf_complete', side_effect=ValueError('missing OCR page')), \
                patch('langchain_community.document_loaders.PyMuPDFLoader') as fallback:
            with self.assertRaisesRegex(ValueError, 'missing OCR'):
                load_pdf('fixture.pdf')
            fallback.assert_not_called()

    def test_pdf_page_count_is_checked_before_text_or_raster_extraction(self):
        import fitz
        import formula_processing
        path = self.env.root / 'pages.pdf'
        with fitz.open() as pdf:
            pdf.new_page(); pdf.new_page(); pdf.save(path)
        with patch.object(formula_processing, 'MAX_PDF_PAGES', 1), \
                patch.object(formula_processing, 'load_pdf_with_formulas') as extractor:
            with self.assertRaises(ExtractionLimitError):
                formula_processing.load_pdf_complete(str(path))
            extractor.assert_not_called()

    def test_workspace_reader_reports_size_limit_instead_of_truncating(self):
        from workspace_document_reader import WorkspaceDocumentReader
        path = self.env.root / 'long.txt'
        path.write_text('A' * 100, encoding='utf-8')
        with self.assertRaises(ExtractionLimitError):
            WorkspaceDocumentReader(max_output_characters=50).read(path)

    def test_drive_download_buffer_is_bounded_before_accumulation(self):
        from gdrive_extractor import LimitedDownload
        buffer = LimitedDownload(5)
        buffer.write(b'1234')
        with self.assertRaises(ExtractionLimitError):
            buffer.write(b'56')
        self.assertEqual(buffer.getvalue(), b'1234')

    def test_drive_download_budget_covers_all_files_including_partial_downloads(self):
        from gdrive_extractor import LimitedDownload
        state = {'bytes': 0, 'deadline': time.monotonic() + 30}
        first, second = LimitedDownload(5, state), LimitedDownload(5, state)
        with patch('gdrive_extractor.MAX_DRIVE_BYTES', 6):
            first.write(b'1234')
            with self.assertRaises(ExtractionLimitError):
                second.write(b'567')
        self.assertEqual(state['bytes'], 4)
        self.assertEqual(second.getvalue(), b'')

    def test_drive_partial_import_reports_extraction_failures(self):
        from langchain_core.documents import Document
        extractor = SimpleNamespace(process_folder=lambda url: [Document(page_content='FIXTURE',
            metadata={'source': 'good.txt', 'gdrive_id': 'good'})],
            last_report={'attempted': 2, 'failed': [{'id': 'bad', 'name': 'broken.pdf', 'error': 'ValueError'}],
                         'unsupported': 0, 'limited': False})
        with patch.object(main, 'GDRIVE_EXTRACTOR', extractor), \
                patch.object(main, 'document_service', SimpleNamespace(upload=lambda *args: 7)), \
                authorized_client(main.app) as client:
            response = client.post('/upload-gdrive', json={'folder_url': 'fixture'})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['status'], 'partial')
        self.assertEqual(response.json()['indexed_files'], 1)
        self.assertEqual(response.json()['failed'][0]['name'], 'broken.pdf')

    def test_drive_total_failure_explains_missing_files_and_limit(self):
        extractor = SimpleNamespace(process_folder=lambda url: [],
            last_report={'attempted': 1, 'failed': [{'id': 'bad', 'name': 'broken.pdf', 'error': 'ValueError'}],
                         'unsupported': 0, 'limited': True})
        with patch.object(main, 'GDRIVE_EXTRACTOR', extractor), authorized_client(main.app) as client:
            response = client.post('/upload-gdrive', json={'folder_url': 'fixture'})
        self.assertEqual(response.status_code, 422)
        self.assertIn('broken.pdf', response.json()['detail'])
        self.assertIn('лимит', response.json()['detail'])

    def test_document_logging_does_not_include_page_text(self):
        import chroma_utils
        from langchain_core.documents import Document
        with patch.object(chroma_utils, 'embedding_model', object()), patch.object(chroma_utils, 'vectorstore', object()), \
                patch.object(chroma_utils, 'IndexTransactions') as transactions, \
                patch('langchain_community.document_loaders.TextLoader') as loader, \
                patch('db_utils.get_document', return_value=None):
            loader.return_value.load.return_value = [Document(page_content='PRIVATE_DOCUMENT_PREVIEW_MARKER')]
            with self.assertLogs('chroma_utils', level='INFO') as logs:
                self.assertTrue(chroma_utils.index_document_to_chroma('fixture.txt', 1))
            self.assertNotIn('PRIVATE_DOCUMENT_PREVIEW_MARKER', '\n'.join(logs.output))


class ServerChatTests(unittest.TestCase):
    def setUp(self):
        self.env = UserEnvironment()
        self.addCleanup(self.env.close)
        with closing(self.env.connection()) as conn, conn:
            conn.execute('''CREATE TABLE application_logs (id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,user_query TEXT,gpt_response TEXT,model TEXT,knowledge_epoch INTEGER)''')
            conn.execute('CREATE TABLE application_state(key TEXT PRIMARY KEY,value TEXT)')
        self.env.core.response_saver = self.env.assistant.save_response
        self.main_patch = patch.object(main, 'assistant_core', self.env.core)
        self.chat_patch = patch.object(chat_routes, 'get_db_connection', self.env.connection)
        self.main_patch.start(); self.chat_patch.start()
        self.addCleanup(self.main_patch.stop); self.addCleanup(self.chat_patch.stop)
        self.client = authorized_client(main.app)
        self.addCleanup(self.client.close)
        self.question = 'Прочитай Без названия 4.md'

    def initial_chat(self):
        response = self.client.put('/chats/review', json={'id': 'review', 'title': 'Мирон', 'session_id': None,
            'updated_at': '2026-10-15T10:00:00Z', 'messages': [
                {'id': 'user-message', 'role': 'user', 'content': self.question, 'timestamp': '2026-10-15T10:00:00Z'}]})
        self.assertEqual(response.status_code, 200)
        return response.json()

    def ask(self, path='/assistant/input', **extra):
        return self.client.post(path, json={'question': self.question, 'chat_id': 'review',
            'request_id': 'user-message', 'workspace_id': self.env.workspace.id, 'model': 'FAST', **extra})

    def test_backend_commits_chat_history_and_task_without_frontend_put(self):
        self.initial_chat()
        response = self.ask()
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        chat = self.client.get('/chats/review').json()
        self.assertEqual(len(chat['messages']), 2)
        self.assertIn('NOTE_FOUR', chat['messages'][-1]['content'])
        self.assertEqual(chat['messages'][-1]['reply_to'], 'user-message')
        self.assertEqual(chat['session_id'], result['session_id'])
        self.assertEqual(chat['version'], result['chat_version'])
        self.assertEqual(self.env.assistant.get_task(result['task_id']).status, 'COMPLETED')
        with closing(self.env.connection()) as conn:
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM application_logs').fetchone()[0], 1)

    def test_retry_returns_stored_result_without_running_skill_again(self):
        self.initial_chat()
        first = self.ask().json()
        with patch.object(self.env.skill, 'execute', side_effect=AssertionError('duplicate execution')):
            second = self.ask()
        self.assertEqual(second.status_code, 200, second.text)
        self.assertEqual(second.json()['task_id'], first['task_id'])
        self.assertEqual(len(self.client.get('/chats/review').json()['messages']), 2)

    def test_reusing_request_id_for_another_question_is_rejected(self):
        self.initial_chat(); self.ask()
        response = self.ask(question='Прочитай Без названия 11.md')
        self.assertEqual(response.status_code, 409)

    def test_history_failure_rolls_back_visible_answer(self):
        self.initial_chat()
        with closing(self.env.connection()) as conn, conn:
            conn.execute("CREATE TRIGGER fail_history BEFORE INSERT ON application_logs BEGIN SELECT RAISE(ABORT,'fixture'); END")
        response = self.ask()
        self.assertEqual(response.status_code, 502)
        self.assertEqual(len(self.client.get('/chats/review').json()['messages']), 1)
        with closing(self.env.connection()) as conn:
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM assistant_responses').fetchone()[0], 0)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM assistant_tasks WHERE status='COMPLETED'").fetchone()[0], 0)

    def test_streamed_text_keeps_normal_mode_and_returns_persisted_answer(self):
        self.initial_chat()
        response = self.ask('/assistant/stream')
        packets = [json.loads(line) for line in response.text.splitlines()]
        self.assertTrue(any(packet['type'] == 'task' for packet in packets))
        result = next(packet['response'] for packet in packets if packet['type'] == 'result')
        self.assertEqual(''.join(packet['text'] for packet in packets if packet['type'] == 'token'), result['answer'])
        self.assertEqual(len(self.client.get('/chats/review').json()['messages']), 2)

    def test_stale_client_cannot_overwrite_new_answer(self):
        before = self.initial_chat(); self.ask()
        self.assertEqual(self.client.put('/chats/review', json=before).status_code, 409)
        self.assertEqual(len(self.client.get('/chats/review').json()['messages']), 2)

    def test_chat_pages_return_summaries_without_dropping_older_dialogues(self):
        self.initial_chat()
        with closing(self.env.connection()) as conn, conn:
            for index in range(106):
                chat = {'id': str(index), 'title': 'Fixture ' + str(index), 'session_id': None,
                    'updated_at': '2026-10-15T10:00:00Z', 'messages': [{'role': 'user', 'content': 'LARGE_BODY'}]}
                conn.execute('INSERT INTO workspace_chats(id,data,updated_at) VALUES(?,?,?)',
                             (chat['id'], json.dumps(chat), chat['updated_at']))
        seen, cursor = [], None
        while True:
            page = self.client.get('/chats/page', params={'cursor': cursor} if cursor else {}).json()
            self.assertLessEqual(len(page['items']), 50)
            self.assertTrue(all(item['messages'] == [] for item in page['items']))
            seen.extend(item['id'] for item in page['items'])
            cursor = page['next_cursor']
            if not cursor:
                break
        self.assertEqual(len(seen), 107)
        self.assertEqual(len(set(seen)), 107)
        self.assertEqual(len(self.client.get('/chats/0').json()['messages']), 1)
        self.assertEqual(len(self.client.get('/chats/page', params={'search': 'мирон'}).json()['items']), 1)


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='fortuna-migration-test-')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.legacy, self.target = self.root / 'legacy', self.root / 'target'
        self.legacy.mkdir()

    def test_interrupted_copy_is_retried_without_publishing_partial_store(self):
        source = self.legacy / 'chroma_db'
        source.mkdir()
        (source / 'first').write_text('FIRST'); (source / 'second').write_text('SECOND')
        original, calls = data_migration.shutil.copy2, []
        def interrupted(*args, **kwargs):
            calls.append(True)
            if len(calls) == 2:
                raise OSError('fixture interruption')
            return original(*args, **kwargs)
        with patch.object(data_migration.shutil, 'copy2', side_effect=interrupted):
            with self.assertRaises(OSError):
                data_migration.initialise_data(self.target, self.legacy)
        self.assertFalse((self.target / 'chroma_db').exists())
        data_migration.initialise_data(self.target, self.legacy)
        self.assertEqual((self.target / 'chroma_db/second').read_text(), 'SECOND')
        data_migration.initialise_data(self.target, self.legacy)
        self.assertTrue((source / 'second').exists())

    def test_sqlite_backup_includes_committed_wal_data(self):
        source = sqlite3.connect(self.legacy / 'rag_app.db')
        self.addCleanup(source.close)
        source.execute('PRAGMA journal_mode=WAL')
        source.execute('PRAGMA wal_autocheckpoint=0')
        source.execute('CREATE TABLE fixture(value TEXT)')
        source.execute("INSERT INTO fixture VALUES('WAL_CONTENT')"); source.commit()
        data_migration.initialise_data(self.target, self.legacy)
        with closing(sqlite3.connect(self.target / 'rag_app.db')) as copied:
            self.assertEqual(copied.execute('SELECT value FROM fixture').fetchone()[0], 'WAL_CONTENT')

    def test_existing_user_database_is_not_mixed_with_legacy_vectors(self):
        self.target.mkdir()
        with closing(sqlite3.connect(self.target / 'rag_app.db')) as conn:
            conn.execute('CREATE TABLE user_history(value TEXT)')
        (self.legacy / 'chroma_db').mkdir()
        (self.legacy / 'chroma_db/old_vectors').write_text('OLD')
        data_migration.initialise_data(self.target, self.legacy)
        self.assertFalse((self.target / 'chroma_db').exists())
        with closing(sqlite3.connect(self.target / 'rag_app.db')) as conn:
            self.assertTrue(conn.execute("SELECT 1 FROM sqlite_master WHERE name='user_history'").fetchone())

    def test_promotion_interruption_resumes_the_same_snapshot(self):
        with closing(sqlite3.connect(self.legacy / 'rag_app.db')) as conn:
            conn.execute('CREATE TABLE fixture(value TEXT)')
        (self.legacy / 'chroma_db').mkdir()
        (self.legacy / 'chroma_db/vector').write_text('VECTOR')
        original = data_migration.os.replace
        def interrupted(source, destination):
            if Path(destination) == self.target / 'chroma_db':
                raise OSError('fixture promotion interruption')
            return original(source, destination)
        with patch.object(data_migration.os, 'replace', side_effect=interrupted):
            with self.assertRaises(OSError):
                data_migration.initialise_data(self.target, self.legacy)
        self.assertTrue((self.target / 'rag_app.db').exists())
        data_migration.initialise_data(self.target, self.legacy)
        self.assertEqual((self.target / 'chroma_db/vector').read_text(), 'VECTOR')


class InternalEditingTests(unittest.TestCase):
    def setUp(self):
        self.env = UserEnvironment()
        self.addCleanup(self.env.close)

    def test_workspace_uses_internal_editing_then_checks_final_text(self):
        (self.env.root / 'project.txt').write_text('Код проекта Орион: 731.', encoding='utf-8')
        generated = Mock(side_effect=[SimpleNamespace(text=text, metrics=SimpleNamespace(finish_reason='stop'))
            for text in ('Код проекта Орион: 731.', 'Код Ориона — 731.')])
        provider = SimpleNamespace(generate=generated)
        checked = []
        def review(question, answer, *args, **kwargs):
            checked.append(answer)
            return answer, VerificationResult(status='SUPPORTED', checked_answer=answer)
        with patch('model_runtime.get_llm_provider', return_value=provider), \
                patch('settings.VERIFIER_ENABLED', True), patch('settings.ANSWER_REVIEW_MODE', 'contextual'), \
                patch('answer_review.review_answer', side_effect=review):
            response, _ = asyncio.run(self.env.core.process_input(QueryInput(question='Какой код проекта Орион?',
                model='FAST', workspace_id=self.env.workspace.id, use_multi_agent=True)))
        self.assertEqual(checked, ['Код Ориона — 731.'])
        self.assertEqual(response.edited, 'Код Ориона — 731.')
        self.assertIn('EDIT_RESPONSE', [action.action_type for action in response.actions])
        self.assertNotIn('WAITING_FOR_USER', str(response.plan_status))
        packet = json.loads(generated.call_args_list[1].kwargs['messages'][-1]['content'])
        self.assertIn('731', packet['sources'])
        self.assertEqual(packet['question'], 'Какой код проекта Орион?')

    def test_balanced_pipeline_runs_all_stages_with_same_sources(self):
        from answer_editing import edit_sourced_answer
        from assistant_core import ExecutionContext
        from assistant_models import TaskStatus
        session = self.env.assistant.ensure_session()
        task = self.env.assistant.create_task(session.id, 'question')
        self.env.assistant.transition(task, TaskStatus.RUNNING)
        provider = Mock()
        provider.generate.side_effect = [SimpleNamespace(text=text, metrics=SimpleNamespace(finish_reason='stop'))
            for text in ('EDIT', 'ENHANCED', 'FINAL')]
        with patch('model_runtime.get_llm_provider', return_value=provider):
            result = asyncio.run(edit_sourced_answer('question', 'DRAFT', 'SOURCE_MARKER', 'BALANCED',
                ExecutionContext(self.env.assistant, task)))
        self.assertEqual(result['final'], 'FINAL')
        self.assertEqual(result['enhanced'], ['ENHANCED'])
        for call in provider.generate.call_args_list:
            self.assertEqual(json.loads(call.kwargs['messages'][-1]['content'])['sources'], 'SOURCE_MARKER')

    def test_incomplete_internal_edit_retains_completed_draft(self):
        from answer_editing import edit_sourced_answer
        from assistant_core import ExecutionContext
        from assistant_models import TaskStatus
        session = self.env.assistant.ensure_session()
        task = self.env.assistant.create_task(session.id, 'question')
        self.env.assistant.transition(task, TaskStatus.RUNNING)
        provider = SimpleNamespace(generate=lambda **kwargs: SimpleNamespace(text='BROKEN',
            metrics=SimpleNamespace(finish_reason='length')))
        with patch('model_runtime.get_llm_provider', return_value=provider):
            result = asyncio.run(edit_sourced_answer('question', 'COMPLETE DRAFT', 'SOURCE', 'FAST',
                ExecutionContext(self.env.assistant, task)))
        self.assertEqual(result['final'], 'COMPLETE DRAFT')
        self.assertEqual(result['edited'], '')

    def test_library_editor_retains_source_labels_and_checks_edited_answer(self):
        from assistant_core import ExecutionContext
        from assistant_models import TaskStatus
        from knowledge_skill import KnowledgeServices, KnowledgeSkill
        from query_router import RouteDecision, QueryStrategy
        from langchain_core.documents import Document
        session = self.env.assistant.ensure_session()
        task = self.env.assistant.create_task(session.id, 'question')
        self.env.assistant.transition(task, TaskStatus.RUNNING)
        document = Document(page_content='SOURCE_MARKER', metadata={'source': 'Named Source.txt', 'chunk_id': 'chunk-1'})
        source = {'document_id': 1, 'chunk_id': 'chunk-1', 'text': 'SOURCE_MARKER',
                  'metadata': document.metadata, 'score': 1, 'found_by': ['fixture']}
        verified = Mock(side_effect=lambda q, answer, *args: (answer, None))
        services = KnowledgeServices(route=lambda *args: RouteDecision(strategy=QueryStrategy.DOCUMENT_QA, confidence=1),
            history=lambda *args: [], general_chat=None, multi_chain=None,
            rag_chain=lambda **kwargs: SimpleNamespace(invoke=lambda inputs: {'answer': 'DRAFT', 'context': [document]}),
            document_result=lambda value: source, verify=verified)
        provider = Mock()
        provider.generate.return_value = SimpleNamespace(text='EDITED', metrics=SimpleNamespace(finish_reason='stop'))
        with patch('model_runtime.get_llm_provider', return_value=provider):
            result = asyncio.run(KnowledgeSkill(services).execute({'question': 'question', 'model': 'FAST',
                'use_multi_agent': True, '_chat_history': []}, ExecutionContext(self.env.assistant, task)))
        packet = json.loads(provider.generate.call_args.kwargs['messages'][-1]['content'])
        self.assertIn('Named Source.txt', packet['sources'])
        self.assertIn('chunk-1', packet['sources'])
        self.assertEqual(verified.call_args.args[1], 'EDITED')
        self.assertEqual(result.data['draft'], 'DRAFT')


if __name__ == '__main__':
    unittest.main()
