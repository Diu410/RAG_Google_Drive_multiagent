"""Print isolated review observations; these are not security acceptance tests.

Run from the repository root with .venv/Scripts/python.exe. All files and
databases are temporary. The system opener, inference and Google Drive are
replaced with controlled adapters; no external document is accessed.
"""
import asyncio
import json
import zipfile
from types import SimpleNamespace
from unittest.mock import patch

from support import UserEnvironment
from support import authorized_client as TestClient
from langchain_core.documents import Document
import main
from chroma_utils import load_pdf
from formula_processing import load_docx_with_formulas
from python_multipart import QuerystringParser


def observe():
    env = UserEnvironment()
    try:
        observations = {}
        with patch.object(main, 'workspace_repository', env.repository), \
                patch.object(main, 'filesystem_provider', env.filesystem):
            # Deliberately omit the lifespan context: never start model services.
            client = TestClient(main.app)
            try:
                other_root = env.directory / 'Other workspace'
                other_root.mkdir()
                response = client.post('/workspaces/select',
                    headers={'Origin': 'https://review.invalid'},
                    content=json.dumps({'root_path': str(other_root)}).encode())
                observations['foreign_origin_post_without_content_type'] = {
                    'status': response.status_code,
                    'cors_allows_read': 'access-control-allow-origin' in response.headers,
                    'workspace_changed': env.repository.active().root_path == str(other_root),
                }
                uploads = []
                def upload_stub(*args):
                    uploads.append(True)
                    return 1
                with patch.object(main, 'document_service',
                                  SimpleNamespace(upload=upload_stub)):
                    response = client.post('/upload-doc',
                        headers={'Origin': 'https://review.invalid'},
                        files={'file': ('review.txt', b'REVIEW_ONLY', 'text/plain')})
                observations['foreign_origin_multipart_upload'] = {
                    'status': response.status_code,
                    'cors_allows_read': 'access-control-allow-origin' in response.headers,
                    'upload_handler_invoked': bool(uploads),
                }
                parsed_bytes = []
                original_write = QuerystringParser.write
                def parser_write(parser, data):
                    parsed_bytes.append(len(data))
                    return original_write(parser, data)
                with patch.object(QuerystringParser, 'write', parser_write):
                    response = client.post('/upload-doc', content=b'a;' * 1024,
                        headers={'Content-Type': 'application/x-www-form-urlencoded'})
                observations['urlencoded_parser_before_upload_validation'] = {
                    'status': response.status_code,
                    'parsed_bytes': sum(parsed_bytes),
                }
                response = client.get(f'/workspaces/{env.workspace.id}/file',
                    params={'path': 'Без названия 4.md'},
                    headers={'Host': 'review.invalid'})
                observations['foreign_host_file_read'] = {
                    'status': response.status_code,
                    'fixture_returned': 'NOTE_FOUR' in response.json().get('content', ''),
                }
            finally:
                client.close()

        executable = env.root / 'review.cmd'
        executable.write_text('@echo REVIEW_ONLY\n', encoding='utf-8')
        try:
            env.filesystem.open_file(env.workspace.id, executable.name)
        except PermissionError:
            pass
        observations['script_handed_to_system_opener'] = bool(
            env.opened and env.opened[-1] == str(executable))
        env.opened.clear()
        from assistant_models import AssistantError
        try:
            response = asyncio.run(env.ask('Открой файл review.cmd'))
            task_status = response.task_status
        except AssistantError:
            task_status = 'REJECTED'
        observations['assistant_open_script'] = {
            'opener_called': bool(env.opened),
            'task_status': task_status,
        }

        docx = env.root / 'review.docx'
        text = 'A' * (2 * 1024 * 1024)
        xml = ('<w:document xmlns:w="http://schemas.openxmlformats.org/'
               'wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>' + text +
               '</w:t></w:r></w:p></w:body></w:document>')
        with zipfile.ZipFile(docx, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('word/document.xml', xml)
        try:
            extracted = load_docx_with_formulas(str(docx))
        except ValueError:
            extracted = []
        observations['docx_expansion'] = {
            'archive_bytes': docx.stat().st_size,
            'extracted_characters': sum(len(item.page_content) for item in extracted),
        }

        with patch('chroma_utils.load_pdf_complete',
                   side_effect=ValueError('OCR returned no text for PDF page 2')), \
                patch('langchain_community.document_loaders.PyMuPDFLoader',
                      return_value=SimpleNamespace(load=lambda: [
                          Document(page_content='PAGE_ONE_ONLY', metadata={'page': 0})])):
            try:
                documents = load_pdf('synthetic-review.pdf')
            except ValueError:
                documents = []
        observations['pdf_coverage_failure_falls_back_to_partial_text'] = {
            'returned_pages': [item.metadata['page'] for item in documents],
            'partial_coverage_flag': any('coverage' in key or 'incomplete' in key
                for item in documents for key in item.metadata),
        }
        return observations
    finally:
        env.close()


if __name__ == '__main__':
    print(json.dumps(observe(), ensure_ascii=True, indent=2))
