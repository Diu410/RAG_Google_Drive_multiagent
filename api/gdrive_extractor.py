from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from gdrive_auth import get_google_drive_credentials
import io
import os
import re
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredHTMLLoader,TextLoader
from langchain_core.documents import Document
from formula_processing import load_docx_with_formulas, load_html_with_formulas, load_pdf_complete
from typing import List, Tuple
import tempfile
import logging
import time
from extraction_limits import ExtractionLimitError, MAX_EXTRACTED_CHARACTERS, check_extraction_budget
from settings import MAX_UPLOAD_BYTES

MAX_DRIVE_FILES = int(os.getenv('MAX_DRIVE_FILES', '200'))
MAX_DRIVE_DEPTH = int(os.getenv('MAX_DRIVE_DEPTH', '12'))
MAX_DRIVE_BYTES = int(os.getenv('MAX_DRIVE_BYTES', str(256 * 1024 * 1024)))
MAX_DRIVE_SECONDS = float(os.getenv('MAX_DRIVE_SECONDS', '300'))


class LimitedDownload(io.BytesIO):
    def __init__(self, limit, state=None):
        super().__init__()
        self.limit = limit
        self.state = state

    def write(self, data):
        if self.tell() + len(data) > self.limit:
            raise ExtractionLimitError('Drive download exceeds size budget')
        if self.state and (self.state['bytes'] + len(data) > MAX_DRIVE_BYTES
                           or time.monotonic() >= self.state['deadline']):
            raise ExtractionLimitError('Drive import budget exceeded')
        written = super().write(data)
        if self.state:
            self.state['bytes'] += written
        return written

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GoogleDriveExtractor:
    """Модуль для извлечения текста из файлов или папок на Google Drive."""

    def __init__(self, credentials_path: str, token_path: str = 'token.json'):
        """Инициализация с путем к файлу credentials JSON и token JSON."""
        self.scopes = ['https://www.googleapis.com/auth/drive.readonly']
        logger.info("Initializing GoogleDriveExtractor")
        self.credentials = get_google_drive_credentials(credentials_path, token_path)
        self.service = build('drive', 'v3', credentials=self.credentials)
        transport = getattr(self.service._http, 'http', self.service._http)
        transport.timeout = 30
        self.supported_extensions = ['.pdf', '.docx', '.html', '.txt']  # <-- Добавь '.txt'

    def extract_folder_id_from_url(self, url: str) -> str:
        """Извлекает ID папки из ссылки Google Drive."""
        logger.info(f"Extracting folder ID from URL: {url}")
        pattern = r'(?:https?:\/\/)?drive\.google\.com\/(?:drive\/folders\/|folderview\?id=)([a-zA-Z0-9_-]+)'
        match = re.search(pattern, url)
        if match:
            folder_id = match.group(1)
            logger.info(f"Extracted folder ID: {folder_id}")
            return folder_id
        logger.error(f"Invalid Google Drive folder URL: {url}")
        raise ValueError(f"Invalid Google Drive folder URL: {url}")

    def list_files_in_folder(self, folder_id: str) -> List[dict]:
        """Получает список файлов и папок в папке Google Drive."""
        logger.info(f"Listing files and folders in folder with ID: {folder_id}")
        try:
            query = f"'{folder_id}' in parents and trashed=false"
            items = []
            page_token = None
            while True:
                results = self.service.files().list(
                    q=query, fields="nextPageToken,files(id,name,mimeType)",
                    pageSize=1000, pageToken=page_token,
                ).execute()
                items.extend(results.get('files', []))
                state = getattr(self, '_import_state', None)
                if state and len(items) > MAX_DRIVE_FILES:
                    state['limited'] = True
                    return items[:MAX_DRIVE_FILES]
                page_token = results.get('nextPageToken')
                if not page_token:
                    break
            logger.info(f"Found {len(items)} items (files and folders) in folder {folder_id}")
            return items
        except Exception as e:
            logger.error(f"Error listing items in folder {folder_id}: {str(e)}")
            raise Exception(f"Error listing items in folder {folder_id}: {str(e)}")

    def download_file(self, file_id: str) -> Tuple[io.BytesIO, str]:
        """Скачивает файл с Google Drive в память и возвращает его содержимое и имя."""
        logger.info(f"Downloading file with ID: {file_id}")
        try:
            file_metadata = self.service.files().get(fileId=file_id, fields='name,mimeType').execute()
            file_name = file_metadata['name']
            logger.info(f"File name: {file_name}")

            if file_metadata['mimeType'] == 'application/vnd.google-apps.document':
                request = self.service.files().export_media(fileId=file_id, mimeType='text/plain')
                file_name += '.txt'
            else:
                request = self.service.files().get_media(fileId=file_id)
            state = getattr(self, '_import_state', None)
            available = MAX_DRIVE_BYTES - state['bytes'] if state else MAX_UPLOAD_BYTES
            file_buffer = LimitedDownload(min(MAX_UPLOAD_BYTES, available), state)
            downloader = MediaIoBaseDownload(file_buffer, request, chunksize=1024 * 1024)
            done = False
            while not done:
                if state and time.monotonic() >= state['deadline']:
                    raise ExtractionLimitError('Drive import time limit exceeded')
                status, done = downloader.next_chunk()
                logger.info(f"Download progress: {status.progress() * 100}%")

            file_buffer.seek(0)
            return file_buffer, file_name
        except ExtractionLimitError:
            raise
        except Exception as e:
            logger.error(f"Error downloading file {file_id}: {str(e)}")
            raise Exception(f"Error downloading file {file_id}: {str(e)}")

    def extract_text(self, file_buffer: io.BytesIO, file_name: str) -> List[Document]:
        logger.info(f"Extracting text from {file_name}")
        temp_file_path = None
        try:
            extension = os.path.splitext(file_name)[1].lower()
            logger.info(f"File extension: {extension}")

            if extension not in self.supported_extensions:
                logger.info(f"Skipping unsupported file: {file_name}")
                return []

            # Temp file
            with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as f:
                temp_file_path = f.name
                f.write(file_buffer.getvalue())

            if extension == '.pdf':
                documents = load_pdf_complete(temp_file_path)
            elif extension == '.docx':
                documents = load_docx_with_formulas(temp_file_path)
            elif extension == '.html':
                documents = load_html_with_formulas(temp_file_path)
            elif extension == '.txt':  # <-- Добавь это
                from langchain_community.document_loaders import TextLoader  # Импорт внутри или в начало
                loader = TextLoader(temp_file_path, encoding='utf-8')  # Для RU
                documents = loader.load()
            else:
                os.unlink(temp_file_path)
                return []

            os.unlink(temp_file_path)

            documents = [doc for doc in documents if doc.page_content.strip()]
            check_extraction_budget(sum(len(doc.page_content) for doc in documents))
            if not documents:
                logger.warning(f"No valid text extracted from {file_name}")
                return []

            for doc in documents:
                doc.metadata['source'] = file_name

            return documents
        except Exception:
            logger.warning('Drive extraction failed: file=%s', file_name)
            raise
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    def process_file(self, file_id: str) -> List[Document]:
        """Скачивает файл и извлекает текст."""
        logger.info(f"Processing file with ID: {file_id}")
        file_buffer, file_name = self.download_file(file_id)
        documents = self.extract_text(file_buffer, file_name)
        if not documents:
            raise ValueError('No usable text was extracted')
        for doc in documents:
            doc.metadata['gdrive_id'] = file_id
        return documents

    def process_folder_recursive(self, folder_id: str, depth: int = 0) -> List[Document]:
        """Рекурсивно обрабатывает все файлы в папке и ее подпапках."""
        logger.info(f"Processing folder recursively with ID: {folder_id}")
        state = getattr(self, '_import_state', None)
        if state and (depth > MAX_DRIVE_DEPTH or folder_id in state['visited']):
            state['limited'] = True
            return []
        if state:
            state['visited'].add(folder_id)
        all_documents = []
        items = self.list_files_in_folder(folder_id)

        for item in items:
            if state and (state['attempted'] >= MAX_DRIVE_FILES or state['bytes'] >= MAX_DRIVE_BYTES
                          or time.monotonic() >= state['deadline']):
                state['limited'] = True
                break
            item_id = item['id']
            item_name = item['name']
            mime_type = item['mimeType']

            if mime_type == 'application/vnd.google-apps.folder':
                # Рекурсивно обрабатываем вложенную папку
                logger.info(f"Found subfolder: {item_name} (ID: {item_id})")
                try:
                    subfolder_documents = self.process_folder_recursive(item_id, depth + 1)
                    all_documents.extend(subfolder_documents)
                except Exception as exc:
                    if state:
                        state['failed'].append({'id': item_id, 'name': item_name, 'error': type(exc).__name__})
            else:
                # Обрабатываем файл
                extension = os.path.splitext(item_name)[1].lower()
                if extension not in self.supported_extensions and mime_type != 'application/vnd.google-apps.document':
                    logger.info(f"Skipping unsupported file: {item_name}")
                    if state:
                        state['unsupported'] += 1
                    continue
                try:
                    if state:
                        state['attempted'] += 1
                    documents = self.process_file(item_id)
                    characters = sum(len(doc.page_content) for doc in documents)
                    if state and state['characters'] + characters > MAX_EXTRACTED_CHARACTERS * 10:
                        state['limited'] = True
                        break
                    if state:
                        state['characters'] += characters
                    all_documents.extend(documents)
                    logger.info(f"Successfully processed file: {item_name}")
                except Exception as e:
                    if state:
                        state['failed'].append({'id': item_id, 'name': item_name, 'error': type(e).__name__})
                    logger.error(f"Failed to process file {item_name}: {str(e)}")
                    continue

        return all_documents

    def process_folder(self, folder_url: str) -> List[Document]:
        """Обрабатывает все поддерживаемые файлы в папке Google Drive и ее подпапках."""
        logger.info(f"Processing folder with URL: {folder_url}")
        folder_id = self.extract_folder_id_from_url(folder_url)
        self._import_state = {'attempted': 0, 'bytes': 0, 'characters': 0, 'visited': set(),
            'deadline': time.monotonic() + MAX_DRIVE_SECONDS, 'failed': [], 'unsupported': 0, 'limited': False}
        self.last_report = self._import_state
        all_documents = self.process_folder_recursive(folder_id)

        if not all_documents:
            logger.warning(f"No valid documents extracted from folder {folder_id}")
            return []

        logger.info(f"Total documents extracted from folder and subfolders: {len(all_documents)}")
        return all_documents

if __name__ == "__main__":
    try:
        credentials_path = '../credentials.json'
        extractor = GoogleDriveExtractor(credentials_path)
        # Пример обработки папки
        folder_url = "https://drive.google.com/drive/folders/1i23Gn0j3cobW72i9TpTJMNE1A6NOewoI"
        documents = extractor.process_folder(folder_url)
        for doc in documents:
            logger.info('Extracted document: source=%s characters=%s', doc.metadata['source'], len(doc.page_content))
    except Exception as e:
        logger.error(f"Failed to process folder: {str(e)}")
# Комментарий: Этот модуль реализует извлечение текста из файлов Google Drive (PDF, DOCX, HTML). Класс GoogleDriveExtractor использует Google Drive API для рекурсивной обработки папок и файлов. Файлы скачиваются в память, текст извлекается с помощью LangChain loaders, временные файлы удаляются. Поддерживает только указанные форматы, логирует процесс. Интегрируется с main.py через эндпоинт /upload-gdrive-doc.
