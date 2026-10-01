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
            file_buffer = io.BytesIO()
            downloader = MediaIoBaseDownload(file_buffer, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                logger.info(f"Download progress: {status.progress() * 100}%")

            file_buffer.seek(0)
            return file_buffer, file_name
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
            if not documents:
                logger.warning(f"No valid text extracted from {file_name}")
                return []

            for doc in documents:
                doc.metadata['source'] = file_name

            return documents
        except Exception as e:
            logger.error(f"Error extracting text from {file_name}: {str(e)}")
            return []
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    def process_file(self, file_id: str) -> List[Document]:
        """Скачивает файл и извлекает текст."""
        logger.info(f"Processing file with ID: {file_id}")
        file_buffer, file_name = self.download_file(file_id)
        documents = self.extract_text(file_buffer, file_name)
        for doc in documents:
            doc.metadata['gdrive_id'] = file_id
        return documents

    def process_folder_recursive(self, folder_id: str) -> List[Document]:
        """Рекурсивно обрабатывает все файлы в папке и ее подпапках."""
        logger.info(f"Processing folder recursively with ID: {folder_id}")
        all_documents = []
        items = self.list_files_in_folder(folder_id)

        for item in items:
            item_id = item['id']
            item_name = item['name']
            mime_type = item['mimeType']

            if mime_type == 'application/vnd.google-apps.folder':
                # Рекурсивно обрабатываем вложенную папку
                logger.info(f"Found subfolder: {item_name} (ID: {item_id})")
                subfolder_documents = self.process_folder_recursive(item_id)
                all_documents.extend(subfolder_documents)
            else:
                # Обрабатываем файл
                extension = os.path.splitext(item_name)[1].lower()
                if extension not in self.supported_extensions and mime_type != 'application/vnd.google-apps.document':
                    logger.info(f"Skipping unsupported file: {item_name}")
                    continue
                try:
                    documents = self.process_file(item_id)
                    all_documents.extend(documents)
                    logger.info(f"Successfully processed file: {item_name}")
                except Exception as e:
                    logger.error(f"Failed to process file {item_name}: {str(e)}")
                    continue

        return all_documents

    def process_folder(self, folder_url: str) -> List[Document]:
        """Обрабатывает все поддерживаемые файлы в папке Google Drive и ее подпапках."""
        logger.info(f"Processing folder with URL: {folder_url}")
        folder_id = self.extract_folder_id_from_url(folder_url)
        all_documents = self.process_folder_recursive(folder_id)

        if not all_documents:
            logger.warning(f"No valid documents extracted from folder {folder_id}")
            raise ValueError(f"No valid documents extracted from folder {folder_id}")

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
            logger.info(f"Document from {doc.metadata['source']} (first 200 chars): {doc.page_content[:200]}")
    except Exception as e:
        logger.error(f"Failed to process folder: {str(e)}")
# Комментарий: Этот модуль реализует извлечение текста из файлов Google Drive (PDF, DOCX, HTML). Класс GoogleDriveExtractor использует Google Drive API для рекурсивной обработки папок и файлов. Файлы скачиваются в память, текст извлекается с помощью LangChain loaders, временные файлы удаляются. Поддерживает только указанные форматы, логирует процесс. Интегрируется с main.py через эндпоинт /upload-gdrive-doc.
