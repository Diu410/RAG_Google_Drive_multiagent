# chroma_utils.py - КОД ДЛЯ GEMINI
import os
os.environ['GRPC_VERBOSITY'] = 'ERROR'  # Подавляет ALTS/GRPC шум
os.environ['GOOGLE_API_USE_MTLS_ENDPOINT'] = 'never'  # Если MTLS глючит
from langchain_community.document_loaders import PyMuPDFLoader  # Добавь импорт
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredHTMLLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_chroma import Chroma

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
print(f"Loaded GOOGLE_API_KEY: {'***' if GOOGLE_API_KEY else 'None'}")  # Маскирует ключ для безопасности
vectorstore = None
retriever = None

try:
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not set. Check your .env file.")

    # Инициализация Embeddings С ЯВНОЙ ПЕРЕДАЧЕЙ КЛЮЧА
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",  # <-- Измени на это (stable, dims до 3072)
        google_api_key=GOOGLE_API_KEY)

    vectorstore = Chroma(collection_name="rag_collection", embedding_function=embedding_model,
                         persist_directory="./chroma_db")

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        # <-- Добавь: MMR (maximal marginal relevance) — баланс diversity + relevance, чтобы не пропускать разбросанные части
        search_kwargs={"k": 30, "fetch_k": 100}  # <-- k=30 (топ-результаты), fetch_k=100 (кандидаты для MMR)
    )
    print("Gemini Embeddings и Chroma успешно инициализированы.")


except Exception as e:
    print(f"ВНИМАНИЕ: Не удалось инициализировать GoogleGenerativeAIEmbeddings или Chroma. Ошибка: {e}.")
    # Устанавливаем None, чтобы предотвратить сбой дальнейшей работы
    embedding_model = None
    vectorstore = None
    retriever = None


def index_document_to_chroma(input_data, file_id):
    if embedding_model is None or vectorstore is None:
        print("Ошибка: Модель эмбеддингов/векторное хранилище не инициализировано. Индексация невозможна.")
        return False

    try:
        if isinstance(input_data, str):
            file_path = input_data
            file_extension = os.path.splitext(file_path)[1].lower()

            if file_extension == '.pdf':
                try:
                    # Сначала native text
                    loader = PyMuPDFLoader(file_path)
                    documents = loader.load()

                    def clean_text(page_content):
                        if not page_content:
                            return ""
                        # Удаляем лишние пробелы, переносы, артефакты PDF
                        import re
                        cleaned = re.sub(r'\n\s*\n', '\n\n', page_content)  # Склеиваем двойные переносы
                        cleaned = re.sub(r'\s+', ' ', cleaned)  # Один пробел вместо множества
                        cleaned = re.sub(r'^\s+|\s+$', '', cleaned, flags=re.MULTILINE)  # Трим по строкам
                        cleaned = re.sub(r'Page \d+', '', cleaned)  # Удаляем "Page X" артефакты
                        return cleaned.strip()

                    for doc in documents:
                        doc.page_content = clean_text(doc.page_content)

                    # Дебаг: Покажи пример
                    print(
                        f"После очистки: {len(documents)} страниц, {sum(len(doc.page_content) for doc in documents)} символов.")
                    if documents:
                        print(f"Пример первой страницы (после clean): {documents[0].page_content[:400]}...")

                    total_chars = sum(len(doc.page_content) for doc in documents)
                    if total_chars == 0:
                        print("Native PDF пуст — пробуем OCR с Unstructured.")
                        from langchain_community.document_loaders import \
                            UnstructuredPDFLoader  # Импорт внутри, если нужно
                        loader = UnstructuredPDFLoader(file_path, strategy="ocr_only", languages=["rus"])  # RU OCR
                        documents = loader.load()
                        total_chars = sum(len(doc.page_content) for doc in documents)
                        print(f"OCR добавил {total_chars} символов.")
                except Exception as e:
                    print(f"PDF loader/OCR error: {e}")
                    return False

            elif file_extension == '.docx':
                loader = Docx2txtLoader(file_path)
            elif file_extension == '.html':
                loader = UnstructuredHTMLLoader(file_path)
            elif file_extension == '.txt':
                from langchain_community.document_loaders import TextLoader  # Импорт в начало файла, если нет
                loader = TextLoader(file_path, encoding='utf-8')  # Для RU-текста
            else:
                print(f"Unsupported extension: {file_extension}")
                return False

            documents = loader.load()
            # Дебаг для локального файла (file_path доступен)
            total_chars = sum(len(doc.page_content) for doc in documents)
            print(f"Loader для {file_path}: {len(documents)} страниц, {total_chars} символов текста.")
            if total_chars < 500:
                print("Мало текста! PDF-скан? Перейди на PyMuPDFLoader.")
        elif isinstance(input_data, list):
            documents = input_data
            # Дебаг для GDrive (нет file_path, но лог общий)
            total_chars = sum(len(doc.page_content) for doc in documents)
            print(f"Loader для GDrive (file_id {file_id}): {len(documents)} документов, {total_chars} символов текста.")
            if total_chars < 500:
                print("Мало текста из GDrive! Проверь extractor.")
        else:
            raise ValueError("Input must be a file path (str) or a list of Documents")

        # Сплиттер (после loader)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        # Дебаг чанков (file_path только если str)
        valid_chunks = [c for c in chunks if len(c.page_content.strip()) > 50]
        if isinstance(input_data, str):
            print(f"Создано {len(chunks)} чанков (валидных: {len(valid_chunks)}) для {file_path}.")
        else:
            print(f"Создано {len(chunks)} чанков (валидных: {len(valid_chunks)}) для GDrive file_id {file_id}.")
        if len(valid_chunks) == 0:
            print("ОШИБКА: Нет валидных чанков! Проверь loader/splitter.")

        for chunk in valid_chunks:
            chunk.metadata["file_id"] = file_id

        if valid_chunks:
            vectorstore.add_documents(valid_chunks)
            collection_count = vectorstore._collection.count()
            print(f"Добавлено {len(valid_chunks)} чанков. Общее в Chroma: {collection_count}")
            return True
        else:
            return False

    except Exception as e:
        print(f"Error indexing document: {str(e)}")
        return False

def delete_doc_from_chroma(file_id):
    if vectorstore is None:
        print("Ошибка: Векторное хранилище не инициализировано. Удаление невозможно.")
        return False

    try:
        vectorstore.delete(where={"file_id": file_id})
        return True
    except Exception as e:
        print(f"Error deleting document from Chroma: {str(e)}")
        return False