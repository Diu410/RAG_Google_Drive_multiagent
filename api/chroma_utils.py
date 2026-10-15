import os
import re
import logging
import traceback
import hashlib
import uuid
from contextlib import closing
from langchain_chroma import Chroma
from embedding_provider import get_embedding_provider
from langchain_core.documents import Document
from settings import CHROMA_PATH, FAST_RETRIEVAL_K
from formula_processing import (clean_formula_safe_text, load_docx_with_formulas,
                                load_html_with_formulas, load_pdf_complete,
                                split_structured_documents)
from index_transactions import IndexTransactions, IndexRecoveryError, consistent_read, invalidate_index, serialized_mutation
from db_utils import get_db_connection
logger = logging.getLogger(__name__)

os.environ['GRPC_VERBOSITY'] = 'ERROR'

vectorstore = None
workspace_vectorstore = None
retriever = None


def collection_name(family):
    with closing(get_db_connection()) as conn, conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS index_collections (
            family TEXT PRIMARY KEY, name TEXT NOT NULL, fingerprint TEXT NOT NULL,
            previous_name TEXT)''')
        row = conn.execute('SELECT name FROM index_collections WHERE family=?', (family,)).fetchone()
        return row['name'] if row else family


try:
    embedding_model = get_embedding_provider()

    vectorstore = Chroma(
        collection_name=collection_name('rag_collection'),
        embedding_function=embedding_model,
        persist_directory=CHROMA_PATH
    )
    workspace_vectorstore = Chroma(
        collection_name=collection_name('workspace_collection'),
        embedding_function=embedding_model,
        persist_directory=CHROMA_PATH,
    )
    embedding_model.bind_collection(vectorstore._collection)
    embedding_model.bind_collection(workspace_vectorstore._collection)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": FAST_RETRIEVAL_K, "fetch_k": 16, "lambda_mult": 0.7}
    )
    logger.info("✅ Embedding provider и Chroma успешно инициализированы.")

except Exception as e:
    logger.error(f"❌ Ошибка инициализации RAG: {e}\n{traceback.format_exc()}")
    embedding_model = None
    vectorstore = None
    workspace_vectorstore = None
    retriever = None


def load_pdf_ocr(file_path: str):
    """OCR для сканированных PDF через pytesseract."""
    import pytesseract
    from pdf2image import convert_from_path
    from pytesseract import Output

    if os.getenv("TESSERACT_CMD"):
        pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_CMD"]
    poppler_path = os.getenv("POPPLER_PATH")

    logger.info("PDF сканированный — запускаю OCR...")
    images = convert_from_path(file_path, poppler_path=poppler_path)
    documents = []
    for i, img in enumerate(images):
        data = pytesseract.image_to_data(img, lang="rus+eng", output_type=Output.DICT)
        confident, confidences = [], []
        for token, raw_confidence in zip(data.get("text", []), data.get("conf", [])):
            try:
                confidence = float(raw_confidence)
            except (TypeError, ValueError):
                confidence = -1
            if token.strip() and confidence >= 0:
                confidences.append(confidence)
                if confidence >= 45:
                    confident.append(token)
        text = " ".join(confident)
        average = sum(confidences) / len(confidences) if confidences else 0
        if text.strip():
            documents.append(Document(page_content=text, metadata={
                "page": i, "source": file_path, "block_type": "text",
                "ocr_status": "OCR_UNCERTAIN" if average < 60 else "OCR_OK",
                "formula_status": "uncertain" if average < 60 else "unknown",
            }))
        logger.info(f"  OCR стр {i}: {len(text)} символов")
    return documents

def load_pdf(file_path: str):
    # Completeness failures must propagate; a text-only retry can lose pages.
    from extraction_limits import extraction_budget
    with extraction_budget():
        return load_pdf_complete(file_path)




def clean_text_refined(text):
    return clean_formula_safe_text(text)


@serialized_mutation
def index_document_to_chroma(input_data, file_id, *, commit=None, source_name=None):
    if embedding_model is None or vectorstore is None:
        logger.error("❌ embedding_model или vectorstore не инициализированы — индексация невозможна.")
        return False

    try:
        transactions = IndexTransactions()
        transactions.recover({'rag_collection': vectorstore})
        documents = []

        if isinstance(input_data, str):
            file_path = input_data
            ext = os.path.splitext(file_path)[1].lower()
            logger.info(f"📄 Загружаю файл: {file_path} (расширение: {ext})")

            if ext == '.pdf':
                documents = load_pdf(file_path)
            elif ext == '.docx':
                documents = load_docx_with_formulas(file_path)
            elif ext == '.txt':
                from langchain_community.document_loaders import TextLoader
                documents = TextLoader(file_path, encoding='utf-8').load()
            elif ext == '.html':
                documents = load_html_with_formulas(file_path)
            else:
                logger.warning(f"Неподдерживаемый тип файла: {ext}")
                return False

            logger.info(f"✅ Загружено {len(documents)} страниц из файла.")
            for i, doc in enumerate(documents):
                logger.info('Extracted page %s: characters=%s', i, len(doc.page_content))

        elif isinstance(input_data, list):
            documents = input_data
            logger.info(f"📋 Получено {len(documents)} документов напрямую.")

        for doc in documents:
            doc.page_content = clean_text_refined(doc.page_content)
        from extraction_limits import check_extraction_budget
        check_extraction_budget(sum(len(doc.page_content) for doc in documents))

        # Сплиттер
        logger.info("✂️ Разбиваю на чанки...")
        chunks = split_structured_documents(documents)
        valid_chunks = [c for c in chunks if c.page_content.strip()]
        logger.info(f"📦 Чанков после фильтрации: {len(valid_chunks)} (из {len(chunks)} всего)")

        # Store a stable, user-facing source label.  Temporary upload paths are
        # neither useful to the model nor safe to expose in an answer.
        from db_utils import get_document
        document_record = get_document(file_id)
        source_name = source_name or (document_record["filename"] if document_record else f"document-{file_id}")

        chunk_ids = []
        for index, chunk in enumerate(valid_chunks):
            chunk_id = f"{file_id}:{index}:{hashlib.sha256(chunk.page_content.encode('utf-8')).hexdigest()[:16]}"
            chunk.metadata["file_id"] = file_id
            chunk.metadata["document_id"] = file_id
            chunk.metadata["chunk_id"] = chunk_id
            chunk.metadata["source"] = source_name
            chunk.metadata["document_name"] = source_name
            chunk_ids.append(chunk_id)

        if not valid_chunks:
            logger.warning("⚠️ Нет валидных чанков для индексации.")
            return False

        # Батчевая индексация
        BATCH_SIZE = 50
        total = len(valid_chunks)
        logger.info(f"🚀 Начинаю индексацию {total} чанков батчами по {BATCH_SIZE}...")

        transactions.replace(vectorstore, 'rag_collection', {'file_id': file_id},
                                    valid_chunks, chunk_ids, batch_size=BATCH_SIZE, commit=commit)

        # Graph extraction is deliberately best-effort. A broken/offline LLM must
        # never invalidate the already completed vector index.
        if document_record:
            try:
                from graph_extraction import GraphExtractor
                GraphExtractor().extract_document(valid_chunks, file_id, source_name)
            except Exception:
                logger.exception("Graph extraction unavailable for file_id=%s; keeping normal RAG index", file_id)

        logger.info(f"✅ Готово: {total} чанков добавлено (file_id={file_id}).")
        return True



    except Exception as e:
        logger.error(f"🔥 Ошибка индексации: {e}\n{traceback.format_exc()}")
        # The transactional writer owns cleanup of its new revision. Deleting
        # by owner here would destroy the previously working index.
        if isinstance(e, IndexRecoveryError):
            raise RuntimeError('Index cleanup is pending; source retained for recovery') from e
        return False


def rebuild_embedding_indexes():
    """Migrate both embedding spaces, keeping old collections until committed.

    Chunk IDs stay stable so graph provenance survives a model migration.
    Supports dimension changes by writing into entirely new collections.
    """
    from document_service import document_lock
    from index_transactions import index_lock
    from embedding_provider import LocalEmbeddings
    global vectorstore, workspace_vectorstore, retriever, embedding_model
    with document_lock, index_lock:
        if vectorstore is None or workspace_vectorstore is None:
            raise RuntimeError('Vector stores are unavailable')
        provider = LocalEmbeddings()
        fingerprint = provider.fingerprint()
        old_stores = {'rag_collection': vectorstore, 'workspace_collection': workspace_vectorstore}
        if all((store._collection.metadata or {}).get('embedding_fingerprint') == fingerprint
               for store in old_stores.values()):
            return {'status': 'unchanged', 'indexed_chunks': 0}
        new_names = {family: family + '_r_' + uuid.uuid4().hex for family in old_stores}
        transactions = IndexTransactions()
        transactions.recover(old_stores)
        operation = transactions.begin('rag_collection', 'collection_switch', {'new_names': new_names})
        staged = {}
        total = 0
        try:
            for family, old in old_stores.items():
                new = Chroma(collection_name=new_names[family], embedding_function=provider,
                             persist_directory=CHROMA_PATH,
                             collection_metadata={'embedding_fingerprint': fingerprint})
                staged[family] = new
                offset = 0
                while True:
                    records = old.get(include=['documents', 'metadatas'], limit=50, offset=offset)
                    ids = records.get('ids') or []
                    if not ids:
                        break
                    documents = [Document(page_content=text, metadata=metadata or {})
                                 for text, metadata in zip(records['documents'], records['metadatas'])]
                    new.add_documents(documents, ids=ids)
                    total += len(ids)
                    offset += len(ids)
            with closing(get_db_connection()) as conn, conn:
                conn.execute('BEGIN IMMEDIATE')
                for family in old_stores:
                    conn.execute('''INSERT INTO index_collections(family,name,fingerprint,previous_name)
                        VALUES(?,?,?,?) ON CONFLICT(family) DO UPDATE SET
                        previous_name=index_collections.name,name=excluded.name,fingerprint=excluded.fingerprint''',
                        (family, new_names[family], fingerprint, old_stores[family]._collection.name))
                conn.execute("UPDATE index_operations SET state='COMMITTED' WHERE id=?", (operation,))
        except BaseException:
            transactions.recover(old_stores)
            raise
        provider.bind_collection(staged['rag_collection']._collection)
        provider.bind_collection(staged['workspace_collection']._collection)
        vectorstore, workspace_vectorstore = staged['rag_collection'], staged['workspace_collection']
        embedding_model = provider
        retriever = vectorstore.as_retriever(search_type='mmr', search_kwargs={
            'k': FAST_RETRIEVAL_K, 'fetch_k': max(16, FAST_RETRIEVAL_K * 2), 'lambda_mult': .7})
        # The compatibility modules still expose injectable retriever globals.
        import langchain_utils
        langchain_utils.vectorstore = vectorstore
        langchain_utils.current_retriever = retriever
        transactions.finish(operation)
        invalidate_index()
        return {'status': 'completed', 'indexed_chunks': total}


@consistent_read
def delete_doc_from_chroma(file_id):
    if vectorstore is None:
        return False
    try:
        vectorstore.delete(where={"file_id": file_id})
        invalidate_index()
        logger.info(f"🗑️ Документ file_id={file_id} удалён из Chroma.")
        return True
    except Exception as e:
        logger.error(f"Ошибка удаления из Chroma: {e}")
        return False


@consistent_read
def purge_orphaned_documents(valid_file_ids):
    """Remove vector chunks whose owning SQLite document no longer exists.

    Older application versions could delete the SQLite row while leaving its
    Chroma chunks behind.  Only records carrying our document ownership
    metadata are considered, so this does not turn into an unrestricted
    collection wipe.
    """
    if vectorstore is None:
        return 0
    valid = {str(value) for value in valid_file_ids}
    records = vectorstore.get(include=["metadatas"])
    ids = records.get("ids") or []
    metadatas = records.get("metadatas") or []
    orphan_ids = []
    for chunk_id, metadata in zip(ids, metadatas):
        metadata = metadata or {}
        owner = metadata.get("file_id", metadata.get("document_id"))
        if owner is not None and str(owner) not in valid:
            orphan_ids.append(chunk_id)
    for offset in range(0, len(orphan_ids), 500):
        vectorstore.delete(ids=orphan_ids[offset:offset + 500])
    if orphan_ids:
        invalidate_index()
        logger.warning("Removed %s orphaned Chroma chunks", len(orphan_ids))
    return len(orphan_ids)
