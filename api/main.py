# main.py - ФИНАЛЬНЫЙ КОД
import sys
import os
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GRPC_TRACE'] = 'off'
import asyncio
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import uuid
import logging
from pydantic_models import QueryInput, QueryResponse, DocumentInfo, DeleteFileRequest, UploadGDriveRequest
from langchain_utils import get_rag_chain
from db_utils import insert_application_logs, get_chat_history, get_all_documents, insert_document_record, \
    delete_document_record
from chroma_utils import index_document_to_chroma, delete_doc_from_chroma
from gdrive_extractor import GoogleDriveExtractor
from langchain_utils import debug_chroma_search
import shutil
import tempfile
from multi_agent_chain import get_multi_agent_chain
from dotenv import load_dotenv

# Загружаем .env
load_dotenv()

logging.basicConfig(filename='app.log', level=logging.INFO)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    GDRIVE_EXTRACTOR = GoogleDriveExtractor('credentials.json')  # <-- Раскомментируй
    logging.info("GDrive Extractor успешно инициализирован.")
except Exception as e:
    logging.warning(f"GDrive init failed: {e}. Установи credentials.json и авторизуйся.")
    GDRIVE_EXTRACTOR = None


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(query_input: QueryInput):
    session_id = query_input.session_id or str(uuid.uuid4())
    model_name = query_input.model.value
    logging.info(f"Session ID: {session_id}, User Query: {query_input.question}, Model: {model_name}")

    # 1. Получаем историю чата
    chat_history = await asyncio.to_thread(get_chat_history, session_id)

    # 2. Получаем RAG-цепочку
    rag_chain = get_rag_chain(model=model_name)

    # Добавь здесь:
    multi_agent_func = get_multi_agent_chain()
    logging.info(f"DEBUG multi: func = {multi_agent_func is not None}, use_multi = {query_input.use_multi_agent}")

    if query_input.use_multi_agent and multi_agent_func:
        logging.info("DEBUG: Running multi-agent...")
        try:
            response = await asyncio.to_thread(multi_agent_func, query_input.question, chat_history)
            intermediate = f"Draft: {response['draft'][:200]}...\nEdited: {response['edited'][:200]}...\nEnhanced: {len(response['enhanced'])} variants"
            logging.info(f"DEBUG multi response keys: {response.keys()}")
            answer = response["final"]
            context_text = response["context"]
            logging.info(f"DEBUG multi: Final len = {len(answer)} chars")
            # Return сразу — без single RAG
            await asyncio.to_thread(insert_application_logs, session_id, query_input.question, answer, model_name)
            return QueryResponse(
                answer=answer,
                session_id=session_id,
                model=query_input.model,
                context=context_text
            )
        except Exception as e:
            logging.error(f"Multi-agent error: {e} — fallback to single RAG")
            # Только fallback, если краш
            rag_response = rag_chain.invoke({"input": query_input.question, "chat_history": chat_history})
            answer = rag_response["answer"]
            context_text = "\n\n".join([doc.page_content for doc in rag_response.get("context", [])])
    else:
        logging.info("DEBUG: Running single RAG")
        response = rag_chain.invoke({"input": query_input.question, "chat_history": chat_history})
        answer = response["answer"]
        context_text = "\n\n".join([doc.page_content for doc in response.get("context", [])])

    # Общий return (только для single или fallback)
    await asyncio.to_thread(insert_application_logs, session_id, query_input.question, answer, model_name)
    return QueryResponse(
        answer=answer,
        session_id=session_id,
        model=query_input.model,
        context=context_text
    )

    # 3. ПРОВЕРКА: Если LLM не инициализировался (решает ошибку 'NoneType')
    if rag_chain is None:
        logging.error(f"Query error in session {session_id}: RAG Chain failed to initialize (LLM/DB issue)")
        raise HTTPException(
            status_code=503,
            detail="LLM or Database components failed to initialize. Check GEMINI_API_KEY."
        )
    logging.info(f"Query: {query_input.question}. История: {len(chat_history)} сообщений.")
    try:
        # 4. Выполняем запрос
        response = rag_chain.invoke({
            "input": query_input.question,
            "chat_history": chat_history,
        })

        # Дебаг: Фикс для list
        context_docs = response.get("context", [])
        context_str = "\n\n---\n\n".join([doc.page_content for doc in context_docs])
        logging.info(
            f"Контекст вернул {len(context_docs)} чанков (str len: {len(context_str)} символов) для query '{query_input.question}'.")

        # 5. Извлекаем ответ и контекст
        answer = response["answer"]
        context_text = "\n\n".join([doc.page_content for doc in context_docs])  # Уже есть, ок

        # 6. Сохраняем историю и логи
        await asyncio.to_thread(insert_application_logs, session_id, query_input.question, answer, model_name)

        return QueryResponse(
            answer=answer,
            session_id=session_id,
            model=query_input.model,
            context=context_text
        )

    except Exception as e:
        logging.error(f"Query processing error in session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing error: {str(e)}")

@app.post("/upload-doc")
async def upload_document(file: UploadFile = File(...)):
    logging.info(f"DEBUG backend: /upload-doc called with filename = {file.filename}, size = {file.size}")
    if file.filename.endswith(('.pdf', '.docx', '.html', '.txt')):
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, file.filename)

        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            file_id = await asyncio.to_thread(insert_document_record, file.filename)

            success = await asyncio.to_thread(index_document_to_chroma, file_path, file_id)

            if success:
                logging.info(f"File '{file.filename}' uploaded and indexed successfully with ID {file_id}.")
                return {"message": "File uploaded and indexed successfully", "file_id": file_id}
            else:
                await asyncio.to_thread(delete_document_record, file_id)
                logging.error(f"Failed to index file {file.filename} into Chroma. Deleting DB record.")
                raise HTTPException(status_code=500, detail="Failed to index document (Embedding/Chroma issue).")

        except Exception as e:
            logging.error(f"Upload error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type.")


@app.post("/upload-gdrive")
async def upload_gdrive_folder(request: UploadGDriveRequest):
    if GDRIVE_EXTRACTOR is None:
        raise HTTPException(status_code=501, detail="Google Drive functionality is disabled because 'credentials.json' is missing or GDrive Extractor failed to initialize.")

    logging.info(f"Starting GDrive upload for URL: {request.folder_url}")
    try:
        documents = await asyncio.to_thread(GDRIVE_EXTRACTOR.process_folder, request.folder_url)

        if not documents:
            raise ValueError("No valid documents extracted from folder")

        # Фикс: Группируем docs по filename, чтобы избежать дублей
        from collections import defaultdict
        grouped_docs = defaultdict(list)
        for doc in documents:
            filename = doc.metadata.get("source", "GDrive File")
            grouped_docs[filename].append(doc)

        indexed_count = 0
        for filename, docs_group in grouped_docs.items():
            # Один file_id на уникальный filename
            file_id = await asyncio.to_thread(insert_document_record, filename)

            # Индексируем все docs из группы (если loader вернул несколько на файл)
            success = await asyncio.to_thread(index_document_to_chroma, docs_group, file_id)

            if success:
                indexed_count += 1
                logging.info(f"Indexed {filename} with {len(docs_group)} docs, file_id {file_id}")
            else:
                await asyncio.to_thread(delete_document_record, file_id)
                logging.warning(f"Failed to index {filename}, deleted record {file_id}")

        logging.info(f"Successfully indexed {indexed_count} unique files out of {len(grouped_docs)} from Google Drive.")
        return {"message": f"Successfully processed and indexed {indexed_count} unique documents from Google Drive.",
                "total_unique_files": len(grouped_docs), "indexed_files": indexed_count}

    except Exception as e:
        logging.error(f"GDrive upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"GDrive processing error: {str(e)}")


@app.get("/list-docs", response_model=list[DocumentInfo])
async def list_documents():
    try:
        documents = await asyncio.to_thread(get_all_documents)
        return documents
    except Exception as e:
        logging.error(f"List documents error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/delete-doc")
async def delete_document(request: DeleteFileRequest):
    try:
        chroma_delete_success = await asyncio.to_thread(delete_doc_from_chroma, request.file_id)

        if chroma_delete_success:
            db_delete_success = await asyncio.to_thread(delete_document_record, request.file_id)
            if db_delete_success:
                logging.info(f"Successfully deleted document with file_id {request.file_id} from the system.")
                return {"message": f"Successfully deleted document with file_id {request.file_id} from the system."}
            else:
                logging.error(
                    f"Deleted from Chroma but failed to delete document with file_id {request.file_id} from the database.")
                raise HTTPException(status_code=500,
                                    detail="Document deleted from Chroma, but failed to remove DB record.")
        else:
            logging.error(f"Failed to delete document with file_id {request.file_id} from Chroma.")
            raise HTTPException(status_code=500,
                                detail=f"Failed to delete document with file_id {request.file_id} from Chroma.")
    except Exception as e:
        logging.error(f"Delete error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))