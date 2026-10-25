from settings import API_DIR, DATA_DIR, MAX_UPLOAD_BYTES
# main.py - ФИНАЛЬНЫЙ КОД
import sys
import os
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GRPC_TRACE'] = 'off'
import asyncio
import threading
import base64
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Body, Response, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
from pathlib import Path
from typing import Literal
from datetime import datetime, timezone
from pydantic_models import (QueryInput, QueryResponse, DocumentInfo, DeleteFileRequest,
                             UploadGDriveRequest, SearchRequest, SearchResult, ReindexRequest)
from langchain_utils import get_rag_chain, answer_general_chat
from db_utils import insert_application_logs, get_chat_history, get_all_documents, insert_document_record, \
    delete_document_record, get_document_source, bump_knowledge_epoch
from chroma_utils import index_document_to_chroma, delete_doc_from_chroma, purge_orphaned_documents
from gdrive_extractor import GoogleDriveExtractor
from langchain_utils import debug_chroma_search
import shutil
import tempfile
from multi_agent_chain import get_multi_agent_chain
from dotenv import load_dotenv
from query_router import query_router
from model_runtime import get_llm_provider
from embedding_provider import get_embedding_provider
from verifier import verify_answer, verify_knowledge_answer
from workspace import FilesystemProvider, WorkspaceRepository, WorkspaceSkill
from workspace_vector_index import WorkspaceVectorIndex
from workspace_graph import WorkspaceGraphRepository
from conversation_context import ConversationResolver
from tts_provider import TTSUnavailable, tts_provider
from llm_contract import GenerationCancelled, current_token_sink
from document_service import DocumentService, create_document_schema
from index_transactions import IndexTransactions
from local_access import LocalAccessMiddleware



# Загружаем .env
load_dotenv()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Also clean up a partially completed startup, before a request is served.
    try:
        await start_model_services()
        yield
    finally:
        await stop_model_services()


app = FastAPI(title="Фортуна", lifespan=lifespan)

from agent_steps import router as agent_router
app.include_router(agent_router)
from chat_routes import router as chat_router
app.include_router(chat_router)
from graph_routes import router as graph_router
app.include_router(graph_router)

app.add_middleware(LocalAccessMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://tauri.localhost",
        "https://tauri.localhost",
        "tauri://localhost",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Accept", "Content-Type", "X-Fortuna-Token", "X-Fortuna-Client"],
)

GDRIVE_EXTRACTOR = None
_drive_lock = asyncio.Lock()
embedding_migration_task = None
create_document_schema()
document_service = DocumentService(
    index=lambda *args, **kwargs: index_document_to_chroma(*args, **kwargs),
    insert=lambda *args: insert_document_record(*args),
    delete_record=lambda *args: delete_document_record(*args),
    store_getter=lambda: __import__('chroma_utils').vectorstore,
    delete_vectors=lambda *args: delete_doc_from_chroma(*args),
    source_getter=lambda *args: get_document_source(*args),
)


def recover_indexes():
    from chroma_utils import vectorstore, workspace_vectorstore
    IndexTransactions().recover({'rag_collection': vectorstore, 'workspace_collection': workspace_vectorstore})


async def migrate_embedding_spaces():
    try:
        embeddings = get_embedding_provider()
        if embeddings.manager:
            await asyncio.to_thread(embeddings.manager.ensure_ready)
        from chroma_utils import rebuild_embedding_indexes
        await asyncio.to_thread(rebuild_embedding_indexes)
    except Exception:
        logger.exception('Embedding index migration unavailable; legacy index preserved for retry')

async def start_model_services():
    global embedding_migration_task
    try:
        await asyncio.to_thread(recover_indexes)
        await asyncio.to_thread(assistant_repository.recover_interrupted_tasks)
        documents = await asyncio.to_thread(get_all_documents)
        await asyncio.to_thread(purge_orphaned_documents, [item["id"] for item in documents])
    except Exception:
        logging.exception("Could not reconcile Chroma with the document database")
    get_llm_provider().manager.start_background()
    embeddings = get_embedding_provider()
    if embeddings.manager:
        embeddings.manager.start_background()
    embedding_migration_task = asyncio.create_task(migrate_embedding_spaces())


async def stop_model_services():
    global embedding_migration_task
    if embedding_migration_task:
        embedding_migration_task.cancel()
        await asyncio.gather(embedding_migration_task, return_exceptions=True)
        embedding_migration_task = None
    provider = get_llm_provider()
    try:
        await asyncio.to_thread(provider.close)
    finally:
        try:
            if provider.manager:
                await asyncio.to_thread(provider.manager.stop)
        finally:
            embeddings = get_embedding_provider()
            if embeddings.manager:
                await asyncio.to_thread(embeddings.manager.stop)


@app.get("/api/model-status")
async def model_status():
    from model_profiles import PROFILES, get_profile
    from chat_model_selection import selection_status
    status = await asyncio.to_thread(get_llm_provider().health)
    return {**status, **selection_status(), 'profiles': list(PROFILES), 'default_profile': get_profile().name,
            'message': 'Модель готова' if status['ready'] else status.get('error') or 'Запуск модели...'}


from pydantic import BaseModel


class ChatModelSelectionInput(BaseModel):
    model_id: str


@app.post('/api/chat-model')
async def select_chat_model(request: ChatModelSelectionInput):
    from chat_model_selection import switch_chat_model
    async with assistant_core.deployment_lock:
        if assistant_core.active_tasks:
            raise HTTPException(status_code=409, detail='Дождитесь завершения текущего ответа перед сменой модели.')
        try:
            return await asyncio.to_thread(switch_chat_model, request.model_id, get_llm_provider())
        except AssistantError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@app.post("/assistant/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    return {'cancelled': assistant_core.cancel_task(task_id)}


@app.get("/assistant/tasks")
async def active_tasks():
    return await asyncio.to_thread(lambda: [assistant_core.repository.get_task(task_id)
                                           for task_id in list(assistant_core.active_tasks)])


@app.get("/assistant/tasks/{task_id}")
async def task_status(task_id: str):
    try:
        return {'task': await asyncio.to_thread(assistant_core.repository.get_task, task_id),
                'actions': await asyncio.to_thread(assistant_core.repository.actions, task_id)}
    except LookupError:
        raise HTTPException(404, 'Task not found')


@app.get("/api/health", tags=["system"])
async def health() -> dict[str, str]:
    """Cheap process-readiness probe used by the desktop shell."""
    return {"status": "ok"}


@app.post('/rebuild-search-index')
async def rebuild_search_index():
    from chroma_utils import rebuild_embedding_indexes
    try:
        return await asyncio.to_thread(rebuild_embedding_indexes)
    except Exception as exc:
        logging.exception('Embedding index migration failed; previous collections retained')
        raise HTTPException(503, 'Не удалось обновить поисковый индекс. Проверьте embedding-модель и журнал backend.') from exc


@app.get("/voice/status")
async def voice_status():
    return tts_provider.status()


@app.post("/voice/synthesize")
async def synthesize_voice(payload: dict = Body(...)):
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(422, "text is required")
    if len(text) > 12000:
        raise HTTPException(413, "Text is too long for one voice turn")
    try:
        audio = await asyncio.to_thread(tts_provider.synthesize, text)
        return Response(content=audio, media_type="audio/wav",
                        headers={"Cache-Control": "no-store"})
    except TTSUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc


@app.post("/voice/stream")
async def stream_voice(payload: dict = Body(...)):
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(422, "text is required")

    def chunks():
        try:
            announced = False
            for sample_rate, tensor in tts_provider.stream_pcm(text[:4000]):
                if not announced:
                    yield json.dumps({"type": "format", "sample_rate": sample_rate}) + "\n"
                    announced = True
                samples = tensor.detach().cpu().float().reshape(-1).clamp(-1, 1)
                pcm = (samples.numpy() * 32767).astype("<i2").tobytes()
                yield json.dumps({"type": "audio", "pcm": base64.b64encode(pcm).decode("ascii")}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
        except TTSUnavailable as exc:
            yield json.dumps({"type": "error", "message": str(exc)}, ensure_ascii=False) + "\n"

    return StreamingResponse(chunks(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


workspace_repository = WorkspaceRepository()
filesystem_provider = FilesystemProvider(workspace_repository)
from workspace_documents import WorkspaceDocuments
from workspace_repository_tools import RepositoryTools
from local_tools_skill import LocalToolsSkill
from local_tools_routes import local_tools_router
workspace_documents = WorkspaceDocuments(filesystem_provider)
repository_tools = RepositoryTools(filesystem_provider)
local_tools_skill = LocalToolsSkill(workspace_documents, repository_tools)
app.include_router(local_tools_router(workspace_documents, repository_tools))
workspace_vector_index = WorkspaceVectorIndex(
    filesystem_provider, graph_repository=WorkspaceGraphRepository()
)


@app.get("/workspaces/active")
async def active_workspace():
    workspace = await asyncio.to_thread(workspace_repository.active)
    return {"workspace": workspace}


@app.post("/workspaces/select")
async def select_workspace(payload: dict = Body(...)):
    root_path = payload.get("root_path")
    if not isinstance(root_path, str) or not root_path.strip():
        raise HTTPException(422, "root_path is required")
    try:
        workspace = await asyncio.to_thread(workspace_repository.select, root_path)
        await asyncio.to_thread(filesystem_provider.sync_index, workspace.id)
        return await asyncio.to_thread(filesystem_provider.list_directory, workspace.id, "")
    except (ValueError, FileNotFoundError, PermissionError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/workspaces/{workspace_id}/files")
async def list_workspace_files(workspace_id: str, path: str = ""):
    try:
        return await asyncio.to_thread(filesystem_provider.list_directory, workspace_id, path)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (ValueError, FileNotFoundError, NotADirectoryError, PermissionError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/workspaces/{workspace_id}/file")
async def read_workspace_file(workspace_id: str, path: str):
    try:
        workspace, relative_path, content = await asyncio.to_thread(
            filesystem_provider.read_text_file, workspace_id, path
        )
        return {"workspace_id": workspace.id, "path": relative_path, "content": content}
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (ValueError, FileNotFoundError, PermissionError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/workspaces/{workspace_id}/sync")
async def sync_workspace(workspace_id: str):
    try:
        return await asyncio.to_thread(filesystem_provider.sync_index, workspace_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (ValueError, FileNotFoundError, PermissionError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/workspaces/{workspace_id}/index")
async def workspace_index(workspace_id: str, path: str = ""):
    try:
        await asyncio.to_thread(filesystem_provider.sync_index, workspace_id)
        return await asyncio.to_thread(workspace_repository.indexed_files, workspace_id, path)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (ValueError, FileNotFoundError, PermissionError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/workspaces/{workspace_id}/vector-sync")
async def sync_workspace_vectors(workspace_id: str, extract_relations: bool = Query(default=True),
                                 mode: Literal['update', 'retry_failed'] = Query(default='update')):
    try:
        options = {}
        if not extract_relations:
            options['extract_relations'] = False
        if mode != 'update':
            options['mode'] = mode
        result = await asyncio.to_thread(workspace_vector_index.sync, workspace_id, **options)
        return result.__dict__
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (ValueError, FileNotFoundError, PermissionError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/workspaces/{workspace_id}/index-status")
async def workspace_index_status(workspace_id: str):
    try:
        scan = await asyncio.to_thread(filesystem_provider.sync_index, workspace_id)
        status = await asyncio.to_thread(workspace_vector_index.status, workspace_id)
        return {**status, 'truncated': scan.truncated, 'scan_errors': len(scan.errors),
                'checked_at': datetime.now(timezone.utc).isoformat()}
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (ValueError, FileNotFoundError, PermissionError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/assistant/input", response_model=QueryResponse)
@app.post("/query", response_model=QueryResponse)
async def query_endpoint(query_input: QueryInput, response: Response):
    result, elapsed_ms = await assistant_core.process_input(query_input)
    response.headers["X-Process-Time-Ms"] = str(elapsed_ms)
    return result


@app.post("/assistant/stream")
async def stream_query_endpoint(query_input: QueryInput):
    """Stream conversational model tokens and expose cancellation early."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=256)
    task_id: str | None = None
    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()
    streamed: list[str] = []
    disconnected = threading.Event()
    pending_puts = set()
    puts_lock = threading.Lock()

    async def publish(item):
        await queue.put(item)

    def answer_chunks(value: str, size: int = 48):
        """Keep transport incremental without exposing intermediate model generations."""
        start = 0
        while start < len(value):
            end = min(start + size, len(value))
            if end < len(value):
                boundary = max(value.rfind(" ", start, end), value.rfind("\n", start, end))
                if boundary > start:
                    end = boundary + 1
            yield value[start:end]
            start = end

    def publish_token(value: str):
        if not value:
            return
        if disconnected.is_set():
            raise GenerationCancelled()
        item = {"type": "token", "text": value}
        if threading.get_ident() == loop_thread:
            queue.put_nowait(item)
            streamed.append(value)
            return
        with puts_lock:
            if disconnected.is_set():
                raise GenerationCancelled()
            future = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
            pending_puts.add(future)
        try:
            future.result(timeout=30)
            streamed.append(value)
        except Exception as exc:
            future.cancel()
            raise GenerationCancelled() from exc
        finally:
            with puts_lock:
                pending_puts.discard(future)

    async def run():
        nonlocal task_id
        token = current_token_sink.set(publish_token if query_input.live_mode and not query_input.workspace_id else None)
        try:
            await publish({"type": "stage", "stage": "thinking"})
            def started(task):
                nonlocal task_id
                task_id = task.id
                queue.put_nowait({"type": "task", "task_id": task.id})
            def progress(message):
                def push():
                    # Status updates can be coalesced/dropped under backpressure;
                    # final results and tokens retain their delivery protocol.
                    if not disconnected.is_set() and not queue.full():
                        queue.put_nowait({"type":"progress","message":message})
                loop.call_soon_threadsafe(push)
            response, _ = await assistant_core.process_input(
                query_input,
                source="voice" if query_input.live_mode else "text", on_task_started=started,on_progress=progress,
            )
            if not streamed:
                await publish({"type": "stage", "stage": "answering"})
                for value in answer_chunks(response.answer):
                    await publish({"type": "token", "text": value})
            elif "".join(streamed) != response.answer:
                # Tool-only and fallback paths may not stream their final text.
                await publish({"type": "reset"})
                for value in answer_chunks(response.answer):
                    await publish({"type": "token", "text": value})
            await publish({"type": "result", "response": response.model_dump(mode="json")})
        except (asyncio.CancelledError, GenerationCancelled):
            if task_id:
                assistant_core.cancel_task(task_id)
            raise
        except Exception as exc:
            logger.exception("Live assistant turn failed")
            await publish({"type": "error", "message": str(exc)})
        finally:
            current_token_sink.reset(token)
            # Completion is part of the protocol, including when the consumer
            # is slow. Disconnect cancels this producer and its pending put.
            if not disconnected.is_set():
                await queue.put(None)

    async def events():
        task = asyncio.create_task(run())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield json.dumps(item, ensure_ascii=False) + "\n"
        finally:
            # A disconnected consumer cannot drain a full queue. Unblock both
            # the completion marker and token publishers before joining run().
            with puts_lock:
                disconnected.set()
                for future in pending_puts:
                    future.cancel()
            if not task.done():
                if task_id:
                    assistant_core.cancel_task(task_id)
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    return StreamingResponse(events(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@app.post('/assistant/live')
async def live_query_endpoint(query_input: QueryInput):
    return await stream_query_endpoint(query_input.model_copy(update={'live_mode': True}))


async def read_limited(file):
    size = 0
    while chunk := await file.read(1024 * 1024):
        size += len(chunk)
        if size > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Upload exceeds size limit")
        yield chunk


@app.post("/upload-doc")
async def upload_document(file: UploadFile = File(...)):
    file_id = None
    try:
        filename = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
        ext = os.path.splitext(filename)[1].lower()
        if ext not in {".pdf", ".docx", ".html", ".txt"}:
            raise HTTPException(400, "Unsupported file type")
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "upload" + ext)
            with open(path, "wb") as target:
                async for chunk in read_limited(file):
                    await asyncio.to_thread(target.write, chunk)
            source_bytes = await asyncio.to_thread(Path(path).read_bytes)
            try:
                file_id = await asyncio.to_thread(document_service.upload, filename, source_bytes, 'raw', path)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
        return {"message": "File uploaded and indexed successfully", "file_id": file_id}
    except HTTPException:
        raise
    except Exception:
        logging.exception("Upload failed for document %s", file_id)
        raise HTTPException(500, "Indexing failed. Check backend logs.")
    finally:
        await file.close()


# Optional browser preview for developers only. Desktop production loads the
# bundled React assets directly from Tauri and exposes FastAPI as API-only.
if os.getenv("KONTUR_DESKTOP") != "1":
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse
    _frontend = API_DIR.parent / 'frontend' / 'dist'
    if (_frontend / 'assets').is_dir():
        app.mount('/assets', StaticFiles(directory=str(_frontend / 'assets')), name='frontend-assets')

    @app.get('/', include_in_schema=False)
    def frontend_index():
        if not (_frontend / 'index.html').is_file():
            raise HTTPException(503, 'Build frontend with npm run build in frontend/')
        return FileResponse(_frontend / 'index.html')


@app.post("/upload-gdrive")
async def upload_gdrive_folder(request: UploadGDriveRequest):
    global GDRIVE_EXTRACTOR
    try:
        async with _drive_lock:
            if GDRIVE_EXTRACTOR is None:
                token = str(DATA_DIR / "token.json")
                if not os.path.exists(token):
                    raise HTTPException(503, "Run api/gdrive_auth.py to authorize Google Drive first")
                GDRIVE_EXTRACTOR = await asyncio.to_thread(GoogleDriveExtractor, str(DATA_DIR / "credentials.json"), token)
            documents = await asyncio.to_thread(GDRIVE_EXTRACTOR.process_folder, request.folder_url)
            report = getattr(GDRIVE_EXTRACTOR, 'last_report', {})
            import_report = {key: report.get(key, default) for key, default in (
                ('attempted', len(documents)), ('failed', []), ('unsupported', 0), ('limited', False))}
            import_report['failed'] = list(import_report['failed'])

        if not documents:
            names = ', '.join(str(item.get('name', '')) for item in import_report['failed'])
            raise HTTPException(422, 'Ни один документ не импортирован. ' +
                ('Не обработаны: ' + names + '. ' if names else '') +
                ('Достигнут лимит импорта.' if import_report['limited'] else ''))

        # Фикс: Группируем docs по filename, чтобы избежать дублей
        from collections import defaultdict
        grouped_docs = defaultdict(list)
        for doc in documents:
            filename = doc.metadata.get("source", "GDrive File")
            grouped_docs[doc.metadata.get("gdrive_id", filename)].append(doc)

        indexed_count = 0
        for _, docs_group in grouped_docs.items():
            filename = docs_group[0].metadata.get("source", "GDrive File")
            # Один file_id на уникальный filename
            serialized = json.dumps([
                {"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs_group
            ], ensure_ascii=False, default=str).encode("utf-8")
            try:
                remote_id = docs_group[0].metadata.get('gdrive_id')
                file_id = await asyncio.to_thread(document_service.upload, filename, serialized,
                    'documents_json', docs_group, 'gdrive:' + str(remote_id) if remote_id else None)
                indexed_count += 1
                logging.info(f"Indexed {filename} with {len(docs_group)} docs, file_id {file_id}")
            except ValueError:
                logging.warning('Failed to index Drive document %s; previous revision preserved', filename)
                import_report['failed'].append({'id': str(remote_id or ''), 'name': filename, 'error': 'Indexing failed'})

        if indexed_count == 0:
            names = ', '.join(str(item.get('name', '')) for item in import_report['failed'])
            raise HTTPException(422, 'Не удалось проиндексировать документы: ' + names)
        logging.info(f"Successfully indexed {indexed_count} unique files out of {len(grouped_docs)} from Google Drive.")
        return {"message": f"Successfully processed and indexed {indexed_count} unique documents from Google Drive.",
                "total_unique_files": len(grouped_docs), "indexed_files": indexed_count,
                'status': 'partial' if import_report['failed'] or import_report['limited'] else 'completed',
                **import_report}

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"GDrive upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"GDrive processing error: {str(e)}")


@app.get("/list-docs", response_model=list[DocumentInfo])
async def list_documents():
    try:
        documents = await asyncio.to_thread(get_all_documents)
        return documents
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"List documents error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def _document_result(document):
    metadata = dict(document.metadata)
    found_by = metadata.pop("found_by", "semantic")
    score = metadata.pop("score", 0.0)
    return {
        "document_id": metadata.get("document_id", metadata.get("file_id")),
        "chunk_id": metadata.get("chunk_id", ""),
        "text": document.page_content,
        "metadata": metadata,
        "score": score,
        "found_by": [value for value in found_by.split(",") if value],
    }


@app.post("/search", response_model=list[SearchResult])
async def search_documents(request: SearchRequest):
    """Search document chunks without running the answer-generating LLM."""
    from chroma_utils import retriever, vectorstore
    from hybrid_context_extractor import HybridRetriever
    if vectorstore is None or (request.mode.value != "lexical" and retriever is None):
        raise HTTPException(503, "Search unavailable. Check Chroma and the embedding provider.")
    searcher = HybridRetriever(
        retriever, vectorstore, request.mode.value, False, request.limit,
        lexical_weight=1.25 if request.mode.value == "hybrid" else 1.0,
    )
    try:
        return [item.as_dict() for item in await asyncio.to_thread(searcher.search, request.query)]
    except Exception:
        logging.exception("Document search failed")
        raise HTTPException(502, "Document search failed. Check backend logs.")


@app.post("/delete-doc")
async def delete_document(request: DeleteFileRequest):
    try:
        await asyncio.to_thread(document_service.delete, request.file_id)
        return {"message": f"Deleted document {request.file_id}."}
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        logging.exception("Document deletion failed; recovery journal preserves the source")
        raise HTTPException(500, "Document deletion failed. Check backend logs.") from exc


def _reindex_document(file_id: int) -> None:
    document_service.reindex(file_id)


@app.post("/reindex-docs")
async def reindex_documents(request: ReindexRequest):
    completed, failed = [], []
    for file_id in request.file_ids:
        try:
            await asyncio.to_thread(_reindex_document, file_id)
            completed.append(file_id)
        except (LookupError, ValueError) as exc:
            failed.append({"file_id": file_id, "error": str(exc)})
        except Exception as exc:
            logging.exception("Reindex failed for document %s", file_id)
            failed.append({"file_id": file_id, "error": str(exc)})
    status = "completed" if not failed else "partial" if completed else "failed"
    return {"status": status, "reindexed": completed, "failed": failed}

@app.post("/transcribe")
async def transcribe_audio_endpoint(file: UploadFile = File(...)):
    """Принимает аудиофайл, возвращает распознанный текст через Whisper."""
    try:
        from whisper_utils import transcribe_audio
        audio_bytes = bytearray()
        async for chunk in read_limited(file):
            audio_bytes.extend(chunk)
        if not audio_bytes:
            raise HTTPException(422, "Audio is empty")
        text = await asyncio.to_thread(transcribe_audio, bytes(audio_bytes), file.filename or "audio.wav")
        if not text:
            raise HTTPException(status_code=422, detail="Не удалось распознать речь")
        return {"text": text}
    except RuntimeError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Transcribe error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await file.close()


# Only dependency wiring lives in the HTTP application. Core has no FastAPI dependency.
from assistant_core import AssistantCore
from assistant_models import AssistantError
from assistant_repository import AssistantRepository
from agent_planner import LLMAgentPlanner
from repository_agent import RepositoryAgentSkill
from knowledge_skill import KnowledgeSkill, KnowledgeServices
from fastapi.responses import JSONResponse

assistant_repository = AssistantRepository()
# Assistant sessions may be created by the additive Core migration above.
# Re-run the idempotent workspace migration to add their workspace link on
# databases that did not have Assistant Core yet when this module was imported.
workspace_repository.migrate()
assistant_core = AssistantCore(
    assistant_repository,
    KnowledgeSkill(KnowledgeServices(
        route=lambda *a, **kw: query_router.route(*a, **kw),
        history=lambda *a, **kw: get_chat_history(*a, **kw),
        general_chat=lambda *a, **kw: answer_general_chat(*a, **kw),
        multi_chain=lambda: get_multi_agent_chain(),
        rag_chain=lambda **kw: get_rag_chain(**kw),
        document_result=lambda doc: _document_result(doc),
        verify=lambda *a, **kw: verify_knowledge_answer(*a, **kw),
    )),
    save_history=lambda *a: insert_application_logs(*a),
    workspace_skill=WorkspaceSkill(filesystem_provider, vector_indexer=workspace_vector_index),
    local_tools_skill=local_tools_skill,
    repository_agent_skill=RepositoryAgentSkill(filesystem_provider),
    history_loader=lambda session_id: get_chat_history(session_id),
    conversation_resolver=ConversationResolver(),
    agent_planner=LLMAgentPlanner(),
    response_saver=assistant_repository.save_response,
)
from conversation_memory import ConversationMemory
from db_utils import get_chat_archive, get_knowledge_epoch
from contextlib import closing
import settings as _context_settings

def _conversation_epoch():
    with closing(assistant_repository.connection_factory()) as conn:return get_knowledge_epoch(conn)

if _context_settings.CONVERSATION_COMPACTION_ENABLED:
    assistant_core.conversation_memory=ConversationMemory(assistant_repository,get_chat_archive,_conversation_epoch)


@app.exception_handler(AssistantError)
async def assistant_error_handler(request, exc):
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})
