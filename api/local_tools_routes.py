"""Authenticated local-tool routes; mounted behind the application's access gate."""
import asyncio
from typing import Literal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from assistant_models import AssistantError


class DocumentProposalInput(BaseModel):
    operation: Literal['create', 'edit', 'convert']
    path: str = Field(min_length=1, max_length=500)
    text: str = Field(default='', max_length=1_048_576)
    old: str = Field(default='', max_length=1_048_576)
    new: str = Field(default='', max_length=1_048_576)
    expected_count: int = Field(default=1, ge=1, le=100)
    target_path: str = Field(default='', max_length=500)


class ApplyInput(BaseModel):
    proposal_id: str = Field(pattern=r'^[0-9a-f]{32}$')


def local_tools_router(documents, repositories):
    router = APIRouter(prefix='/workspaces/{workspace_id}')

    async def call(function, *args):
        try:
            return await asyncio.to_thread(function, *args)
        except AssistantError:
            raise
        except LookupError as exc:
            raise HTTPException(404, 'Рабочая папка не найдена.') from exc
        except FileExistsError as exc:
            raise HTTPException(409, 'Файл назначения уже существует.') from exc
        except (ValueError, OSError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @router.get('/repository')
    async def overview(workspace_id: str):
        return await call(repositories.overview, workspace_id)

    @router.get('/search')
    async def search(workspace_id: str, query: str, directory: str = '', max_results: int = 50):
        return await call(repositories.search, workspace_id, query, directory, max_results)

    @router.get('/lines')
    async def lines(workspace_id: str, path: str, start: int = 1, end: int = 200):
        return await call(documents.read_lines, workspace_id, path, start, end)

    @router.get('/source')
    async def source(workspace_id: str, path: str, revision: str, start: int, end: int):
        try:
            return await call(documents.filesystem.source_excerpt, workspace_id, path, revision, start, end)
        except HTTPException as exc:
            # Do not expose absolute filesystem paths in missing-source errors.
            if exc.status_code == 400:
                raise HTTPException(400, 'Не удалось прочитать указанный фрагмент источника.') from exc
            raise

    @router.post('/documents/preview')
    async def preview(workspace_id: str, payload: DocumentProposalInput):
        if payload.operation == 'convert':
            return await call(documents.prepare_convert, workspace_id, payload.path, payload.target_path)
        if payload.operation == 'edit':
            return await call(documents.prepare_edit, workspace_id, payload.path,
                              payload.old, payload.new, payload.expected_count)
        return await call(documents.prepare_create, workspace_id, payload.path, payload.text)

    @router.post('/documents/apply')
    async def apply(workspace_id: str, payload: ApplyInput):
        return await call(documents.apply, workspace_id, payload.proposal_id)

    return router
