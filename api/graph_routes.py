"""Read-only, bounded API for local knowledge-graph visualization."""
from fastapi import APIRouter, HTTPException, Query

from graph_service import GraphService
from settings import MAX_GRAPH_DEPTH, MAX_GRAPH_RESULTS

router = APIRouter(prefix="/graph", tags=["graph"])


def _document_graph_call(workspace_id, method, *args):
    from semantic_graph import SemanticDocumentGraph
    from assistant_models import AssistantError
    from chroma_utils import vectorstore, workspace_vectorstore
    filesystem = None
    try:
        if workspace_id:
            from workspace import FilesystemProvider, WorkspaceRepository
            filesystem = FilesystemProvider(WorkspaceRepository())
            filesystem.sync_index(workspace_id)
        graph = SemanticDocumentGraph(workspace_vectorstore if workspace_id else vectorstore,
                                      workspace_id, filesystem=filesystem)
        return getattr(graph, method)(*args)
    except AssistantError as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(409, str(exc) if method == 'explain_link' else 'Источник карты недоступен. Обновите выбранную папку.') from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(409, 'Источник карты недоступен. Обновите выбранную папку.') from exc


@router.get('/map')
def document_map(workspace_id: str | None = Query(default=None, min_length=1, max_length=128),
                 limit: int = Query(default=200, ge=1, le=500)):
    return _document_graph_call(workspace_id, 'map', limit)


@router.get('/link')
def document_link(source_id: str = Query(min_length=1, max_length=1024),
                  target_id: str = Query(min_length=1, max_length=1024),
                  workspace_id: str | None = Query(default=None, min_length=1, max_length=128)):
    return _document_graph_call(workspace_id, 'explain_link', source_id, target_id)


def _store():
    from chroma_utils import vectorstore
    return vectorstore


def _workbench(workspace_id):
    from graph_workbench import graph_workbench
    return graph_workbench(workspace_id)


def _graph_call(workspace_id, method, *args):
    from assistant_models import AssistantError
    try:
        return getattr(_workbench(workspace_id), method)(*args)
    except AssistantError as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc
    except (LookupError, OSError, ValueError) as exc:
        raise HTTPException(409, 'Источник графа недоступен. Обновите выбранную папку.') from exc


@router.get('/browse/entities')
def browse_entities(workspace_id: str | None = Query(default=None, min_length=1, max_length=128),
                    query: str = Query(default='', max_length=200),
                    limit: int = Query(default=20, ge=1, le=50)):
    return _graph_call(workspace_id, 'entities', query, limit)


@router.get('/explore')
def explore(workspace_id: str | None = Query(default=None, min_length=1, max_length=128),
            entity_ids: list[int] = Query(default=[], max_length=20),
            depth: int = Query(default=1, ge=1, le=2),
            document_id: str | None = Query(default=None, max_length=1024)):
    if any(value <= 0 or value > 9223372036854775807 for value in entity_ids):
        raise HTTPException(422, 'entity_ids must be positive SQLite IDs')
    return _graph_call(workspace_id, 'explore', entity_ids, depth, document_id)


@router.get('/connect')
def connect(source_id: int = Query(gt=0, le=9223372036854775807),
            target_id: int = Query(gt=0, le=9223372036854775807),
            workspace_id: str | None = Query(default=None, min_length=1, max_length=128),
            max_hops: int = Query(default=4, ge=1, le=6)):
    return _graph_call(workspace_id, 'connect', source_id, target_id, max_hops)


@router.get("/entities")
def entities(query: str = Query(default="", max_length=200),
             limit: int = Query(default=20, ge=1, le=MAX_GRAPH_RESULTS)):
    """Return a bounded entity picker for the graph UI."""
    service = GraphService()
    rows = service.find_entity(query, limit) if query.strip() else service.repository.all_entities(limit)
    return [{"id": int(row["id"]), "label": row["display_name"], "type": row["entity_type"]}
            for row in rows[:limit]]


@router.get('/subgraph')
def entities_subgraph(entity_ids: list[int] = Query(min_length=1, max_length=MAX_GRAPH_RESULTS),
                      depth: int = Query(default=1, ge=1, le=MAX_GRAPH_DEPTH),
                      limit: int = Query(default=MAX_GRAPH_RESULTS, ge=1, le=MAX_GRAPH_RESULTS)):
    return GraphService().entities_subgraph(entity_ids, depth, limit, _store())


@router.get("/entity/{entity_id}")
def entity_details(entity_id: int):
    result = GraphService().entity_subgraph(entity_id, 1, MAX_GRAPH_RESULTS, _store())
    if result is None:
        raise HTTPException(404, "Entity is no longer available")
    return result


@router.get("/entity/{entity_id}/neighbors")
def entity_neighbors(entity_id: int,
                     depth: int = Query(default=1, ge=1, le=MAX_GRAPH_DEPTH),
                     limit: int = Query(default=MAX_GRAPH_RESULTS, ge=1, le=MAX_GRAPH_RESULTS)):
    result = GraphService().entity_subgraph(entity_id, depth, limit, _store())
    if result is None:
        raise HTTPException(404, "Entity is no longer available")
    return result


@router.get("/document/{document_id}")
def document_details(document_id: int,
                     limit: int = Query(default=MAX_GRAPH_RESULTS, ge=1, le=MAX_GRAPH_RESULTS)):
    result = GraphService().document_subgraph(document_id, limit, _store())
    if result is None:
        raise HTTPException(404, "Document is no longer available")
    return result
