"""Read-only, bounded API for local knowledge-graph visualization."""
from fastapi import APIRouter, HTTPException, Query

from graph_service import GraphService
from settings import MAX_GRAPH_DEPTH, MAX_GRAPH_RESULTS

router = APIRouter(prefix="/graph", tags=["graph"])


def _store():
    from chroma_utils import vectorstore
    return vectorstore


@router.get("/entities")
def entities(query: str = Query(default="", max_length=200),
             limit: int = Query(default=20, ge=1, le=MAX_GRAPH_RESULTS)):
    """Return a bounded entity picker for the graph UI."""
    service = GraphService()
    rows = service.find_entity(query, limit) if query.strip() else service.repository.all_entities(limit)
    return [{"id": int(row["id"]), "label": row["display_name"], "type": row["entity_type"]}
            for row in rows[:limit]]


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
