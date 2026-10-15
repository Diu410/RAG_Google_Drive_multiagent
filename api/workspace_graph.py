"""Workspace-scoped knowledge graph persistence and retrieval."""
from __future__ import annotations

from contextlib import closing

from db_utils import get_db_connection
from entity_normalization import normalize_entity_name


class WorkspaceGraphRepository:
    def __init__(self, connection_factory=get_db_connection):
        self.connection_factory = connection_factory
        self.create_schema()

    def create_schema(self) -> None:
        with closing(self.connection_factory()) as conn, conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS workspace_graph_entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workspace_id TEXT NOT NULL,
                    canonical_name TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    UNIQUE(workspace_id,canonical_name,entity_type));
                CREATE TABLE IF NOT EXISTS workspace_graph_relations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workspace_id TEXT NOT NULL,
                    source_entity_id INTEGER NOT NULL REFERENCES workspace_graph_entities(id),
                    relation_type TEXT NOT NULL,
                    target_entity_id INTEGER NOT NULL REFERENCES workspace_graph_entities(id),
                    file_key TEXT NOT NULL,
                    document_name TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    evidence TEXT NOT NULL DEFAULT '',
                    UNIQUE(workspace_id,source_entity_id,relation_type,target_entity_id,file_key,chunk_id));
                CREATE INDEX IF NOT EXISTS idx_workspace_graph_entity
                    ON workspace_graph_entities(workspace_id,canonical_name);
                CREATE INDEX IF NOT EXISTS idx_workspace_graph_file
                    ON workspace_graph_relations(workspace_id,file_key);
                CREATE INDEX IF NOT EXISTS idx_workspace_graph_source
                    ON workspace_graph_relations(workspace_id,source_entity_id);
                CREATE INDEX IF NOT EXISTS idx_workspace_graph_target
                    ON workspace_graph_relations(workspace_id,target_entity_id);
            """)

    @staticmethod
    def _entity(conn, workspace_id: str, name: str, kind: str) -> int:
        canonical = normalize_entity_name(name)
        if not canonical:
            raise ValueError("Entity name is empty after normalization")
        kind = (kind or "OTHER").upper()[:80]
        conn.execute("""INSERT OR IGNORE INTO workspace_graph_entities
            (workspace_id,canonical_name,display_name,entity_type) VALUES(?,?,?,?)""",
            (workspace_id, canonical, name.strip(), kind))
        return int(conn.execute("""SELECT id FROM workspace_graph_entities
            WHERE workspace_id=? AND canonical_name=? AND entity_type=?""",
            (workspace_id, canonical, kind)).fetchone()["id"])

    def replace_workspace_file_graph(self, workspace_id: str, file_key: str, entities: list[dict],
                                     relations: list[dict], document_name: str) -> None:
        with closing(self.connection_factory()) as conn, conn:
            conn.execute("DELETE FROM workspace_graph_relations WHERE workspace_id=? AND file_key=?",
                         (workspace_id, file_key))
            ids = {}
            for item in entities:
                name, kind = item["name"].strip(), item.get("type", "OTHER").upper()
                ids[(normalize_entity_name(name), kind)] = self._entity(conn, workspace_id, name, kind)
            for relation in relations:
                source = self._resolve(relation["source"], relation.get("source_type"), ids)
                target = self._resolve(relation["target"], relation.get("target_type"), ids)
                if source is None or target is None:
                    continue
                conn.execute("""INSERT OR IGNORE INTO workspace_graph_relations
                    (workspace_id,source_entity_id,relation_type,target_entity_id,file_key,document_name,
                     chunk_id,confidence,evidence) VALUES(?,?,?,?,?,?,?,?,?)""",
                    (workspace_id, source, relation.get("relation", "RELATED_TO").upper()[:120], target,
                     file_key, document_name, relation["chunk_id"],
                     max(0.0, min(1.0, float(relation.get("confidence", 1.0)))),
                     relation.get("evidence", "")[:2000]))
            self._cleanup(conn, workspace_id)

    @staticmethod
    def _resolve(name: str, kind: str | None, ids: dict) -> int | None:
        canonical = normalize_entity_name(name)
        if kind and (canonical, kind.upper()) in ids:
            return ids[(canonical, kind.upper())]
        matches = [value for (candidate, _), value in ids.items() if candidate == canonical]
        return matches[0] if len(set(matches)) == 1 else None

    @staticmethod
    def _cleanup(conn, workspace_id: str) -> None:
        conn.execute("""DELETE FROM workspace_graph_entities WHERE workspace_id=? AND id NOT IN (
            SELECT source_entity_id FROM workspace_graph_relations WHERE workspace_id=?
            UNION SELECT target_entity_id FROM workspace_graph_relations WHERE workspace_id=?)""",
            (workspace_id, workspace_id, workspace_id))

    def delete_missing(self, workspace_id: str, active_file_keys: set[str]) -> int:
        with closing(self.connection_factory()) as conn, conn:
            rows = conn.execute("SELECT DISTINCT file_key FROM workspace_graph_relations WHERE workspace_id=?",
                                (workspace_id,)).fetchall()
            stale = [row["file_key"] for row in rows if row["file_key"] not in active_file_keys]
            if stale:
                conn.executemany("DELETE FROM workspace_graph_relations WHERE workspace_id=? AND file_key=?",
                                 [(workspace_id, key) for key in stale])
            self._cleanup(conn, workspace_id)
            return len(stale)

    def search(self, workspace_id: str, query: str, limit: int = 12) -> list[dict]:
        normalized = normalize_entity_name(query)
        with closing(self.connection_factory()) as conn:
            entities = conn.execute("""SELECT id,canonical_name FROM workspace_graph_entities
                WHERE workspace_id=? ORDER BY length(canonical_name) DESC""", (workspace_id,)).fetchall()
            entity_ids = [int(row["id"]) for row in entities
                          if len(row["canonical_name"]) >= 2 and row["canonical_name"] in normalized]
            if not entity_ids:
                return []
            marks = ",".join("?" for _ in entity_ids)
            rows = conn.execute(f"""SELECT r.*,s.display_name source_name,t.display_name target_name
                FROM workspace_graph_relations r
                JOIN workspace_graph_entities s ON s.id=r.source_entity_id
                JOIN workspace_graph_entities t ON t.id=r.target_entity_id
                WHERE r.workspace_id=? AND (r.source_entity_id IN ({marks}) OR r.target_entity_id IN ({marks}))
                ORDER BY r.confidence DESC,r.id LIMIT ?""",
                (workspace_id, *entity_ids, *entity_ids, limit)).fetchall()
            return [dict(row) for row in rows]


class WorkspaceGraphAdapter:
    """Give the generic GraphExtractor a workspace-bound persistence contract."""
    def __init__(self, repository: WorkspaceGraphRepository, workspace_id: str):
        self.repository = repository
        self.workspace_id = workspace_id

    def replace_document_graph(self, document_id: str, entities, relations, document_name: str):
        self.repository.replace_workspace_file_graph(
            self.workspace_id, str(document_id), list(entities), list(relations), document_name
        )
