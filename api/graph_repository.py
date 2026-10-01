"""SQLite persistence for the local knowledge graph and its provenance."""
from __future__ import annotations

from contextlib import closing
from typing import Iterable

from db_utils import get_db_connection
from entity_normalization import normalize_entity_name


class GraphRepository:
    def __init__(self, connection_factory=get_db_connection):
        self.connection_factory = connection_factory
        self.create_schema()

    def create_schema(self) -> None:
        with closing(self.connection_factory()) as conn, conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS knowledge_entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    canonical_name TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(canonical_name, entity_type)
                );
                CREATE TABLE IF NOT EXISTS knowledge_relations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_entity_id INTEGER NOT NULL REFERENCES knowledge_entities(id),
                    relation_type TEXT NOT NULL,
                    target_entity_id INTEGER NOT NULL REFERENCES knowledge_entities(id),
                    document_id INTEGER NOT NULL,
                    chunk_id TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 1.0 CHECK(confidence >= 0 AND confidence <= 1),
                    evidence TEXT NOT NULL DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(source_entity_id, relation_type, target_entity_id, document_id, chunk_id)
                );
                CREATE INDEX IF NOT EXISTS idx_graph_rel_source ON knowledge_relations(source_entity_id);
                CREATE INDEX IF NOT EXISTS idx_graph_rel_target ON knowledge_relations(target_entity_id);
                CREATE INDEX IF NOT EXISTS idx_graph_rel_document ON knowledge_relations(document_id);
                CREATE INDEX IF NOT EXISTS idx_graph_entity_name ON knowledge_entities(canonical_name);
            """)

    def upsert_entity(self, name: str, entity_type: str) -> int:
        with closing(self.connection_factory()) as conn, conn:
            return self._upsert_entity(conn, name, entity_type)

    @staticmethod
    def _upsert_entity(conn, name: str, entity_type: str) -> int:
        canonical = normalize_entity_name(name)
        if not canonical:
            raise ValueError("Entity name is empty after normalization")
        entity_type = (entity_type or "OTHER").strip().upper()[:80]
        conn.execute(
            "INSERT OR IGNORE INTO knowledge_entities(canonical_name, display_name, entity_type) VALUES(?,?,?)",
            (canonical, name.strip(), entity_type),
        )
        row = conn.execute(
            "SELECT id FROM knowledge_entities WHERE canonical_name=? AND entity_type=?",
            (canonical, entity_type),
        ).fetchone()
        return int(row["id"])

    def add_relation(self, source_id: int, relation_type: str, target_id: int,
                     document_id: int, chunk_id: str, confidence: float = 1.0,
                     evidence: str = "") -> None:
        with closing(self.connection_factory()) as conn, conn:
            self._add_relation(conn, source_id, relation_type, target_id, document_id,
                               chunk_id, confidence, evidence)

    @staticmethod
    def _add_relation(conn, source_id, relation_type, target_id, document_id,
                      chunk_id, confidence=1.0, evidence=""):
        conn.execute("""
            INSERT OR IGNORE INTO knowledge_relations
            (source_entity_id, relation_type, target_entity_id, document_id, chunk_id, confidence, evidence)
            VALUES(?,?,?,?,?,?,?)
        """, (source_id, relation_type.strip().upper()[:120], target_id, document_id,
              chunk_id, max(0.0, min(1.0, confidence)), evidence[:2000]))

    def replace_document_graph(self, document_id: int, entities: Iterable[dict],
                               relations: Iterable[dict], document_name: str) -> None:
        """Replace one document's edges atomically; shared entities remain intact."""
        entities, relations = list(entities), list(relations)
        with closing(self.connection_factory()) as conn, conn:
            conn.execute("DELETE FROM knowledge_relations WHERE document_id=?", (document_id,))
            ids: dict[tuple[str, str], int] = {}
            for item in entities:
                name, kind = item["name"].strip(), item.get("type", "OTHER").upper()
                ids[(normalize_entity_name(name), kind)] = self._upsert_entity(conn, name, kind)
            document_entity = self._upsert_entity(conn, document_name, "DOCUMENT")
            for rel in relations:
                source = self._resolve(rel["source"], rel.get("source_type"), ids)
                target = self._resolve(rel["target"], rel.get("target_type"), ids)
                if source is None or target is None:
                    continue
                self._add_relation(conn, source, rel.get("relation", "RELATED_TO"), target, document_id,
                                   rel["chunk_id"], float(rel.get("confidence", 1.0)), rel.get("evidence", ""))
            chunk_by_entity = {}
            for rel in relations:
                for key in ("source", "target"):
                    canonical = normalize_entity_name(rel.get(key, ""))
                    chunk_by_entity.setdefault(canonical, rel.get("chunk_id", f"document:{document_id}"))
            for (canonical, _), entity_id in ids.items():
                if entity_id != document_entity:
                    self._add_relation(conn, document_entity, "MENTIONS", entity_id, document_id,
                                       chunk_by_entity.get(canonical, f"document:{document_id}"))
        self.cleanup_orphans()

    def _resolve(self, name: str, kind: str | None, ids: dict) -> int | None:
        canonical = normalize_entity_name(name)
        if kind and (canonical, kind.upper()) in ids:
            return ids[(canonical, kind.upper())]
        matches = [value for (candidate, _), value in ids.items() if candidate == canonical]
        return matches[0] if len(set(matches)) == 1 else None

    def delete_document(self, document_id: int, connection=None) -> None:
        if connection is not None:
            connection.execute('DELETE FROM knowledge_relations WHERE document_id=?', (document_id,))
            connection.execute('''DELETE FROM knowledge_entities WHERE id NOT IN (
                SELECT source_entity_id FROM knowledge_relations UNION SELECT target_entity_id FROM knowledge_relations)''')
            return
        with closing(self.connection_factory()) as conn, conn:
            conn.execute("DELETE FROM knowledge_relations WHERE document_id=?", (document_id,))
        self.cleanup_orphans()

    def cleanup_orphans(self) -> None:
        with closing(self.connection_factory()) as conn, conn:
            conn.execute("""DELETE FROM knowledge_entities WHERE id NOT IN (
                SELECT source_entity_id FROM knowledge_relations UNION SELECT target_entity_id FROM knowledge_relations
            )""")

    def find_entities(self, text: str, limit: int = 20) -> list[dict]:
        canonical = normalize_entity_name(text)
        if not canonical:
            return []
        escaped = canonical.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        with closing(self.connection_factory()) as conn, conn:
            rows = conn.execute("""SELECT * FROM knowledge_entities
                WHERE canonical_name=? OR canonical_name LIKE ? ESCAPE '\\'
                ORDER BY canonical_name=? DESC, length(canonical_name), id LIMIT ?""",
                (canonical, f"%{escaped}%", canonical, limit)).fetchall()
        return [dict(row) for row in rows]

    def all_entities(self, limit: int = 1000) -> list[dict]:
        with closing(self.connection_factory()) as conn, conn:
            return [dict(row) for row in conn.execute(
                "SELECT * FROM knowledge_entities ORDER BY length(canonical_name) DESC LIMIT ?", (limit,)).fetchall()]

    def get_entity(self, entity_id: int) -> dict | None:
        with closing(self.connection_factory()) as conn, conn:
            row = conn.execute("SELECT id, display_name, entity_type FROM knowledge_entities WHERE id=?",
                               (entity_id,)).fetchone()
        return dict(row) if row else None

    def relations_for_entities(self, entity_ids: list[int], limit: int) -> list[dict]:
        if not entity_ids:
            return []
        marks = ",".join("?" for _ in entity_ids)
        sql = f"""SELECT r.*, s.display_name source_name, s.entity_type source_type,
            t.display_name target_name, t.entity_type target_type, d.filename document_name
            FROM knowledge_relations r JOIN knowledge_entities s ON s.id=r.source_entity_id
            JOIN knowledge_entities t ON t.id=r.target_entity_id
            LEFT JOIN document_store d ON d.id=r.document_id
            WHERE r.source_entity_id IN ({marks}) OR r.target_entity_id IN ({marks})
            ORDER BY r.confidence DESC, r.id LIMIT ?"""
        with closing(self.connection_factory()) as conn, conn:
            return [dict(row) for row in conn.execute(sql, (*entity_ids, *entity_ids, limit)).fetchall()]

    def relations_for_document(self, document_id: int, limit: int) -> list[dict]:
        sql = """SELECT r.*, s.display_name source_name, s.entity_type source_type,
            t.display_name target_name, t.entity_type target_type, d.filename document_name
            FROM knowledge_relations r JOIN knowledge_entities s ON s.id=r.source_entity_id
            JOIN knowledge_entities t ON t.id=r.target_entity_id
            LEFT JOIN document_store d ON d.id=r.document_id
            WHERE r.document_id=? ORDER BY r.confidence DESC, r.id LIMIT ?"""
        with closing(self.connection_factory()) as conn, conn:
            return [dict(row) for row in conn.execute(sql, (document_id, limit)).fetchall()]
