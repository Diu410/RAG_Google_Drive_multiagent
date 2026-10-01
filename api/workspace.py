"""Read-only workspace persistence and filesystem access."""
from __future__ import annotations

import json
import asyncio
import hashlib
import difflib
import os
import re
import subprocess
import sys
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from assistant_models import AssistantError, ToolResult
from db_utils import get_db_connection
from text_normalization import lexical_key
from workspace_planner import WorkspaceOperation, WorkspacePlan, WorkspacePlanner
from workspace_router import WorkspaceRouteDecision, WorkspaceToolRouter
from workspace_document_reader import (PLAIN_TEXT_EXTENSIONS, SUPPORTED_WORKSPACE_EXTENSIONS,
                                       WorkspaceDocumentReader)
from file_reference_resolver import FileReferenceResolver
from response_budget import ResponseBudgetPlanner
from command_intent import requests_file_open, command_text, requests_unsupported_file_change


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Workspace(BaseModel):
    id: str
    name: str
    root_path: str
    created_at: str
    updated_at: str
    status: str = "ACTIVE"
    metadata: dict = Field(default_factory=dict)


class WorkspaceEntry(BaseModel):
    name: str
    relative_path: str
    kind: str
    size: int | None = None
    modified_at: str | None = None
    navigable: bool = True
    readable: bool = False


class WorkspaceListing(BaseModel):
    workspace: Workspace
    path: str = ""
    parent: str | None = None
    entries: list[WorkspaceEntry]
    truncated: bool = False


class WorkspaceFileRecord(BaseModel):
    workspace_id: str
    relative_path: str
    name: str
    extension: str
    size: int
    modified_at: str
    modified_ns: int
    filesystem_created_at: str = ""
    filesystem_created_ns: int = 0
    content_hash: str | None = None
    readable: bool = False
    discovered_at: str
    indexed_at: str
    vector_hash: str | None = None
    vector_status: str = "PENDING"
    vector_error: str | None = None
    graph_hash: str | None = None
    graph_status: str = "PENDING"
    graph_error: str | None = None


class WorkspaceSyncResult(BaseModel):
    workspace_id: str
    scanned: int
    created: int
    updated: int
    deleted: int
    unchanged: int
    truncated: bool = False
    errors: list[str] = Field(default_factory=list)


class WorkspaceRepository:
    def __init__(self, connection_factory=get_db_connection):
        self.connection_factory = connection_factory
        self.migrate()

    def migrate(self):
        with closing(self.connection_factory()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("""CREATE TABLE IF NOT EXISTS assistant_workspaces (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, root_path TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL, status TEXT NOT NULL,
                metadata TEXT NOT NULL, is_active INTEGER NOT NULL DEFAULT 0)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS assistant_workspace_files (
                workspace_id TEXT NOT NULL, relative_path TEXT NOT NULL, name TEXT NOT NULL,
                extension TEXT NOT NULL, size INTEGER NOT NULL, modified_at TEXT NOT NULL,
                modified_ns INTEGER NOT NULL, filesystem_created_at TEXT NOT NULL,
                filesystem_created_ns INTEGER NOT NULL, content_hash TEXT, readable INTEGER NOT NULL,
                discovered_at TEXT NOT NULL, indexed_at TEXT NOT NULL,
                vector_hash TEXT, vector_status TEXT NOT NULL DEFAULT 'PENDING', vector_error TEXT,
                graph_hash TEXT, graph_status TEXT NOT NULL DEFAULT 'PENDING', graph_error TEXT,
                PRIMARY KEY(workspace_id, relative_path),
                FOREIGN KEY(workspace_id) REFERENCES assistant_workspaces(id) ON DELETE CASCADE)""")
            file_columns = {row[1] for row in conn.execute("PRAGMA table_info(assistant_workspace_files)")}
            if "filesystem_created_at" not in file_columns:
                conn.execute("ALTER TABLE assistant_workspace_files ADD COLUMN filesystem_created_at TEXT NOT NULL DEFAULT ''")
            if "filesystem_created_ns" not in file_columns:
                conn.execute("ALTER TABLE assistant_workspace_files ADD COLUMN filesystem_created_ns INTEGER NOT NULL DEFAULT 0")
            if "vector_hash" not in file_columns:
                conn.execute("ALTER TABLE assistant_workspace_files ADD COLUMN vector_hash TEXT")
            if "vector_status" not in file_columns:
                conn.execute("ALTER TABLE assistant_workspace_files ADD COLUMN vector_status TEXT NOT NULL DEFAULT 'PENDING'")
            if "vector_error" not in file_columns:
                conn.execute("ALTER TABLE assistant_workspace_files ADD COLUMN vector_error TEXT")
            if "graph_hash" not in file_columns:
                conn.execute("ALTER TABLE assistant_workspace_files ADD COLUMN graph_hash TEXT")
            if "graph_status" not in file_columns:
                conn.execute("ALTER TABLE assistant_workspace_files ADD COLUMN graph_status TEXT NOT NULL DEFAULT 'PENDING'")
            if "graph_error" not in file_columns:
                conn.execute("ALTER TABLE assistant_workspace_files ADD COLUMN graph_error TEXT")
            conn.execute("CREATE INDEX IF NOT EXISTS assistant_workspace_files_name "
                         "ON assistant_workspace_files(workspace_id,name)")
            columns = {row[1] for row in conn.execute("PRAGMA table_info(assistant_sessions)")}
            if columns and "workspace_id" not in columns:
                conn.execute("ALTER TABLE assistant_sessions ADD COLUMN workspace_id TEXT")
                columns.add("workspace_id")
            if columns and "workspace_path" not in columns:
                conn.execute("ALTER TABLE assistant_sessions ADD COLUMN workspace_path TEXT NOT NULL DEFAULT ''")
                columns.add("workspace_path")
            if columns and "workspace_file" not in columns:
                conn.execute("ALTER TABLE assistant_sessions ADD COLUMN workspace_file TEXT NOT NULL DEFAULT ''")
                columns.add("workspace_file")
            if "workspace_id" in columns:
                conn.execute("CREATE INDEX IF NOT EXISTS assistant_sessions_workspace ON assistant_sessions(workspace_id)")

    @staticmethod
    def _decode(row) -> Workspace:
        if row is None:
            raise LookupError("Workspace not found")
        data = dict(row)
        data.pop("is_active", None)
        data["metadata"] = json.loads(data["metadata"])
        return Workspace(**data)

    def select(self, root_path: str) -> Workspace:
        root = Path(root_path).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise ValueError("Выбранный путь не является папкой")
        now = _now()
        with closing(self.connection_factory()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM assistant_workspaces WHERE root_path=?", (str(root),)).fetchone()
            if row:
                workspace_id = row["id"]
                conn.execute("UPDATE assistant_workspaces SET name=?,updated_at=?,status='ACTIVE' WHERE id=?",
                             (root.name or str(root), now, workspace_id))
            else:
                workspace_id = str(uuid.uuid4())
                conn.execute("INSERT INTO assistant_workspaces VALUES(?,?,?,?,?,?,?,0)",
                             (workspace_id, root.name or str(root), str(root), now, now, "ACTIVE", "{}"))
            conn.execute("UPDATE assistant_workspaces SET is_active=CASE WHEN id=? THEN 1 ELSE 0 END", (workspace_id,))
            return self._decode(conn.execute("SELECT * FROM assistant_workspaces WHERE id=?", (workspace_id,)).fetchone())

    def get(self, workspace_id: str) -> Workspace:
        with closing(self.connection_factory()) as conn:
            return self._decode(conn.execute("SELECT * FROM assistant_workspaces WHERE id=?", (workspace_id,)).fetchone())

    def active(self) -> Workspace | None:
        with closing(self.connection_factory()) as conn:
            row = conn.execute("SELECT * FROM assistant_workspaces WHERE is_active=1 LIMIT 1").fetchone()
            return self._decode(row) if row else None

    def bind_session(self, session_id: str, workspace_id: str, current_directory: str | None = None,
                     active_file: str | None = None):
        with closing(self.connection_factory()) as conn, conn:
            if current_directory is None and active_file is None:
                cursor = conn.execute("UPDATE assistant_sessions SET workspace_id=?,updated_at=? WHERE id=?",
                                      (workspace_id, _now(), session_id))
            elif active_file is None:
                cursor = conn.execute(
                    "UPDATE assistant_sessions SET workspace_id=?,workspace_path=?,updated_at=? WHERE id=?",
                    (workspace_id, current_directory.strip("/"), _now(), session_id),
                )
            elif current_directory is None:
                cursor = conn.execute(
                    "UPDATE assistant_sessions SET workspace_id=?,workspace_file=?,updated_at=? WHERE id=?",
                    (workspace_id, active_file.strip("/"), _now(), session_id),
                )
            else:
                cursor = conn.execute(
                    "UPDATE assistant_sessions SET workspace_id=?,workspace_path=?,workspace_file=?,updated_at=? "
                    "WHERE id=?",
                    (workspace_id, current_directory.strip("/"), active_file.strip("/"), _now(), session_id),
                )
            if cursor.rowcount != 1:
                raise LookupError("Session not found")

    def session_directory(self, session_id: str, workspace_id: str) -> str:
        with closing(self.connection_factory()) as conn:
            row = conn.execute(
                "SELECT workspace_path FROM assistant_sessions WHERE id=? AND workspace_id=?",
                (session_id, workspace_id),
            ).fetchone()
            return str(row["workspace_path"] or "") if row else ""

    def session_file(self, session_id: str, workspace_id: str) -> str:
        with closing(self.connection_factory()) as conn:
            row = conn.execute(
                "SELECT workspace_file FROM assistant_sessions WHERE id=? AND workspace_id=?",
                (session_id, workspace_id),
            ).fetchone()
            return str(row["workspace_file"] or "") if row else ""

    def indexed_files(self, workspace_id: str, relative_directory: str = "") -> list[WorkspaceFileRecord]:
        prefix = relative_directory.strip("/")
        with closing(self.connection_factory()) as conn:
            if prefix:
                rows = conn.execute(
                    "SELECT * FROM assistant_workspace_files WHERE workspace_id=? "
                    "AND relative_path LIKE ? ESCAPE '\\' ORDER BY relative_path COLLATE NOCASE",
                    (workspace_id, prefix.replace("%", "\\%").replace("_", "\\_") + "/%"),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM assistant_workspace_files WHERE workspace_id=? "
                    "ORDER BY relative_path COLLATE NOCASE", (workspace_id,),
                ).fetchall()
        return [WorkspaceFileRecord(**{**dict(row), "readable": bool(row["readable"])}) for row in rows]

    def update_vector_state(self, workspace_id: str, relative_path: str, *, content_hash: str | None,
                            status: str, error: str | None = None) -> None:
        with closing(self.connection_factory()) as conn, conn:
            conn.execute("""UPDATE assistant_workspace_files
                SET vector_hash=?,vector_status=?,vector_error=?
                WHERE workspace_id=? AND relative_path=?""",
                (content_hash, status, error[:1000] if error else None, workspace_id, relative_path))

    def update_graph_state(self, workspace_id: str, relative_path: str, *, content_hash: str | None,
                           status: str, error: str | None = None) -> None:
        with closing(self.connection_factory()) as conn, conn:
            conn.execute("""UPDATE assistant_workspace_files
                SET graph_hash=?,graph_status=?,graph_error=?
                WHERE workspace_id=? AND relative_path=?""",
                (content_hash, status, error[:1000] if error else None, workspace_id, relative_path))


class FilesystemProvider:
    """Lists entries without allowing access outside the selected root."""
    def __init__(self, repository: WorkspaceRepository, max_entries: int = 500, opener=None,
                 document_reader: WorkspaceDocumentReader | None = None,
                 file_reference_resolver: FileReferenceResolver | None = None):
        self.repository = repository
        self.max_entries = max_entries
        self.opener = opener or self._open_with_system
        self.readable_extensions = SUPPORTED_WORKSPACE_EXTENSIONS
        self.max_text_bytes = 1024 * 1024
        self.max_document_bytes = 50 * 1024 * 1024
        self.document_reader = document_reader or WorkspaceDocumentReader()
        self.file_reference_resolver = file_reference_resolver or FileReferenceResolver()
        self.ignored_directory_names = {
            ".git", ".idea", ".obsidian", ".trash", "node_modules", "target",
            "__pycache__", ".venv", "venv", "dist", "build",
        }

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.blake2b(digest_size=20)
        with path.open("rb") as stream:
            while chunk := stream.read(256 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _open_with_system(path: str) -> None:
        if sys.platform == "win32":
            os.startfile(path)  # type: ignore[attr-defined]
            return
        command = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.Popen([command, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def sync_index(self, workspace_id: str, max_files: int = 20_000) -> WorkspaceSyncResult:
        """Synchronize persistent metadata, hashing only new or changed readable files."""
        workspace = self.repository.get(workspace_id)
        root = Path(workspace.root_path).resolve(strict=True)
        existing = {item.relative_path: item for item in self.repository.indexed_files(workspace_id)}
        seen: set[str] = set()
        records: list[dict] = []
        created = updated = unchanged = 0
        truncated = False
        errors = []
        now = _now()
        def failed_walk(error):
            errors.append(str(error))

        for directory, names, files in os.walk(root, followlinks=False, onerror=failed_walk):
            names[:] = self._visible_directories(directory, names)
            for name in files:
                path = Path(directory) / name
                if path.is_symlink():
                    continue
                if len(seen) >= max_files:
                    truncated = True
                    break
                try:
                    stat = path.stat()
                except OSError as exc:
                    errors.append(str(exc))
                    continue
                relative = path.relative_to(root).as_posix()
                seen.add(relative)
                old = existing.get(relative)
                size_limit = (self.max_text_bytes if path.suffix.casefold() in PLAIN_TEXT_EXTENSIONS
                              else self.max_document_bytes)
                readable = path.suffix.casefold() in self.readable_extensions and stat.st_size <= size_limit
                changed = old is None or old.size != stat.st_size or old.modified_ns != stat.st_mtime_ns
                content_hash = old.content_hash if old else None
                if changed and readable:
                    try:
                        content_hash = self._hash_file(path)
                    except OSError:
                        content_hash = None
                elif not readable:
                    content_hash = None
                if old is None:
                    created += 1
                elif changed:
                    updated += 1
                else:
                    unchanged += 1
                records.append({
                    "workspace_id": workspace_id, "relative_path": relative, "name": name,
                    "extension": path.suffix.casefold(), "size": stat.st_size,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                    "modified_ns": stat.st_mtime_ns, "content_hash": content_hash,
                    "filesystem_created_at": datetime.fromtimestamp(stat.st_ctime, timezone.utc).isoformat(),
                    "filesystem_created_ns": stat.st_ctime_ns,
                    "readable": int(readable), "discovered_at": old.discovered_at if old else now,
                    "indexed_at": now,
                })
            if truncated:
                break
        # A missing observation is evidence of deletion only after a complete
        # scan. Preserve previous records when permissions or I/O fail.
        deleted_paths = set(existing) - seen if not truncated and not errors else set()
        with closing(self.repository.connection_factory()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.executemany("""INSERT INTO assistant_workspace_files
                (workspace_id,relative_path,name,extension,size,modified_at,modified_ns,content_hash,
                 filesystem_created_at,filesystem_created_ns,readable,discovered_at,indexed_at) VALUES
                (:workspace_id,:relative_path,:name,:extension,:size,:modified_at,:modified_ns,:content_hash,
                 :filesystem_created_at,:filesystem_created_ns,
                 :readable,:discovered_at,:indexed_at)
                ON CONFLICT(workspace_id,relative_path) DO UPDATE SET
                 name=excluded.name,extension=excluded.extension,size=excluded.size,
                 modified_at=excluded.modified_at,modified_ns=excluded.modified_ns,
                 filesystem_created_at=excluded.filesystem_created_at,
                 filesystem_created_ns=excluded.filesystem_created_ns,
                 content_hash=excluded.content_hash,readable=excluded.readable,indexed_at=excluded.indexed_at""",
                records,
            )
            if deleted_paths:
                conn.executemany(
                    "DELETE FROM assistant_workspace_files WHERE workspace_id=? AND relative_path=?",
                    [(workspace_id, path) for path in deleted_paths],
                )
            conn.execute("UPDATE assistant_workspaces SET updated_at=? WHERE id=?", (now, workspace_id))
        return WorkspaceSyncResult(
            workspace_id=workspace_id, scanned=len(seen), created=created, updated=updated,
            deleted=len(deleted_paths), unchanged=unchanged, truncated=truncated, errors=errors,
        )

    def version_files(self, workspace_id: str, query: str) -> list[tuple[WorkspaceFileRecord, tuple[int, ...]]]:
        """Return one matching document family ordered newest first."""
        self.sync_index(workspace_id)
        terms = {_term_key(term) for term in re.findall(r"[\w-]+", query, re.UNICODE)
                 if len(term) >= 3 and term.casefold() not in _QUERY_STOPWORDS | {
                     "последний", "последняя", "последнюю", "версия", "версию", "файл", "найди",
                 }}
        candidates = []
        version_re = re.compile(r"(?:^|[\s_.-])(?:v|ver(?:sion)?|версия)?\s*(\d+(?:\.\d+)*)", re.IGNORECASE)
        for item in self.repository.indexed_files(workspace_id):
            name_terms = {_term_key(term) for term in re.findall(r"[\w-]+", Path(item.name).stem, re.UNICODE)}
            score = len(terms & name_terms)
            if terms and not score:
                continue
            versions = version_re.findall(Path(item.name).stem)
            version = tuple(int(part) for part in versions[-1].split(".")) if versions else ()
            candidates.append((score, bool(version), version, item.modified_ns, item))
        if not candidates:
            raise AssistantError("Не найден файл, подходящий под описание.", 404)
        candidates.sort(key=lambda value: (value[0], value[1], value[2], value[3]), reverse=True)
        best_score = candidates[0][0]
        return [(item, version) for score, _, version, _, item in candidates if score == best_score]

    def latest_file(self, workspace_id: str, query: str) -> tuple[WorkspaceFileRecord, str]:
        """Choose the best matching file, preferring explicit version numbers then mtime."""
        if re.search(r"\b(?:измен[её]н\w*|изменени\w*|редактирован\w*)\b", query, re.IGNORECASE):
            self.sync_index(workspace_id)
            scope = self.find_directory(workspace_id, query, require_marker=False)
            files = [item for item in self.repository.indexed_files(workspace_id)
                     if item.readable and (not scope or item.relative_path.startswith(scope + "/"))]
            if not files:
                location = f" в папке `{scope}`" if scope else " в текущей директории"
                raise AssistantError(f"Не найдено поддерживаемых документов{location}.", 404)
            selected = max(files, key=lambda item: (item.modified_ns, item.relative_path.casefold()))
            location = f" внутри `{scope}`" if scope else ""
            return selected, (f"выбран документ{location} с самой поздней датой изменения "
                              f"({selected.modified_at})")
        if re.search(r"\b(?:создан\w*|нов\w*\s+по\s+создани\w*)\b", query, re.IGNORECASE):
            self.sync_index(workspace_id)
            ignored_prefixes = ("послед", "создан", "нов", "данн", "текущ", "директор", "папк", "документ", "файл", "найд")
            terms = {
                _term_key(term) for term in re.findall(r"[\w-]+", query, re.UNICODE)
                if len(term) >= 3 and not term.casefold().startswith(ignored_prefixes)
            }
            ranked = []
            for item in self.repository.indexed_files(workspace_id):
                if not item.readable:
                    continue
                name_terms = {_term_key(term) for term in re.findall(r"[\w-]+", Path(item.name).stem, re.UNICODE)}
                score = len(terms & name_terms)
                if terms and not score:
                    continue
                ranked.append((score, item.filesystem_created_ns, item.modified_ns, item))
            if not ranked:
                raise AssistantError("В текущей директории не найдено поддерживаемых документов.", 404)
            ranked.sort(key=lambda value: (value[0], value[1], value[2]), reverse=True)
            selected = ranked[0][3]
            reason = ("выбран поддерживаемый документ с самым поздним временем создания "
                      f"({selected.filesystem_created_at})")
            return selected, reason
        selected, version = self.version_files(workspace_id, query)[0]
        has_version = bool(version)
        if has_version:
            reason = f"выбрана наибольшая явная версия {'.'.join(map(str, version))}"
        else:
            reason = "явных номеров версии нет; выбран файл с самой поздней датой изменения"
        return selected, reason

    def compare_latest(self, workspace_id: str, query: str, max_diff_lines: int = 240) -> dict:
        versions = self.version_files(workspace_id, query)
        if len(versions) < 2:
            raise AssistantError("Для сравнения нужна как минимум ещё одна версия файла.", 409)
        current, _ = versions[0]
        previous, _ = versions[1]
        _, _, current_text = self.read_text_file(workspace_id, current.relative_path)
        _, _, previous_text = self.read_text_file(workspace_id, previous.relative_path)
        diff_lines = list(difflib.unified_diff(
            previous_text.splitlines(), current_text.splitlines(),
            fromfile=previous.relative_path, tofile=current.relative_path, lineterm="",
        ))
        truncated = len(diff_lines) > max_diff_lines
        shown = diff_lines[:max_diff_lines]
        additions = sum(1 for line in diff_lines if line.startswith("+") and not line.startswith("+++"))
        deletions = sum(1 for line in diff_lines if line.startswith("-") and not line.startswith("---"))
        return {
            "current": current, "previous": previous, "diff": "\n".join(shown),
            "additions": additions, "deletions": deletions, "truncated": truncated,
        }

    def open_file(self, workspace_id: str, relative_path: str) -> str:
        workspace = self.repository.get(workspace_id)
        root = Path(workspace.root_path).resolve(strict=True)
        path = self._resolve(root, relative_path)
        if not path.is_file():
            raise FileNotFoundError("Файл не найден")
        self.opener(str(path))
        return path.relative_to(root).as_posix()

    def _visible_directories(self, directory: str, names: list[str], include_hidden: bool = False) -> list[str]:
        return [name for name in names
                if (include_hidden or not name.startswith("."))
                and name.casefold() not in self.ignored_directory_names
                and not (Path(directory) / name).is_symlink()]

    @staticmethod
    def _resolve(root: Path, relative_path: str) -> Path:
        relative = Path(relative_path or ".")
        if relative.is_absolute():
            raise PermissionError("Разрешён только путь внутри workspace")
        candidate = root.joinpath(relative)
        # A symlink is visible in its parent, but cannot be entered. Checking
        # every component prevents a link from escaping and re-entering root.
        current = root
        for part in relative.parts:
            if part in ("", "."):
                continue
            if part == "..":
                raise PermissionError("Выход за пределы workspace запрещён")
            current = current / part
            if current.is_symlink():
                raise PermissionError("Переход по символическим ссылкам запрещён")
        resolved = candidate.resolve(strict=True)
        if os.path.commonpath((str(root), str(resolved))) != str(root):
            raise PermissionError("Выход за пределы workspace запрещён")
        return resolved

    def list_directory(self, workspace_id: str, relative_path: str = "") -> WorkspaceListing:
        workspace = self.repository.get(workspace_id)
        root = Path(workspace.root_path).resolve(strict=True)
        directory = self._resolve(root, relative_path)
        if not directory.is_dir():
            raise NotADirectoryError("Путь не является папкой")
        items = []
        for child in directory.iterdir():
            is_link = child.is_symlink()
            try:
                stat = child.stat(follow_symlinks=False)
                modified = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
            except OSError:
                stat, modified = None, None
            kind = "symlink" if is_link else "directory" if child.is_dir() else "file"
            relative = child.relative_to(root).as_posix()
            items.append(WorkspaceEntry(
                name=child.name, relative_path=relative, kind=kind,
                size=stat.st_size if stat and kind == "file" else None,
                modified_at=modified, navigable=kind == "directory",
                readable=kind == "file" and child.suffix.casefold() in self.readable_extensions,
            ))
        items.sort(key=lambda item: (item.kind != "directory", item.name.casefold()))
        path = directory.relative_to(root).as_posix()
        path = "" if path == "." else path
        parent = None if not path else Path(path).parent.as_posix()
        if parent == ".":
            parent = ""
        return WorkspaceListing(workspace=workspace, path=path, parent=parent,
                                entries=items[:self.max_entries], truncated=len(items) > self.max_entries)

    def read_text_file(self, workspace_id: str, relative_path: str) -> tuple[Workspace, str, str]:
        workspace = self.repository.get(workspace_id)
        root = Path(workspace.root_path).resolve(strict=True)
        path = self._resolve(root, relative_path)
        if not path.is_file():
            raise FileNotFoundError("Файл не найден")
        if path.suffix.casefold() not in self.readable_extensions:
            raise ValueError("Формат файла не поддерживается для чтения")
        size = path.stat().st_size
        limit = self.max_text_bytes if path.suffix.casefold() in PLAIN_TEXT_EXTENSIONS else self.max_document_bytes
        if size > limit:
            raise ValueError(f"Файл превышает допустимый размер {limit // (1024 * 1024)} МБ")
        content = self.document_reader.read(path)
        return workspace, path.relative_to(root).as_posix(), content

    def find_text_file(self, workspace_id: str, question: str, model: str | None = None,
                       current_directory: str = "") -> str:
        self.sync_index(workspace_id)
        candidates = [item.relative_path for item in self.repository.indexed_files(workspace_id)
                      if item.readable][:2000]
        workspace = self.repository.get(workspace_id)
        root = Path(workspace.root_path).resolve(strict=True)
        try:
            explicit_directory = self.find_directory(workspace_id, question, require_marker=False)
        except AssistantError:
            # A full filename may contain words equal to several directory
            # names. Exact file resolution below has priority in that case.
            explicit_directory = ""
        preferred_directory = explicit_directory or current_directory.strip("/")
        if preferred_directory:
            try:
                directory = self._resolve(root, preferred_directory)
                direct_candidates = [
                    path.relative_to(root).as_posix() for path in directory.iterdir()
                    if path.is_file() and not path.is_symlink()
                    and path.suffix.casefold() in self.readable_extensions
                ]
                candidates = list(dict.fromkeys(direct_candidates + candidates))[:2000]
            except (OSError, PermissionError, FileNotFoundError):
                pass
        lowered = question.casefold().replace("\\", "/")
        explicit = re.search(
            r"(?:\b(?:выведи|покажи)\s+содержим\w*|\b(?:прочитай|прочти|открой)\b)"
            r"\s+(?:файл\w*\s+)?(.+?)\s*[?.!]*$",
            question, re.IGNORECASE,
        )
        if explicit:
            requested_label = explicit.group(1).strip()
            if explicit_directory:
                # The directory is a scope, not part of the filename:
                # "открой app.json в папке .obsidian" must resolve app.json.
                requested_label = re.sub(
                    r"\s+в\s+(?:папк|директор)\w*\s+.+$", "", requested_label,
                    flags=re.IGNORECASE,
                ).strip()
            requested = requested_label.casefold().replace("\\", "/")
            exact = [path for path in candidates if requested in {
                path.casefold(), Path(path).name.casefold(), Path(path).stem.casefold(),
            }]
            if len(exact) == 1:
                return exact[0]
            if len(exact) > 1:
                options = "\n".join(f"- `{path}`" for path in exact[:20])
                raise AssistantError(
                    "Найдено несколько файлов с таким именем. Выберите относительный путь:\n" + options, 409
                )
            requested_terms = {_term_key(term) for term in re.findall(r"[\w-]+", requested, re.UNICODE)}
            requested_numbers = re.findall(r"\d+", requested)
            eligible = [path for path in candidates
                        if re.findall(r"\d+", Path(path).stem.casefold()) == requested_numbers]
            directory_scope = explicit_directory or current_directory.strip("/")
            if directory_scope:
                eligible = [path for path in eligible if path.startswith(directory_scope + "/")]
            initial = re.search(
                r"начина\w*\s+на\s+букв\w*\s*[«\"']?([\w])", requested_label, re.IGNORECASE
            )
            if initial:
                letter = initial.group(1).casefold().replace("ё", "е")
                eligible = [path for path in eligible
                            if Path(path).stem.casefold().replace("ё", "е").startswith(letter)]
                if len(eligible) == 1:
                    return eligible[0]
            similar = sorted(
                eligible,
                key=lambda path: (
                    -difflib.SequenceMatcher(None, requested, Path(path).stem.casefold()).ratio(),
                    -len(requested_terms & {_term_key(term) for term in re.findall(
                        r"[\w-]+", path, re.UNICODE
                    )}),
                    path.casefold(),
                ),
            )
            lexical_matches = [path for path in similar if requested_terms & {
                _term_key(term) for term in re.findall(r"[\w-]+", path, re.UNICODE)
            }]
            if len(lexical_matches) == 1:
                candidate_terms = {_term_key(term) for term in re.findall(
                    r"[\w-]+", Path(lexical_matches[0]).stem, re.UNICODE
                )}
                meaningful = {term for term in requested_terms if len(term) >= 3}
                if len(meaningful & candidate_terms) >= 2:
                    return lexical_matches[0]
            similar = (lexical_matches or similar)[:20]
            if model and similar:
                try:
                    decision = self.file_reference_resolver.resolve(requested_label, similar, model)
                    if (decision.confidence >= 0.75 and decision.selected_path in similar
                            and re.findall(r"\d+", Path(decision.selected_path).stem.casefold()) == requested_numbers):
                        return decision.selected_path
                except Exception:
                    pass
            suffix = ("\nПохожие файлы:\n" + "\n".join(f"- `{path}`" for path in similar)) if similar else ""
            raise AssistantError(f"Файл `{requested_label}` не найден.{suffix}", 404)
        # Resolve from most to least specific. A shorter stem such as
        # "Без названия" must not make "Без названия 4" ambiguous.
        path_matches = [path for path in candidates if "/" in path and path.casefold() in lowered]
        mentioned = path_matches
        if not mentioned:
            mentioned = [path for path in candidates if Path(path).name.casefold() in lowered]
        if not mentioned:
            stem_matches = [path for path in candidates if re.search(
                rf"(?<![\w-]){re.escape(Path(path).stem.casefold())}(?![\w-])", lowered
            )]
            if stem_matches:
                longest = max(len(Path(path).stem) for path in stem_matches)
                mentioned = [path for path in stem_matches if len(Path(path).stem) == longest]
        if len(mentioned) == 1:
            return mentioned[0]
        if not mentioned and len(candidates) == 1:
            return candidates[0]
        if len(mentioned) > 1:
            options = "\n".join(f"- `{path}`" for path in mentioned[:20])
            raise AssistantError(
                "Найдено несколько файлов с таким именем. Выберите вариант или укажите относительный путь:\n"
                + options, 409,
            )
        examples = ", ".join(candidates[:5])
        suffix = f" Доступны: {examples}." if examples else " Поддерживаемых текстовых файлов нет."
        raise AssistantError("Укажите имя MD или текстового файла." + suffix, 409)

    def find_directory(self, workspace_id: str, question: str, require_marker: bool = True) -> str:
        """Resolve a directory explicitly mentioned in a natural-language query."""
        if (require_marker and not re.search(
                r"\b(?:папк|директор|workspace|воркспейс)\w*\b", question, re.IGNORECASE)):
            return ""
        workspace = self.repository.get(workspace_id)
        root = Path(workspace.root_path).resolve(strict=True)
        lowered = question.casefold().replace("\\", "/")
        candidates: list[str] = []
        for directory, names, _ in os.walk(root, followlinks=False):
            discoverable = [name for name in names if not (Path(directory) / name).is_symlink()]
            for name in discoverable:
                relative = (Path(directory) / name).relative_to(root).as_posix()
                candidates.append(relative)
                if len(candidates) >= 2000:
                    break
            names[:] = self._visible_directories(directory, names)
            if len(candidates) >= 2000:
                break
        mentioned = [path for path in candidates if any(
            re.search(rf'(?<![\w-]){re.escape(label.casefold())}(?![\w-])', lowered)
            for label in (path, Path(path).name))]
        if not mentioned:
            question_terms = {_term_key(term) for term in re.findall(r"[\w-]+", question, re.UNICODE)}
            mentioned = [path for path in candidates if (
                (name_terms := {_term_key(term) for term in re.findall(
                    r"[\w-]+", Path(path).name, re.UNICODE
                )}) and name_terms <= question_terms
            )]
        if len(mentioned) == 1:
            return mentioned[0]
        if len(mentioned) > 1:
            shallowest_depth = min(path.count("/") for path in mentioned)
            shallowest = [path for path in mentioned if path.count("/") == shallowest_depth]
            if len(shallowest) == 1:
                return shallowest[0]
        # A full relative path is more specific than its named parents.
        most_specific = [path for path in mentioned
                         if not any(other != path and other.startswith(path + "/") for other in mentioned)]
        if len(most_specific) == 1:
            return most_specific[0]
        if len(mentioned) > 1:
            options = "\n".join(f"- `{path}`" for path in mentioned[:20])
            raise AssistantError(
                "Найдено несколько папок с таким именем. Выберите вариант или укажите относительный путь:\n"
                + options, 409,
            )
        return ""

    def list_tree(self, workspace_id: str, relative_path: str = "") -> WorkspaceListing:
        """Return a bounded recursive listing rooted at a workspace directory."""
        workspace = self.repository.get(workspace_id)
        root = Path(workspace.root_path).resolve(strict=True)
        base = self._resolve(root, relative_path)
        if not base.is_dir():
            raise NotADirectoryError("Путь не является папкой")
        items: list[WorkspaceEntry] = []
        truncated = False
        for directory, names, files in os.walk(base, followlinks=False):
            names[:] = self._visible_directories(
                directory, names, include_hidden=bool(relative_path and Path(relative_path).name.startswith("."))
            )
            for name, kind in [(name, "directory") for name in names] + [(name, "file") for name in files]:
                path = Path(directory) / name
                if path.is_symlink():
                    continue
                if len(items) >= self.max_entries:
                    truncated = True
                    break
                try:
                    stat = path.stat()
                    modified = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
                except OSError:
                    stat, modified = None, None
                items.append(WorkspaceEntry(
                    name=name, relative_path=path.relative_to(root).as_posix(), kind=kind,
                    size=stat.st_size if stat and kind == "file" else None,
                    modified_at=modified, navigable=kind == "directory",
                    readable=kind == "file" and path.suffix.casefold() in self.readable_extensions,
                ))
            if truncated:
                break
        items.sort(key=lambda item: (item.relative_path.casefold(), item.kind != "directory"))
        path = base.relative_to(root).as_posix()
        path = "" if path == "." else path
        parent = None if not path else Path(path).parent.as_posix()
        return WorkspaceListing(workspace=workspace, path=path, parent="" if parent == "." else parent,
                                entries=items, truncated=truncated)

    def searchable_files(self, workspace_id: str, max_files: int = 500,
                         max_total_bytes: int = 10 * 1024 * 1024, relative_directory: str = "",
                         coverage: dict | None = None, synchronize: bool = True):
        """Yield indexed current text files with a bounded amount of filesystem I/O."""
        workspace = self.repository.get(workspace_id)
        root = Path(workspace.root_path).resolve(strict=True)
        base = self._resolve(root, relative_directory)
        if not base.is_dir():
            raise NotADirectoryError("Путь не является папкой")
        if synchronize:
            self.sync_index(workspace_id)
        coverage = coverage if coverage is not None else {}
        eligible = [item for item in self.repository.indexed_files(workspace_id, relative_directory) if item.readable]
        coverage.update(available=len(eligible), inspected=0, omitted=0, failed=0)
        used = 0
        found = 0
        for item in eligible:
            if found >= max_files:
                coverage['omitted'] += 1
                continue
            try:
                _, _, content = self.read_text_file(workspace_id, item.relative_path)
            except (OSError, UnicodeError, ValueError, PermissionError):
                coverage['failed'] += 1
                continue
            content_bytes = len(content.encode("utf-8"))
            if used + content_bytes > max_total_bytes:
                coverage['omitted'] += 1
                continue
            coverage['inspected'] += 1
            yield item.relative_path, content
            used += content_bytes
            found += 1


_LIST_RE = re.compile(
    r"\b(?:покажи|перечисли|список)\b.*\b(?:файл|папк|директор|workspace|воркспейс)\w*\b|"
    r"\bкакие\s+(?:здесь\s+|тут\s+)?(?:файлы|папки)\b|"
    r"\bчто\b.*\b(?:папк|директор|workspace|воркспейс)\w*\b|"
    r"\b(?:открой|покажи)\w*\s+(?:папк|директор)\w*\b.*\b(?:внутри|содержим|что)\b|"
    r"\b(?:list|show)\b.*\b(?:files|folders|directory|workspace)\b|"
    r"\bwhat\b.*\b(?:directory|folder|workspace)\b",
    re.IGNORECASE,
)
_READ_RE = re.compile(
    r"\b(?:read|open|прочитай|прочти|прочесть|открой|посмотри|загляни|разбери|проанализируй|изучи|покажи\s+содержимое|что\s+(?:написано|находится)|содержимое)\b.*"
    r"(?:\.(?:md|markdown|txt|pdf|docx|html?|jsonl?|ya?ml|toml|csv|tsv|log|xml|ini|cfg|conf|rtf|"
    r"py|jsx?|tsx?|css|scss|sql|sh|bat|cmd|ps1)\b|"
    r"текстов\w*\s+файл|md\s+файл)",
    re.IGNORECASE,
)
_NAMED_FILE_RE = re.compile(
    r"[^\\/:*?\"<>|\r\n]+\.(?:md|markdown|txt|pdf|docx|html?|jsonl?|ya?ml|toml|csv|tsv|log|xml|"
    r"ini|cfg|conf|rtf|py|jsx?|tsx?|css|scss|sql|sh|bat|cmd|ps1)\b",
    re.IGNORECASE,
)
_OPEN_FILE_RE = re.compile(
    r"\bоткрой\s+файл\w*\s+.+|\b(?:открой|open)\b.*" + _NAMED_FILE_RE.pattern,
    re.IGNORECASE,
)
_DISPLAY_FILE_RE = re.compile(
    r"\b(?:выведи|покажи)\s+содержим\w*(?:\s+файл\w*)?\s+.+|"
    r"\b(?:прочитай|прочти)\s+(?:файл\w*\s+)?.+|"
    r"\bоткрой\s+(?!файл\w*\b)(?:без\s+названия\s+\d+|[^?\r\n]*\d+)\s*$",
    re.IGNORECASE,
)
_ANALYZE_RE = re.compile(r"\b(?:разбери|проанализируй|изучи)\b", re.IGNORECASE)
_SUMMARY_RE = re.compile(
    r"(?:о\s+ч[её]м|содержани\w*|описан\w*|кратк\w*|перескаж\w*|простыми\s+словами)\b",
    re.IGNORECASE,
)
_FILE_FOLLOW_UP_RE = re.compile(
    r"\b(?:перескаж\w*|объясни|резюмируй|подробн\w*|простыми\s+словами|его\s+содержим\w*|"
    r"это(?:т|го)?\s+файл)\b",
    re.IGNORECASE,
)
_CONTENT_DATE_RE = re.compile(
    r"\b(?:дат\w*|когда|год\w*)\b.*\b(?:контекст|содержим|текст|внутри|указан|упомянут)\w*|"
    r"\b(?:контекст|содержим|текст|внутри|указан|упомянут)\w*.*\b(?:дат\w*|когда|год\w*)\b",
    re.IGNORECASE,
)
_DATE_VALUE_RE = re.compile(
    r"\b(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|(?:19|20)\d{2}|"
    r"\d{1,2}\s+(?:январ\w*|феврал\w*|март\w*|апрел\w*|ма[йя]\w*|июн\w*|"
    r"июл\w*|август\w*|сентябр\w*|октябр\w*|ноябр\w*|декабр\w*))\b",
    re.IGNORECASE,
)
_DIRECTORY_ANALYZE_RE = re.compile(
    r"\b(?:разбери|проанализируй|изучи)\b.*\b(?:папк|директор|workspace|воркспейс)\w*",
    re.IGNORECASE,
)
_RECURSIVE_RE = re.compile(r"\b(?:все|внутри|вложенн\w*|рекурсивн\w*|дерево)\b", re.IGNORECASE)
_QUERY_STOPWORDS = {
    "что", "кто", "где", "как", "какой", "какая", "какие", "знаешь", "скажи",
    "расскажи", "про", "обо", "это", "этот", "эта", "есть", "мне", "можешь",
    "the", "what", "who", "where", "about", "tell", "does", "know",
}


def _term_key(term: str) -> str:
    return lexical_key(term)


def _message_parts(message) -> tuple[str, str]:
    if isinstance(message, dict):
        return str(message.get("role", "")), str(message.get("content", ""))
    return str(getattr(message, "type", getattr(message, "role", ""))), str(getattr(message, "content", ""))


def _source_excerpt_for_fallback(question: str, answer: str, sources: list[dict], verification):
    """Return source text only when an answer cannot be verified as a whole."""
    from verifier import _quote_source_ids

    source_by_id = {source["chunk_id"]: source["text"] for source in sources}
    quoted = [claim.evidence_quote for claim in (verification.claims if verification else [])
              if claim.evidence_quote]
    quoted.extend(re.findall(r"«([^»]{12,})»", answer))
    for candidate in quoted:
        matching_ids = _quote_source_ids(candidate, source_by_id)
        if matching_ids:
            source = next(item for item in sources if item["chunk_id"] == matching_ids[0])
            return source["metadata"]["source"], candidate.strip(" \t\r\n\"'«»")

    terms = {_term_key(term) for term in re.findall(r"[\w-]{4,}", question, re.UNICODE)
             if term.casefold() not in _QUERY_STOPWORDS}
    for source in sources:
        content = source["text"]
        positions = [match.start() for match in re.finditer(r"[\w-]{4,}", content, re.UNICODE)
                     if _term_key(match.group()) in terms]
        if positions:
            position = min(positions)
            start = max(0, position - 80)
            end = min(len(content), position + 180)
            excerpt = ("…" if start else "") + content[start:end].strip() + ("…" if end < len(content) else "")
            return source["metadata"]["source"], excerpt
    return None


class WorkspaceSkill:
    name = "workspace"
    description = ("Read-only access to the active local folder: list files and folders, read documents, "
                   "answer questions from their contents, find the latest version, compare versions, "
                   "read filesystem dates and explicitly open a selected file. execute interprets the user's "
                   "question; it is not limited to listing files. No deletion, renaming or writing.")
    input_schema = {"type": "object", "properties": {"workspace_id": {"type": "string"}}}
    permissions = ("workspace:read",)
    operations = ("execute", "find_latest", "compare_versions", "open_file", "list_file_dates")
    operation_contracts = {
        "find_latest": {"inputs": ["target_query"],
                         "outputs": ["selected_path", "file_path", "path"]},
        "compare_versions": {"inputs": ["target_query"],
                              "outputs": ["selected_path", "file_path", "path"]},
        "open_file": {"inputs": ["path"], "outputs": ["selected_path", "file_path", "path"]},
        "list_file_dates": {"inputs": [], "outputs": []},
        "execute": {"inputs": ["question"], "outputs": []},
    }

    def __init__(self, filesystem: FilesystemProvider, planner: WorkspacePlanner | None = None,
                 router: WorkspaceToolRouter | None = None, vector_indexer=None):
        self.filesystem = filesystem
        self.repository = filesystem.repository
        self.planner = planner or WorkspacePlanner()
        self.router = router or WorkspaceToolRouter()
        self.vector_indexer = vector_indexer

    def plan(self, text: str, contextual_text: str | None = None) -> WorkspacePlan | None:
        return self.planner.plan(text, contextual_text)

    def route_ambiguous(self, text: str, model: str) -> WorkspaceRouteDecision:
        return self.router.route(text, model)

    def metadata_plan(self, text: str) -> WorkspacePlan:
        return self.planner.metadata_plan(text)

    def resolution_error_result(self, message: str, workspace_id: str,
                                original_question: str | None = None) -> ToolResult:
        choices = re.findall(r"- `([^`]+)`", message)
        interaction = ({
            "kind": "workspace_file_choice", "choices": choices,
            "original_question": original_question or "",
        } if choices else None)
        return ToolResult(success=True, tool=self.name, summary="Workspace file needs clarification", data={
            "final": message, "context": "", "results": [],
            "strategy": "WORKSPACE_FILE_CHOICES", "workspace_id": workspace_id,
            "pending_interaction": interaction,
            "plan_status": "WAITING_FOR_USER" if interaction else None,
            "pending_step": interaction,
        })

    def clarification_result(self, workspace_id: str, unsupported=False) -> ToolResult:
        return ToolResult(success=True, tool=self.name, summary='Workspace intent needs clarification', data={
            'final': ('Удалять, переименовывать и перемещать файлы в рабочей папке пока нельзя. '
                      if unsupported else '') + 'Уточните, что нужно сделать: показать список файлов, прочитать конкретный файл, '
                     'найти последнюю версию или ответить на вопрос по содержимому? Укажите файл или папку, если это важно.',
            'context': '', 'results': [], 'strategy': 'WORKSPACE_CLARIFICATION', 'workspace_id': workspace_id,
        })

    async def execute_agent_step(self, operation: str, inputs: dict, execution) -> ToolResult:
        """Granular operations used by the general dependency-aware agent executor."""
        if operation == "execute":
            return await self.execute(inputs, execution)
        workspace_id = inputs.get("workspace_id")
        workspace = await asyncio.to_thread(self.repository.get, workspace_id) if workspace_id else await asyncio.to_thread(self.repository.active)
        if workspace is None:
            raise AssistantError("Сначала выберите workspace в боковой панели.", 409)
        target_query = inputs.get("target_query") or inputs.get("question", "")
        if operation == "find_latest":
            async with execution.action("FIND_LATEST_FILE", {"workspace_id": workspace.id}) as action:
                selected, reason = await asyncio.to_thread(self.filesystem.latest_file, workspace.id, target_query)
                action.output = {"path": selected.relative_path, "reason": reason}
            return ToolResult(success=True, tool=self.name, summary="Latest file found", data={
                "selected_path": selected.relative_path, "file_path": selected.relative_path,
                "path": selected.relative_path,
                "final": (f"Последняя подходящая версия: `{selected.relative_path}`.\n\n"
                          f"Основание выбора: {reason}. Изменён: {selected.modified_at}."),
                "strategy": "WORKSPACE_LATEST", "workspace_id": workspace.id,
            })
        if operation == "compare_versions":
            async with execution.action("COMPARE_FILE_VERSIONS", {"workspace_id": workspace.id}) as action:
                comparison = await asyncio.to_thread(self.filesystem.compare_latest, workspace.id, target_query)
                action.output = {
                    "current": comparison["current"].relative_path,
                    "previous": comparison["previous"].relative_path,
                    "additions": comparison["additions"], "deletions": comparison["deletions"],
                    "truncated": comparison["truncated"],
                }
            answer = (f"Сравнение с `{comparison['previous'].relative_path}`: добавлено строк — "
                      f"{comparison['additions']}, удалено — {comparison['deletions']}.")
            answer += (f"\n\n```diff\n{comparison['diff']}\n```" if comparison["diff"]
                       else "\n\nТекстовых различий не найдено.")
            if comparison["truncated"]:
                answer += "\n\n_Показана только первая часть различий._"
            return ToolResult(success=True, tool=self.name, summary="Versions compared", data={
                "selected_path": comparison["current"].relative_path,
                "file_path": comparison["current"].relative_path,
                "path": comparison["current"].relative_path, "final": answer,
                "strategy": "WORKSPACE_COMPARE", "workspace_id": workspace.id,
            })
        if operation == "open_file":
            if not requests_file_open(execution.task.user_input):
                raise AssistantError("Открытие файла не запрошено. Уточните, какой файл нужно открыть.", 409)
            relative_path = inputs.get("path") or inputs.get("selected_path")
            if not relative_path:
                raise AssistantError("План открытия не содержит выбранного файла.", 422)
            async with execution.action("OPEN_FILE", {
                "workspace_id": workspace.id, "path": relative_path,
            }) as action:
                opened = await asyncio.to_thread(self.filesystem.open_file, workspace.id, relative_path)
                action.output = {"path": opened}
            return ToolResult(success=True, tool=self.name, summary="File opened", data={
                "selected_path": opened, "file_path": opened, "path": opened,
                "final": f"Файл `{opened}` открыт в системном приложении.",
                "strategy": "WORKSPACE_OPEN", "preserve_previous_strategy": True,
                "workspace_id": workspace.id,
                "current_directory": "" if Path(opened).parent.as_posix() == "."
                else Path(opened).parent.as_posix(),
            })
        if operation == "list_file_dates":
            async with execution.action("LIST_FILE_DATES", {"workspace_id": workspace.id}) as action:
                sync = await asyncio.to_thread(self.filesystem.sync_index, workspace.id)
                files = [item for item in await asyncio.to_thread(self.repository.indexed_files, workspace.id) if item.readable]
                action.output = {"file_count": len(files), "created": sync.created,
                                 "updated": sync.updated, "deleted": sync.deleted}
            shown = sorted(files, key=lambda value: value.relative_path.casefold())[:500]
            lines = [f"- `{item.relative_path}` — создан: {item.filesystem_created_at}; изменён: {item.modified_at}"
                     for item in shown]
            answer = "Системные даты файлов:\n\n" + "\n".join(lines) if lines else (
                "В workspace нет поддерживаемых документов с доступными системными датами."
            )
            if lines:
                answer += ("\n\nЭто даты файловой системы. Они не обязательно совпадают с фактической "
                           "датой написания текста или первоначального создания документа.")
            return ToolResult(success=True, tool=self.name, summary="File dates listed", data={
                "final": answer, "strategy": "WORKSPACE_FILE_DATES", "workspace_id": workspace.id,
            })
        raise AssistantError(f"Workspace не поддерживает операцию `{operation}`.", 422)

    def matches(self, text: str, contextual_text: str | None = None) -> bool:
        commands = command_text(contextual_text or text)
        return bool(self.plan(text, contextual_text) or _READ_RE.search(commands)
                    or _NAMED_FILE_RE.fullmatch((contextual_text or text).strip())
                    or _DISPLAY_FILE_RE.search(commands)
                    or _LIST_RE.search(commands)
                    or self.requires_context(text))

    def requires_context(self, text: str) -> bool:
        return bool(_SUMMARY_RE.search(text) or _DIRECTORY_ANALYZE_RE.search(text)
                    or _CONTENT_DATE_RE.search(text))

    def relevant_context(self, question: str, workspace_id: str | None = None,
                         broad: bool = False, max_characters: int | None = None,
                         active_file: str = "", semantic: bool = True) -> list[dict]:
        workspace = self.repository.get(workspace_id) if workspace_id else self.repository.active()
        if workspace is None:
            return []
        # File-reference resolution must see files created since the previous
        # workspace request, before broad content retrieval starts.
        scan = self.filesystem.sync_index(workspace.id)
        indexed = self.repository.indexed_files(workspace.id)
        limit = max_characters or 12000
        if active_file and _FILE_FOLLOW_UP_RE.search(question):
            active = next((item for item in indexed if item.relative_path == active_file and item.readable), None)
            if active:
                _, _, content = self.filesystem.read_text_file(workspace.id, active.relative_path)
                return [{"path": active.relative_path, "content": content[:limit], "score": 20.0,
                         "matched": [], "found_by": ["active_conversation_file"]}]
        scope = self.filesystem.find_directory(workspace.id, question)
        reference = re.search(
            r"\sв\s+(?:файл\w*\s+)?([^?!.]+)\s*[?!.]*$", question, re.IGNORECASE
        )
        if reference:
            label = reference.group(1).strip()
            label_terms = {_term_key(term) for term in re.findall(r"[\w-]+", label, re.UNICODE)}
            resembles_file = any(
                len(label_terms & {_term_key(term) for term in re.findall(
                    r"[\w-]+", Path(item.name).stem, re.UNICODE
                )}) >= min(2, len(label_terms))
                for item in indexed if label_terms
            )
            if resembles_file:
                path = self.filesystem.find_text_file(
                    workspace.id, f"Прочитай файл {label}", None
                )
                _, _, content = self.filesystem.read_text_file(workspace.id, path)
                limit = max_characters or 12000
                return [{"path": path, "content": content[:limit], "score": 10.0,
                         "matched": [], "found_by": ["workspace_file_reference"]}]
        terms = {
            _term_key(term) for term in re.findall(r"[\w-]+", question, re.UNICODE)
            if len(term) >= 3 and term.casefold() not in _QUERY_STOPWORDS
        }
        content_date = bool(_CONTENT_DATE_RE.search(question))
        semantic_date = content_date or (broad and bool(re.search(
            r"\b(?:дат\w*|когда|написан\w*|год\w*)\b", question, re.IGNORECASE
        )))
        # Only broad requests about the folder as a whole should include files
        # without a query match. A summary of a person or topic needs retrieval.
        summarize = bool(
            _DIRECTORY_ANALYZE_RE.search(question) or content_date or
            (_SUMMARY_RE.search(question) and re.search(
                r"\b(?:файл|документ|папк|директори|workspace)\w*", question, re.IGNORECASE,
            ))
        )
        if not terms and not summarize:
            return []
        # A title explicitly named without an extension scopes the answer to
        # that file. Do this before broad summaries to avoid mixing neighbors.
        title_matches = []
        for item in indexed:
            if not item.readable:
                continue
            title_terms = {_term_key(term) for term in re.findall(r"[\w-]+", Path(item.name).stem, re.UNICODE)}
            overlap_count = sum(
                1 for title_term in title_terms
                if any(min(len(query_term), len(title_term)) >= 4 and (
                    query_term.startswith(title_term) or title_term.startswith(query_term)
                ) for query_term in terms)
            )
            if overlap_count >= 2:
                title_matches.append((overlap_count, len(title_terms), item))
        title_matches.sort(key=lambda value: (-value[0], value[1], value[2].relative_path.casefold()))
        if title_matches and (len(title_matches) == 1 or title_matches[0][:2] != title_matches[1][:2]):
            selected_file = title_matches[0][2]
            _, _, content = self.filesystem.read_text_file(workspace.id, selected_file.relative_path)
            return [{"path": selected_file.relative_path, "content": content[:limit], "score": 15.0,
                     "matched": sorted(terms), "found_by": ["workspace_file_title"]}]
        ranked = []
        coverage = {}
        for path, content in self.filesystem.searchable_files(workspace.id, relative_directory=scope,
                                                             coverage=coverage, synchronize=False):
            content_terms = {_term_key(term) for term in re.findall(r"[\w-]+", content, re.UNICODE)}
            path_terms = {_term_key(term) for term in re.findall(r"[\w-]+", path, re.UNICODE)}
            matched_content = terms & content_terms
            matched_path = terms & path_terms
            has_date = bool(_DATE_VALUE_RE.search(content)) if semantic_date else False
            if matched_content or matched_path or summarize:
                # A descriptive filename is strong metadata: exact title words
                # should beat incidental mentions in unrelated documents.
                score = len(matched_content) + 3 * len(matched_path) + (2 if has_date else 0)
                ranked.append({"path": path, "content": content, "score": score,
                               "matched": sorted(matched_content)})
        ranked.sort(key=lambda item: (-item["score"], item["path"].casefold()))
        selected = ranked[:8] if summarize else ranked[:3]
        if semantic and self.vector_indexer is not None and not scope and not summarize and len(selected) < 3:
            try:
                # Retrieval must not run a full vector/graph rebuild. That operation
                # can require many LLM calls and is available via vector-sync.
                semantic_matches = self.vector_indexer.search(workspace.id, question, limit=3)
                selected_paths = {item["path"] for item in selected}
                current_vectors = {
                    item.relative_path for item in indexed
                    if item.readable and item.content_hash
                    and item.vector_status == "INDEXED" and item.vector_hash == item.content_hash
                }
                for item in semantic_matches:
                    if item["path"] in current_vectors and item["path"] not in selected_paths:
                        selected.append(item)
                        selected_paths.add(item["path"])
                    if len(selected) >= 3:
                        break
            except Exception:
                # Lexical reading still works if embeddings are unavailable.
                pass
        default_total = (900 if semantic_date else 600 if summarize else 1600) * max(1, len(selected))
        total_budget = max_characters or default_total
        per_file = max(240, total_budget // max(1, len(selected)))
        for item in selected:
            item['coverage'] = {**coverage, 'selected': len(selected), 'matched_files': len(ranked),
                                'scan_truncated': scan.truncated, 'scan_errors': len(scan.errors)}
            content = item["content"]
            position = 0
            if semantic_date:
                date_match = _DATE_VALUE_RE.search(content)
                if date_match:
                    position = max(0, date_match.start() - per_file // 3)
            elif item["matched"]:
                lowered = content.casefold().replace("ё", "е")
                positions = [lowered.find(term) for term in item["matched"]]
                positions = [value for value in positions if value >= 0]
                if positions:
                    position = max(0, min(positions) - per_file // 3)
            item["content"] = content[position:position + per_file]
        return selected

    def response_budget(self, question: str, model: str | None = None):
        return ResponseBudgetPlanner().plan(question, model)

    async def execute(self, inputs: dict, execution) -> ToolResult:
        workspace_id = inputs.get("workspace_id")
        workspace = await asyncio.to_thread(self.repository.get, workspace_id) if workspace_id else await asyncio.to_thread(self.repository.active)
        if workspace is None:
            raise AssistantError("Сначала выберите workspace в боковой панели.", 409)
        if requests_unsupported_file_change(execution.task.user_input):
            return self.clarification_result(workspace.id, unsupported=True)
        if inputs.get("_workspace_resolution_error"):
            return self.resolution_error_result(
                inputs["_workspace_resolution_error"], workspace.id, inputs.get("question")
            )
        question = inputs.get("_workspace_command_question", inputs.get('_retrieval_question', inputs.get("question", "")))
        plan = inputs.get("_workspace_plan") or self.plan(inputs.get("question", ""), question)
        decision_data = inputs.get('_workspace_semantic_decision')
        commands = command_text(question)
        known_command = bool(_LIST_RE.search(commands) or _READ_RE.search(commands)
                             or _NAMED_FILE_RE.fullmatch(question.strip()) or _DISPLAY_FILE_RE.search(question))
        if (not plan and not inputs.get('_workspace_matches')
                and not inputs.get('_workspace_context_requested') and (decision_data or not known_command)):
            if decision_data:
                decision = WorkspaceRouteDecision.model_validate(decision_data)
            else:
                try:
                    async with execution.action('ROUTE_WORKSPACE', {'question': question}) as action:
                        decision = await asyncio.to_thread(self.route_ambiguous, question, inputs.get('model', 'BALANCED'))
                        action.output = decision.model_dump(mode='json')
                except Exception:
                    return self.clarification_result(workspace.id)
            if decision.confidence < .65 or decision.operation == 'clarify':
                return self.clarification_result(workspace.id)
            if decision.operation in {'find_latest', 'compare_versions', 'list_file_dates'}:
                return await self.execute_agent_step(decision.operation, {**inputs, 'target_query': question}, execution)
            if decision.operation == 'read_file':
                inputs = {**inputs, '_workspace_read_requested': True}
            elif decision.operation != 'list_directory':
                if decision.route.value in {'GENERAL_CHAT', 'DOCUMENT_LIBRARY'}:
                    return self.clarification_result(workspace.id)
                budget = inputs.get('_response_budget') or self.response_budget(question, inputs.get('model', 'BALANCED'))
                try:
                    matches = await asyncio.to_thread(self.relevant_context, question, workspace.id,
                                                      False, budget.max_source_characters,
                                                      inputs.get('_workspace_active_file', ''))
                except AssistantError as exc:
                    if exc.status_code not in {404, 409}:
                        raise
                    return self.resolution_error_result(str(exc), workspace.id, question)
                inputs = {**inputs, '_workspace_matches': matches, '_workspace_context_requested': True,
                          '_response_budget': budget}
        if plan:
            if isinstance(plan, dict):
                plan = WorkspacePlan.model_validate(plan)
            async with execution.action("PLAN_WORKSPACE", {"objective": plan.objective}) as action:
                action.output = {
                    "target_query": plan.target_query,
                    "steps": [step.model_dump(mode="json") for step in plan.steps],
                }
            selected = reason = None
            answer = ""
            strategy = "WORKSPACE_LATEST"
            for step in plan.steps:
                if step.operation == WorkspaceOperation.LIST_FILE_DATES:
                    async with execution.action(step.operation.value, {
                        "workspace_id": workspace.id, "step_id": step.id,
                    }) as action:
                        sync = await asyncio.to_thread(self.filesystem.sync_index, workspace.id)
                        files = [item for item in await asyncio.to_thread(self.repository.indexed_files, workspace.id) if item.readable]
                        action.output = {
                            "file_count": len(files), "created": sync.created,
                            "updated": sync.updated, "deleted": sync.deleted,
                        }
                    if files:
                        shown_files = sorted(files, key=lambda value: value.relative_path.casefold())[:500]
                        lines = [
                            f"- `{item.relative_path}` — создан: {item.filesystem_created_at}; "
                            f"изменён: {item.modified_at}"
                            for item in shown_files
                        ]
                        answer = (
                            "Системные даты файлов:\n\n" + "\n".join(lines)
                            + "\n\nЭто даты файловой системы. Они не обязательно совпадают с фактической "
                              "датой написания текста или первоначального создания документа."
                        )
                        if len(files) > len(shown_files):
                            answer += f"\n\nПоказаны первые {len(shown_files)} из {len(files)} документов."
                    else:
                        answer = "В workspace нет поддерживаемых документов с доступными системными датами."
                    strategy = "WORKSPACE_FILE_DATES"
                elif step.operation == WorkspaceOperation.FIND_LATEST:
                    async with execution.action(step.operation.value, {
                        "workspace_id": workspace.id, "step_id": step.id,
                        "depends_on": step.depends_on,
                    }) as action:
                        selected, reason = await asyncio.to_thread(self.filesystem.latest_file, workspace.id, plan.target_query)
                        action.output = {"path": selected.relative_path, "reason": reason}
                    answer = (f"Последняя подходящая версия: `{selected.relative_path}`.\n\n"
                              f"Основание выбора: {reason}. Изменён: {selected.modified_at}.")
                elif step.operation == WorkspaceOperation.COMPARE_VERSIONS:
                    if selected is None:
                        raise AssistantError("План сравнения не содержит выбранного файла.", 500)
                    async with execution.action("COMPARE_FILE_VERSIONS", {
                        "workspace_id": workspace.id, "current": selected.relative_path,
                        "step_id": step.id, "depends_on": step.depends_on,
                    }) as action:
                        comparison = await asyncio.to_thread(self.filesystem.compare_latest, workspace.id, plan.target_query)
                        action.output = {
                            "current": comparison["current"].relative_path,
                            "previous": comparison["previous"].relative_path,
                            "additions": comparison["additions"], "deletions": comparison["deletions"],
                            "truncated": comparison["truncated"],
                        }
                    answer += (f"\n\nСравнение с `{comparison['previous'].relative_path}`: "
                               f"добавлено строк — {comparison['additions']}, удалено — {comparison['deletions']}.")
                    answer += (f"\n\n```diff\n{comparison['diff']}\n```" if comparison["diff"]
                               else "\n\nТекстовых различий не найдено.")
                    if comparison["truncated"]:
                        answer += "\n\n_Показана только первая часть различий._"
                    strategy = "WORKSPACE_COMPARE"
                elif step.operation == WorkspaceOperation.OPEN_SELECTED:
                    if not requests_file_open(execution.task.user_input):
                        return self.clarification_result(workspace.id)
                    if selected is None:
                        raise AssistantError("План открытия не содержит выбранного файла.", 500)
                    async with execution.action(step.operation.value, {
                        "workspace_id": workspace.id, "path": selected.relative_path,
                        "step_id": step.id, "depends_on": step.depends_on,
                    }) as action:
                        opened = await asyncio.to_thread(self.filesystem.open_file, workspace.id, selected.relative_path)
                        action.output = {"path": opened}
                    answer += f"\n\nФайл `{opened}` открыт в системном приложении."
            return ToolResult(success=True, tool=self.name, summary="Latest workspace file found", data={
                "final": answer, "context": "", "strategy": strategy,
                "workspace_id": workspace.id,
            })
        if _OPEN_FILE_RE.search(question) and requests_file_open(execution.task.user_input):
            model = inputs.get("model")
            profile = model.value if hasattr(model, "value") else model
            try:
                async with execution.action("RESOLVE_FILE_REFERENCE", {"question": question}) as action:
                    relative_path = await asyncio.to_thread(self.filesystem.find_text_file,
                        workspace.id, question, profile, inputs.get("_workspace_current_directory", "")
                    )
                    action.output = {"path": relative_path}
            except AssistantError as exc:
                if exc.status_code in {404, 409}:
                    return self.resolution_error_result(str(exc), workspace.id, question)
                raise
            async with execution.action("OPEN_FILE", {
                "workspace_id": workspace.id, "path": relative_path,
            }) as action:
                opened = await asyncio.to_thread(self.filesystem.open_file, workspace.id, relative_path)
                action.output = {"path": opened}
            return ToolResult(success=True, tool=self.name, summary="Workspace file opened", data={
                "final": f"Файл `{opened}` открыт в системном приложении.", "context": "",
                "strategy": "WORKSPACE_OPEN", "workspace_id": workspace.id,
                "active_file": opened,
                "current_directory": "" if Path(opened).parent.as_posix() == "."
                else Path(opened).parent.as_posix(),
            })
        matches = inputs.get("_workspace_matches") or []
        if matches:
            budget = inputs.get("_response_budget") or self.response_budget(
                inputs.get("question", ""),
                getattr(inputs.get("model"), "value", inputs.get("model")),
            )
            sources = []
            context_parts = []
            async with execution.action("SEARCH_WORKSPACE", {"workspace_id": workspace.id}) as action:
                for item in matches:
                    excerpt = item["content"]
                    context_parts.append(f"Файл {item['path']}:\n{excerpt}")
                    sources.append({
                        "document_id": workspace.id,
                        "chunk_id": f"workspace:{item['path']}",
                        "text": excerpt,
                        "metadata": {"source": item["path"], "workspace_id": workspace.id,
                                     "coverage": item.get('coverage', {})},
                        "score": float(item["score"]),
                        "found_by": item.get("found_by", ["workspace"]),
                    })
                action.output = {"source_count": len(sources), "paths": [item["path"] for item in matches]}
            from model_runtime import get_llm_provider
            model = inputs.get("model")
            profile = model.value if hasattr(model, "value") else model
            context = "\n\n".join(context_parts)
            history_messages = []
            history_used = 0
            for message in reversed((inputs.get("_chat_history") or [])[-10:]):
                role, content = _message_parts(message)
                mapped_role = "user" if role.casefold() in {"human", "user"} else "assistant"
                if content.strip():
                    remaining = budget.max_history_characters - history_used
                    if remaining <= 0:
                        break
                    excerpt = content[-remaining:]
                    history_messages.append({"role": mapped_role, "content": excerpt})
                    history_used += len(excerpt)
            history_messages.reverse()
            async with execution.action("GENERATE_RESPONSE", {"source_count": len(sources)}) as action:
                generated = await asyncio.to_thread(get_llm_provider().generate, [
                    {"role": "system", "content": (
                        "Ответь на вопрос только по содержимому файлов активной локальной папки. "
                        "Учитывай предыдущие реплики, чтобы понимать местоимения и уточняющие вопросы, "
                        "но прежние ответы ассистента не считай источниками фактов. "
                        "Отделяй прямо сказанное в файлах от собственной интерпретации. "
                        "Не превращай метафору, намерение или образ в буквальное определение, событие или установленную цель. "
                        "Если объясняешь возможный смысл, пометь его как осторожную трактовку и назови текстовую опору. "
                        "Если точный смысл, участник или результат в файлах не раскрыт, прямо скажи об этом; "
                        "это не мешает привести относящийся к вопросу фрагмент и осторожно его обсудить. "
                        "Не добавляй мотивы, последствия и детали, которых нет в источниках. "
                        "Для короткого вопроса дай прямой краткий ответ, а не обзор всех связанных тем; "
                        "не составляй список свойств, если пользователь его не просил. "
                        "Укажи имена использованных файлов."
                    )},
                    *history_messages,
                    {"role": "user", "content": f"Текущий вопрос: {inputs['question']}\n\nИсточники:\n{context}"},
                ], profile=profile, max_tokens=budget.max_output_tokens)
                answer = generated.text
                action.output = {"source_count": len(sources), "input_characters": len(context),
                                 "max_output_tokens": budget.max_output_tokens,
                                 "detail": budget.detail.value, "coverage": budget.coverage.value,
                                 "budget_reason": budget.reason}
            verification = None
            from verifier import verify_answer
            from settings import VERIFIER_ENABLED
            async with execution.action("VERIFY_RESPONSE"):
                answer, verification = await asyncio.to_thread(
                    verify_answer, inputs["question"], answer, sources, "WORKSPACE_QA", profile,
                )
            if VERIFIER_ENABLED and (verification is None or verification.status != "SUPPORTED"):
                excerpt = _source_excerpt_for_fallback(inputs["question"], answer, sources, verification)
                if excerpt:
                    path, quote = excerpt
                    answer = (f"В файле «{path}» сказано: «{quote}». "
                              "Более подробный ответ по найденным файлам я не могу надёжно подтвердить.")
                else:
                    answer = "Не могу подтвердить ответ по найденным файлам. Уточните вопрос или добавьте источник."
            coverage = next((item.get('coverage') for item in matches if item.get('coverage')), {})
            if (coverage.get('omitted') or coverage.get('failed') or coverage.get('scan_truncated')
                    or coverage.get('scan_errors') or
                    (self.requires_context(inputs['question']) and coverage.get('matched_files', 0) > len(matches))):
                answer += (f"\n\nОхват ограничен: просмотрено {coverage.get('inspected', 0)} из "
                           f"{coverage.get('available', 0)} доступных файлов; ответ основан на "
                           f"{len(sources)} выбранных источниках. Полный обзор папки не выполнен.")
            return ToolResult(success=True, tool=self.name, summary="Workspace question answered", data={
                "final": answer, "context": context, "results": sources, "verification": verification,
                "strategy": "WORKSPACE_QA", "workspace_id": workspace.id,
                "active_file": sources[0]["metadata"]["source"] if len(sources) == 1 else None,
            })
        if inputs.get("_workspace_context_requested"):
            return ToolResult(success=True, tool=self.name, summary="Workspace search returned no files", data={
                "final": "В указанной папке не найдено поддерживаемых текстовых файлов с информацией для ответа.",
                "context": "", "results": [], "strategy": "WORKSPACE_QA", "workspace_id": workspace.id,
            })
        if (inputs.get('_workspace_read_requested') or _READ_RE.search(commands) or _NAMED_FILE_RE.fullmatch(question.strip())
                or _DISPLAY_FILE_RE.search(commands)):
            model = inputs.get("model")
            profile = model.value if hasattr(model, "value") else model
            try:
                async with execution.action("RESOLVE_FILE_REFERENCE", {"question": question}) as action:
                    relative_path = await asyncio.to_thread(self.filesystem.find_text_file,
                        workspace.id, question, profile, inputs.get("_workspace_current_directory", "")
                    )
                    action.output = {"path": relative_path}
            except AssistantError as exc:
                if exc.status_code in {404, 409}:
                    return self.resolution_error_result(str(exc), workspace.id, question)
                raise
            async with execution.action("READ_FILE", {"workspace_id": workspace.id, "path": relative_path}) as action:
                _, relative_path, content = await asyncio.to_thread(self.filesystem.read_text_file, workspace.id, relative_path)
                action.output = {"path": relative_path, "characters": len(content)}
            if _ANALYZE_RE.search(question):
                from model_runtime import get_llm_provider
                model = inputs.get("model")
                profile = model.value if hasattr(model, "value") else model
                budget = inputs.get("_response_budget") or self.response_budget(question, profile)
                source = content[:budget.max_source_characters]
                async with execution.action("GENERATE_RESPONSE", {"source": relative_path}) as action:
                    generated = await asyncio.to_thread(get_llm_provider().generate, [
                        {"role": "system", "content": "Отвечай только по содержимому локального файла. Не выдумывай отсутствующие факты."},
                        {"role": "user", "content": f"Запрос: {inputs['question']}\n\nФайл {relative_path}:\n{source}"},
                    ], profile=profile, max_tokens=budget.max_output_tokens)
                    answer = generated.text
                    action.output = {"source": relative_path, "input_characters": len(source),
                                     "max_output_tokens": budget.max_output_tokens,
                                     "detail": budget.detail.value, "coverage": budget.coverage.value,
                                     "source_truncated": len(source) < len(content)}
                strategy = "WORKSPACE_ANALYZE"
            else:
                answer = f"Содержимое `{relative_path}`:\n\n{content}"
                strategy = "WORKSPACE_READ"
            return ToolResult(success=True, tool=self.name, summary="Workspace file read", data={
                "final": answer, "context": "", "strategy": strategy,
                "workspace_id": workspace.id,
                "active_file": relative_path,
                "current_directory": "" if Path(relative_path).parent.as_posix() == "."
                else Path(relative_path).parent.as_posix(),
            })
        question = inputs.get("question", "")
        try:
            scope = await asyncio.to_thread(self.filesystem.find_directory, workspace.id, question)
        except AssistantError as exc:
            if exc.status_code not in {404, 409}:
                raise
            return self.resolution_error_result(str(exc), workspace.id, question)
        recursive = bool(scope or _RECURSIVE_RE.search(question))
        if not scope and re.search(r"\b(?:текущ\w*|данн\w*|этой|эту)\s+(?:директор|папк)\w*\b",
                                   question, re.IGNORECASE):
            scope = inputs.get('_workspace_current_directory', '')
        action_type = "LIST_DIRECTORY_RECURSIVE" if recursive else "LIST_DIRECTORY"
        async with execution.action(action_type, {"workspace_id": workspace.id, "path": scope}) as action:
            listing = (await asyncio.to_thread(self.filesystem.list_tree, workspace.id, scope) if recursive
                       else await asyncio.to_thread(self.filesystem.list_directory, workspace.id, scope))
            action.output = {"path": scope, "entry_count": len(listing.entries), "truncated": listing.truncated}
        if listing.entries:
            lines = [f"- {'📁' if item.kind == 'directory' else '📄'} {item.relative_path}"
                     for item in listing.entries]
            location = f"папке `{scope}`" if scope else f"workspace «{workspace.name}»"
            answer = f"В {location}:\n\n" + "\n".join(lines)
            if listing.truncated:
                answer += f"\n\nПоказаны первые {len(listing.entries)} элементов."
        else:
            answer = f"Workspace «{workspace.name}» пуст."
        return ToolResult(success=True, tool=self.name, summary="Workspace listed", data={
            "final": answer, "context": "", "strategy": "WORKSPACE_LIST",
            "workspace_id": workspace.id, "current_directory": scope,
        })
