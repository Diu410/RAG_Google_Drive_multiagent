"""Shared document extraction adapter for files discovered inside a workspace."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path


PLAIN_TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".json", ".jsonl", ".yaml", ".yml", ".toml",
    ".csv", ".tsv", ".log", ".xml", ".ini", ".cfg", ".conf", ".rtf",
    ".py", ".js", ".jsx", ".ts", ".tsx", ".css", ".scss", ".sql",
    ".sh", ".bat", ".cmd", ".ps1",
}
STRUCTURED_DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".html", ".htm"}
SUPPORTED_WORKSPACE_EXTENSIONS = PLAIN_TEXT_EXTENSIONS | STRUCTURED_DOCUMENT_EXTENSIONS


class WorkspaceDocumentReader:
    """Extract text through the same structured loaders used by Document RAG."""

    def __init__(self, max_cache_entries: int = 32, max_output_characters: int = 2_000_000):
        self.max_cache_entries = max_cache_entries
        self.max_output_characters = max_output_characters
        self._cache: OrderedDict[tuple[str, int, int], str] = OrderedDict()

    @staticmethod
    def _decode_plain(path: Path) -> str:
        raw = path.read_bytes()
        if b"\x00" in raw:
            raise ValueError("Файл выглядит как бинарный")
        for encoding in ("utf-8-sig", "cp1251"):
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError("Не удалось определить текстовую кодировку файла")

    @staticmethod
    def _structured_documents(path: Path):
        suffix = path.suffix.casefold()
        if suffix == ".pdf":
            # This adapter deliberately reuses the production PDF extraction,
            # including text-layer fallback and OCR behavior.
            from chroma_utils import load_pdf
            return load_pdf(str(path))
        from formula_processing import load_docx_with_formulas, load_html_with_formulas
        if suffix == ".docx":
            return load_docx_with_formulas(str(path))
        if suffix in {".html", ".htm"}:
            return load_html_with_formulas(str(path))
        raise ValueError(f"Неподдерживаемый формат: {suffix}")

    def read(self, path: Path) -> str:
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        suffix = path.suffix.casefold()
        if suffix in PLAIN_TEXT_EXTENSIONS:
            content = self._decode_plain(path)
        elif suffix in STRUCTURED_DOCUMENT_EXTENSIONS:
            parts = [document.page_content.strip() for document in self._structured_documents(path)
                     if document.page_content and document.page_content.strip()]
            content = "\n\n".join(parts)
        else:
            raise ValueError(f"Неподдерживаемый формат: {suffix}")
        if len(content) > self.max_output_characters:
            content = content[:self.max_output_characters]
        self._cache[key] = content
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_cache_entries:
            self._cache.popitem(last=False)
        return content
