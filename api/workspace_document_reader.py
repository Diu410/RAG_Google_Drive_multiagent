"""Shared document extraction adapter for files discovered inside a workspace."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
from pathlib import Path
import threading
from extraction_limits import ExtractionLimitError, check_extraction_budget
from file_policy import is_secret_file, has_credential_content, FilePolicyError
from file_revision import file_revision


PLAIN_TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".json", ".jsonl", ".yaml", ".yml", ".toml",
    ".csv", ".tsv", ".log", ".xml", ".ini", ".cfg", ".conf", ".rtf",
    ".py", ".js", ".jsx", ".ts", ".tsx", ".css", ".scss", ".sql",
    ".sh", ".bat", ".cmd", ".ps1",
    ".rs", ".go", ".java", ".kt", ".kts", ".c", ".h", ".cpp", ".hpp",
    ".cs", ".rb", ".php", ".swift", ".vue", ".svelte", ".astro", ".mjs", ".cjs",
    ".ipynb", ".rst", ".tex", ".cmake", ".dockerfile", ".graphql", ".proto",
}
STRUCTURED_DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".html", ".htm"}
SUPPORTED_WORKSPACE_EXTENSIONS = PLAIN_TEXT_EXTENSIONS | STRUCTURED_DOCUMENT_EXTENSIONS
PLAIN_TEXT_NAMES = {'readme', 'license', 'licence', 'notice', 'authors', 'changelog',
                    'dockerfile', 'containerfile', 'makefile', '.gitignore', '.gitattributes', '.editorconfig'}


def is_plain_text_path(path):
    path = Path(path)
    return path.suffix.casefold() in PLAIN_TEXT_EXTENSIONS or path.name.casefold() in PLAIN_TEXT_NAMES


def is_workspace_document(path):
    return is_plain_text_path(path) or Path(path).suffix.casefold() in STRUCTURED_DOCUMENT_EXTENSIONS


@dataclass(frozen=True)
class SourceBlock:
    id: str
    start: int
    end: int
    page: int | None = None  # One-based PDF page; never an invented DOCX page.
    section: str | None = None
    extraction_method: str | None = None


@dataclass(frozen=True)
class ExtractedDocument:
    revision: str
    text: str
    blocks: tuple[SourceBlock, ...]
    plain_text: bool = False

    def locations(self, start: int, end: int) -> list[dict]:
        if not 0 <= start <= end <= len(self.text):
            raise ValueError('Диапазон выходит за пределы извлечённого текста.')
        output = []
        for block in self.blocks:
            if block.end <= start or block.start >= end:
                continue
            item = {'block_id': block.id, 'start': max(start, block.start), 'end': min(end, block.end)}
            if block.page is not None:
                item['page'] = block.page
            if block.section:
                item['section'] = block.section
            if block.extraction_method:
                item['extraction_method'] = block.extraction_method
            if self.plain_text:
                item['line_start'] = self.text.count('\n', 0, item['start']) + 1
                item['line_end'] = self.text.count('\n', 0, max(item['start'], item['end'] - 1)) + 1
            output.append(item)
        return output


class WorkspaceDocumentReader:
    """Extract text through the same structured loaders used by Document RAG."""

    def __init__(self, max_cache_entries: int = 32, max_output_characters: int = 2_000_000):
        self.max_cache_entries = max_cache_entries
        self.max_output_characters = max_output_characters
        self._cache: OrderedDict[tuple[str, str], ExtractedDocument] = OrderedDict()
        self._cache_lock = threading.RLock()

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
        return self.read_document(path).text

    def read_document(self, path: Path) -> ExtractedDocument:
        if is_secret_file(path) or has_credential_content(path):
            raise FilePolicyError('Файл с учётными данными исключён из доступа ассистента')
        revision = file_revision(path)
        key = (str(path), revision)
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached
        suffix = path.suffix.casefold()
        blocks = []
        if is_plain_text_path(path):
            content = self._decode_plain(path)
            blocks.append(SourceBlock(self._block_id(revision, 0, content), 0, len(content)))
        elif suffix in STRUCTURED_DOCUMENT_EXTENSIONS:
            parts, offset = [], 0
            for document in self._structured_documents(path):
                text = document.page_content.strip()
                if not text:
                    continue
                if parts:
                    offset += 2
                metadata = document.metadata
                page = metadata.get('page') if suffix == '.pdf' else None
                blocks.append(SourceBlock(self._block_id(revision, len(parts), text), offset, offset + len(text),
                    page=page + 1 if isinstance(page, int) and not isinstance(page, bool) and page >= 0 else None,
                    section=str(metadata['section']) if metadata.get('section') else None,
                    extraction_method=metadata.get('extraction_method')))
                parts.append(text)
                offset += len(text)
            content = "\n\n".join(parts)
        else:
            raise ValueError(f"Неподдерживаемый формат: {suffix}")
        if len(content) > self.max_output_characters:
            raise ExtractionLimitError('Extracted text exceeds size limit; partial content was not returned')
        check_extraction_budget(len(content))
        if file_revision(path) != revision:
            raise ValueError('Файл изменился во время чтения. Повторите запрос.')
        # A file revision replaces older cached extractions of the same path.
        result = ExtractedDocument(revision, content, tuple(blocks), is_plain_text_path(path))
        with self._cache_lock:
            for previous in list(self._cache):
                if previous[0] == str(path):
                    self._cache.pop(previous, None)
            self._cache[key] = result
            self._cache.move_to_end(key)
            while len(self._cache) > self.max_cache_entries:
                self._cache.popitem(last=False)
        return result

    @staticmethod
    def _block_id(revision, position, text):
        return hashlib.sha256(f'{revision}:{position}:{text}'.encode('utf-8')).hexdigest()[:24]
