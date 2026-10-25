"""Reviewable local document changes, revision checks and text conversion.

Proposals contain exact bytes and expire; inference never receives a write
capability. Only apply() changes files. Existing text edits keep a byte backup.
"""
from __future__ import annotations

from collections import OrderedDict
import difflib
import hashlib
from html import escape
import io
import os
from pathlib import Path, PureWindowsPath
import re
import threading
import time
import uuid
import zipfile

from assistant_models import AssistantError
from file_policy import is_secret_file, FilePolicyError
from file_revision import file_revision


EDITABLE = {'.txt', '.md', '.markdown', '.csv', '.tsv'}
EXPORTS = EDITABLE | {'.html', '.docx', '.pdf'}
BACKUPS = '.fortuna-backups'
MAX_TEXT_BYTES = 1024 * 1024
MAX_OUTPUT_BYTES = 20 * 1024 * 1024


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _html(text):
    # Source HTML/Markdown is escaped: no remote resources, scripts or local URLs.
    return ('<!doctype html><html><head><meta charset="utf-8"></head><body>' +
        ''.join('<p style="white-space:pre-wrap">' + escape(line) + '</p>'
                for line in text.split('\n')) + '</body></html>')


def convert_text(text, suffix):
    if len(text.encode('utf-8')) > MAX_TEXT_BYTES:
        raise ValueError('Для конвертации допускается не более 1 МиБ извлечённого текста.')
    if re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', text):
        raise ValueError('Документ содержит неподдерживаемые управляющие символы.')
    if suffix in EDITABLE:
        return text.encode('utf-8')
    if suffix == '.html':
        return _html(text).encode('utf-8')
    if suffix == '.docx':
        output = io.BytesIO()
        body = ''.join('<w:p><w:r><w:t xml:space="preserve">' + escape(line) +
                       '</w:t></w:r></w:p>' for line in text.split('\n'))
        with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('[Content_Types].xml',
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                '<Default Extension="xml" ContentType="application/xml"/>'
                '<Override PartName="/word/document.xml" '
                'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>')
            archive.writestr('_rels/.rels',
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
                'Target="word/document.xml"/></Relationships>')
            archive.writestr('word/document.xml', '<?xml version="1.0" encoding="UTF-8"?>'
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                '<w:body>' + body + '<w:sectPr/></w:body></w:document>')
        return output.getvalue()
    if suffix == '.pdf':
        import pymupdf
        output = io.BytesIO()
        story = pymupdf.Story(html=_html(text))
        writer = pymupdf.DocumentWriter(output)
        page = pymupdf.paper_rect('a4')
        started = time.monotonic()
        try:
            for _ in range(100):
                device = writer.begin_page(page)
                more, _ = story.place(page + (40, 40, -40, -40))
                story.draw(device)
                writer.end_page()
                if not more:
                    break
                if time.monotonic() - started > 15:
                    raise ValueError('Конвертация PDF превысила 15 секунд.')
            else:
                raise ValueError('Конвертация PDF превысила 100 страниц.')
        finally:
            writer.close()
        return output.getvalue()
    raise ValueError('Поддерживаются TXT, MD, CSV, TSV, HTML, DOCX и PDF.')


class WorkspaceDocuments:
    editable_formats = EDITABLE
    output_formats = EXPORTS

    def __init__(self, filesystem):
        self.filesystem = filesystem
        self._proposals = OrderedDict()
        self._lock = threading.RLock()

    def _root(self, workspace_id):
        return Path(self.filesystem.repository.get(workspace_id).root_path).resolve(strict=True)

    def _target(self, root, relative):
        if not isinstance(relative, str) or not relative or len(relative) > 500:
            raise ValueError('Укажите относительный путь нового документа.')
        # Reject NTFS streams, devices and platform-dependent path spellings too.
        if ('\\' in relative or ':' in relative or PureWindowsPath(relative).is_absolute()
                or any(ord(c) < 32 for c in relative)):
            raise FilePolicyError('Используйте относительный путь с разделителем /.')
        parts = relative.split('/')
        if any(not part or part in {'.', '..'} or part.startswith('.') or part.endswith((' ', '.'))
               or is_secret_file(part) or PureWindowsPath(part).is_reserved()
               or part.casefold() in self.filesystem.ignored_directory_names for part in parts):
            raise FilePolicyError('Этот путь недоступен для записи документов.')
        target = root.joinpath(*parts)
        parent = self.filesystem._resolve(root, '/'.join(parts[:-1]))
        if not parent.is_dir():
            raise ValueError('Родительская папка должна существовать.')
        if target.exists() or target.is_symlink():
            self.filesystem._resolve(root, relative)
        if target.suffix.casefold() not in self.output_formats:
            raise ValueError('Этот формат недоступен для записи документов.')
        return target

    def read_lines(self, workspace_id, path, start=1, end=200):
        if not 1 <= start <= end or end - start >= 500:
            raise ValueError('Укажите диапазон до 500 строк, начиная с 1.')
        _, relative, text = self.filesystem.read_text_file(workspace_id, path)
        lines = text.splitlines()
        result, characters = [], 0
        for number in range(start, min(end, len(lines)) + 1):
            line = lines[number - 1]
            if characters + len(line) > 32_000:
                break
            result.append({'line': number, 'text': line})
            characters += len(line)
        last = result[-1]['line'] if result else start - 1
        return {'path': relative, 'lines': result, 'total_lines': len(lines),
                'start': start, 'end': last, 'truncated': last < min(end, len(lines)),
                'partial': start > 1 or last < len(lines),
                'line_basis': 'extracted text; source lines for plain text'}

    def _remember(self, workspace_id, path, raw, before=None, source=None, note=''):
        if len(raw) > MAX_OUTPUT_BYTES:
            raise ValueError('Результат превышает 20 МиБ.')
        root = self._root(workspace_id)
        target = self._target(root, path)
        if before is None and target.exists():
            raise AssistantError('Файл назначения уже существует. Укажите новое имя.', 409)
        identifier = uuid.uuid4().hex
        if before is not None:
            if _digest(target.read_bytes()) != before:
                raise AssistantError('Файл изменился до предпросмотра. Повторите запрос.', 409)
            old = self.filesystem.document_reader._decode_plain(target)
            new = raw.decode('utf-8-sig') if raw.startswith(b'\xef\xbb\xbf') else raw.decode(
                'utf-8' if _is_utf8(raw) else 'cp1251')
            changes = list(difflib.unified_diff(old.splitlines(), new.splitlines(),
                fromfile=path, tofile=path, lineterm=''))
        else:
            changes = []
        preview = '\n'.join(changes[:200])[:16_000]
        stat = root.stat()
        proposal = {'workspace_id': workspace_id, 'path': path, 'raw': raw,
            'before': before, 'source': source, 'created': time.monotonic(),
            'root_identity': (stat.st_dev, stat.st_ino),
            'preview': {'proposal_id': identifier, 'path': path, 'operation': 'edit' if before is not None else 'create',
                'bytes': len(raw), 'diff': preview, 'diff_truncated': len(changes) > 200 or len('\n'.join(changes)) > 16_000,
                'note': note, 'expires_in_seconds': 600}}
        with self._lock:
            self._proposals[identifier] = proposal
            while len(self._proposals) > 24 or sum(len(p['raw']) for p in self._proposals.values()) > 32 * 1024 * 1024:
                self._proposals.popitem(last=False)
        return proposal['preview']

    def prepare_create(self, workspace_id, path, text):
        if not isinstance(text, str):
            raise ValueError('text должен быть строкой.')
        suffix = Path(path).suffix.casefold()
        raw = convert_text(text, '.txt' if suffix in self.editable_formats else suffix)
        return self._remember(workspace_id, path, raw, note='Создание нового документа.')

    def prepare_edit(self, workspace_id, path, old, new, expected_count=1, *, expected_revision=None):
        root = self._root(workspace_id)
        target = self._target(root, path)
        if target.suffix.casefold() not in self.editable_formats:
            raise ValueError('Точная замена доступна для TXT, MD, CSV и TSV. Конвертируйте документ в текст.')
        if not isinstance(old, str) or not old or not isinstance(new, str) or not 1 <= expected_count <= 100:
            raise ValueError('Укажите непустой исходный текст, замену и ожидаемое число совпадений от 1 до 100.')
        if target.stat().st_size > MAX_TEXT_BYTES:
            raise ValueError('Редактируемый файл превышает 1 МиБ.')
        before = target.read_bytes()
        if expected_revision is not None and _digest(before) != expected_revision:
            raise AssistantError('Файл изменился после чтения. Прочитайте текущую версию.', 409)
        text = self.filesystem.document_reader._decode_plain(target)
        if target.read_bytes() != before:
            raise AssistantError('Файл изменился во время чтения. Повторите предпросмотр.', 409)
        if text.count(old) != expected_count:
            raise AssistantError(f'Найдено совпадений: {text.count(old)}; ожидалось: {expected_count}. Уточните замену.', 409)
        updated = text.replace(old, new)
        if updated == text:
            raise ValueError('Замена не меняет документ.')
        encoding = 'utf-8-sig' if before.startswith(b'\xef\xbb\xbf') else 'utf-8' if _is_utf8(before) else 'cp1251'
        raw = updated.encode(encoding)
        if len(raw) > MAX_TEXT_BYTES:
            raise ValueError('Изменённый файл превышает 1 МиБ.')
        return self._remember(workspace_id, path, raw, before=_digest(before), note='Перед заменой сохраняется резервная копия.')

    def prepare_convert(self, workspace_id, source_path, target_path):
        root = self._root(workspace_id)
        source = self.filesystem._resolve(root, source_path)
        revision = file_revision(source)
        _, _, text = self.filesystem.read_text_file(workspace_id, source_path)
        raw = convert_text(text, Path(target_path).suffix.casefold())
        if file_revision(source) != revision:
            raise AssistantError('Исходник изменился. Повторите конвертацию.', 409)
        return self._remember(workspace_id, target_path, raw,
            source=(source.relative_to(root).as_posix(), revision),
            note='Конвертация извлечённого текста. Оформление, изображения и структура таблиц не сохраняются.')

    def apply(self, workspace_id, proposal_id):
        with self._lock:
            proposal = self._proposals.get(proposal_id)
            if not proposal or proposal['workspace_id'] != workspace_id:
                raise AssistantError('Предпросмотр не найден для этой папки. Подготовьте его заново.', 409)
            if time.monotonic() - proposal['created'] > 600:
                self._proposals.pop(proposal_id)
                raise AssistantError('Срок предпросмотра истёк. Подготовьте его заново.', 409)
            root = self._root(workspace_id)
            stat = root.stat()
            if (stat.st_dev, stat.st_ino) != proposal['root_identity']:
                raise AssistantError('Рабочая папка изменилась. Подготовьте предпросмотр заново.', 409)
            target = self._target(root, proposal['path'])
            if proposal['source']:
                source, revision = proposal['source']
                if file_revision(self.filesystem._resolve(root, source)) != revision:
                    raise AssistantError('Исходник изменился после конвертации. Повторите предпросмотр.', 409)
            backup = None
            if proposal['before'] is not None:
                if not target.is_file() or _digest(target.read_bytes()) != proposal['before']:
                    raise AssistantError('Файл изменился после предпросмотра. Изменения не применены.', 409)
                backup_dir = root / BACKUPS
                if not backup_dir.exists():
                    backup_dir.mkdir()
                self.filesystem._resolve(root, BACKUPS)
                backup = backup_dir / (uuid.uuid4().hex + target.suffix)
                with backup.open('xb') as stream:
                    stream.write(target.read_bytes())
                    stream.flush()
                    os.fsync(stream.fileno())
                if _digest(backup.read_bytes()) != proposal['before']:
                    raise AssistantError('Файл изменился во время подготовки копии. Повторите предпросмотр.', 409)
            elif target.exists():
                raise AssistantError('Файл назначения уже существует. Изменения не применены.', 409)
            temporary = target.parent / ('.fortuna-' + uuid.uuid4().hex + '.tmp')
            try:
                with temporary.open('xb') as stream:
                    stream.write(proposal['raw'])
                    stream.flush()
                    os.fsync(stream.fileno())
                target = self._target(root, proposal['path'])
                if proposal['before'] is None:
                    # Atomic no-clobber publication, including concurrent writers.
                    os.link(temporary, target)
                else:
                    if _digest(target.read_bytes()) != proposal['before']:
                        raise AssistantError('Файл изменился перед записью. Повторите предпросмотр.', 409)
                    os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
            self._proposals.pop(proposal_id)
            from workspace import workspace_scan_cache
            cache = workspace_scan_cache.get()
            if cache is not None:
                cache.clear()
            return {'path': proposal['path'], 'bytes': len(proposal['raw']),
                    'operation': proposal['preview']['operation'],
                    'backup_path': backup.relative_to(root).as_posix() if backup else None,
                    'note': proposal['preview']['note'], 'applied': True}


def _is_utf8(raw):
    try:
        raw.decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False
