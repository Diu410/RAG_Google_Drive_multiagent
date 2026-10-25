"""Bounded, non-executing repository discovery and literal source search."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from pathspec import GitIgnoreSpec


class RepositoryIgnore:
    """Per-scan nested .gitignore rules, relative to their own directories.

    No global Git config, hooks, external processes or paths outside the root.
    Explicit file reads remain possible; these rules govern discovery only.
    """
    def __init__(self, root, is_link):
        self.root, self.is_link, self.specs = root, is_link, {}

    def _load(self, directory):
        if directory not in self.specs:
            path = directory / '.gitignore'
            if path.exists() and not self.is_link(path):
                if path.stat().st_size > 256 * 1024:
                    raise ValueError('.gitignore exceeds 256 KiB')
                lines = path.read_text(encoding='utf-8-sig').splitlines()
                if len(lines) > 10_000:
                    raise ValueError('.gitignore exceeds 10000 lines')
                self.specs[directory] = GitIgnoreSpec.from_lines(lines)
            else:
                self.specs[directory] = None
        return self.specs[directory]

    def ignored(self, path, is_directory=False):
        ignored = False
        directories = [self.root]
        relative = path.parent.relative_to(self.root)
        current = self.root
        for part in relative.parts:
            current = current / part
            directories.append(current)
        for directory in directories:
            spec = self._load(directory)
            if spec:
                result = spec.check_file(path.relative_to(directory).as_posix() + ('/' if is_directory else ''))
                if result.include is not None:
                    ignored = result.include
        return ignored


class RepositoryTools:
    def __init__(self, filesystem):
        self.filesystem = filesystem

    def overview(self, workspace_id):
        sync = self.filesystem.sync_index(workspace_id)
        files = self.filesystem.repository.indexed_files(workspace_id)
        readable = [item for item in files if item.readable]
        return {'workspace_id': workspace_id, 'files': len(files), 'readable_files': len(readable),
            'extensions': dict(Counter(item.extension or '(no extension)' for item in readable).most_common(40)),
            'paths': [item.relative_path for item in readable[:160]],
            'paths_truncated': len(readable) > 160, 'truncated': sync.truncated,
            'errors': sync.errors, 'ignore_policy': 'nested .gitignore + built-in exclusions'}

    def search(self, workspace_id, query, directory='', max_results=50):
        if not isinstance(query, str) or not query.strip() or len(query) > 300:
            raise ValueError('Введите искомый текст длиной от 1 до 300 символов.')
        if not isinstance(max_results, int) or not 1 <= max_results <= 100:
            raise ValueError('Количество результатов должно быть от 1 до 100.')
        workspace = self.filesystem.repository.get(workspace_id)
        root = Path(workspace.root_path).resolve(strict=True)
        scope = self.filesystem._resolve(root, directory)
        if not scope.is_dir():
            raise ValueError('Область поиска должна быть папкой.')
        prefix = scope.relative_to(root).as_posix()
        prefix = '' if prefix == '.' else prefix + '/'
        sync = self.filesystem.sync_index(workspace_id)
        candidates = [item for item in self.filesystem.repository.indexed_files(workspace_id)
                      if item.readable and item.relative_path.startswith(prefix)]
        matches, errors, searched = [], [], 0
        truncated = sync.truncated or len(candidates) > 2000
        needle = query.casefold()
        for item in candidates[:2000]:
            try:
                _, _, content = self.filesystem.read_text_file(workspace_id, item.relative_path)
            except (OSError, ValueError) as exc:
                errors.append({'path': item.relative_path, 'error': type(exc).__name__})
                continue
            searched += 1
            for number, line in enumerate(content.splitlines(), 1):
                if needle in line.casefold():
                    if len(matches) == max_results:
                        truncated = True
                        break
                    matches.append({'path': item.relative_path, 'line': number,
                                    'text': line[:600], 'line_truncated': len(line) > 600})
            if truncated and len(matches) == max_results:
                break
        return {'workspace_id': workspace_id, 'query': query, 'directory': prefix.rstrip('/'),
                'matches': matches, 'searched_files': searched, 'candidate_files': len(candidates),
                'truncated': truncated, 'errors': errors, 'scan_errors': sync.errors,
                'line_basis': 'extracted text; source lines for plain text'}
