"""Explicit source selection, separate from navigation and write authority."""
from __future__ import annotations

import os
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def source_path(value: str) -> str:
    if (len(value) > 500 or '\\' in value or ':' in value
            or any(ord(c) < 32 for c in value)
            or value.startswith('/')
            or any(part in {'', '.', '..'} for part in value.split('/'))):
        raise ValueError('Укажите относительный путь внутри рабочей папки с разделителем /.')
    return value


class WorkspaceSourceScope(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    kind: Literal['workspace', 'directory', 'files'] = 'workspace'
    directory: str = Field(default='', max_length=500)
    files: tuple[str, ...] = Field(default=(), max_length=100)

    @model_validator(mode='after')
    def coherent_selection(self):
        if self.kind == 'workspace':
            if self.directory or self.files:
                raise ValueError('Область всей папки не принимает дополнительные пути.')
        elif self.kind == 'directory':
            if not self.directory or self.files:
                raise ValueError('Выберите одну подпапку без списка файлов.')
            source_path(self.directory)
        else:
            if self.directory or not self.files:
                raise ValueError('Выберите от одного до ста файлов без подпапки.')
            for path in self.files:
                source_path(path)
            if len({os.path.normcase(path) for path in self.files}) != len(self.files):
                raise ValueError('Список источников содержит повторяющиеся файлы.')
        return self

    def contains(self, path: str) -> bool:
        path = os.path.normcase(path).replace('\\', '/')
        if self.kind == 'workspace':
            return True
        if self.kind == 'files':
            return path in {os.path.normcase(item).replace('\\', '/') for item in self.files}
        directory = os.path.normcase(self.directory).replace('\\', '/')
        return directory in [str(parent) for parent in PurePosixPath(path).parents]

    def fingerprint(self) -> dict:
        # Selection order does not change the source set or invalidate a retry.
        return {'kind': self.kind, 'directory': self.directory,
                'files': sorted(self.files, key=os.path.normcase)}
