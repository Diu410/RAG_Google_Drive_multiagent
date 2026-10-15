"""Typed deterministic plans for safe local-workspace operations."""
from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, Field
from command_intent import requests_file_open, negates_comparison, command_text


class WorkspaceOperation(str, Enum):
    FIND_LATEST = "FIND_LATEST_FILE"
    COMPARE_VERSIONS = "COMPARE_FILE_VERSIONS"
    OPEN_SELECTED = "OPEN_FILE"
    READ_SELECTED = "READ_FILE"
    LIST_FILE_DATES = "LIST_FILE_DATES"


class WorkspacePlanStep(BaseModel):
    id: str
    operation: WorkspaceOperation
    depends_on: list[str] = Field(default_factory=list)


class WorkspacePlan(BaseModel):
    objective: str
    target_query: str
    steps: list[WorkspacePlanStep]


_LATEST = re.compile(r"\b(?:последн\w*|актуальн\w*|свеж\w*)\b", re.IGNORECASE)
_DOCUMENT = re.compile(r"\b(?:верси\w*|файл\w*|отч[её]т\w*|документ\w*)\b", re.IGNORECASE)
_COMPARE = re.compile(r"\b(?:сравни\w*|изменил\w*|разниц\w*|предыдущ\w*)\b", re.IGNORECASE)
_OPEN = re.compile(r"\bоткрой\w*\b", re.IGNORECASE)
_CONTENT_CONTEXT = re.compile(
    r"\b(?:в\s+(?:контекст|содержим|текст)\w*|внутри\s+(?:документ|файл)\w*|"
    r"указан\w*|упомянут\w*)\b", re.IGNORECASE,
)
_FILE_DATES = re.compile(
    r"\b(?:системн\w*\s+дат\w*|метаданн\w*|свойств\w*\s+файл\w*|"
    r"дат\w*\s+(?:создани\w*|изменени\w*)\s+(?:файл|документ)\w*)\b",
    re.IGNORECASE,
)


class WorkspacePlanner:
    """Build an executable dependency graph for the operations currently supported."""

    @staticmethod
    def metadata_plan(objective: str, target_query: str | None = None) -> WorkspacePlan:
        return WorkspacePlan(
            objective=objective,
            target_query=target_query or objective,
            steps=[WorkspacePlanStep(id="list_file_dates", operation=WorkspaceOperation.LIST_FILE_DATES)],
        )

    def plan(self, text: str, contextual_text: str | None = None) -> WorkspacePlan | None:
        command = (contextual_text or text).strip()
        action_text = command_text(command)
        if _FILE_DATES.search(action_text) and not _CONTENT_CONTEXT.search(command):
            return self.metadata_plan(text, command)
        has_document = bool(_DOCUMENT.search(command))
        wants_compare = (bool(_COMPARE.search(action_text)) and has_document and not negates_comparison(text)
                         and bool(re.search(r'\b(?:верси\w*|предыдущ\w*|изменил\w*)\b', command, re.IGNORECASE)))
        wants_latest = bool(_LATEST.search(action_text)) and has_document
        # Device actions require an explicit verb in the user's actual message;
        # a contextualizer is allowed to resolve the target, not invent consent.
        wants_open = requests_file_open(text)
        read_commands = command_text(text)
        wants_read = bool(re.search(r'\b(?:прочитай|прочти|покажи\s+содержимое)\b', read_commands))
        if re.search(r'\bне\s+(?:прочитай|прочти|читай|показывай)\b', read_commands):
            wants_read = False
        if not (wants_latest or wants_compare):
            return None

        explicit_pair = len(re.findall(r'(?<![\w.])\d+(?:\.\d+)*(?![\w.])', command)) == 2
        steps = ([] if wants_compare and explicit_pair and not wants_latest else
                 [WorkspacePlanStep(id="find_latest", operation=WorkspaceOperation.FIND_LATEST)])
        dependency = "find_latest" if steps else None
        if wants_compare:
            steps.append(WorkspacePlanStep(
                id="compare_versions", operation=WorkspaceOperation.COMPARE_VERSIONS,
                depends_on=[dependency] if dependency else [],
            ))
            dependency = "compare_versions"
        if wants_open:
            steps.append(WorkspacePlanStep(
                id="open_selected", operation=WorkspaceOperation.OPEN_SELECTED,
                depends_on=[dependency],
            ))
        if wants_read:
            steps.append(WorkspacePlanStep(
                id="read_selected", operation=WorkspaceOperation.READ_SELECTED,
                depends_on=[dependency],
            ))
        return WorkspacePlan(objective=text, target_query=command, steps=steps)
