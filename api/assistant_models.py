"""Transport-independent contracts for assistant execution (not message history)."""
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskStatus(str, Enum):
    PENDING = 'PENDING'
    RUNNING = 'RUNNING'
    COMPLETED = 'COMPLETED'
    FAILED = 'FAILED'
    CANCELLED = 'CANCELLED'


class Session(BaseModel):
    id: str
    chat_id: str | None = None
    created_at: str
    updated_at: str
    status: str = 'ACTIVE'
    metadata: dict[str, Any] = Field(default_factory=dict)


class Task(BaseModel):
    id: str
    session_id: str
    user_input: str
    task_type: str = 'QUERY'
    status: TaskStatus = TaskStatus.PENDING
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Action(BaseModel):
    id: str
    task_id: str
    action_type: str
    status: TaskStatus = TaskStatus.PENDING
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None


class ToolResult(BaseModel):
    success: bool
    tool: str
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class EventType(str, Enum):
    TASK_STARTED = 'TASK_STARTED'
    ACTION_STARTED = 'ACTION_STARTED'
    ACTION_COMPLETED = 'ACTION_COMPLETED'
    ACTION_FAILED = 'ACTION_FAILED'
    ACTION_CANCELLED = 'ACTION_CANCELLED'
    TASK_COMPLETED = 'TASK_COMPLETED'
    TASK_FAILED = 'TASK_FAILED'
    TASK_CANCELLED = 'TASK_CANCELLED'


class AssistantEvent(BaseModel):
    type: EventType
    task_id: str
    session_id: str
    action_id: str | None = None
    timestamp: str = Field(default_factory=utc_now)


class AssistantError(Exception):
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code
