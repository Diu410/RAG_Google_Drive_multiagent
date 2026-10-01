# api/pydantic_models.py
from pydantic import BaseModel, Field, field_validator
from enum import Enum
from datetime import datetime
from assistant_models import Action, AssistantEvent, TaskStatus
from settings import DEFAULT_CHAT_MODEL

class ModelName(str, Enum):
    FAST = "FAST"
    BALANCED = "BALANCED"
    REASONING = "REASONING"

class SearchMode(str, Enum):
    SEMANTIC = "semantic"
    LEXICAL = "lexical"
    HYBRID = "hybrid"

class SearchResult(BaseModel):
    document_id: int | str | None
    chunk_id: str
    text: str
    metadata: dict
    score: float
    found_by: list[str]

class VerificationClaim(BaseModel):
    text: str
    status: str
    source_chunk_ids: list[str] = Field(default_factory=list)
    basis: str | None = None
    evidence_quote: str | None = None

class ConfidenceComponents(BaseModel):
    retrieval: float = Field(ge=0.0, le=1.0)
    claim_support: float = Field(ge=0.0, le=1.0)
    source_agreement: float = Field(ge=0.0, le=1.0)

class VerificationResult(BaseModel):
    status: str
    claims: list[VerificationClaim] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    conflicting_claims: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    components: ConfidenceComponents | None = None
    warning: str | None = None

class QueryInput(BaseModel):
    question: str = Field(min_length=1, max_length=32000)
    session_id: str | None = Field(default=None, max_length=128)
    chat_id: str | None = Field(default=None, min_length=1, max_length=128)
    model: ModelName = Field(default=ModelName(DEFAULT_CHAT_MODEL))
    use_multi_agent: bool = Field(default=False)
    search_mode: SearchMode | None = Field(default=None)
    workspace_id: str | None = Field(default=None, max_length=128)
    live_mode: bool = Field(default=False)

    @field_validator('question')
    @classmethod
    def nonblank_question(cls, value):
        if not value.strip():
            raise ValueError('Question must not be blank')
        return value.strip()

class QueryResponse(BaseModel):
    answer: str
    session_id: str
    model: ModelName
    context: str
    draft: str = ""
    edited: str = ""
    enhanced: list[str] = Field(default_factory=list)
    results: list[SearchResult] = Field(default_factory=list)
    strategy: str | None = None
    verification: VerificationResult | None = None
    graph: dict | None = None
    plan_status: str | None = None
    pending_step: dict | None = None
    task_id: str | None = None
    task_status: TaskStatus | None = None
    actions: list[Action] = Field(default_factory=list)
    events: list[AssistantEvent] = Field(default_factory=list)

class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=32000)
    mode: SearchMode = Field(default=SearchMode.HYBRID)
    limit: int = Field(default=8, ge=1, le=100)

    @field_validator('query')
    @classmethod
    def nonblank_query(cls, value):
        if not value.strip():
            raise ValueError('Query must not be blank')
        return value.strip()

class DocumentInfo(BaseModel):
    id: int
    filename: str
    upload_timestamp: datetime
    can_reindex: bool = False

class DeleteFileRequest(BaseModel):
    file_id: int = Field(gt=0)

class ReindexRequest(BaseModel):
    file_ids: list[int] = Field(min_length=1, max_length=100)

    @field_validator('file_ids')
    @classmethod
    def valid_unique_ids(cls, values):
        if any(value <= 0 for value in values):
            raise ValueError('file_ids must be positive')
        return list(dict.fromkeys(values))

class UploadGDriveRequest(BaseModel):
    folder_url: str
