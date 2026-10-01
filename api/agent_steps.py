from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator
from pydantic_models import ModelName, SearchResult, VerificationResult
from settings import DEFAULT_CHAT_MODEL
from typing import List, Optional
from multi_agent_chain import draft_agent, edit_agent, enhance_agent, merge_agent
from query_router import query_router, retrieval_plan, QueryStrategy, RetrievalPlan
from langchain_utils import answer_general_chat
from verifier import verify_answer

router = APIRouter(prefix="/agent", tags=["agents"])


class TextRequest(BaseModel):
    @field_validator('input', 'draft', 'edited', 'question', 'answer', check_fields=False)
    @classmethod
    def meaningful_text(cls, value):
        if not value.strip() or len(value) > 32000:
            raise ValueError('Text must be nonblank and contain at most 32000 characters')
        return value


class DraftRequest(TextRequest):
    model: ModelName = ModelName(DEFAULT_CHAT_MODEL)
    input: str
    context: str = ""
    chat_history: List = Field(default_factory=list)


class EditRequest(TextRequest):
    model: ModelName = ModelName(DEFAULT_CHAT_MODEL)
    draft: str
    comment: str = ""


class EnhanceRequest(TextRequest):
    model: ModelName = ModelName(DEFAULT_CHAT_MODEL)
    edited: str
    comment: str = ""


class MergeRequest(TextRequest):
    model: ModelName = ModelName(DEFAULT_CHAT_MODEL)
    edited: str
    enhanced: List[str]
    comment: str = ""


class AgentStepResponse(BaseModel):
    result: str
    variants: List[str] = Field(default_factory=list)  # Для enhance — список вариантов
    strategy: str | None = None
    context: str = ""
    results: List[SearchResult] = Field(default_factory=list)
    verification: VerificationResult | None = None


class VerifyStepRequest(TextRequest):
    model: ModelName = ModelName(DEFAULT_CHAT_MODEL)
    question: str
    answer: str
    strategy: str
    results: List[SearchResult] = Field(default_factory=list)


@router.post("/draft", response_model=AgentStepResponse)
async def run_draft(req: DraftRequest):
    import asyncio
    route = await asyncio.to_thread(query_router.route, req.input, req.model.value)
    if route.strategy == QueryStrategy.GENERAL_CHAT and not req.context:
        answer = await asyncio.to_thread(answer_general_chat, req.input, req.chat_history, req.model.value)
        return AgentStepResponse(result=answer, strategy=route.strategy.value)
    plan = retrieval_plan(route) or RetrievalPlan()
    state = {
        "model": req.model.value,
        "input": req.input,
        "chat_history": req.chat_history,
        "context": req.context,
        "results": [],
        "draft": "", "edited": "", "enhanced": [], "final": "",
        "search_mode": plan.mode, "result_k": plan.result_k,
        "lexical_weight": plan.lexical_weight, "query_variants": plan.query_variants,
    }
    result = await asyncio.to_thread(draft_agent, state)
    return AgentStepResponse(result=result["draft"], strategy=route.strategy.value,
                             context=result.get("context", ""), results=result.get("results", []))


@router.post("/edit", response_model=AgentStepResponse)
async def run_edit(req: EditRequest):
    import asyncio
    # Передаём комментарий пользователя агенту
    state = {
        "model": req.model.value,
        "input": "", "chat_history": [], "context": "",
        "draft": req.draft,
        "user_comment": req.comment,  # агент учтёт это
        "edited": "", "enhanced": [], "final": ""
    }
    result = await asyncio.to_thread(edit_agent, state)
    return AgentStepResponse(result=result["edited"])


@router.post("/enhance", response_model=AgentStepResponse)
async def run_enhance(req: EnhanceRequest):
    import asyncio
    state = {
        "model": req.model.value,
        "input": "", "chat_history": [], "context": "", "draft": "",
        "edited": req.edited,
        "user_comment": req.comment,
        "enhanced": [], "final": ""
    }
    result = await asyncio.to_thread(enhance_agent, state)
    return AgentStepResponse(result=result["enhanced"][0], variants=result["enhanced"])


@router.post("/merge", response_model=AgentStepResponse)
async def run_merge(req: MergeRequest):
    import asyncio
    state = {
        "model": req.model.value,
        "input": "", "chat_history": [], "context": "", "draft": "",
        "edited": req.edited,
        "enhanced": req.enhanced,
        "user_comment": req.comment,
        "final": ""
    }
    result = await asyncio.to_thread(merge_agent, state)
    return AgentStepResponse(result=result["final"])


@router.post("/verify", response_model=AgentStepResponse)
async def run_final_verification(req: VerifyStepRequest):
    """Verify only the final text accepted in the manual workflow."""
    import asyncio
    answer, verification = await asyncio.to_thread(
        verify_answer, req.question, req.answer,
        [item.model_dump() for item in req.results], req.strategy, req.model.value,
    )
    return AgentStepResponse(result=answer, strategy=req.strategy, verification=verification)
