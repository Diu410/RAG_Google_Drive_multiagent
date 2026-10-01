"""Thin adapter around the existing router, RAG, graph and verifier services.

RAG_PIPELINE is deliberately a single high-level action: the existing chain owns
retrieval and generation. No source chunks are copied to the execution journal.
"""
import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Callable
from assistant_models import AssistantError, ToolResult
from pydantic_models import QueryInput
from query_router import retrieval_plan, QueryStrategy


@dataclass
class KnowledgeServices:
    route: Callable
    history: Callable
    general_chat: Callable
    multi_chain: Callable
    rag_chain: Callable
    document_result: Callable
    verify: Callable


_GREETING_RE = re.compile(
    r"^\s*(?:привет(?:ик|ствую)?|здравствуй(?:те)?|добрый\s+(?:день|вечер|утро)|hello|hi|hey)\s*[!.?]*\s*$",
    re.IGNORECASE,
)


def _instant_reply(question: str) -> str | None:
    """Messages requiring no document lookup should never wait for an LLM."""
    if not _GREETING_RE.fullmatch(question):
        return None
    if re.search(r"\b(?:hello|hi|hey)\b", question, re.IGNORECASE):
        return "Hello! How can I help?"
    return "Привет! Чем могу помочь?"


class KnowledgeSkill:
    name = 'knowledge'
    description = 'Answer using the existing local RAG / GraphRAG pipeline.'
    input_schema = QueryInput.model_json_schema()
    permissions = ('knowledge:read', 'model:inference')

    def __init__(self, services: KnowledgeServices):
        self.services = services

    async def execute(self, inputs: dict, execution) -> ToolResult:
        query_input = QueryInput.model_validate(inputs)
        session_id = execution.task.session_id
        model = query_input.model.value
        retrieval_question = inputs.get('_retrieval_question') or query_input.question
        async with execution.action('ROUTE_QUERY') as action:
            route = await asyncio.to_thread(self.services.route, retrieval_question, model)
            action.output = {'strategy': route.strategy.value}
        plan = retrieval_plan(
            route, query_input.search_mode.value if query_input.search_mode is not None else None
        )
        use_graph = route.strategy in {QueryStrategy.GRAPH_QUERY, QueryStrategy.DEEP_ANALYSIS}
        history = inputs.get('_chat_history')
        if history is None:
            async with execution.action('LOAD_HISTORY'):
                history = await asyncio.to_thread(self.services.history, session_id)
        answer = _instant_reply(query_input.question) if route.strategy == QueryStrategy.GENERAL_CHAT else None
        if route.strategy == QueryStrategy.GENERAL_CHAT and answer is None:
            async with execution.action('GENERATE_RESPONSE'):
                arguments = (query_input.question, history, model, True) if query_input.live_mode else (
                    query_input.question, history, model,
                )
                answer = await asyncio.to_thread(self.services.general_chat, *arguments)
        result = ({"final": answer or "", "context": "", "strategy": route.strategy.value}
                  if route.strategy == QueryStrategy.GENERAL_CHAT else None)
        if result is None and query_input.use_multi_agent and not use_graph:
            try:
                async with execution.action('MULTI_AGENT_RAG'):
                    result = await asyncio.to_thread(
                        self.services.multi_chain(), retrieval_question, history,
                        model=model, search_mode=plan.mode, result_k=plan.result_k,
                        lexical_weight=plan.lexical_weight, query_variants=plan.query_variants,
                    )
                    result["strategy"] = route.strategy.value
            except Exception:
                logging.exception("Multi-agent failed; falling back to single RAG")
        if result is None:
            async with execution.action('RAG_PIPELINE', {'search_mode': plan.mode, 'use_graph': use_graph}) as action:
                chain = self.services.rag_chain(
                    model=model, search_mode=plan.mode, result_k=plan.result_k,
                    lexical_weight=plan.lexical_weight, query_variants=plan.query_variants,
                    use_graph=use_graph, graph_query=retrieval_question,
                    graph_entities=route.entities, graph_depth=2 if route.strategy == QueryStrategy.DEEP_ANALYSIS else 1,
                    live_mode=query_input.live_mode,
                )
                if chain is None:
                    raise AssistantError('RAG unavailable. Check model provider and Chroma.', 503)
                chain_response = await asyncio.to_thread(
                    chain.invoke, {"input": retrieval_question, "chat_history": history[-6:]}
                )
                result = {
                    "final": chain_response["answer"],
                    "context": "\n\n".join(d.page_content for d in chain_response.get("context", [])),
                    "results": [self.services.document_result(d) for d in chain_response.get("context", [])],
                    "strategy": route.strategy.value,
                }
                if not result["results"]:
                    result["final"] = "В локальных документах недостаточно информации для ответа."
                action.output = {'source_count': len(result['results'])}
            if use_graph:
                try:
                    async with execution.action('GRAPH_SEARCH'):
                        from graph_service import GraphService
                        graph_result = await asyncio.to_thread(lambda: GraphService().search(
                            retrieval_question, route.entities,
                            2 if route.strategy == QueryStrategy.DEEP_ANALYSIS else 1,
                        ))
                        result["graph"] = graph_result.as_dict() if graph_result.relations else None
                except Exception:
                    logging.exception("Graph metadata unavailable; answer used Hybrid Search fallback")
        verification = None
        if not query_input.live_mode and route.strategy != QueryStrategy.GENERAL_CHAT and result.get("results"):
            async with execution.action('VERIFY_RESPONSE'):
                result["final"], verification = await asyncio.to_thread(
                    self.services.verify,
                    query_input.question,
                    result["final"],
                    result["results"],
                    route.strategy.value,
                    model,
                )
        result["verification"] = verification
        return ToolResult(success=True, tool=self.name, summary="Knowledge query completed", data=result)
