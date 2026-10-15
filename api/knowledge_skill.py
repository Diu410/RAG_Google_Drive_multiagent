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
from query_router import retrieval_plan, QueryStrategy, RouteDecision


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
            route = (RouteDecision(strategy=QueryStrategy.GRAPH_QUERY, confidence=1.0)
                     if inputs.get('_graph_focus') else
                     await asyncio.to_thread(self.services.route, retrieval_question, model))
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
        if result is None:
            async with execution.action('RAG_PIPELINE', {'search_mode': plan.mode, 'use_graph': use_graph,
                                                       'automatic_graph': True}) as action:
                focus_arguments = {'graph_focus': inputs['_graph_focus']} if inputs.get('_graph_focus') else {}
                chain = self.services.rag_chain(
                    model=model, search_mode=plan.mode, result_k=plan.result_k,
                    lexical_weight=plan.lexical_weight, query_variants=plan.query_variants,
                    use_graph=use_graph, graph_query=retrieval_question,
                    graph_entities=route.entities, graph_depth=2 if route.strategy == QueryStrategy.DEEP_ANALYSIS else 1,
                    live_mode=query_input.live_mode,
                    **focus_arguments,
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
                if use_graph:
                    result["graph"] = chain_response.get("graph")
                if not result["results"]:
                    result["final"] = "В локальных документах недостаточно информации для ответа."
                action.output = {'source_count': len(result['results']),
                                 'graph_fast_path': any(source.get('metadata', {}).get('graph_fast_path')
                                                        for source in result['results']),
                                 'graph_source_count': sum('semantic_graph' in source.get('found_by', [])
                                                           for source in result['results'])}
        if query_input.use_multi_agent and not query_input.live_mode and result.get('results'):
            from answer_editing import edit_sourced_answer
            editing_context = '\n\n'.join(
                f"[Источник: {(source.get('metadata') or {}).get('source', source.get('document_id', ''))}; "
                f"chunk: {source.get('chunk_id', '')}]\n{source.get('text', '')}" for source in result['results'])
            revised = await edit_sourced_answer(query_input.question, result['final'], editing_context, model, execution)
            result.update(revised)
        verification = None
        if not query_input.live_mode and route.strategy != QueryStrategy.GENERAL_CHAT and result.get("results"):
            draft = result['final']
            async with execution.action('VERIFY_RESPONSE') as action:
                result["final"], verification = await asyncio.to_thread(
                    self.services.verify,
                    query_input.question,
                    result["final"],
                    result["results"],
                    route.strategy.value,
                    model,
                )
                action.output = {'checked_answer': result['final'],
                    'verification': verification.model_dump(mode='json') if verification else None}
            if result['final'] != draft:
                result.setdefault('draft', draft)
        result["verification"] = verification
        return ToolResult(success=True, tool=self.name, summary="Knowledge query completed", data=result)
