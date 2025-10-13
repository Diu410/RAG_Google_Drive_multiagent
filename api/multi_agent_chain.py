# multi_agent_chain.py
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, List, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage
from hybrid_context_extractor import HybridRetriever
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from typing import Optional
import operator
from dotenv import load_dotenv
import os

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# State для графа (что передаётся между агентами)
class AgentState(TypedDict):
    input: str  # Исходный query
    chat_history: List[BaseMessage]  # История чата
    context: str  # RAG-контекст (из retriever)
    draft: str  # Выход Draft Agent
    edited: str  # Выход Edit
    enhanced: List[str]  # Выходы Enhance (несколько вариантов)
    final: str  # Выход Merge


# Инициализация LLM и retriever (твой HybridRetriever)
def get_llm(model="gemini-2.5-pro", temperature=0.0):
    return ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=GOOGLE_API_KEY)


def get_retriever():
    from chroma_utils import retriever
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = get_llm("gemini-2.5-flash")  # Для history-aware
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Reformulate the question to stand alone, using chat history if needed. Do not answer."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    base_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    return HybridRetriever(base_retriever)


# Промпты для агентов (на основе твоего qa_prompt)
def get_draft_prompt():
    return ChatPromptTemplate.from_template(
        "Generate a raw draft based on context and query. Keep it simple and complete.\nContext: {context}\nQuery: {input}"
    )


def get_edit_prompt():
    return ChatPromptTemplate.from_template(
        "Edit the draft for clarity, grammar, structure. Fix errors, make coherent.\nDraft: {draft}"
    )


def get_enhance_prompt():
    return ChatPromptTemplate.from_template(
        "Enhance the edited text for engagement, depth, creativity. Add details if needed.\nEdited: {edited}"
    )


def get_merge_prompt():
    return ChatPromptTemplate.from_template(
        "Merge the best parts from edited and enhanced versions. Prioritize coherence and quality. Output final text.\nEdited: {edited}\nEnhanced: {enhanced}"
    )


# Агенты как функции (узлы графа)
def draft_agent(state: AgentState):
    print("DEBUG multi: Starting draft_agent...")
    retriever = get_retriever()
    context_docs = retriever(state["input"], state["chat_history"])
    context = "\n\n".join([doc.page_content for doc in context_docs])
    print(f"DEBUG multi: Context ready ({len(context)} chars)")
    llm = get_llm("gemini-2.5-flash", 0.7)  # Быстрый, креативный
    chain = get_draft_prompt() | llm | StrOutputParser()
    draft = chain.invoke({"input": state["input"], "context": context})
    print("DEBUG multi: Draft done!")
    return {"draft": draft, "context": context}


def edit_agent(state: AgentState):
    llm = get_llm("gemini-2.5-pro", 0.1)  # Точный редактор
    print("DEBUG multi: Starting edit_agent...")
    chain = get_edit_prompt() | llm | StrOutputParser()
    edited = chain.invoke({"draft": state["draft"]})
    print("DEBUG multi: Edit done!")
    return {"edited": edited}


def enhance_agent(state: AgentState):
    print("DEBUG multi: Starting enhance_agent...")
    llm = get_llm("gemini-2.5-pro", 0.3)
    chain = get_enhance_prompt() | llm | StrOutputParser()
    enhanced1 = chain.invoke({"edited": state["edited"]})
    enhanced2 = chain.invoke({"edited": state["edited"] + " Make it more creative."})
    enhanced = [enhanced1, enhanced2]
    print(f"DEBUG multi: Enhance done! {len(enhanced)} variants, len1 = {len(enhanced1)} chars")
    return {"enhanced": enhanced}

def merge_agent(state: AgentState):
    print("DEBUG multi: Starting merge_agent...")
    llm = get_llm("gemini-2.5-pro", 0.0)
    enhanced_str = "\n--- Variant 1 ---\n".join(state["enhanced"])  # <-- Чекни, list ли enhanced
    if not state["enhanced"]:
        print("DEBUG multi: ERROR: enhanced is empty!")
        return {"final": state["edited"]}  # Fallback
    chain = get_merge_prompt() | llm | StrOutputParser()
    final = chain.invoke({"edited": state["edited"], "enhanced": enhanced_str})
    print(f"DEBUG multi: Merge done! Final len = {len(final)} chars")
    return {"final": final}


# Построение графа
def create_multi_agent_graph():
    workflow = StateGraph(AgentState)

    # Узлы
    workflow.add_node("draft", draft_agent)
    workflow.add_node("edit", edit_agent)
    workflow.add_node("enhance", enhance_agent)
    workflow.add_node("merge", merge_agent)

    # Ребра (последовательно, enhance параллельно, но здесь sequentially)
    workflow.set_entry_point("draft")
    workflow.add_edge("draft", "edit")
    workflow.add_edge("edit", "enhance")
    workflow.add_edge("enhance", "merge")
    workflow.add_edge("merge", END)

    # Компиляция
    graph = workflow.compile()
    return graph


# Основная функция run (интеграция в main.py)
def run_multi_agent(input_query: str, chat_history: List[BaseMessage]):
    graph = create_multi_agent_graph()
    initial_state = {
        "input": input_query,
        "chat_history": chat_history,
        "draft": "",
        "edited": "",
        "enhanced": [],
        "final": ""
    }
    result = graph.invoke(initial_state)
    return {
        "final": result["final"],
        "draft": result["draft"],
        "edited": result["edited"],
        "enhanced": result["enhanced"],
        "context": result["context"]  # Для expander
    }


# Интеграция в langchain_utils.py (get_multi_agent_chain)
def get_multi_agent_chain():
    if not GOOGLE_API_KEY:
        return None
    # Возвращает функцию run_multi_agent
    return run_multi_agent