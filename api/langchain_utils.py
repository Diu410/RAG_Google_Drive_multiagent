# langchain_utils.py - КОД ДЛЯ GEMINI
import os
# Меняем ChatOllama на Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from hybrid_context_extractor import HybridRetriever
from chroma_utils import retriever
from typing import Optional
from dotenv import load_dotenv
from chroma_utils import vectorstore
# --- ПРИНУДИТЕЛЬНАЯ ЗАГРУЗКА И ПРОВЕРКА КЛЮЧА ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# --------------------------------------------------

def debug_chroma_search(query):
    """Дебаг: Прямой поиск в Chroma без wrapper."""
    if vectorstore is None:
        print("Chroma не init!")
        return
    try:
        docs = vectorstore.similarity_search(query, k=5)
        print(f"Direct Chroma search для '{query}': {len(docs)} результатов.")
        if docs:
            print(f"Первый матч: {docs[0].page_content[:150]}...")
        else:
            print("Direct search 0! Пустой DB или плохой match.")
    except Exception as e:
        print(f"Error in direct search: {e}")

# Убедимся, что retriever существует
current_retriever: Optional = retriever

# Инструкция для LLM (остается прежней)
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

# Формат промта для LLM (остается прежним)
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Формат Финального промта для LLM (остается прежним)
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI script assistant. Use the following context to answer the user's question. "
               "If the query asks for lists (e.g. 'all variants', 'all names'), enumerate ALL items from the context without summarizing or omitting. "
               "Be exhaustive and accurate."),  # <-- Добавь инструкцию для списков
    ("system", "Context: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])


def get_rag_chain(model="gemini-2.5-flash"):  # Изменяем модель по умолчанию
    """Создает и возвращает полную RAG-цепочку."""

    if current_retriever is None:
        return None

    try:
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is not available for LLM initialization.")

        # Используем Gemini LLM С ЯВНОЙ ПЕРЕДАЧЕЙ КЛЮЧА
        llm = ChatGoogleGenerativeAI(model=model, temperature=0.0)

        base_retriever = create_history_aware_retriever(llm, current_retriever, contextualize_q_prompt)

        def debug_chroma_search(query):
            from chroma_utils import vectorstore  # Импорт для доступа
            if vectorstore is None:
                print("Chroma не init!")
                return
            try:
                docs = vectorstore.similarity_search(query, k=5)  # Прямой вызов, без hybrid/history
                print(f"Direct Chroma search для '{query}': {len(docs)} результатов.")
                if docs:
                    print(f"Первый матч: {docs[0].page_content[:150]}... (score не выводится, но если >0 — ок)")
                else:
                    print("Direct search тоже 0! Проблема в embeddings или пустой DB.")
            except Exception as e:
                print(f"Error in direct search: {e}")

        hybrid_retriever_obj = HybridRetriever(base_retriever)
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(hybrid_retriever_obj, question_answer_chain)

        return rag_chain

    except Exception as e:
        print(f"Критическая ошибка при инициализации RAG-цепочки с Gemini: {str(e)}")
        return None