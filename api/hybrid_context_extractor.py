# hybrid_context_extractor.py
import re
import os
from typing import List, Dict, Optional, Union
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv  # <-- Добавь импорт

# Загрузи env в начале модуля
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Инициализация с явным ключом (чтобы избежать ADC)
try:
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not set.")
    EMBEDDING_MODEL = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
except Exception as e:
    EMBEDDING_MODEL = None
    print(f"ВНИМАНИЕ: Не удалось инициализировать GoogleGenerativeAIEmbeddings. Проверьте GOOGLE_API_KEY. Ошибка: {e}")


def hybrid_extract_context(query: str, chunks: List[str], max_chunks: int = 100):
    cleaned_chunks = [chunk.strip() for chunk in chunks if
                      len(chunk.strip()) > 30]

    # Дедуп: Только точный, без fuzzy (чтобы сохранить похожие варианты)
    unique_chunks = list(dict.fromkeys(cleaned_chunks))

    return unique_chunks[:max_chunks]


class HybridRetriever:
    """
    Класс-обертка для LangChain Retriever, выполняющий постобработку контекста
    (часть функционала Retrieval Agent).
    """

    def __init__(self, base_retriever):
        """
        Инициализируется базовым ретривером (History-Aware Retriever).
        """
        self.base_retriever = base_retriever
        self.last_context = ""

        # ИСПРАВЛЕНИЕ: Теперь Union импортирован, ошибка NameError исчезнет.

    def __call__(self, query: Union[str, Dict], chat_history: List = None):
        """
        Основной метод вызова, совместимый с интерфейсом LangChain.
        """
        if chat_history is None:
            chat_history = []

        if isinstance(query, dict):
            query_str = query.get("input", "")
        else:
            query_str = query

        # 2. Вызов базового ретривера (History-Aware Retriever)
        initial_chunks = self.base_retriever.invoke({"input": query_str, "chat_history": chat_history})

        # 3. Преобразование объектов Document в простой текст
        texts = [chunk.page_content for chunk in initial_chunks if hasattr(chunk, "page_content")]

        # 4. Гибридная очистка и дедупликация
        refined_texts = hybrid_extract_context(query_str, texts)

        print(f"Retrieval: {len(initial_chunks)} исходных чанков → {len(refined_texts)} после фильтра.")
        if refined_texts:
            print(f"Первый релевантный: {refined_texts[0][:100]}...")
        else:
            print("ВНИМАНИЕ: Нет релевантных чанков для query!")

        # 5. Сохраняем для отладки и UI (Streamlit Expander)
        self.last_context = "\n\n---\n\n".join(refined_texts)

        # 6. Преобразуем очищенные тексты обратно в объекты Document
        refined_docs = [Document(page_content=txt, metadata={}) for txt in refined_texts]

        return refined_docs

    # Обеспечиваем совместимость с интерфейсом Retriever
    def similarity_search(self, query: str, **kwargs):
        chat_history = kwargs.get("chat_history", [])
        return self.__call__(query, chat_history)

    # Необходимый метод для работы с .with_config() в LangChain
    def with_config(self, **kwargs):
        return self