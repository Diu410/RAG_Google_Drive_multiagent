import re
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from langchain_core.documents import Document

# Инициализация модели и токенизатора
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()


def compute_cosine_similarity(vec1, vec2): #Сравнивает два вектора насколько они похожи друг на друга
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def get_transformer_embedding(text): #Преобразуем текст в эмбединг
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Берем среднее по токенам последнего скрытого состояния
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embeddings


def get_keyword_score(query, text):
    # Простой поиск по ключевым словам: разбиваем запрос на слова и считаем совпадения
    query_tokens = re.findall(r'\w+', query.lower())
    text_lower = text.lower()
    score = sum(text_lower.count(token) for token in query_tokens)
    return score


def hybrid_extract_context(query, chunks, top_k=2, weight_transformer=0.7, weight_keyword=0.3):
    query_embedding = get_transformer_embedding(query)
    scores = []

    # Вычисляем оценки для каждого чанка
    for chunk in chunks:
        chunk_embedding = get_transformer_embedding(chunk)
        transformer_score = compute_cosine_similarity(query_embedding, chunk_embedding)

        keyword_score = get_keyword_score(query, chunk)
        # Нормализация keyword_score (можно настроить)
        keyword_score = keyword_score / (len(query.split()) or 1)

        final_score = weight_transformer * transformer_score + weight_keyword * (keyword_score/10)
        scores.append(final_score)

    # Сортировка чанков по финальному баллу
    scores = np.array(scores)
    top_indices = scores.argsort()[-top_k:][::-1]
    selected_chunks = [chunks[i] for i in top_indices]

    return selected_chunks
class HybridRetriever:
    def __init__(self, base_retriever):
        self.base_retriever = base_retriever

    def __call__(self, query, chat_history=""):
        # Если query пришёл как словарь, извлекаем текст из поля "input"
        if isinstance(query, dict):
            query = query.get("input", "")
        # Получаем первоначальные результаты от базового retriever
        initial_chunks = self.base_retriever.invoke({"input": query, "chat_history": chat_history})
        # Преобразуем каждый элемент в строку: если это объект Document (или dict) с полем page_content, берем его
        texts = []
        for chunk in initial_chunks:
            if isinstance(chunk, dict):
                texts.append(chunk.get("page_content", str(chunk)))
            elif hasattr(chunk, "page_content"):
                texts.append(chunk.page_content)
            else:
                texts.append(str(chunk))

        # Применяем гибридное извлечение для уточнения контекста
        refined_chunks = hybrid_extract_context(query, texts)

        # Сохраняем объединённый контекст
        self.last_context = "\n\n".join(refined_chunks)

        # Оборачиваем каждую строку в объект Document с пустыми метаданными
        refined_docs = [Document(page_content=txt, metadata={}) for txt in refined_chunks]
        return refined_docs

    def similarity_search(self, query: str, **kwargs):
        # Если есть chat_history в kwargs, извлекаем его, иначе – пустая строка
        chat_history = kwargs.get("chat_history", "")
        return self.__call__(query, chat_history)

    def with_config(self, **kwargs):
        # Если нужна дополнительная конфигурация, здесь можно её добавить
        return self
