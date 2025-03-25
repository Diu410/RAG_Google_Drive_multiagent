from langchain_community.chat_models import ChatOllama #Библиотека да Ламы
from hybrid_context_extractor import hybrid_extract_context
from langchain_core.prompts import ChatPromptTemplate #Позволяет создавать шаблон для промта
from langchain_core.prompts import MessagesPlaceholder #Добавлет место куда можно подставить историю чата
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from hybrid_context_extractor import HybridRetriever
from chroma_utils import vectorstore

retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
#vectorstore преобразуется в retriever – компонент, который по запросу будет возвращать релевантные фрагменты документов.


#Инструкция для LLM
contextualize_q_system_prompt = (
     "Given a chat history and the latest user question "
     "which might reference context in the chat history, "
     "formulate a standalone question which can be understood "
     "without the chat history. Do NOT answer the question, "
     "just reformulate it if needed and otherwise return it as is."
 )
""" Этот текст инструктирует модель взять вопрос, который может быть неясен без контекста, 
    и переписать его так, чтобы он стал самостоятельным.
"""


#Формат промта для LLM чтобы он сделал вопрос самостоятельным
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

#Формат Финального промта для LLM
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI script assistant. Use the following context to answer the user's question."),
    ("system", "Context: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])
def get_rag_chain(model="llama3.2"):
    llm = ChatOllama(model=model)

    #Базовый ретривер
    base_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    #Гибридный
    hybrid_retriever_obj = HybridRetriever(base_retriever)

    # Создаем цепочку для генерации ответа на основе финального промта
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Собираем всю цепочку
    rag_chain = create_retrieval_chain(hybrid_retriever_obj, question_answer_chain)

    #СТАРАЯ ЦЕПОЧКА
    #rag_chain = create_retrieval_chain(base_retriever, question_answer_chain)
    return rag_chain, hybrid_retriever_obj
