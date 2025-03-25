from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredHTMLLoader #Используется для извлечения текста из PDF, DOCX, HTML
from langchain_text_splitters import RecursiveCharacterTextSplitter #Делит большой текст на Чанки (части)
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings #Преобразует текст в векторное представление (эмбединги)
#Эмбеддинги позволяют сравнивать тексты по смыслу, т.е. находить похожие по значению отрывки.
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma #Само векторное хранилище
from typing import List
from langchain_core.documents import Document #Используется, чтобы передавать текстовые части вместе с идентификатором файла.

""""Chroma Utils отвечает за подготовку и индексацию документов для семантического поиска
Другими словами, его задача — взять загруженный документ, извлечь из него текст, разбить этот текст на части (чанки)
и сохранить эти части в векторном хранилище, чтобы потом можно было быстро находить релевантные фрагменты по смыслу."""

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200, length_function=len)
#chunk_overlap - Это насколько чанки текста будут залазить друг на друга (пересекаться)

#embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
embedding_function = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")


vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)


def load_and_split_document(file_path: str) -> List[Document]:
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith('.html'):
        loader = UnstructuredHTMLLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    documents = loader.load()
    return text_splitter.split_documents(documents) #Возращает список объектов Document, каждый из которых содержит фрагмент текста


def index_document_to_chroma(file_path: str, file_id: int) -> bool:
    try:
        splits = load_and_split_document(file_path)

        for split in splits:
            split.metadata['file_id'] = file_id #Для каждого чанка в его метаданные добавляется file_id чтобы связать с целым файлом

        vectorstore.add_documents(splits)
        return True
    except Exception as e:
        print(f"Error indexing document: {e}")
        return False


def delete_doc_from_chroma(file_id: int):
    try:
        docs = vectorstore.get(where={"file_id": file_id})
        print(f"Found {len(docs['ids'])} document chunks for file_id {file_id}")

        vectorstore._collection.delete(where={"file_id": file_id})
        print(f"Deleted all documents with file_id {file_id}")

        return True
    except Exception as e:
        print(f"Error deleting document with file_id {file_id} from Chroma: {str(e)}")
        return False