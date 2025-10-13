import requests
import streamlit as st

"""Функции для взаимодействия с API FastAPI."""
# Изменение: Обновлено описание модуля
def get_api_response(question, session_id, model, use_multi_agent=False):  # <-- Добавь arg
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    data = {
        "question": question,
        "model": model,
        "use_multi_agent": use_multi_agent  # <-- Добавь флаг в data
    }
    if session_id:
        data["session_id"] = session_id

    try:
        response = requests.post("http://localhost:8000/query", headers=headers, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API request failed with status code {response.status_code}: {response.text}")
            return None
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

def upload_document(file):
    """Загружает локальный файл (PDF, DOCX, HTML) на сервер."""
    print("Uploading file...")
    try:
        # Multipart для файлов (не JSON!)
        files = {"file": (file.name, file, file.type)}  # file — UploadedFile из Streamlit
        response = requests.post("http://localhost:8000/upload-doc", files=files)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to upload file. Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"An error occurred while uploading the file: {str(e)}")
        return None


def list_documents():
    try:
        response = requests.get("http://localhost:8000/list-docs")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch document list. Error: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        st.error(f"An error occurred while fetching the document list: {str(e)}")
        return []

def delete_document(file_id):
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    data = {"file_id": file_id}

    try:
        response = requests.post("http://localhost:8000/delete-doc", headers=headers, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to delete document. Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"An error occurred while deleting the document: {str(e)}")
        return None

def upload_gdrive_folder(folder_url):
    """Отправляет URL папки Google Drive в API для обработки."""
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    data = {"folder_url": folder_url}

    try:
        # ТОЧНО /upload-gdrive (без -folder!)
        response = requests.post("http://localhost:8000/upload-gdrive", headers=headers, json=data)
        response.raise_for_status()  # Авто-error на 4xx/5xx
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to upload Google Drive folder. Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
        return None
    except Exception as e:
        st.error(f"An error occurred while uploading the Google Drive folder: {str(e)}")
        return None