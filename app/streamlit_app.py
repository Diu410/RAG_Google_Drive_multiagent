# app/streamlit_app.py
import streamlit as st #Через него мы создаём веб приложение.

# --- ФИКС: ДОБАВЛЯЕМ КОРЕНЬ ПРОЕКТА В PATH ДЛЯ ИМПОРТА МОДЕЛЕЙ ИЗ 'api' ---
import sys
import os

# Получаем путь к родительской папке (корню RAG_GoogleDisk_ScriptAssistant)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
# -------------------------------------------------------------------------


from sidebar import display_sidebar #Функция для отображения боковой панели
from chat_interface import display_chat_interface

st.title("Ассистент по работе с текстом") #Навзвание приложения

if "documents" not in st.session_state:
    st.session_state.documents = []
#Если это новая сессия создаём списки для сообщений и айди
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = None

display_sidebar()#Включаем боковую панель

display_chat_interface()#Включаем чат