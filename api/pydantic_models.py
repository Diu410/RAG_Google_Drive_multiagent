# api/pydantic_models.py
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

class ModelName(str, Enum): #Все модели которые может выбрать пользователь.
    GEMINI_25_FLASH = "gemini-2.5-flash"
    GEMINI_25_PRO = "gemini-2.5-pro"

class QueryInput(BaseModel): #Данные которые пользователь обязан дать
    question: str
    session_id: str = Field(default=None) #не обязательный параметр. Пользователь его не передаёт, мы их генерируем в main.
    model: ModelName = Field(default=ModelName.GEMINI_25_FLASH)
    use_multi_agent: bool = Field(default=False)

class QueryResponse(BaseModel): #Данные которые обязаны вернуть пользователю
    answer: str
    session_id: str
    model: ModelName
    context: str


class DocumentInfo(BaseModel):  #Данные которые нужны чтобы вывести файл
    id: int
    filename: str
    upload_timestamp: datetime

class DeleteFileRequest(BaseModel): #Данные которые нужны для удаления файла
    file_id: int

# Изменение: Модель для загрузки папок с Google Drive
class UploadGDriveRequest(BaseModel):
     folder_url: str