# RAG-ассистент для документов

React + TypeScript + FastAPI, Assistant Core, локальные GGUF-модели через llama-server, Chroma и SQLite. Поддерживаются обычный RAG, GraphRAG, автоматическая цепочка draft → edit → enhance → merge и ручное прохождение этапов. Ollama не требуется для production runtime.

Подробности model provider, профилей, streaming, cancellation и настройки моделей: [docs/llama-runtime.md](docs/llama-runtime.md). Архитектура задач: [docs/assistant-core.md](docs/assistant-core.md).

## Desktop-приложение (Tauri)

Desktop-оболочка использует тот же React UI и тот же FastAPI backend. При запуске она выбирает свободный loopback-порт, скрыто запускает standalone backend, ждёт `GET /api/health` и только затем показывает рабочий интерфейс. Backend помещается в Windows Job Object и завершается вместе с приложением. Второй запуск переводит фокус на существующее окно и не создаёт ещё один backend.

Для desktop-разработки нужны Rust toolchain, Visual Studio Build Tools с workload «Desktop development with C++» и Windows 10/11 SDK, а также существующее Python-окружение `.venv`. Основные команды из корня проекта:

```powershell
npm run frontend:dev     # React/Vite; backend отдельно на 8000
npm run backend:dev      # FastAPI с reload на 8000
npm run desktop:dev      # Tauri + Vite; backend стартует из корневой .venv
npm run desktop:build    # React dist + PyInstaller sidecar + Tauri installer
```

Backend dev из корня проекта:

```powershell
.venv\Scripts\python.exe -m uvicorn main:app --app-dir api --host 127.0.0.1 --port 8000 --reload
```

Production desktop хранит SQLite, Chroma, настройки, credentials и логи в системном AppData приложения `local.kontur.rag`. При первом старте существующие данные из каталога `api` копируются туда только при отсутствии соответствующих целевых файлов; исходники не изменяются и не удаляются. Секреты и рабочие базы не включаются в installer или React bundle. Обычный dev-режим продолжает использовать каталог `api`. Desktop capabilities не дают приложению бесконтрольного доступа к диску: системный picker WebView сохраняет существующий upload workflow.

Если backend не готов за 180 секунд или завершился, окно показывает причину, хвост `backend.log` и кнопку повторного запуска. Первый запуск standalone backend может занять больше времени из-за распаковки и проверки Windows Defender. Микрофон использует прежние `getUserMedia`/`MediaRecorder`; WebView запросит системное разрешение при первом использовании.

## Запуск после установки

Запустите «Контур» из меню Start или через ярлык. Откроется обычное desktop-окно; backend и React UI запускаются автоматически без PowerShell, npm, uvicorn и браузера. Закрытие окна завершает скрытый backend. История, документы и индекс сохраняются в AppData между запусками.

Backend автоматически запускает llama-server скрыто и пишет вывод в `llama-server.log`. Embeddings используют отдельный сервер с `embedding-server.log`. Уже доступные серверы используются повторно. Загрузка модели не блокирует интерфейс; статус виден в заголовке чата. При закрытии завершаются только принадлежащие приложению процессы.

## Первый запуск на новом компьютере

Проверенное окружение: Windows, Python 3.12, Node.js 24 с npm, llama.cpp CUDA для NVIDIA GPU. Команды ниже выполняются в PowerShell из корня проекта.

### 1. Python-зависимости

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r api/requirements.txt
```

Активация виртуального окружения не требуется: команды явно используют Python из `.venv`.

### 2. Локальные модели

Разместите официальный Windows CUDA llama-server и DLL из соответствующего релиза в `runtime/llama`. Сохраните совместимую небольшую GGUF-модель чата (например Qwen 3.5 4B Q4_K_M) в `api/models/chat.gguf`, Nomic embedding GGUF — в `api/models/nomic-embed-text.gguf`. Пути переопределяются настройками из `api/.env.example`. Модели не входят в Git и installer. На новом компьютере положите их в каталог данных приложения по пути из сообщения о недостающей модели. Подробности и ссылки на модели — в [документации runtime](docs/llama-runtime.md).

### 3. Сборка desktop installer

```powershell
npm ci
npm run desktop:build
```

Сборка создаёт React assets, standalone `kontur-backend.exe`, Tauri executable и NSIS installer. Установите полученный `.exe` из `frontend/src-tauri/target/release/bundle/nsis/`, затем запускайте приложение из Start Menu.

### После изменений в интерфейсе

Пересоберите installer:

```powershell
npm run desktop:build
```

Если изменились зависимости или lock-файлы, перед сборкой выполните `npm ci` в корне и `npm ci --prefix frontend`.

## Хранение истории и работа с чатом

История хранится в таблице `workspace_chats` существующей SQLite-базы. При первом обращении импортируется `app/chat_history.json`, если он существует; исходный архив сохраняется. Импорт не повторяется после удаления диалога. Названия, сообщения, временные метки и серверный идентификатор сохраняются; сортировка меняется только при сообщении пользователя. Интерфейс предназначен для личного локального использования: авторизация и разделение истории между пользователями не реализованы.

Кнопка `⋮` открывает меню поверх списка, Escape и клик снаружи закрывают его. Микрофон запрашивает системное разрешение WebView. Для записи используется MediaRecorder; для распознавания нужен настроенный Whisper.

Версии прямых Python-зависимостей зафиксированы по локальному окружению. Whisper для голосового ввода устанавливается отдельно:

```powershell
.venv\Scripts\python.exe -m pip install faster-whisper
```

Альтернатива — `openai-whisper`; для него нужен FFmpeg. Модели распознавания скачиваются при первом использовании.

## llama.cpp / model provider

See [runtime configuration, profiles and cancellation](docs/llama-runtime.md) and
[api/.env.example](api/.env.example). Chat inference and embeddings use separate
providers. The default deployment automatically starts llama-server for both;
there is no mandatory Ollama dependency, startup, warmup or Python SDK.

Profiles: `FAST`, `BALANCED` (default), `REASONING`. All initially share the
configured small GGUF; context defaults to 4096, GPU offload to `auto`, CPU
threads to 6. The embedded GGUF chat template handles system/user/assistant
messages. Structured routing, verification and graph extraction use JSON schema.

Task cancellation: `POST /assistant/tasks/{task_id}/cancel`. Provider cancellation
closes the actual HTTP stream, then Task/Action transition to CANCELLED.
`GET /api/model-status` reports readiness independently from backend health.
Metrics include TTFT, request/generation duration, tokens/sec and token usage.

SQLite and Chroma paths stay unchanged. Existing history is preserved. Changing
the embedding model requires reindexing; switching the same Nomic weights from
the old backend to llama.cpp preserves the embedding space.

## Режим разработки React

Для автоматического обновления интерфейса при редактировании нужны два терминала. Терминал 1 — API из корня проекта, **на порту 8000**, который задан в прокси Vite:

```powershell
.venv\Scripts\python.exe -m uvicorn main:app --app-dir api --host 127.0.0.1 --port 8000 --reload
```

Терминал 2 — Vite, также начните из корня проекта:

```powershell
cd frontend
npm run dev
```

При чистой web-разработке можно открыть адрес Vite, напечатанный в терминале. Это только developer convenience: production не открывает и не использует браузер. Для проверки desktop-окна используйте `npm run desktop:dev`.

## Интерфейс «Контур»

Основной экран — чат: история и поиск слева, переписка по центру и закреплённое поле сообщения внизу. У каждого диалога меню `⋮` для переименования и удаления. Кнопка в верхнем левом углу сворачивает и раскрывает панель. Выбор модели и скачивание переписки находятся сверху. «Документы» в левой панели открывает библиотеку поверх переписки.

Режим ответа выбирается внизу, рядом с микрофоном. В пошаговой редактуре можно изменить текст, передать пожелания следующему этапу, выбрать вариант или принять текущую редакцию. Скрепка добавляет вложения; микрофон запускает запись, повторное нажатие останавливает её и вставляет распознанный текст в поле. Enter отправляет сообщение, Shift+Enter добавляет новую строку. Вложения индексируются перед обработкой сообщения.

В окне документов доступны загрузка нескольких файлов, импорт Drive и удаление с подтверждением. Новый чат сохраняет библиотеку документов.

## Если не запускается

| Симптом | Что проверить |
| --- | --- |
| Desktop startup сообщает об ошибке | Откройте детали ошибки в стартовом экране и проверьте `backend.log` в AppData приложения; затем нажмите «Повторить». |
| Второй запуск не создаёт окно | Уже запущенный экземпляр должен получить фокус; проверьте панель задач Windows. |
| Ошибка подключения в Vite dev | Проверьте отдельный backend dev на порту 8000. Это не относится к установленному desktop приложению. |
| Модель не отвечает | Проверьте статус модели, пути GGUF и `llama-server.log` в каталоге данных. При недостатке VRAM уменьшите `LLAMA_GPU_LAYERS` или оставьте `auto`. |
| Микрофон недоступен | Разрешите приложению доступ к микрофону в Windows Privacy settings и установите Whisper для распознавания. |
| Google Drive недоступен | Выполните авторизацию из раздела ниже. |

## Форматы документов

Поддерживаются PDF, DOCX, HTML и UTF-8 TXT, включая короткие непустые тексты. Максимальный размер загрузки по умолчанию — 200 МиБ; сервер проверяет лимит и для аудио. HTML индексируется как текст без выполнения скриптов.

Для PDF без текстового слоя применяется OCR. Установите Tesseract с языками `rus` и `eng`, а также Poppler. Если они не доступны через PATH, задайте `TESSERACT_CMD` (путь к исполняемому файлу) и `POPPLER_PATH` (путь к каталогу bin). Смешанные PDF, где лишь часть страниц является сканами, требуют дополнительной доработки OCR.

## Google Drive

Создайте OAuth-клиент типа Desktop app с включённым Drive API и сохраните credentials в `api/credentials.json`. Авторизуйтесь отдельно:

```powershell
.venv\Scripts\python.exe api/gdrive_auth.py
```

Команда откроет браузер и создаст `api/token.json`. Сервер не запускает интерактивную авторизацию при импорте. Затем укажите ссылку на папку в интерфейсе. Импорт обходит подпапки, читает все страницы списка файлов и поддерживает экспорт Google Docs в текст. Файлы с одинаковыми именами различаются по Drive ID. Повторный импорт создаёт новые записи; автоматической синхронизации изменений пока нет.

## Ограничения

Приложение предназначено для локального использования: API не имеет аутентификации и разграничения доступа к документам. React сохраняет сообщения всех режимов в SQLite. Незавершённая пошаговая редактура хранится в памяти страницы и теряется при обновлении. Её принятый финальный текст сохраняется в диалоге, но не добавляется в отдельный журнал контекста обычной модели.

При сбое индексации выполняется удаление частичного индекса. Если сам откат не удался, запись документа остаётся доступной для повторного удаления. SQLite и Chroma не поддерживают общую транзакцию, поэтому аварийное завершение процесса между операциями может потребовать сверки хранилищ.

## Структура проекта

- `frontend/` — React-интерфейс и Tauri desktop shell.
- `api/` — FastAPI, поиск, работа с моделями и хранилища данных.
- `app/chat_history.json` — прежний архив истории, если существует; оставлен для одноразового импорта. Код старого интерфейса удалён.

Базы, `.env`, credentials и токены не относятся к временным файлам. Не удаляйте их при очистке проекта.
