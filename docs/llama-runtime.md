# Local inference runtime

Production chat inference uses `LlamaCppProvider` through the provider-neutral
`LLMProvider` contract. Assistant Core and KnowledgeSkill contain no llama.cpp
requests. Router, Verifier and GraphExtractor call `structured_generate` and
retain their existing malformed-output fallbacks. The existing LangChain RAG
and editing chains use `ProviderChatModel`, a thin adapter around the same
provider. `generate` consumes the provider stream only when a complete response
is needed. `stream` yields token deltas immediately through a bounded queue.

## Assets and configuration

Install an official Windows CUDA build of [llama.cpp](https://github.com/ggml-org/llama.cpp/releases)
with its matching runtime DLLs in `runtime/llama`, or set
`LLAMA_SERVER_EXECUTABLE`. The migration was tested with build **b11060**, CUDA
12.4. Preserve the binary distribution licenses. `desktop:prepare` copies the
server and DLLs into Tauri resources. GGUF models are excluded from Git and the
installer; configuration and model files live in the application data directory.

Place a compatible small chat GGUF in `api/models/chat.gguf` for development.
One suitable 4B asset is
[Qwen3.5-4B-Q4_K_M.gguf](https://huggingface.co/lmstudio-community/Qwen3.5-4B-GGUF/tree/main).
The model is configurable, not hard-coded in the provider. A GGUF taken directly
from another runtime may have incompatible metadata; the original Ollama Qwen
blob here had a three-section RoPE layout rejected by this llama.cpp build.

Use `api/.env.example` as the configuration reference. Relative model paths
resolve under `KONTUR_DATA_DIR` (default `api`, desktop production AppData).
Existing development models are copied once to desktop AppData during upgrade.
For a clean installation, place models at the paths shown by the UI or configure
absolute paths in AppData `.env`. No console command is needed to start models.

| Setting | Default | Purpose |
|---|---|---|
| `LLAMA_SERVER_URL` | `http://127.0.0.1:8081` | Chat server |
| `LLAMA_MODEL_PATH` | `models/chat.gguf` | External GGUF |
| `LLAMA_MODEL_ALIAS` | `assistant` | Server-side model alias |
| `LLAMA_CONTEXT_SIZE` | `4096` | Bounded context |
| `LLAMA_GPU_LAYERS` | `auto` | Memory-aware offload; accepts layer count or `all` |
| `LLAMA_THREADS` | `6` | CPU inference threads |
| `LLAMA_BATCH_SIZE` | `256` | Prompt processing batch |
| `LLAMA_MAX_TOKENS` | `2048` | Default output limit |
| `LLAMA_AUTOSTART` | `true` | Start a local child if unavailable |
| `LLAMA_REQUEST_TIMEOUT` | `120` | HTTP inactivity timeout, seconds |
| `LLAMA_STARTUP_TIMEOUT` | `180` | Model load deadline, seconds |

Defaults target the RTX 4050 Laptop's 6 GB VRAM and i5-12500H without requiring
full GPU offload. llama.cpp's `auto` fitting can use partial offload when another
application occupies VRAM. Increasing context consumes memory; no 32K/128K
context is requested implicitly. No chat prompt template is assembled by the
client: roles and messages are sent to `/v1/chat/completions`, with `--jinja`
using the template embedded in the GGUF.

`FAST`, `BALANCED`, `REASONING` are user-facing profiles. Initially all share one
GGUF/context/offload; FAST caps output at 1024, BALANCED at 2048, and REASONING
enables template thinking. Router, verification and graph extraction explicitly
disable thinking for structured JSON. `LLAMA_PROFILES_FILE` may point to JSON
overrides keyed by profile name (fields from `ModelProfile`). A profile with a
different model path/alias/context/offload requires restarting the server with
that deployment profile (`LLAMA_DEFAULT_PROFILE`); runtime hot swapping is not
implemented. Unsupported profile names are rejected rather than sent as models.

## Embeddings and existing Chroma

`EmbeddingProvider` is independent of `LLMProvider`. Default embeddings use a
second llama-server at `127.0.0.1:8082`, running Nomic on CPU so chat inference
can use VRAM. Place the same Nomic GGUF used by the old index at
`models/nomic-embed-text.gguf`. The migration copied the existing Nomic weights;
sample comparison against the former service gave cosine similarity 0.99995
with the same 768 dimensions. Text and task prefixes are kept unchanged.

Do not substitute an unrelated 768-dimensional model: matching dimension alone
does not mean matching embedding space. A different embedding model requires
reindexing. Neither SQLite history nor existing Chroma collections are rewritten
by provider migration. The optional legacy adapter can be explicitly selected
with `EMBEDDING_BACKEND=ollama`, `EMBEDDING_SERVER_URL=http://127.0.0.1:11434`;
it uses HTTP and requires neither the Ollama Python SDK nor langchain-ollama.
It is disabled by default and never used for chat inference.

## Readiness, process ownership, errors

FastAPI process health (`/api/health`) stays independent of model loading.
Startup begins model loading on background threads. `/api/model-status` exposes
`process_started`, `server_ready`, `model_loaded`, `ready`, `owned`, paths and a
readable message; the chat header shows startup/readiness or the actual error.
Generation checks readiness before making a completion request. `/health=200`
and the configured alias in `/v1/models` are required for managed servers.
An existing server still loading is awaited, never replaced by another child.

Children launch with `CREATE_NO_WINDOW`, stdout/stderr in `llama-server.log` and
`embedding-server.log`. Graceful backend shutdown closes owned children. In the
Windows desktop, the existing kill-on-close Job Object contains the backend
and its descendants, covering forced termination too. External servers are not
in that job and are not stopped. Missing GGUF/executable errors include the
expected path; load failures include exit code and log path. Provider failures
become HTTP 503 and failed Task/Action, retaining chat history.

## Streaming and cancellation

The HTTP reader runs on a dedicated asyncio loop. Synchronous chains consume a
bounded queue of SSE deltas (16 items); there is no full-answer buffer in
`stream`. Reasoning deltas are separate from answer text. `generate` collects
answer text, while `structured_generate` sends a JSON schema in OpenAI-compatible
`response_format` and checks JSON before consumer-specific validation.

`cancel(generation_id)` also accepts the owning task/action ID. It sets a
cancellation signal and cancels the asynchronous HTTP read, causing `aclose`
on the socket stream. A consumer waiting for the next token wakes promptly;
the server receives the disconnect rather than continuing to send an unwanted
answer. Closing the iterator also cancels its underlying HTTP request.
Task scope propagates through `asyncio.to_thread` and LangChain's context so
every model call remains cancellable. Cancellation is not swallowed by Router,
Verifier, multi-agent fallback or graph best-effort error handling.

`GET /assistant/tasks` lists running tasks; `GET /assistant/tasks/{id}` gives
persisted state/actions. `POST /assistant/tasks/{id}/cancel` requests cancellation.
The active Action and Task become CANCELLED; the original request returns 409.
Cancelling between model calls prevents the next call. This is not rollback of
already committed actions, nor a generic interruption mechanism for unrelated
blocking filesystem/embedding operations. No Live Mode/UI streaming is added.

Each generation records `llm_request_ms` (response-header latency),
`time_to_first_token_ms`, `generation_total_ms`, `tokens_per_second`,
`prompt_tokens`, `completion_tokens`. Token counts come from the server usage;
unavailable values on cancelled streams are null, not invented estimates.
Logs include generation/task/action IDs and timings, not message contents.
