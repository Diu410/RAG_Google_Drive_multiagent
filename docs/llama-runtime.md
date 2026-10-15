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
different alias/context/offload requires restarting the server with that
deployment profile (`LLAMA_DEFAULT_PROFILE`). Installed 4B/9B chat weights can
be selected in the UI; the application restarts its owned chat server when
switching weights. Unsupported profile names are rejected rather than sent as models.

## Larger local model

The available larger asset is
[Qwen3.5-9B Q4_K_M](https://huggingface.co/lmstudio-community/Qwen3.5-9B-GGUF/tree/main),
based on [Qwen's official 9B model](https://huggingface.co/Qwen/Qwen3.5-9B).
Keep both `api/models/chat.gguf` (4B) and `api/models/Qwen3.5-9B-Q4_K_M.gguf`.
The chat header has a local-model selector, separate from the response profile.
Without a saved UI choice, development defaults to 4B. Both files remain on
disk, while one chat server occupies CPU/GPU memory at a time. A choice persists
in `KONTUR_DATA_DIR/chat-model-selection.json` and applies to all three profiles.
Missing assets appear disabled. Switching during an assistant task or active
inference returns 409; an existing external server is never terminated. After
a load failure the selector permits choosing the other model.

Additional supported selector IDs are `qwen9b-q3`
(`models/Qwen3.5-9B-Q3_K_M.gguf`) and `gemma4e4b`
(`models/gemma-4-E4B_q4_0-it.gguf`). They require separately installed assets.
The Q3 asset is [Unsloth's quantization of Qwen3.5-9B](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/tree/main).
The E4B asset is [Google's official QAT GGUF](https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf/tree/main).
E4B denotes effective inference parameters; its text GGUF is 5.15 GB, not the
size of an ordinary 4B model. No multimodal projector is needed for document text.

The API adds `POST /api/chat-model` with one of these installed-model IDs.
`GET /api/model-status` returns `chat_models` and `selected_model_id` alongside
readiness. Selection writes are atomic; a failed write restores the old
deployment. A saved UI choice takes precedence over the initial model path.

For structured calls with reasoning disabled, the provider gives the decoder
an explicit zero thought budget. It enables the reasoning-capable template for
these JSON calls so llama.cpp can recognise and close spontaneous thought
channels before applying the JSON grammar. Ordinary text keeps the non-thinking
template: forcing the reasoning template there can turn subsequent analysis
into visible prose. Enabled reasoning retains its finite token budget.
Token counting uses the template selected for the actual request. This behaviour
was checked with build b11060 and real Gemma 4 E4B requests, including a failed
audit that previously used 1000 tokens entirely for hidden thoughts.

Measured speed and limits of answer quality are recorded in
[the local model comparison](local-model-alternatives.md).

For an installation without a saved UI choice, one can also configure
`LLAMA_MODEL_PATH=models/Qwen3.5-9B-Q4_K_M.gguf` in the active data directory's
`.env`. Restart the backend to load the new setting. A desktop installation's
AppData `.env` takes precedence over the development `api/.env`.

The downloaded file is 5,627,044,256 bytes; SHA-256:
`cd76ec205963b3b33350093e6904d9de16c4e666fd104e1f632d25c7f15f2a13`.
Pinned GGUF repository revision: `1379f25c6b505a3fc737bd7818cb09389cf807c1`.
Keep the original 4B at `api/models/chat.gguf`; to return to it, choose 4B in
the UI selector. Do not replace Nomic embeddings.
Changing the chat GGUF requires no reindexing.

On the tested RTX 4050 Laptop with 6 GB VRAM and 16 GB RAM, 9B uses partial
CPU/GPU offload with `LLAMA_GPU_LAYERS=auto` and context 4096. Observed generation
was approximately 13 tokens/second, compared with approximately 50 for 4B;
fact answers with routing and verification can take more than two minutes.
Direct file operations bypass most model calls and remain fast. These are local
measurements, not throughput guarantees. More parameters do not ensure correct
tool choice: the original 9B evaluation also exposed routing and verification
failures. See [the evaluation report](user-query-evaluation.md) for fixes,
tested scenarios and limitations.

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
`preparation_ms` (readiness and prompt budgeting), `request_total_ms` (including preparation),
`time_to_first_token_ms`, `generation_total_ms`, `tokens_per_second`,
`prompt_tokens`, `completion_tokens`, `finish_reason`. Token counts come from the server usage;
unavailable values on cancelled streams are null, not invented estimates.
Logs include generation/task/action IDs and timings, not message contents.

Structured output that ends with `finish_reason=length` raises
`StructuredOutputTruncated`; JSON grammar does not guarantee completion within
the output budget. The answer verifier checks long answers in bounded parts
and retries incomplete parts at sentence/line boundaries with a depth limit.
It never accepts completed claims from an otherwise truncated JSON. See
[tool-routing.md](tool-routing.md) for the “Тест 4” reproduction and limitations.

Source-scoped workspace overviews enable thinking with a per-request
`reasoning_budget_tokens` limit. The provider reserves at least 128 output tokens
after this budget and rejects invalid budgets or a budget with thinking disabled.
The default request behavior remains unchanged. The overview composer rejects an
empty or truncated answer. Its verifier uses a smaller reasoning budget and
constrains generated assessments to the original complete answer sentences;
changed, duplicated or omitted statements cannot certify the answer.
This requires the bundled llama-server support for the request field; see
[the server request conversion](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server-common.cpp).

Tokenizer requests can fail while a model loads or the server restarts. The
provider retries on subsequent counts and caches only successful tokenizer
results. Conservative UTF-8 byte estimates remain temporary and are never
cached: an early connection error must not poison the entire backend session.
Changing chat weights still clears the exact-count cache.

A verification request that cannot retain any sources within its input budget
returns `UNAVAILABLE`, not `INSUFFICIENT_INFORMATION`. Workspace retains this
technical cause in the final answer and action journal without repeating
synthesis or suggesting that the user's question needs clarification. A
warm-server evaluation alone does not exercise startup recovery.
