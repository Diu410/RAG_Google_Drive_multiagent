# Assistant Core foundation

The existing desktop UI remains a chat. `frontend/src/assistant.ts` sends normal
chat and automatic editing requests to `POST /assistant/input`. `POST /query`
remains a compatible alias. Both endpoints only validate the request, call
`AssistantCore.process_input`, and attach the existing timing header.

```text
Input adapter -> AssistantCore -> Task / execution context
                              -> KnowledgeSkill -> existing Query Router
                                                -> general chat / RAG / multi-agent RAG
                                                -> GraphRAG / Verifier
                              -> existing conversation history -> response
```

`AssistantCore` has no FastAPI dependency. Dependencies are wired in `main.py`.
Non-HTTP callers can supply the same validated `QueryInput` and a source label
to `process_input(request, source="system")`. This label records provenance; it
does not authorize new capabilities. Only existing knowledge/chat execution is
implemented. The interactive writing workshop's `/agent/*` endpoints remain
unchanged; they are explicit, user-driven editing operations, not a planner.

## Persistence and migration

`AssistantRepository` applies version 1 in one SQLite transaction on backend
initialization. It adds `assistant_sessions`, `assistant_tasks`,
`assistant_actions`, indexes and `assistant_migrations`. No existing table or
message is rewritten. Re-running migration is safe. Existing workspace chat IDs
are linked to their original model session IDs; sessions found only in
`application_logs` are imported without a chat association. Chats without a
model session receive one. Late-created/imported chats resolve on first input.

The frontend sends both `chat_id` and the existing `session_id`. A chat has at
most one assistant session, and a session at most one chat. Conflicting
associations return 409 rather than mixing histories. `chat_id` is a logical
association, not a foreign key: deleting a chat retains the execution journal.

Conversation context still comes from `application_logs`; UI messages still
come from `workspace_chats`. Tasks store the requested input, status and source;
they are not another message history. Actions contain compact execution data,
never document chunks, prompts or generated responses.

Tasks/actions enforce `PENDING -> RUNNING -> COMPLETED | FAILED | CANCELLED`
(a pending execution can also be cancelled). Updates compare the persisted
status to reject conflicting transitions. Timestamps are UTC. Caught pipeline
errors mark the active action and task failed without inserting an error message
into conversation history. Technical errors in the journal contain only the
exception class and a short description; tracebacks remain in backend logs.

The existing multi-agent-to-RAG and graph metadata fallbacks are preserved.
A recovered action can be `FAILED` while its parent task is `COMPLETED`.
Cancellation propagates through a provider-neutral task scope and closes active
HTTP generation streams. See [runtime details](llama-runtime.md). Process termination recovery and retry/idempotency
are future work; this stage does not introduce a background task scheduler.

## Extension contracts

`Skill` declares name, description, input schema, permissions and asynchronous
`execute(inputs, context) -> ToolResult`. `SkillRegistry` registers named skills
and rejects duplicates. It currently contains only `KnowledgeSkill`. Permissions
are capability declarations, not an implemented permission enforcement system.

`KnowledgeSkill` adapts existing services without replacing their internals.
Actions track routing, history loading, general generation, multi-agent RAG,
RAG, graph metadata search, verification and history saving when they run.
`RAG_PIPELINE` intentionally includes retrieval and generation because the
existing chain owns both. `GRAPH_SEARCH` tracks the existing graph metadata
lookup; graph-assisted retrieval itself remains inside `RAG_PIPELINE`.

`ToolResult.data` can carry the existing complete response in memory, but it is
not serialized into Action output. Action output has a compact ToolResult-shaped
envelope containing only stage metadata such as source count.

Typed events include task/action start, completion, failure and cancellation.
The execution context collects them per request and optionally invokes an
observer. Observer errors do not fail a task. Successful responses add optional
`task_id`, `task_status`, `actions`, `events`; HTTP error responses retain their
existing `detail` format. Events are not separately persisted or streamed.

No planner, plugin loader, filesystem/OS/browser control, workspace, Live Mode
or model migration is implemented here.
