import {post,streamJsonLines} from './api';
import type {GraphSelection} from './graphTypes';

export type TaskStatus = 'PENDING' | 'RUNNING' | 'COMPLETED' | 'FAILED' | 'CANCELLED';
export type Session = {
  id: string; chat_id: string | null; created_at: string; updated_at: string;
  status: string; metadata: Record<string, unknown>;
};
export type Task = {
  id: string; session_id: string; user_input: string; task_type: string;
  status: TaskStatus; created_at: string; started_at: string | null;
  completed_at: string | null; error: string | null; metadata: Record<string, unknown>;
};
export type Action = {
  id: string; task_id: string; action_type: string; status: TaskStatus;
  input: Record<string, unknown>; output: Record<string, unknown>;
  started_at: string | null; completed_at: string | null; error: string | null;
};
export type AssistantEvent = {
  type: 'TASK_STARTED' | 'ACTION_STARTED' | 'ACTION_COMPLETED' | 'ACTION_FAILED' |
    'ACTION_CANCELLED' | 'TASK_COMPLETED' | 'TASK_FAILED' | 'TASK_CANCELLED';
  task_id: string; session_id: string; action_id: string | null; timestamp: string;
};
export type ExecutionMetadata = {task_id?: string; task_status?: TaskStatus; actions?: Action[]; events?: AssistantEvent[]};
export type WorkspaceSourceScope =
  | {kind: 'workspace'}
  | {kind: 'directory'; directory: string}
  | {kind: 'files'; files: string[]};
export type AssistantInput = {
  question: string; session_id?: string | null; chat_id?: string;
  request_id?: string;
  model: string; use_multi_agent?: boolean; workspace_id?: string | null; workspace_directory?: string;
  live_mode?: boolean;
  source_scope?: WorkspaceSourceScope | null;
  graph_selection?: GraphSelection;
};

// Shared client entry point for chat and future input surfaces.
export function processAssistantInput<T>(input: AssistantInput) {
  return post<T & ExecutionMetadata>('/assistant/input', input);
}

export function processAssistantLive<T>(input: AssistantInput,onEvent:(event:{type:string;text?:string;stage?:string;task_id?:string;response?:T;message?:string})=>void,signal?:AbortSignal){
  return streamJsonLines('/assistant/live',input,onEvent,{signal,timeoutMs:10*60*1000});
}

export function processAssistantStream<T>(input: AssistantInput,onEvent:(event:{type:string;task_id?:string;response?:T;message?:string})=>void,signal?:AbortSignal){
  return streamJsonLines('/assistant/stream',input,onEvent,{signal,timeoutMs:10*60*1000});
}
