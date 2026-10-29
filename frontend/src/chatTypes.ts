import type {GraphSelection} from './graphTypes';
import type {WorkspaceSourceScope} from './assistant';

export type SearchResult = {
  document_id: number | string | null;
  chunk_id: string;
  text: string;
  metadata: Record<string, unknown>;
  score: number;
  found_by: string[];
};
export type Verification = {
  status: string;
  confidence?: number | null;
  warning?: string | null;
  checked_answer?: string | null;
  answer_scope?: string | null;
  review_mode?: string | null;
  source_chunk_ids: string[];
  unsupported_claims: string[];
  conflicting_claims: string[];
  claims: {
    text: string;
    status: string;
    source_chunk_ids: string[];
    basis?: string | null;
    evidence_quote?: string | null;
    note?: string | null;
    original_text?: string | null;
  }[];
  components?: {
    retrieval: number;
    claim_support: number;
    source_agreement: number;
  } | null;
};
export type GraphInfo = {
  scope?: {workspace_id: string | null; label: string};
  retrieval?: {selected_source_count: number; graph_added_source_count: number};
  entities: { id: number; display_name: string; entity_type: string }[];
  relations: {
    id: number;
    source_name: string;
    source_type: string;
    relation_type: string;
    target_name: string;
    target_type: string;
    document_id: number | string;
    document_name?: string | null;
    chunk_id: string;
    evidence?: string;
  }[];
  documents: { id: number | string; name?: string | null }[];
};
export type Message = {
  id?: string;
  reply_to?: string | null;
  task_id?: string;
  task_policy?: {kind:string;seconds:number;reason:string} | null;
  context_memory?: {version:number;compacted:boolean;archive_preserved:boolean} | null;
  answer_contract?: {source:string;focus:string;allow_interpretation:boolean;topic:string} | null;
  interpretation?: {status:string;text?:string;notice?:string} | null;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  context?: string;
  strategy?: string;
  verification?: Verification | null;
  graph?: GraphInfo | null;
  graphSelection?: GraphSelection;
  workspace_id?: string | null;
  workspace_directory?: string;
  source_scope?: WorkspaceSourceScope | null;
  results?: SearchResult[];
  steps?: { draft?: string; edited?: string; enhanced?: string[] };
};
export type Chat = {
  id: string;
  title: string;
  messages: Message[];
  session_id: string | null;
  updated_at: string;
  version?: number;
  messages_loaded?: boolean;
};
