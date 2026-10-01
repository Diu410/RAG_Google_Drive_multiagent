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
  source_chunk_ids: string[];
  unsupported_claims: string[];
  conflicting_claims: string[];
  claims: {
    text: string;
    status: string;
    source_chunk_ids: string[];
    basis?: string | null;
    evidence_quote?: string | null;
  }[];
  components?: {
    retrieval: number;
    claim_support: number;
    source_agreement: number;
  } | null;
};
export type GraphInfo = {
  entities: { id: number; display_name: string; entity_type: string }[];
  relations: {
    id: number;
    source_name: string;
    source_type: string;
    relation_type: string;
    target_name: string;
    target_type: string;
    document_id: number;
    document_name?: string | null;
    chunk_id: string;
    evidence?: string;
  }[];
  documents: { id: number; name?: string | null }[];
};
export type Message = {
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  context?: string;
  strategy?: string;
  verification?: Verification | null;
  graph?: GraphInfo | null;
  steps?: { draft?: string; edited?: string; enhanced?: string[] };
};
export type Chat = {
  id: string;
  title: string;
  messages: Message[];
  session_id: string | null;
  updated_at: string;
};
