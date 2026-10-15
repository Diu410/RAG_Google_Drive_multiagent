export type GraphSelection = {workspace_id: string | null; relation_ids: number[]};
export type GraphLinkExplanation = {
  similarity: number; threshold: number; passage_similarity: number; sampled: boolean;
  passages: {document_id: string; label: string; chunk_id: string; text: string; truncated: boolean}[];
};

export function graphParams(workspaceId?: string | null) {
  const params = new URLSearchParams();
  if (workspaceId) params.set('workspace_id', workspaceId);
  return params;
}
