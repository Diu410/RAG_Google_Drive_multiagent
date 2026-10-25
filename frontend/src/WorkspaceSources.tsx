import {useState} from 'react';
import {api} from './api';
import type {WorkspaceSourceScope} from './assistant';
import type {SearchResult} from './chatTypes';

export function sourceScopeLabel(scope: WorkspaceSourceScope): string {
  if (scope.kind === 'directory') return `Папка: ${scope.directory} · с вложенными`;
  if (scope.kind === 'files') return scope.files.length === 1 ? scope.files[0] : `Выбрано файлов: ${scope.files.length}`;
  return 'Вся рабочая папка · с вложенными';
}

type SourceLocation = {page?: number; section?: string; line_start?: number; line_end?: number};

function locationLabel(locations: unknown): string {
  if (!Array.isArray(locations)) return '';
  return [...new Set((locations as SourceLocation[]).map(location => {
    if (typeof location.page === 'number') return `стр. ${location.page}`;
    if (typeof location.line_start === 'number') return `строки ${location.line_start}–${location.line_end}`;
    return typeof location.section === 'string' ? location.section : '';
  }).filter(Boolean))].join(', ');
}

function SourceExcerpt({source}: {source: SearchResult}) {
  const [status, setStatus] = useState('');
  const [busy, setBusy] = useState(false);
  const metadata = source.metadata;
  const range = metadata.source_range;
  const canCheck = typeof metadata.workspace_id === 'string' && typeof metadata.source === 'string'
    && typeof metadata.revision === 'string' && Array.isArray(range) && range.length === 2
    && range.every(value => Number.isInteger(value));

  async function checkRevision() {
    if (!canCheck) return;
    setBusy(true);
    setStatus('');
    try {
      const query = new URLSearchParams({path: String(metadata.source), revision: String(metadata.revision),
        start: String(range[0]), end: String(range[1])});
      await api(`/workspaces/${encodeURIComponent(String(metadata.workspace_id))}/source?${query}`);
      setStatus('Версия источника не изменилась. Это проверка версии, не достоверности содержания.');
    } catch (error) {
      setStatus(error instanceof Error ? error.message : 'Не удалось проверить источник.');
    } finally {
      setBusy(false);
    }
  }

  return <details className="workspace-source">
    <summary>{String(metadata.source || source.chunk_id)} {locationLabel(metadata.locations)}</summary>
    <p>Сохранённый фрагмент, использованный при подготовке ответа. Он не заменяется текущим содержимым файла.</p>
    <pre>{source.text}</pre>
    {canCheck && <button type="button" disabled={busy} onClick={() => void checkRevision()}>
      {busy ? 'Проверяю версию…' : 'Проверить актуальность источника'}
    </button>}
    {status && <p role="status">{status}</p>}
  </details>;
}

export function AnswerSources({sources, context}: {sources?: SearchResult[]; context?: string}) {
  // Older chats have only the flattened context. Never show both versions of
  // the same evidence, and retain library sources as well as workspace sources.
  const excerpts = [...new Map((sources || []).map(source => [source.chunk_id, source])).values()];
  if (!excerpts.length && !context) return null;
  return <details className="workspace-sources">
    <summary>Источники ответа{excerpts.length ? ` · ${excerpts.length}` : ''}</summary>
    {excerpts.length ? excerpts.map(source => <SourceExcerpt key={source.chunk_id} source={source} />) : <pre>{context}</pre>}
  </details>;
}
