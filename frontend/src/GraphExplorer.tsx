import {useEffect, useState} from 'react';
import {Search} from 'lucide-react';
import {api} from './api';
import {GraphViewer, type GraphPayload} from './GraphViewer';
import {graphParams} from './graphTypes';

export function GraphExplorer({workspace, onClose}: {
  workspace?: {id: string; name: string} | null; onClose: () => void;
}) {
  const [scope, setScope] = useState(workspace?.id || '');
  const [query, setQuery] = useState('');
  const [graph, setGraph] = useState<GraphPayload | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    let disposed = false;
    setLoading(true); setError(''); setGraph(null);
    api<GraphPayload>(`/graph/map?${graphParams(scope)}`).then(value => {
      if (!Array.isArray(value.nodes) || !Array.isArray(value.edges) || !value.coverage) throw new Error('Не удалось прочитать карту.');
      if (!disposed) setGraph(value);
    }).catch(error => {
      if (!disposed) setError(error instanceof Error ? error.message : 'Карта недоступна.');
    }).finally(() => {if (!disposed) setLoading(false);});
    return () => {disposed = true;};
  }, [scope, revision]);

  return <div className="graph-explorer" aria-busy={loading}>
    <div className="graph-toolbar">
      <div><strong>Связи документов</strong><span>Формируются по смыслу и помогают ассистенту находить контекст</span></div>
      <label><span className="sr-only">Источник графа</span><select aria-label="Источник графа" value={scope}
        onChange={event => {setGraph(null); setLoading(true); setScope(event.target.value); setQuery('');}}>
        {workspace && <option value={workspace.id}>Папка: {workspace.name}</option>}
        <option value="">Общая библиотека</option>
      </select></label>
      <button onClick={() => {setGraph(null);setLoading(true);setRevision(value => value + 1);}} disabled={loading}>Обновить</button>
      <button onClick={onClose}>Закрыть</button>
    </div>
    <div className="graph-filter-bar">
      <label className="graph-search"><Search size={16}/><input aria-label="Поиск документа на карте"
        placeholder="Найти документ…" value={query} onChange={event => setQuery(event.target.value)}/></label>
      {graph && <span>{graph.nodes.length} документов · {graph.edges.length} связей</span>}
    </div>
    {loading && <p className="graph-status" role="status">Загружаю карту…</p>}
    {error && <p className="graph-status graph-error" role="alert">{error}</p>}
    {graph && !graph.nodes.length && <p className="graph-status">В этом источнике пока нет документов.</p>}
    {graph && graph.nodes.length > 0 && <GraphViewer key={`${scope}:${revision}`} graph={graph} query={query} workspaceId={scope}/>}
    {graph && <div className="graph-coverage">
      Связи доступны для {graph.coverage.indexed_documents} из {graph.coverage.documents} документов.
      {graph.coverage.indexed_documents < graph.coverage.documents && ' Для остальных нужна индексация.'}
      {graph.truncated && ' Показана часть карты.'}
      {!graph.edges.length && graph.nodes.length > 0 && ' Достаточно близкие по смыслу документы пока не найдены.'}
    </div>}
  </div>;
}
