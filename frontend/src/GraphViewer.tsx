import {useEffect, useMemo, useRef, useState, type PointerEvent} from 'react';
import {createParticles, stepLayout, type MapEdge, type MapNode} from './graphLayout';
import {api} from './api';
import {graphParams, type GraphLinkExplanation} from './graphTypes';
import './graph.css';

export type GraphPayload = {nodes: MapNode[]; edges: MapEdge[];
  coverage: {documents: number; indexed_documents: number}; truncated: boolean};

export function GraphViewer({graph, query, workspaceId}: {graph: GraphPayload; query: string; workspaceId?: string}) {
  const particles = useMemo(() => createParticles(graph.nodes), [graph.nodes]);
  const byId = useMemo(() => new Map(particles.map(node => [node.id, node])), [particles]);
  const svg = useRef<SVGSVGElement>(null);
  const [frame, setFrame] = useState(0);
  const [hovered, setHovered] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [selectedEdge, setSelectedEdge] = useState<MapEdge | null>(null);
  const [explanation, setExplanation] = useState<GraphLinkExplanation | null>(null);
  const [linkError, setLinkError] = useState('');
  const [camera, setCamera] = useState({x: 0, y: 0, zoom: 1});
  const cameraRef = useRef(camera);
  cameraRef.current = camera;
  const heat = useRef(1);
  const drag = useRef<{id: string | null; x: number; y: number; moved: boolean} | null>(null);
  const wake = useRef<() => void>(() => {});

  useEffect(() => {
    let disposed = false;
    setExplanation(null); setLinkError('');
    if (selectedEdge) {
      const params = graphParams(workspaceId);
      params.set('source_id', selectedEdge.source); params.set('target_id', selectedEdge.target);
      api<GraphLinkExplanation>(`/graph/link?${params}`).then(value => {
        if (!disposed) setExplanation(value);
      }).catch(error => {
        if (!disposed) setLinkError(error instanceof Error ? error.message : 'Связь недоступна. Обновите карту.');
      });
    }
    return () => {disposed = true;};
  }, [selectedEdge, workspaceId]);

  useEffect(() => {
    if (!selectedEdge) return;
    const escape = (event: KeyboardEvent) => {
      const container = svg.current?.closest('.graph-explorer') || svg.current?.parentElement;
      if (event.key !== 'Escape' || !(event.target instanceof Node) || !container?.contains(event.target)) return;
      // The enclosing dialog handles Escape at document capture; close this layer first.
      event.preventDefault(); event.stopPropagation(); setSelectedEdge(null);
    };
    window.addEventListener('keydown', escape, true);
    return () => window.removeEventListener('keydown', escape, true);
  }, [selectedEdge]);

  useEffect(() => {
    let handle = 0, disposed = false;
    heat.current = 1; setSelected(null); setHovered(null);
    setCamera({x: 0, y: 0, zoom: Math.min(1.5, 260 / (35 * Math.sqrt(particles.length) + 40))});
    const animate = () => {
      handle = 0;
      if (disposed || heat.current < .01) return;
      stepLayout(particles, graph.edges, heat.current);
      heat.current *= .97; setFrame(value => value + 1);
      handle = window.requestAnimationFrame(animate);
    };
    wake.current = () => {if (!handle && !disposed) handle = window.requestAnimationFrame(animate);};
    wake.current();
    return () => {disposed = true; window.cancelAnimationFrame(handle);};
  }, [particles, graph.edges]);

  function point(clientX: number, clientY: number) {
    const matrix = svg.current?.getScreenCTM();
    return matrix ? new DOMPoint(clientX, clientY).matrixTransform(matrix.inverse()) : new DOMPoint();
  }
  useEffect(() => {
    const element = svg.current;
    if (!element) return;
    const wheel = (event: WheelEvent) => {
      event.preventDefault();
      const cursor = point(event.clientX, event.clientY), old = cameraRef.current;
      const zoom = Math.max(.2, Math.min(4, old.zoom * Math.exp(-event.deltaY * .001)));
      setCamera({zoom, x: cursor.x - (cursor.x - old.x) * zoom / old.zoom,
        y: cursor.y - (cursor.y - old.y) * zoom / old.zoom});
    };
    element.addEventListener('wheel', wheel, {passive: false});
    return () => element.removeEventListener('wheel', wheel);
  }, []);

  const focus = hovered || selected;
  const neighbors = new Set<string>(focus ? [focus] : []);
  if (focus) graph.edges.forEach(edge => {
    if (edge.source === focus) neighbors.add(edge.target);
    if (edge.target === focus) neighbors.add(edge.source);
  });
  const needle = query.trim().toLocaleLowerCase();
  const matches = new Set(graph.nodes.filter(node => node.label.toLocaleLowerCase().includes(needle)).map(node => node.id));

  function selectEdge(edge: MapEdge) {
    setExplanation(null); setLinkError('');
    setSelectedEdge(old => old?.id === edge.id ? null : edge);
  }

  function pointerDown(event: PointerEvent<SVGSVGElement>) {
    if (event.button !== 0) return;
    if ((event.target as Element).closest('[data-edge-id]')) return;
    const id = (event.target as Element).closest('[data-node-id]')?.getAttribute('data-node-id') || null;
    const cursor = point(event.clientX, event.clientY);
    drag.current = {id, x: cursor.x, y: cursor.y, moved: false};
    svg.current?.setPointerCapture(event.pointerId);
    const node = id ? byId.get(id) : undefined;
    if (node) node.pinned = true;
  }
  function pointerMove(event: PointerEvent<SVGSVGElement>) {
    const active = drag.current;
    if (!active) return;
    const cursor = point(event.clientX, event.clientY), dx = cursor.x - active.x, dy = cursor.y - active.y;
    active.moved ||= Math.hypot(dx, dy) > 2;
    const node = active.id ? byId.get(active.id) : undefined;
    if (node) {
      node.x += dx / cameraRef.current.zoom; node.y += dy / cameraRef.current.zoom;
      heat.current = Math.max(heat.current, .2); wake.current(); setFrame(value => value + 1);
    } else setCamera(old => ({...old, x: old.x + dx, y: old.y + dy}));
    active.x = cursor.x; active.y = cursor.y;
  }
  function pointerUp(event: PointerEvent<SVGSVGElement>) {
    const active = drag.current;
    if (active) {
      const node = active.id ? byId.get(active.id) : undefined;
      if (node) node.pinned = false;
      if (!active.moved) setSelected(old => old === active.id ? null : active.id);
    }
    drag.current = null;
    if (svg.current?.hasPointerCapture(event.pointerId)) svg.current.releasePointerCapture(event.pointerId);
  }
  return <div className="graph-canvas" data-frame={frame} data-settled={heat.current < .01}>
    <svg ref={svg} viewBox="-500 -300 1000 600" aria-label="Карта связей документов"
      onPointerDown={pointerDown} onPointerMove={pointerMove} onPointerUp={pointerUp} onPointerCancel={pointerUp}>
      <g className="graph-camera" transform={`translate(${camera.x} ${camera.y}) scale(${camera.zoom})`}>
        {graph.edges.map(edge => {
          const a = byId.get(edge.source), b = byId.get(edge.target);
          if (!a || !b) return null;
          const active = !focus || edge.source === focus || edge.target === focus;
          const label = `${a.label} ↔ ${b.label}`;
          return <g key={edge.id} data-edge-id={edge.id} role="button" tabIndex={0}
            className={`graph-edge ${active ? '' : 'dimmed'} ${selectedEdge?.id === edge.id ? 'selected' : ''}`}
            aria-label={`Связь: ${label}`} aria-pressed={selectedEdge?.id === edge.id}
            onClick={() => selectEdge(edge)}
            onKeyDown={event => {
              if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault(); selectEdge(edge);
              }
            }}>
            <line className="graph-edge-hit" x1={a.x} y1={a.y} x2={b.x} y2={b.y}/>
            <line className="graph-link" x1={a.x} y1={a.y} x2={b.x} y2={b.y}
              strokeWidth={Math.max(.8, edge.similarity * 1.5)}/>
            <title>{label}: близость {edge.similarity.toFixed(3)} · нажмите для пояснения</title>
          </g>;
        })}
        {particles.map(node => {
          const dimmed = (focus && !neighbors.has(node.id)) || (needle && !matches.has(node.id));
          return <g key={node.id} data-node-id={node.id} transform={`translate(${node.x} ${node.y})`}
            className={`graph-note ${dimmed ? 'dimmed' : ''} ${node.id === focus ? 'focused' : ''} ${node.indexed ? '' : 'unindexed'}`}
            role="button" tabIndex={0} aria-label={node.label} aria-pressed={selected === node.id}
            onPointerEnter={() => setHovered(node.id)} onPointerLeave={() => setHovered(null)}
            onFocus={() => setHovered(node.id)} onBlur={() => setHovered(null)}
            onKeyDown={event => {
              if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault(); setSelected(old => old === node.id ? null : node.id);
              }
            }}>
            <circle className="graph-hit" r={20}/>
            <circle className="graph-dot" r={4 + Math.min(9, Math.sqrt(node.connections) * 2)}/>
            <text y={21} textAnchor="middle">{node.label.length > 34 ? node.label.slice(0, 31) + '…' : node.label}</text>
            <title>{node.label}{node.indexed ? ` · ${node.connections} связей` : ' · ещё не проиндексирован'}</title>
          </g>;
        })}
      </g>
    </svg>
    {selectedEdge && <aside className="graph-link-explanation" aria-label="Пояснение связи">
      <div className="graph-link-heading"><strong>Почему документы связаны</strong>
        <button aria-label="Закрыть пояснение" onClick={() => setSelectedEdge(null)}>×</button></div>
      <p>Линия означает похожее содержание документов.</p>
      {!explanation && !linkError && <p role="status">Сравниваю фрагменты…</p>}
      {linkError && <p className="graph-error" role="alert">{linkError}</p>}
      {explanation && <>
        <p>Близость документов: <strong>{explanation.similarity.toFixed(3)}</strong> · порог {explanation.threshold.toFixed(3)}</p>
        <p>Близость показанных фрагментов: {explanation.passage_similarity.toFixed(3)}</p>
        {explanation.passages.map(passage => <div className="graph-link-passage" key={passage.chunk_id}>
          <strong>{passage.label}</strong><blockquote>{passage.text}{passage.truncated && '…'}</blockquote>
        </div>)}
        <p className="graph-link-note">Показатель сравнивает содержание по шкале от −1 до 1.
          Связь может означать общую тему; причинность и противоречия требуют проверки текста.</p>
        {explanation.sampled && <p className="graph-link-note">Фрагменты выбраны из ограниченной выборки длинных документов.</p>}
      </>}
    </aside>}
    <div className="graph-view-controls">
      <button aria-label="Приблизить" onClick={() => setCamera(old => ({...old, zoom: Math.min(4, old.zoom * 1.25)}))}>+</button>
      <button aria-label="Отдалить" onClick={() => setCamera(old => ({...old, zoom: Math.max(.2, old.zoom / 1.25)}))}>−</button>
      <button onClick={() => {
        const radius = Math.max(120, ...particles.map(node => Math.max(Math.abs(node.x), Math.abs(node.y)) + 40));
        setCamera({x: 0, y: 0, zoom: Math.min(2, 260 / radius)}); setSelected(null);
      }}>Сбросить вид</button>
    </div>
    <p className="graph-hint">{focus ? byId.get(focus)?.label + ' · подсвечены связанные документы' :
      'Перетаскивайте узлы и карту · колесо — масштаб · нажмите на линию, чтобы увидеть пояснение'}</p>
    {needle && !matches.size && <p className="graph-no-match">Документы не найдены</p>}
  </div>;
}
