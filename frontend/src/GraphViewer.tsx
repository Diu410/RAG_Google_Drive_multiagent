import {useCallback,useEffect,useMemo,useState} from 'react';
import {Background,Controls,Handle,MarkerType,MiniMap,Position,ReactFlow,useEdgesState,useNodesState,type Edge,type Node,type NodeProps} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import {api} from './api';
import './graph.css';

export type GraphApiNode={id:string;entity_id?:number;document_id?:number;label:string;type:string;metadata:Record<string,unknown>};
export type GraphProvenance={document_id?:number;document_name?:string|null;chunk_id?:string;evidence?:string;page?:number|null;section?:string|null;source_position?:number|null;text?:string;block_type?:string|null};
export type GraphApiEdge={id:string;source:string;target:string;relation:string;provenance?:GraphProvenance};
export type GraphPayload={nodes:GraphApiNode[];edges:GraphApiEdge[];center_ids:string[];truncated:boolean;limit:number;documents?:string[]};
type FlowData={label:string;entityType:string;apiNode:GraphApiNode;center:boolean};

const FILTER_TYPES=['PERSON','ORGANIZATION','PROJECT','DOCUMENT','TECHNOLOGY','OTHER'];
const accent:Record<string,string>={PERSON:'#66866f',ORGANIZATION:'#6f7f91',PROJECT:'#8b7b60',DOCUMENT:'#7d778b',TECHNOLOGY:'#537f7a',PRODUCT:'#867363',DATE:'#817c68',CONCEPT:'#6c8061',OTHER:'#7c817a'};
const icon:Record<string,string>={PERSON:'●',ORGANIZATION:'▣',PROJECT:'◆',DOCUMENT:'▤',TECHNOLOGY:'⌘',PRODUCT:'◇',DATE:'◷',CONCEPT:'◉',OTHER:'○'};

function KnowledgeNode({data}:NodeProps<Node<FlowData>>){return <div className={'kg-node '+(data.center?'center':'')} style={{'--node-accent':accent[data.entityType]||accent.OTHER} as React.CSSProperties}><Handle type="target" position={Position.Left}/><span aria-hidden>{icon[data.entityType]||icon.OTHER}</span><div><strong>{data.label}</strong><small>{data.entityType}</small></div><Handle type="source" position={Position.Right}/></div>}
const nodeTypes={knowledge:KnowledgeNode};

function validPayload(value:unknown):value is GraphPayload{if(!value||typeof value!=='object')return false;const item=value as Partial<GraphPayload>;return Array.isArray(item.nodes)&&Array.isArray(item.edges)&&item.nodes.every(n=>n&&typeof n.id==='string'&&typeof n.label==='string'&&typeof n.type==='string')&&item.edges.every(e=>e&&typeof e.id==='string'&&typeof e.source==='string'&&typeof e.target==='string'&&typeof e.relation==='string')}

function layout(nodes:GraphApiNode[],edges:GraphApiEdge[],centers:string[]):Node<FlowData>[]{
 const adjacent=new Map<string,string[]>();nodes.forEach(n=>adjacent.set(n.id,[]));edges.forEach(e=>{adjacent.get(e.source)?.push(e.target);adjacent.get(e.target)?.push(e.source)});
 const level=new Map<string,number>(),queue=[...centers.filter(id=>adjacent.has(id))];queue.forEach(id=>level.set(id,0));
 while(queue.length){const current=queue.shift()!;for(const next of adjacent.get(current)||[])if(!level.has(next)){level.set(next,(level.get(current)||0)+1);queue.push(next)}}
 const groups=new Map<number,GraphApiNode[]>();nodes.forEach(n=>{const l=level.get(n.id)??Math.max(2,...level.values())+1;groups.set(l,[...(groups.get(l)||[]),n])});
 return nodes.map(n=>{const l=level.get(n.id)??Math.max(2,...level.values())+1;const group=groups.get(l)!;const index=group.findIndex(item=>item.id===n.id);return{id:n.id,type:'knowledge',position:{x:l*260,y:(index-(group.length-1)/2)*125},data:{label:n.label,entityType:n.type,apiNode:n,center:centers.includes(n.id)}}});
}

function toEdges(edges:GraphApiEdge[]):Edge[]{return edges.map(edge=>({id:edge.id,source:edge.source,target:edge.target,label:edge.relation,data:{apiEdge:edge},markerEnd:{type:MarkerType.ArrowClosed},style:{stroke:'#849584'},labelStyle:{fill:'#536557',fontSize:10,fontWeight:650},labelBgStyle:{fill:'#f9fbf7',fillOpacity:.9}}))}

export function GraphViewer({seedEntityIds=[],documentId,onClose}:{seedEntityIds?:number[];documentId?:number;onClose:()=>void}){
 const [rawNodes,setRawNodes]=useState<GraphApiNode[]>([]),[rawEdges,setRawEdges]=useState<GraphApiEdge[]>([]),[centers,setCenters]=useState<string[]>([]);
 const [nodes,setNodes,onNodesChange]=useNodesState<Node<FlowData>>([]),[edges,setEdges,onEdgesChange]=useEdgesState<Edge>([]);
 const [selectedNode,setSelectedNode]=useState<GraphApiNode|null>(null),[selectedEdge,setSelectedEdge]=useState<GraphApiEdge|null>(null);
 const [depth,setDepth]=useState(1),[filters,setFilters]=useState(new Set(FILTER_TYPES)),[loading,setLoading]=useState(true),[error,setError]=useState(''),[truncated,setTruncated]=useState(false);
 const [expanded,setExpanded]=useState(new Set<string>());
 const merge=useCallback((payload:GraphPayload,replace=false)=>{setRawNodes(previous=>{const map=new Map((replace?[]:previous).map(n=>[n.id,n]));payload.nodes.forEach(n=>map.set(n.id,n));return [...map.values()].slice(0,50)});setRawEdges(previous=>{const map=new Map((replace?[]:previous).map(e=>[e.id,e]));payload.edges.forEach(e=>map.set(e.id,e));return [...map.values()].slice(0,50)});setCenters(previous=>[...new Set([...(replace?[]:previous),...(payload.center_ids||[])])]);setTruncated(value=>value||payload.truncated||payload.nodes.length>50||payload.edges.length>50)},[]);
 const load=useCallback(async(path:string,replace=false)=>{setLoading(true);setError('');try{const payload=await api<unknown>(path);if(!validPayload(payload))throw new Error('Сервер вернул некорректный граф.');merge(payload,replace);if(!payload.nodes.length)setError('Связи для выбранного объекта не найдены.');}catch(e){setError(e instanceof Error?e.message:'Граф временно недоступен.');}finally{setLoading(false)}},[merge]);
 useEffect(()=>{const first=seedEntityIds[0];if(first)void load(`/graph/entity/${first}/neighbors?depth=1`,true);else if(documentId)void load(`/graph/document/${documentId}`,true);else{setLoading(false);setError('Выберите сущность или документ для визуализации.')}},[seedEntityIds,documentId,load]);
 useEffect(()=>{const visible=rawNodes.filter(n=>filters.has(FILTER_TYPES.includes(n.type)?n.type:'OTHER'));const ids=new Set(visible.map(n=>n.id));const visibleEdges=rawEdges.filter(e=>ids.has(e.source)&&ids.has(e.target));setNodes(layout(visible,visibleEdges,centers));setEdges(toEdges(visibleEdges));},[rawNodes,rawEdges,filters,centers,setNodes,setEdges]);
 const incident=useMemo(()=>selectedNode?rawEdges.filter(e=>e.source===selectedNode.id||e.target===selectedNode.id):[],[selectedNode,rawEdges]);
 async function expand(){if(!selectedNode?.entity_id)return;const key=`${selectedNode.entity_id}:${depth}`;if(expanded.has(key))return;await load(`/graph/entity/${selectedNode.entity_id}/neighbors?depth=${depth}`);setExpanded(value=>new Set(value).add(key));}
 function toggleFilter(type:string){setFilters(value=>{const next=new Set(value);next.has(type)?next.delete(type):next.add(type);return next})}
 return <div className="graph-shell" aria-label="Граф знаний"><div className="graph-toolbar"><div><strong>Связи GraphRAG</strong><span>{rawNodes.length} узлов · {rawEdges.length} связей</span></div><label>Глубина <select aria-label="Глубина графа" value={depth} onChange={e=>setDepth(Number(e.target.value))}><option value={1}>1 уровень</option><option value={2}>2 уровня</option></select></label><button onClick={onClose}>Закрыть</button></div>
 <div className="graph-filters" aria-label="Фильтры типов">{FILTER_TYPES.map(type=><label key={type}><input type="checkbox" checked={filters.has(type)} onChange={()=>toggleFilter(type)}/>{type}</label>)}</div>
 {truncated&&<div className="graph-notice">Достигнут лимит отображения. Уточните фильтр или раскройте конкретную сущность.</div>}{error&&<div className="graph-error" role="alert">{error}</div>}
 <div className="graph-workspace"><div className="graph-canvas">{loading&&!rawNodes.length?<div className="graph-empty">Загружаю связи…</div>:!rawNodes.length?<div className="graph-empty">Граф пуст</div>:<ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} onNodesChange={onNodesChange} onEdgesChange={onEdgesChange} onNodeClick={(_,node)=>{setSelectedNode(node.data.apiNode);setSelectedEdge(null)}} onEdgeClick={(_,edge)=>{setSelectedEdge((edge.data as {apiEdge:GraphApiEdge})?.apiEdge||null);setSelectedNode(null)}} fitView fitViewOptions={{padding:.2,maxZoom:1.25}} minZoom={.25} maxZoom={1.8}><Background gap={22} color="#dfe7da"/><MiniMap pannable zoomable nodeColor={node=>accent[(node.data as FlowData).entityType]||accent.OTHER}/><Controls/></ReactFlow>}</div>
 <aside className="graph-details">{selectedNode?<><span className="node-kind">{selectedNode.type}</span><h3>{selectedNode.label}</h3><p>{incident.length?`Связей: ${incident.length}`:'Связей в текущем отображении нет.'}</p>{selectedNode.entity_id&&<button className="primary" onClick={()=>void expand()} disabled={loading||expanded.has(`${selectedNode.entity_id}:${depth}`)}>{expanded.has(`${selectedNode.entity_id}:${depth}`)?'Связи раскрыты':'Раскрыть связи'}</button>}<h4>Связанные объекты</h4>{incident.map(edge=><button className="relation-row" key={edge.id} onClick={()=>{setSelectedEdge(edge);setSelectedNode(null)}}>{edge.relation}</button>)}</>:selectedEdge?<EdgeDetails edge={selectedEdge}/>:<p>Выберите узел или связь, чтобы увидеть детали.</p>}</aside></div></div>
}

function EdgeDetails({edge}:{edge:GraphApiEdge}){const p=edge.provenance||{};return <><span className="node-kind">СВЯЗЬ</span><h3>{edge.relation}</h3><dl><dt>Источник</dt><dd>{p.document_name||'Источник не найден'}</dd>{p.page!=null&&<><dt>Страница</dt><dd>{Number(p.page)+1}</dd></>}{p.section&&<><dt>Раздел</dt><dd>{p.section}</dd></>}{p.chunk_id&&<><dt>Chunk</dt><dd>{p.chunk_id}</dd></>}</dl>{p.evidence&&<blockquote>{p.evidence}</blockquote>}{p.text&&<details><summary>Supporting chunk</summary><pre>{p.text}</pre></details>}</>}
