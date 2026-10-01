import {useEffect,useState} from 'react';
import {Search} from 'lucide-react';
import {api} from './api';
import {GraphViewer} from './GraphViewer';

type EntityOption={id:number;label:string;type:string};

export function GraphExplorer({onClose}:{onClose:()=>void}){
 const [query,setQuery]=useState(''),[entities,setEntities]=useState<EntityOption[]>([]),[selected,setSelected]=useState<number|null>(null);
 const [loading,setLoading]=useState(true),[error,setError]=useState('');
 useEffect(()=>{const timer=window.setTimeout(async()=>{setLoading(true);setError('');try{setEntities(await api<EntityOption[]>(`/graph/entities?query=${encodeURIComponent(query)}&limit=20`));}catch(e){setError(e instanceof Error?e.message:'Не удалось загрузить сущности.');}finally{setLoading(false)}},200);return()=>window.clearTimeout(timer)},[query]);
 if(selected)return <GraphViewer seedEntityIds={[selected]} onClose={onClose}/>;
 return <div className="graph-picker" aria-label="Граф знаний"><div className="graph-toolbar"><div><strong>Граф знаний</strong><span>Выберите сущность, чтобы открыть локальные связи</span></div><button onClick={onClose}>Закрыть</button></div><label className="graph-search"><Search size={17}/><input autoFocus aria-label="Поиск сущности" placeholder="Название сущности…" value={query} onChange={e=>setQuery(e.target.value)}/></label>{error&&<div className="graph-error" role="alert">{error}</div>}<div className="entity-picker-list">{entities.map(entity=><button key={entity.id} onClick={()=>setSelected(entity.id)}><strong>{entity.label}</strong><span>{entity.type}</span></button>)}{!loading&&!entities.length&&<div className="graph-empty">Сущности не найдены. Переиндексируйте документы, чтобы построить граф.</div>}{loading&&<div className="graph-empty">Загружаю сущности…</div>}</div></div>
}
