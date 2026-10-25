import {useEffect, useState} from 'react';
import {api} from './api';
import type {Action} from './assistant';

const labels: Record<string,string> = {
  INSPECT_REPOSITORY:'Обзор проекта', READ_PROJECT_RULES:'Правила проекта',
  REPOSITORY_READ:'Чтение файла', REPOSITORY_SEARCH:'Поиск', REPOSITORY_SKILL:'Навык',
  REPOSITORY_EDIT:'Правка файла', REPOSITORY_WRITE:'Новый файл', REPOSITORY_CHECK:'Проверка проекта',
  REPOSITORY_REWRITE:'Правка файла',
  INVENTORY_SUMMARY:'Состав папки', SUMMARIZE_DOCUMENT_PART:'Подготовка пересказа',
  VERIFY_DOCUMENT_PART:'Сверка с источником', REPAIR_DOCUMENT_PART:'Исправление пересказа',
  REUSE_DOCUMENT_SUMMARY:'Сохранённый проверенный раздел',
  CREATE_DOCUMENT:'Подготовка документа', APPLY_DOCUMENT_CHANGE:'Сохранение документа',
  REVIEW_PREVIOUS_ACTION:'Проверка предыдущего действия',
};

function sourceChecks(value:unknown):{text:string;quote:string;supported:boolean}[]{
  if(!Array.isArray(value))return [];
  return value.flatMap(row=>{
    const claim=row?.claim;
    if(!claim||typeof claim.text!=='string')return [];
    return [{text:claim.text,quote:typeof claim.evidence_quote==='string'?claim.evidence_quote:'',
      supported:claim.status==='SUPPORTED'}];
  });
}

export function RepositoryTaskDetails({taskId}:{taskId:string}) {
  const [open,setOpen]=useState(false);
  const [actions,setActions]=useState<Action[]|null>(null);
  const [error,setError]=useState('');
  useEffect(()=>{
    if(!open||actions)return;
    let current=true;
    api<{actions:Action[]}>('/assistant/tasks/'+encodeURIComponent(taskId))
      .then(result=>{if(current)setActions(result.actions);})
      .catch(()=>{if(current)setError('Не удалось загрузить журнал действий.');});
    return ()=>{current=false;};
  },[open,actions,taskId]);
  return <details className="repository-task" onToggle={event=>setOpen(event.currentTarget.open)}>
    <summary>Действия, изменения и проверки</summary>
    {error&&<p>{error}</p>}
    {!actions&&!error&&<p>Загружаю журнал…</p>}
    {actions?.filter(action=>labels[action.action_type]).map(action=>{
      const data=(action.output?.data||{}) as Record<string,unknown>;
      const path=typeof data.path==='string'?data.path:typeof action.input.path==='string'?action.input.path:'';
      const checks=sourceChecks(action.action_type==='REPAIR_DOCUMENT_PART'?data.correction:data.review);
      return <div key={action.id} className="repository-action">
        <strong>{labels[action.action_type]}</strong>{path&&<> · <code>{path}</code></>}
        {action.status==='FAILED'&&<p>Действие не выполнено.</p>}
        {action.action_type==='INVENTORY_SUMMARY'&&Array.isArray(data.sources)&&<p>Документов для пересказа: {data.sources.length}.</p>}
        {action.action_type==='VERIFY_DOCUMENT_PART'&&<p>{data.status==='SUPPORTED'
          ? 'Пересказ этой части проверен.' : 'Проверка этой части не завершена или содержит замечания.'}</p>}
        {checks.length>0&&<details><summary>Предложения и доказательства</summary>
          {checks.map((check,index)=><div key={index}>
            <p><strong>{check.supported?'Подтверждено':'Не подтверждено'}:</strong> {check.text}</p>
            {check.quote&&<blockquote>{check.quote}</blockquote>}
          </div>)}
        </details>}
        {typeof data.diff==='string'&&data.diff&&<pre>{data.diff}</pre>}
        {action.action_type==='REPOSITORY_CHECK'&&<>
          <p>{String(data.check||'')} · {data.passed?'Пройдена':'Не пройдена'} · код выхода {String(data.exit_code??'—')}
            {data.timed_out===true&&' · лимит времени'}{data.truncated===true&&' · вывод сокращён'}</p>
          {typeof data.output==='string'&&<pre>{data.output}</pre>}
        </>}
      </div>;
    })}
  </details>;
}
