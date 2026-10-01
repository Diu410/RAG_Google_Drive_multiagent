/// <reference types="vite/client" />
let desktopBaseUrl: string | null = null;

export function setApiBaseUrl(value: string | null) {
  desktopBaseUrl = value?.replace(/\/$/, '') || null;
}

export function getApiBaseUrl() {
  const configured = import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, '');
  if (desktopBaseUrl) return desktopBaseUrl;
  if (configured) return configured;
  return import.meta.env.DEV ? '/api' : '';
}

export async function api<T>(path:string, options:RequestInit={}):Promise<T>{
  const response=await fetch(getApiBaseUrl()+path,{...options,headers:options.body instanceof FormData ? options.headers : {'Content-Type':'application/json',...options.headers}});
  if(!response.ok){
    const detail=await response.json().catch(()=>null);
    if((((path==='/assistant/input'||path==='/query'||path.startsWith('/agent/'))&&response.status>=400)||path.startsWith('/workspaces'))&&typeof detail?.detail==='string')throw new Error(detail.detail);
    const messages:Record<number,string>={413:'Файл слишком большой.',422:'Проверьте содержимое запроса или файла.',503:'Сервис недоступен. Проверьте статус модели или подключение Google Drive.',502:'Не удалось получить ответ модели.',501:'Распознавание речи не настроено.'};
    throw new Error(messages[response.status] || `Не удалось выполнить действие (${response.status}).`);
  }
  return response.json();
}
export function post<T>(path:string,data:unknown){return api<T>(path,{method:'POST',body:JSON.stringify(data)});}
export async function upload(path:string,file:File){const body=new FormData();body.append('file',file);return api<any>(path,{method:'POST',body});}
export async function postBlob(path:string,data:unknown):Promise<Blob>{
  const response=await fetch(getApiBaseUrl()+path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
  if(!response.ok)throw new Error((await response.json().catch(()=>null))?.detail||`Voice service unavailable (${response.status}).`);
  return response.blob();
}
export async function streamJsonLines(path:string,data:unknown,onEvent:(value:any)=>void,options:{signal?:AbortSignal;timeoutMs?:number}={}):Promise<void>{
  const timeout=options.timeoutMs ? AbortSignal.timeout(options.timeoutMs) : undefined;
  const signal=options.signal&&timeout ? AbortSignal.any([options.signal,timeout]) : options.signal||timeout;
  const response=await fetch(getApiBaseUrl()+path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data),signal});
  if(!response.ok)throw new Error((await response.json().catch(()=>null))?.detail||`Не удалось открыть поток ответа (${response.status}).`);
  if(!response.body)throw new Error('Браузер не поддерживает потоковый ответ.');
  const reader=response.body.getReader(),decoder=new TextDecoder();let pending='';
  try{while(true){const {done,value}=await reader.read();pending+=decoder.decode(value||new Uint8Array(),{stream:!done});const lines=pending.split('\n');pending=lines.pop()||'';for(const line of lines)if(line.trim())onEvent(JSON.parse(line));if(done)break;}}
  finally{reader.releaseLock();}
  if(pending.trim())onEvent(JSON.parse(pending));
}
