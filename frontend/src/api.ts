/// <reference types="vite/client" />
let desktopBaseUrl: string | null = null;
let accessToken: string | null = null;
let accessRequest: Promise<string> | null = null;

export function setApiAccessToken(value: string | null) {
  accessToken = value;
  accessRequest = null;
}

async function requestHeaders(options: RequestInit = {}) {
  if (!accessToken) {
    accessRequest ||= fetch(getApiBaseUrl() + '/access-token', {
      headers: {'X-Fortuna-Client': 'web'}, cache: 'no-store', signal: AbortSignal.timeout(10000),
    }).then(async response => {
      if (!response.ok) throw new Error('Не удалось подключиться к локальному API. Перезапустите приложение.');
      const value = await response.json();
      if (typeof value.token !== 'string' || !value.token) throw new Error('Некорректный ответ локального API.');
      accessToken = value.token;
      return value.token as string;
    }).catch(error => {accessRequest = null; throw error;});
    await accessRequest;
  }
  const headers = new Headers(options.headers);
  headers.set('X-Fortuna-Token', accessToken!);
  if (options.body && !(options.body instanceof FormData)) headers.set('Content-Type', 'application/json');
  return headers;
}

async function authenticatedFetch(path: string, options: RequestInit = {}) {
  const headers = await requestHeaders(options);
  let response = await fetch(getApiBaseUrl() + path, {...options, headers});
  // Dev reload rotates the process token. A rejected request has not reached
  // its handler, so it can be retried once with a new web bootstrap token.
  if (response.status === 401 && !desktopBaseUrl && !options.signal?.aborted) {
    if (headers.get('X-Fortuna-Token') === accessToken) setApiAccessToken(null);
    response = await fetch(getApiBaseUrl() + path, {...options, headers: await requestHeaders(options)});
  }
  return response;
}

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
  const response=await authenticatedFetch(path,options);
  if(!response.ok){
    const detail=await response.json().catch(()=>null);
    if((((path==='/assistant/input'||path==='/query'||path.startsWith('/agent/'))&&response.status>=400)||path.startsWith('/workspaces')||path.startsWith('/graph/')||path==='/api/chat-model'||path==='/upload-gdrive')&&typeof detail?.detail==='string')throw new Error(detail.detail);
    if(response.status===409&&path.startsWith('/chats/'))throw new Error('Диалог изменился. Откройте его заново перед сохранением.');
    const messages:Record<number,string>={401:'Подключение к серверу изменилось. Перезапустите приложение.',413:'Файл слишком большой.',422:'Проверьте содержимое запроса или файла.',503:'Сервис недоступен. Проверьте статус модели или подключение Google Drive.',502:'Не удалось получить ответ модели.',501:'Распознавание речи не настроено.'};
    throw new Error(messages[response.status] || `Не удалось выполнить действие (${response.status}).`);
  }
  return response.json();
}
export function post<T>(path:string,data:unknown){return api<T>(path,{method:'POST',body:JSON.stringify(data)});}
export async function upload(path:string,file:File){const body=new FormData();body.append('file',file);return api<any>(path,{method:'POST',body});}
export async function postBlob(path:string,data:unknown):Promise<Blob>{
  const options = {method:'POST',body:JSON.stringify(data)};
  const response=await authenticatedFetch(path,options);
  if(!response.ok)throw new Error((await response.json().catch(()=>null))?.detail||`Voice service unavailable (${response.status}).`);
  return response.blob();
}
export async function streamJsonLines(path:string,data:unknown,onEvent:(value:any)=>void,options:{signal?:AbortSignal;timeoutMs?:number}={}):Promise<void>{
  const timeout=options.timeoutMs ? AbortSignal.timeout(options.timeoutMs) : undefined;
  const signal=options.signal&&timeout ? AbortSignal.any([options.signal,timeout]) : options.signal||timeout;
  const request = {method:'POST',body:JSON.stringify(data),signal};
  const response=await authenticatedFetch(path,request);
  if(!response.ok)throw new Error((await response.json().catch(()=>null))?.detail||`Не удалось открыть поток ответа (${response.status}).`);
  if(!response.body)throw new Error('Браузер не поддерживает потоковый ответ.');
  const reader=response.body.getReader(),decoder=new TextDecoder();let pending='';
  try{while(true){const {done,value}=await reader.read();pending+=decoder.decode(value||new Uint8Array(),{stream:!done});const lines=pending.split('\n');pending=lines.pop()||'';for(const line of lines)if(line.trim())onEvent(JSON.parse(line));if(done)break;}}
  finally{reader.releaseLock();}
  if(pending.trim())onEvent(JSON.parse(pending));
}
