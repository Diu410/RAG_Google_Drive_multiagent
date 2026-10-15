import React from 'react';
import {createRoot} from 'react-dom/client';
import App from './App';
import './style.css';
import {setApiBaseUrl,setApiAccessToken} from './api';

type BackendStatus={state:'starting'|'ready'|'error';base_url:string;access_token?:string;message?:string;log?:string};

function StartupError({status,onRetry}:{status:BackendStatus;onRetry:()=>void}){
  return <main className="startup-screen"><section><h1>Не удалось запустить локальный сервер</h1><p>{status.message||'Неизвестная ошибка запуска.'}</p>{status.log&&<details><summary>Журнал запуска</summary><pre>{status.log}</pre></details>}<button onClick={onRetry}>Повторить</button></section></main>;
}

function DesktopBootstrap(){
  const [status,setStatus]=React.useState<BackendStatus>({state:'starting',base_url:''});
  const check=React.useCallback(async()=>{
    try{
      const {invoke}=await import('@tauri-apps/api/core');
      const next=await invoke<BackendStatus>('backend_status');
      if(next.state==='ready'){setApiBaseUrl(next.base_url);setApiAccessToken(next.access_token||null);}
      setStatus(next);
    }catch(error){setStatus({state:'error',base_url:'',message:String(error)});}
  },[]);
  React.useEffect(()=>{void check();const timer=setInterval(()=>void check(),400);return()=>clearInterval(timer);},[check]);
  if(status.state==='starting')return <main className="startup-screen"><section><h1>Фортуна</h1><p>Запуск локальных сервисов...</p></section></main>;
  if(status.state==='error')return <StartupError status={status} onRetry={async()=>{setStatus({state:'starting',base_url:''});const {invoke}=await import('@tauri-apps/api/core');await invoke('retry_backend');void check();}}/>;
  return <App/>;
}

const desktop='__TAURI_INTERNALS__' in window;
createRoot(document.getElementById('root')!).render(desktop?<DesktopBootstrap/>:<App/>);
