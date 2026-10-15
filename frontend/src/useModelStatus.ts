import { useEffect, useRef, useState } from 'react';
import { api, post } from './api';

type ChatModel = { id: string; label: string; available: boolean };
type ModelStatus = { message: string; default_profile?: string; ready?: boolean;
  state?: string; process_started?: boolean; error?: string | null;
  chat_models?: ChatModel[]; selected_model_id?: string | null };

export function useModelStatus() {
  const [model, setModel] = useState('BALANCED');
  const [modelStatus, setModelStatus] = useState('');
  const [chatModels, setChatModels] = useState<ChatModel[]>([]);
  const [selectedChatModel, setSelectedChatModel] = useState('');
  const [switchingModel, setSwitchingModel] = useState(false);
  const [modelReady, setModelReady] = useState(false);
  const [modelLoading, setModelLoading] = useState(false);
  const revision = useRef(0);
  useEffect(() => {
    let disposed = false;
    let initialized = false;
    let timer: ReturnType<typeof setTimeout>;
    const controller = new AbortController();
    const check = async () => {
      const requestRevision = revision.current;
      try {
        const status = await api<ModelStatus>('/api/model-status', { signal: controller.signal });
        if (!disposed && requestRevision === revision.current) {
          setModelStatus(status.message);
          setChatModels(status.chat_models || []);
          setSelectedChatModel(status.selected_model_id || '');
          setModelReady(!!status.ready);
          setModelLoading(status.state === 'loading' || (!!status.process_started && !status.ready && !status.error));
          if (!initialized && status.default_profile) {
            setModel(status.default_profile);
            initialized = true;
          }
        }
      } catch { /* Model startup is allowed to take longer than a poll. */ }
      if (!disposed) timer = setTimeout(check, 2000);
    };
    void check();
    return () => { disposed = true; clearTimeout(timer); controller.abort(); };
  }, []);
  const switchChatModel = async (modelId: string) => {
    revision.current += 1;
    setSwitchingModel(true);
    try {
      const status = await post<ModelStatus>('/api/chat-model', { model_id: modelId });
      revision.current += 1;
      setSelectedChatModel(status.selected_model_id || '');
      setChatModels(status.chat_models || []);
      setModelReady(false);
      setModelLoading(true);
      setModelStatus('Загрузка выбранной модели...');
    } finally { setSwitchingModel(false); }
  };
  return { model, setModel, modelStatus, chatModels, selectedChatModel,
    switchChatModel, switchingModel, modelReady, modelLoading };
}
