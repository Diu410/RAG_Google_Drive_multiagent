import { useEffect, useState } from 'react';
import { api } from './api';

export function useModelStatus() {
  const [model, setModel] = useState('BALANCED');
  const [modelStatus, setModelStatus] = useState('');
  useEffect(() => {
    let disposed = false;
    let initialized = false;
    let timer: ReturnType<typeof setTimeout>;
    const controller = new AbortController();
    const check = async () => {
      try {
        const status = await api<{ message: string; default_profile?: string }>('/api/model-status', { signal: controller.signal });
        if (!disposed) {
          setModelStatus(status.message);
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
  return { model, setModel, modelStatus };
}
