import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from './api';
import type { Chat } from './chatTypes';

export function useChats(onError: (error: unknown) => void) {
  const [chats, setChats] = useState<Chat[]>([]);
  const [active, setActive] = useState<string | null>(() => localStorage.getItem('kontur-active'));
  const [loaded, setLoaded] = useState(false);
  const report = useRef(onError);
  report.current = onError;
  const request = useRef(0);

  const load = useCallback(async () => {
    const generation = ++request.current;
    try {
      const value = await api<Chat[]>('/chats');
      if (generation === request.current) {
        setChats(value);
        setLoaded(true);
      }
    } catch (error) {
      if (generation === request.current) report.current(error);
    }
  }, []);

  useEffect(() => {
    void load();
    return () => { request.current++; };
  }, [load]);

  useEffect(() => {
    if (active) localStorage.setItem('kontur-active', active);
    else localStorage.removeItem('kontur-active');
  }, [active]);

  const save = useCallback(async (chat: Chat) => {
    const saved = await api<Chat>('/chats/' + encodeURIComponent(chat.id), {
      method: 'PUT', body: JSON.stringify(chat),
    });
    setChats(previous => [...previous.filter(item => item.id !== saved.id), saved]
      .sort((a, b) => b.updated_at.localeCompare(a.updated_at)));
  }, []);

  return { chats, setChats, active, setActive, loaded, load, save };
}
