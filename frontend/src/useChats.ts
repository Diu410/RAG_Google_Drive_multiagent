import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from './api';
import type { Chat } from './chatTypes';

type Page = {items: Chat[]; next_cursor: string | null};
export function mergeChatSummary(cached: Chat | undefined, summary: Chat): Chat {
  return cached && (cached.version ?? 0) >= (summary.version ?? 0) ? cached : summary;
}

export function useChats(onError: (error: unknown) => void, search = '') {
  const [chats, setChats] = useState<Chat[]>([]);
  const [active, setActive] = useState<string | null>(() => localStorage.getItem('kontur-active'));
  const [loaded, setLoaded] = useState(false);
  const [loadingChat, setLoadingChat] = useState(false);
  const [nextCursor, setNextCursor] = useState<string | null>(null);
  const [loadingMore, setLoadingMore] = useState(false);
  const [detailRefresh, setDetailRefresh] = useState(0);
  const report = useRef(onError), activeRef = useRef(active);
  const chatsRef = useRef(chats);
  report.current = onError;
  activeRef.current = active;
  chatsRef.current = chats;
  const request = useRef(0), detailRequest = useRef(0);
  const accept = useCallback((chat: Chat) => {
    setChats(previous => {
      const existing = previous.find(item => item.id === chat.id);
      if ((existing?.version ?? 0) > (chat.version ?? 0)) return previous;
      return [...previous.filter(item => item.id !== chat.id), {...chat, messages_loaded: true}]
        .sort((a, b) => b.updated_at.localeCompare(a.updated_at));
    });
  }, []);
  const refreshChangedActive = useCallback((items: Chat[]) => {
    const summary = items.find(item => item.id === activeRef.current);
    const cached = chatsRef.current.find(item => item.id === activeRef.current);
    if (summary && (!cached || cached.messages_loaded === false ||
        (summary.version ?? 0) > (cached.version ?? 0))) setDetailRefresh(value => value + 1);
  }, []);
  const load = useCallback(async () => {
    const generation = ++request.current;
    try {
      const page = await api<Page>('/chats/page?limit=50&search=' + encodeURIComponent(search));
      if (generation === request.current) {
        setChats(previous => {
          const cached = new Map(previous.map(item => [item.id,item]));
          const items = page.items.map(item => mergeChatSummary(cached.get(item.id), item));
          const selected = previous.find(item => item.id === activeRef.current);
          if (selected && !items.some(item => item.id === selected.id)) items.push(selected);
          return items;
        });
        setNextCursor(page.next_cursor);
        setLoaded(true);
        refreshChangedActive(page.items);
      }
    } catch (error) {if (generation === request.current) report.current(error);}
  }, [search, refreshChangedActive]);
  useEffect(() => {
    const timer = setTimeout(() => void load(), search ? 200 : 0);
    return () => {clearTimeout(timer); request.current++;};
  }, [load, search]);
  const loadMore = useCallback(async () => {
    if (!nextCursor || loadingMore) return;
    const generation = request.current;
    setLoadingMore(true);
    try {
      const page = await api<Page>('/chats/page?limit=50&search=' + encodeURIComponent(search) + '&cursor=' + encodeURIComponent(nextCursor));
      if (generation === request.current) {
        setChats(previous => {
          const incoming = new Map(page.items.map(item => [item.id, item]));
          return [...previous.map(item => incoming.has(item.id) ? mergeChatSummary(item, incoming.get(item.id)!) : item),
            ...page.items.filter(item => !previous.some(p => p.id === item.id))];
        });
        setNextCursor(page.next_cursor);
        refreshChangedActive(page.items);
      }
    } catch (error) {report.current(error);} finally {setLoadingMore(false);}
  }, [nextCursor, loadingMore, search, refreshChangedActive]);
  useEffect(() => {
    if (active) localStorage.setItem('kontur-active', active);
    else localStorage.removeItem('kontur-active');
    const generation = ++detailRequest.current;
    if (!active) {setLoadingChat(false); return;}
    setLoadingChat(true);
    api<Chat>('/chats/' + encodeURIComponent(active)).then(chat => {
      if (generation === detailRequest.current) accept(chat);
    }).catch(error => {if (generation === detailRequest.current) report.current(error);})
      .finally(() => {if (generation === detailRequest.current) setLoadingChat(false);});
    return () => {detailRequest.current++;};
  }, [active, accept, detailRefresh]);
  const save = useCallback(async (chat: Chat) => {
    const saved = await api<Chat>('/chats/' + encodeURIComponent(chat.id), {method:'PUT',body:JSON.stringify(chat)});
    accept(saved);
    return saved;
  }, [accept]);
  return {chats,setChats,active,setActive,loaded,loadingChat,load,loadMore,hasMore:!!nextCursor,loadingMore,save,accept};
}
