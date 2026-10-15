import {
  lazy,
  Suspense,
  useEffect,
  useRef,
  useState,
  type FormEvent,
} from "react";
import * as Menu from "@radix-ui/react-dropdown-menu";
import * as Dialog from "@radix-ui/react-dialog";
import {
  ArrowUp,
  Check,
  ChevronLeft,
  Copy,
  Download,
  File,
  FileText,
  Folder,
  FolderOpen,
  GitBranch,
  HardDrive,
  Menu as MenuIcon,
  Mic,
  MoreVertical,
  PanelLeftClose,
  Paperclip,
  Pencil,
  Plus,
  RotateCcw,
  Search,
  Square,
  Trash2,
  X,
} from "lucide-react";
import { api, post, streamJsonLines, upload } from "./api";
import { processAssistantStream, processAssistantLive } from "./assistant";
import { useChats } from "./useChats";
import { useModelStatus } from "./useModelStatus";
import type { Chat, GraphInfo, Message, Verification } from "./chatTypes";
const GraphExplorer = lazy(() => import("./GraphExplorer").then(module => ({ default: module.GraphExplorer })));
import "./router.css";
import "./workspace.css";

type Doc = {
  id: number;
  filename: string;
  upload_timestamp: string;
  can_reindex?: boolean;
};
type WorkspaceIndexStatus = {
  readable_documents: number; indexed_documents: number; failed_documents: number;
  truncated: boolean; scan_errors: number;
};
type WorkspaceIndexResult = {indexed: number; unchanged: number; failed: number; unavailable: boolean};
type Workspace = {
  id: string;
  name: string;
  root_path: string;
  created_at: string;
  updated_at: string;
  status: string;
  metadata: Record<string, unknown>;
};
type WorkspaceEntry = {
  name: string;
  relative_path: string;
  kind: "directory" | "file" | "symlink";
  size: number | null;
  modified_at: string | null;
  navigable: boolean;
  readable: boolean;
};
type WorkspaceListing = {
  workspace: Workspace;
  path: string;
  parent: string | null;
  entries: WorkspaceEntry[];
  truncated: boolean;
};
type WorkspaceFile = { workspace_id: string; path: string; content: string };
const stamp = () => new Date().toISOString();
const title = (s: string) => {
  const t = s
    .replace(
      /^(напиши|составь|подготовь|создай|расскажи|объясни)\s+(мне\s+)?/i,
      "",
    )
    .trim();
  return (t.charAt(0).toUpperCase() + t.slice(1)).slice(0, 65);
};
const date = (s: string) => {
  const d = new Date(s);
  return Number.isNaN(d.valueOf())
    ? s
    : d.toLocaleString("ru-RU", {
        day: "2-digit",
        month: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
      });
};
function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState(false);
  return (
    <button
      className="icon"
      title={
        error
          ? "Буфер недоступен. Выделите текст и нажмите Ctrl+C"
          : "Скопировать сообщение"
      }
      aria-label="Скопировать сообщение"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(text);
          setCopied(true);
          setTimeout(() => setCopied(false), 1800);
        } catch {
          setError(true);
        }
      }}
    >
      {copied ? <Check size={15} /> : <Copy size={15} />}{" "}
      {error && <small>Выделите текст и нажмите Ctrl+C</small>}
    </button>
  );
}
const MarkdownRenderer = lazy(() => import("./SafeMarkdown"));
function SafeMarkdown({ children }: { children: string }) {
  return <Suspense fallback={<span className="formula-raw">{children}</span>}>
    <MarkdownRenderer>{children}</MarkdownRenderer>
  </Suspense>;
}
export default function App() {
  const [error, setError] = useState("");
  const report = (error: unknown) => setError(error instanceof Error ? error.message : "Не удалось выполнить действие. Проверьте подключение к серверу.");
  const [search, setSearch] = useState("");
  const { chats, setChats, active, setActive, loaded, loadingChat, load, loadMore, hasMore, loadingMore, save, accept } = useChats(report, search);
  const { model, setModel, modelStatus, chatModels, selectedChatModel,
    switchChatModel, switchingModel, modelLoading } = useModelStatus();
  const [sidebar, setSidebar] = useState(window.innerWidth > 760),
    [input, setInput] = useState(""),
    [busy, setBusy] = useState(false);
  const [workspace, setWorkspace] = useState<Workspace | null>(null),
    [workspaceOpen, setWorkspaceOpen] = useState(false),
    [workspaceListing, setWorkspaceListing] = useState<WorkspaceListing | null>(
      null,
    ),
    [workspaceFile, setWorkspaceFile] = useState<WorkspaceFile | null>(null),
    [workspaceBusy, setWorkspaceBusy] = useState(false);
  const [workspaceIndexBusy, setWorkspaceIndexBusy] = useState(false);
  const [workspaceIndexStatus, setWorkspaceIndexStatus] = useState<WorkspaceIndexStatus | null>(null);
  const [workspaceIndexMessage, setWorkspaceIndexMessage] = useState('');
  const [documentsOpen, setDocumentsOpen] = useState(false),
    [docs, setDocs] = useState<Doc[]>([]),
    [drive, setDrive] = useState(""),
    [files, setFiles] = useState<File[]>([]);
  const [graphBrowse, setGraphBrowse] = useState(false);
  const [rename, setRename] = useState<Chat | null>(null),
    [renameText, setRenameText] = useState(""),
    [deleting, setDeleting] = useState<Chat | null>(null);
  const [retry, setRetry] = useState<Chat | null>(null),
    [retryLive, setRetryLive] = useState(false);
  const [recording, setRecording] = useState(false),
    [live, setLive] = useState(false),
    [liveRunning, setLiveRunning] = useState(false),
    [voiceDraft, setVoiceDraft] = useState(false),
    [voiceState, setVoiceState] = useState(""),
    [voiceProvider, setVoiceProvider] = useState("Проверяю голос…");
  const recorder = useRef<MediaRecorder | null>(null),
    audio = useRef<HTMLAudioElement | null>(null),
    bottom = useRef<HTMLDivElement>(null),
    fileInput = useRef<HTMLInputElement>(null);
  const recordInterval = useRef<number | null>(null),
    vadFrame = useRef<number | null>(null),
    captureContext = useRef<AudioContext | null>(null);
  const audioContext = useRef<AudioContext | null>(null),
    nextAudioTime = useRef(0),
    voiceQueue = useRef<Promise<void>>(Promise.resolve()),
    audioSources = useRef<Set<AudioBufferSourceNode>>(new Set());
  const liveAbort = useRef<AbortController | null>(null),
    liveTaskId = useRef<string | null>(null),
    liveRun = useRef(0),
    liveRef = useRef(false),
    discardRecording = useRef(false),
    partialRequest = useRef(0);
  const chat = chats.find((c) => c.id === active);
  const locked = busy || recording || switchingModel || loadingChat;
  useEffect(() => {
    api<{ configured: boolean; ready: boolean; fallback?: string | null }>(
      "/voice/status",
    )
      .then((s) =>
        setVoiceProvider(
          s.ready
            ? "CosyVoice готов"
            : s.fallback
              ? "Системный голос"
              : "Озвучивание недоступно",
        ),
      )
      .catch(() => setVoiceProvider("Статус голоса недоступен"));
    return () => {
      recorder.current?.stream.getTracks().forEach((t) => t.stop());
      if (recordInterval.current != null) clearInterval(recordInterval.current);
      if (vadFrame.current != null) cancelAnimationFrame(vadFrame.current);
      void captureContext.current?.close();
      liveAbort.current?.abort();
      stopAudio();
    };
  }, []);
  useEffect(() => {
    api<{ workspace: Workspace | null }>("/workspaces/active")
      .then((r) => setWorkspace(r.workspace))
      .catch(() => {});
  }, []);
  useEffect(() => {
    let disposed = false;
    if (workspaceOpen && workspace && !workspaceIndexBusy) {
      setWorkspaceIndexStatus(null);
      api<WorkspaceIndexStatus>(`/workspaces/${encodeURIComponent(workspace.id)}/index-status`)
        .then(status => {if (!disposed) setWorkspaceIndexStatus(status);})
        .catch(error => {if (!disposed) report(error);});
    }
    return () => {disposed = true;};
  }, [workspace?.id, workspaceOpen, workspaceIndexBusy]);
  useEffect(() => {
    setWorkspaceIndexStatus(null);
    setWorkspaceIndexMessage('');
  }, [workspace?.id]);
  useEffect(() => {
    liveRef.current = live;
    if (live && !recording && !busy && !liveRunning && !recorder.current)
      void record();
  }, [live]);
  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth" });
  }, [
    active,
    chat?.messages.length,
    chat?.messages.at(-1)?.content,
    busy,
  ]);
  function open(c: Chat) {
    setActive(c.id);
    setRetry(null);
    setError("");
    if (window.innerWidth < 760) setSidebar(false);
  }
  function fresh() {
    setActive(null);
    setInput("");
    setRetry(null);
    setFiles([]);
    setError("");
  }
  function stopAudio(invalidate = true) {
    if (invalidate) liveRun.current++;
    audio.current?.pause();
    audio.current = null;
    if ("speechSynthesis" in window) speechSynthesis.cancel();
    for (const source of audioSources.current) {
      try {
        source.stop();
      } catch {}
    }
    audioSources.current.clear();
    nextAudioTime.current = 0;
    voiceQueue.current = Promise.resolve();
  }
  async function speakStream(text: string, run: number) {
    let received = false,
      sampleRate = 24000;
    const playback: Promise<void>[] = [];
    try {
      await streamJsonLines("/voice/stream", { text }, (event) => {
        if (run !== liveRun.current) return;
        if (event.type === "format") sampleRate = event.sample_rate;
        if (event.type === "error") throw new Error(event.message);
        if (event.type === "audio") {
          received = true;
          const raw = atob(event.pcm),
            pcm = new Int16Array(raw.length / 2);
          for (let i = 0; i < pcm.length; i++)
            pcm[i] =
              ((raw.charCodeAt(i * 2) | (raw.charCodeAt(i * 2 + 1) << 8)) <<
                16) >>
              16;
          const context =
            audioContext.current || (audioContext.current = new AudioContext());
          const buffer = context.createBuffer(1, pcm.length, sampleRate),
            channel = buffer.getChannelData(0);
          for (let i = 0; i < pcm.length; i++) channel[i] = pcm[i] / 32768;
          const source = context.createBufferSource();
          audioSources.current.add(source);
          playback.push(
            new Promise((resolve) => {
              source.onended = () => {
                audioSources.current.delete(source);
                resolve();
              };
            }),
          );
          source.buffer = buffer;
          source.connect(context.destination);
          const start = Math.max(
            context.currentTime + 0.03,
            nextAudioTime.current,
          );
          source.start(start);
          nextAudioTime.current = start + buffer.duration;
        }
      });
      if (received || run !== liveRun.current) {
        await Promise.all(playback);
        return;
      }
    } catch {
      if (run !== liveRun.current) return;
      setVoiceProvider("CosyVoice недоступен — системный голос");
    }
    if (
      !("speechSynthesis" in window) ||
      typeof SpeechSynthesisUtterance === "undefined"
    )
      throw new Error("Озвучивание недоступно в этой системе.");
    await new Promise<void>((resolve) => {
      const utterance = new SpeechSynthesisUtterance(
        text.replace(/[`*_#>|]/g, " "),
      );
      utterance.lang = "ru-RU";
      utterance.rate = 1.05;
      utterance.onend = () => resolve();
      utterance.onerror = () => resolve();
      speechSynthesis.speak(utterance);
    });
  }
  async function answerLive(c: Chat) {
    type LiveResponse = {
      answer: string;
      session_id: string;
      chat_version?: number;
      context: string;
      strategy?: string;
      verification?: Verification | null;
      graph?: GraphInfo | null;
      draft: string;
      edited: string;
      enhanced: string[];
    };
    let partial = "",
      sentence = "",
      result: LiveResponse | undefined;
    const responseStamp = stamp();
    stopAudio();
    const run = liveRun.current,
      controller = new AbortController();
    liveAbort.current = controller;
    liveTaskId.current = null;
    setLiveRunning(true);
    setVoiceState("Анализирую запрос…");
    setChats((prev) =>
      prev.map((item) =>
        item.id === c.id
          ? {
              ...c,
              messages: [
                ...c.messages,
                { role: "assistant", content: "", timestamp: responseStamp },
              ],
            }
          : item,
      ),
    );
    try {
        await processAssistantLive<LiveResponse>(
          {
          question: c.messages.at(-1)!.content,
          request_id: c.messages.at(-1)?.id,
          session_id: c.session_id,
          chat_id: c.id,
          model,
          use_multi_agent: false,
          workspace_id: workspace?.id,
          live_mode: true,
        },
        (event) => {
          if (event.type === "task" && event.task_id)
            liveTaskId.current = event.task_id;
          if (event.type === "stage")
            setVoiceState(
              event.stage === "answering"
                ? "Формирую проверенный ответ…"
                : "Анализирую запрос…",
            );
          if (event.type === "reset") {
            partial = "";
            sentence = "";
            stopAudio(false);
          }
          if (event.type === "token" && event.text) {
            partial += event.text;
            sentence += event.text;
            setVoiceState("Отвечаю и озвучиваю…");
            setChats((prev) =>
              prev.map((item) =>
                item.id === c.id
                  ? {
                      ...item,
                      messages: [
                        ...c.messages,
                        {
                          role: "assistant",
                          content: partial,
                          timestamp: responseStamp,
                        },
                      ],
                    }
                  : item,
              ),
            );
            const punctuation = sentence.search(/[.!?…](?:\s|$)/);
            let cut = punctuation >= 34 ? punctuation + 1 : -1;
            if (cut < 0 && sentence.length >= 180) {
              const space = sentence.lastIndexOf(" ", 180);
              cut = space > 40 ? space + 1 : 180;
            }
            if (cut > 0) {
              const ready = sentence.slice(0, cut);
              sentence = sentence.slice(cut);
              voiceQueue.current = voiceQueue.current
                .catch(() => {})
                .then(() => speakStream(ready, run));
            }
          }
          if (event.type === "result" && event.response)
            result = event.response;
          if (event.type === "error")
            throw new Error(
              event.message || "Live-ответ завершился с ошибкой.",
            );
        },
        controller.signal,
      );
      if (!result) throw new Error("Live-ответ не был завершён.");
      if (!partial) {
        partial = result.answer;
        sentence = result.answer;
      }
      if (sentence.trim())
        voiceQueue.current = voiceQueue.current
          .catch(() => {})
          .then(() => speakStream(sentence, run));
      const r = result,
        next = {
          ...c,
          session_id: r.session_id,
          version: r.chat_version ?? c.version,
          messages: [
            ...c.messages,
            {
              role: "assistant" as const,
              content: r.answer,
              timestamp: responseStamp,
              context: r.context,
              strategy: r.strategy,
              verification: r.verification,
              graph: r.graph,
              steps: { draft: r.draft, edited: r.edited, enhanced: r.enhanced },
            },
          ],
        };
      accept(await api<Chat>('/chats/' + encodeURIComponent(c.id)).catch(() => next));
      setRetry(null);
      void voiceQueue.current.catch(report).finally(() => {
        if (run === liveRun.current) {
          setVoiceState("");
          if (liveRef.current) void record();
        }
      });
    } catch (e) {
      setChats((prev) =>
        prev.map((item) =>
          item.id === c.id ? { ...item, messages: c.messages } : item,
        ),
      );
      throw e;
    } finally {
      if (liveAbort.current === controller) liveAbort.current = null;
      liveTaskId.current = null;
      setLiveRunning(false);
    }
  }
  async function cancelLive() {
    const taskId = liveTaskId.current;
    liveAbort.current?.abort();
    stopAudio();
    setVoiceState("Остановлено");
    if (taskId)
      try {
        await post(`/assistant/tasks/${encodeURIComponent(taskId)}/cancel`, {});
      } catch {
        /* disconnect cancellation is also handled server-side */
      }
  }
  async function answer(c: Chat, speakAfter = false) {
    const graphSelection = c.messages.at(-1)?.graphSelection;
    let controller: AbortController | undefined;
    try {
      setRetry(c);
      setRetryLive(speakAfter);
      if (speakAfter) {await answerLive(c); return;}
      controller = new AbortController();
      liveAbort.current = controller;
      liveTaskId.current = null;
      setLiveRunning(true);
      type Result = {answer:string;session_id:string;chat_version?:number;task_id?:string;context:string;
        strategy?:string;verification?:Verification|null;graph?:GraphInfo|null;draft:string;edited:string;enhanced:string[]};
      let result: Result | undefined;
      await processAssistantStream<Result>({
        question:c.messages.at(-1)!.content,request_id:c.messages.at(-1)?.id,
        session_id:c.session_id,chat_id:c.id,model,use_multi_agent:true,
        workspace_id:graphSelection ? graphSelection.workspace_id : workspace?.id,graph_selection:graphSelection,
      }, event => {
        if (event.type === "task" && event.task_id) {liveTaskId.current = event.task_id;setVoiceState('Готовлю ответ…');}
        if (event.type === "result" && event.response) result = event.response;
        if (event.type === "error") throw new Error(event.message || "Не удалось завершить ответ.");
      }, controller.signal);
      if (!result) throw new Error("Ответ не был завершён.");
      const r = result;
      const next: Chat = {...c,session_id:r.session_id,version:r.chat_version ?? c.version,updated_at:stamp(),messages:[...c.messages,{
        role:"assistant",content:r.answer,timestamp:stamp(),reply_to:c.messages.at(-1)?.id,task_id:r.task_id,
        context:r.context,strategy:r.strategy,verification:r.verification,graph:r.graph,
        steps:{draft:r.draft,edited:r.edited,enhanced:r.enhanced},
      }]};
      accept(await api<Chat>('/chats/' + encodeURIComponent(c.id)).catch(() => next));
      setRetry(null);
    } catch (e) {
      // A disconnected response may already be committed. Recover it without generating again.
      let recovered = false;
      try {
        const saved = await api<Chat>("/chats/" + encodeURIComponent(c.id));
        const requestId = c.messages.at(-1)?.id;
        if (requestId && saved.messages.some(message => message.reply_to === requestId)) {
          accept(saved);setRetry(null);recovered = true;
        }
      } catch { /* Preserve the original transport error. */ }
      if (!recovered && !(e instanceof DOMException && e.name === "AbortError")) report(e);
      if (e instanceof DOMException && e.name === "AbortError") setRetry(null);
    } finally {
      if (controller && liveAbort.current === controller) {
        liveAbort.current = null;liveTaskId.current = null;setLiveRunning(false);
      }
      setBusy(false);
      if (!speakAfter) setVoiceState('');
    }
  }
  async function submitText(text: string, voice = false) {
    if (busy || loadingChat || !loaded || !text.trim()) return;
    setBusy(true);
    setError("");
    try {
      const c: Chat = {
        ...(chat || {
          id: crypto.randomUUID(),
          title: title(text),
          messages: [],
          session_id: null,
          updated_at: stamp(),
        }),
        updated_at: stamp(),
        messages: [
          ...(chat?.messages || []),
          { id:crypto.randomUUID(), role: "user", content: text.trim(), timestamp: stamp() },
        ],
      };
      const saved = await save(c);
      setActive(saved.id);
      setInput("");
      await answer(saved, voice);
    } catch (e) {
      report(e);
      setBusy(false);
    }
  }
  async function send(e: FormEvent) {
    e.preventDefault();
    if (locked || !loaded || (!input.trim() && !files.length)) return;
    setBusy(true);
    setError("");
    try {
      for (const file of files) {
        await upload("/upload-doc", file);
        setFiles((prev) => prev.filter((f) => f !== file));
      }
      if (!input.trim()) {
        setBusy(false);
        return;
      }
      const text = input.trim();
      setBusy(false);
      setVoiceDraft(false);
      await submitText(text, live);
    } catch (e) {
      report(e);
      setBusy(false);
    }
  }
  async function refreshDocs() {
    try {
      setDocs(await api<Doc[]>("/list-docs"));
    } catch (e) {
      report(e);
    }
  }
  async function loadWorkspaceFiles(selected: Workspace, path = "") {
    setWorkspaceBusy(true);
    setError("");
    setWorkspaceFile(null);
    try {
      const listing = await api<WorkspaceListing>(
        `/workspaces/${encodeURIComponent(selected.id)}/files?path=${encodeURIComponent(path)}`,
      );
      setWorkspace(selected);
      setWorkspaceListing(listing);
    } catch (e) {
      report(e);
    } finally {
      setWorkspaceBusy(false);
    }
  }
  async function readWorkspaceFile(selected: Workspace, path: string) {
    setWorkspaceBusy(true);
    setError("");
    try {
      setWorkspaceFile(
        await api<WorkspaceFile>(
          `/workspaces/${encodeURIComponent(selected.id)}/file?path=${encodeURIComponent(path)}`,
        ),
      );
    } catch (e) {
      report(e);
    } finally {
      setWorkspaceBusy(false);
    }
  }
  async function chooseWorkspace() {
    setWorkspaceBusy(true);
    setError("");
    try {
      if (!("__TAURI_INTERNALS__" in window))
        throw new Error("Выбор папки доступен в desktop-приложении.");
      const { open } = await import("@tauri-apps/plugin-dialog");
      const selected = await open({
        directory: true,
        multiple: false,
        title: "Выберите workspace",
      });
      if (!selected) return;
      const listing = await post<WorkspaceListing>("/workspaces/select", {
        root_path: selected,
      });
      setWorkspace(listing.workspace);
      setWorkspaceListing(listing);
      setWorkspaceOpen(true);
    } catch (e) {
      report(e);
    } finally {
      setWorkspaceBusy(false);
    }
  }
  async function indexWorkspace() {
    if (!workspace || workspaceIndexBusy) return;
    const selected = workspace;
    setWorkspaceIndexBusy(true);
    setWorkspaceIndexMessage('');
    setError('');
    try {
      const result = await post<WorkspaceIndexResult>(
        `/workspaces/${encodeURIComponent(selected.id)}/vector-sync?extract_relations=false`, {},
      );
      setWorkspaceIndexMessage(result.unavailable ? 'Векторный индекс недоступен. Проверьте embedding-модель.' :
        `Индексация завершена: обновлено ${result.indexed}, без изменений ${result.unchanged}.` +
        (result.failed ? ` Ошибок: ${result.failed}. Проверьте embedding-модель и журнал backend.` : '') +
        ' Откройте граф или обновите карту, чтобы увидеть связи.');
    } catch (error) {report(error);}
    finally {setWorkspaceIndexBusy(false);}
  }
  async function toggleLive() {
    if (liveRef.current) {
      liveRef.current = false;
      setLive(false);
      discardRecording.current = true;
      if (recorder.current?.state === "recording") recorder.current.stop();
      setRecording(false);
      stopAudio();
      setVoiceState("Live выключен");
      return;
    }
    liveRef.current = true;
    setLive(true);
    setVoiceDraft(false);
    setError("");
    await record();
  }
  async function record() {
    if (recording) {
      recorder.current?.stop();
      setRecording(false);
      return;
    }
    if (recorder.current?.state === "recording") return;
    stopAudio();
    if (
      !navigator.mediaDevices?.getUserMedia ||
      typeof MediaRecorder === "undefined"
    ) {
      setError("Запись голоса не поддерживается в этой системе.");
      liveRef.current = false;
      setLive(false);
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const media = new MediaRecorder(stream);
      recorder.current = media;
      discardRecording.current = false;
      const chunks: Blob[] = [];
      const started = Date.now();
      let heardSpeech = false,
        silentSince = started,
        lastPartial = 0,
        partialInFlight = false;
      const mime = media.mimeType || "audio/webm";
      const cleanupCapture = () => {
        if (recordInterval.current != null)
          clearInterval(recordInterval.current);
        recordInterval.current = null;
        if (vadFrame.current != null) cancelAnimationFrame(vadFrame.current);
        vadFrame.current = null;
        void captureContext.current?.close();
        captureContext.current = null;
      };
      const transcribe = async (final = false) => {
        if (!chunks.length || partialInFlight) return "";
        partialInFlight = true;
        const request = ++partialRequest.current;
        try {
          const r = await upload(
            "/transcribe",
            new globalThis.File([...chunks], "live.webm", { type: mime }),
          );
          if (request === partialRequest.current && r.text) setInput(r.text);
          return r.text || "";
        } catch (e) {
          if (final) report(e);
          return "";
        } finally {
          partialInFlight = false;
        }
      };
      media.ondataavailable = (e) => {
        if (e.data.size) chunks.push(e.data);
      };
      media.onstop = async () => {
        cleanupCapture();
        stream.getTracks().forEach((t) => t.stop());
        setRecording(false);
        if (discardRecording.current) {
          discardRecording.current = false;
          return;
        }
        setVoiceState("Собираю фразу…");
        const text = (await transcribe(true)).trim();
        if (!text) {
          setVoiceState(liveRef.current ? "Слушаю…" : "");
          if (liveRef.current) void record();
          return;
        }
        if (liveRef.current) {
          setInput("");
          setVoiceDraft(false);
          setVoiceState("Отправляю фразу…");
          await submitText(text, true);
        } else {
          setInput(text);
          setVoiceDraft(true);
          setVoiceState("Проверьте текст и отправьте");
        }
      };
      media.start(500);
      setRecording(true);
      setVoiceDraft(false);
      setVoiceState("Live слушает… говорите");
      recordInterval.current = window.setInterval(() => {
        const now = Date.now(),
          seconds = Math.floor((now - started) / 1000);
        setVoiceState(
          `${liveRef.current ? "Live слушает" : "Слушаю"}… ${seconds} с${heardSpeech ? " · распознаю речь" : ""}`,
        );
        if (heardSpeech && now - lastPartial > 1300) {
          lastPartial = now;
          void transcribe();
        }
        if (seconds >= 60 && media.state === "recording") media.stop();
      }, 400);
      if (stream instanceof MediaStream) {
        const context = new AudioContext(),
          analyser = context.createAnalyser(),
          source = context.createMediaStreamSource(stream),
          samples = new Uint8Array(analyser.fftSize);
        captureContext.current = context;
        source.connect(analyser);
        const watch = () => {
          analyser.getByteTimeDomainData(samples);
          let energy = 0;
          for (const value of samples) {
            const normalized = (value - 128) / 128;
            energy += normalized * normalized;
          }
          const rms = Math.sqrt(energy / samples.length);
          if (rms > 0.025) {
            heardSpeech = true;
            silentSince = Date.now();
          } else if (
            heardSpeech &&
            Date.now() - silentSince > 1100 &&
            Date.now() - started > 1400 &&
            !partialInFlight &&
            media.state === "recording"
          ) {
            media.stop();
            return;
          }
          vadFrame.current = requestAnimationFrame(watch);
        };
        watch();
      }
    } catch (e) {
      liveRef.current = false;
      setLive(false);
      setError(
        e instanceof DOMException && e.name === "NotAllowedError"
          ? "Доступ к микрофону запрещён. Разрешите его в настройках системы."
          : "Не удалось начать запись с микрофона.",
      );
    }
  }
  function download() {
    if (!chat) return;
    const a = document.createElement("a");
    const url = URL.createObjectURL(
      new Blob(
        [
          chat.messages
            .map(
              (m) =>
                `${m.role === "user" ? "Вы" : "Фортуна"} · ${date(m.timestamp)}\n\n${m.content}`,
            )
            .join("\n\n---\n\n"),
        ],
        { type: "text/markdown" },
      ),
    );
    a.href = url;
    a.download = chat.title + ".md";
    a.click();
    URL.revokeObjectURL(url);
  }
  return (
    <div className={"app " + (sidebar ? "" : "collapsed")}>
      {sidebar && (
        <aside>
          <div className="brand-row">
            <a className="brand" href="#" onClick={(e) => e.preventDefault()}>
              Фортуна<span>ЛИЧНЫЙ ПОМОЩНИК</span>
            </a>
            <button
              className="icon"
              aria-label="Свернуть панель"
              onClick={() => setSidebar(false)}
            >
              <PanelLeftClose size={19} />
            </button>
          </div>
          <button className="new-chat" onClick={fresh} disabled={locked}>
            <Plus size={18} /> Новый диалог
          </button>
          <button
            className="library-link"
            onClick={() => {
              setDocumentsOpen(true);
              void refreshDocs();
            }}
          >
            <FolderOpen size={17} /> Документы <span>{docs.length || ""}</span>
          </button>
          <button className="library-link" onClick={() => setGraphBrowse(true)}>
            <GitBranch size={17} /> Граф знаний
          </button>
          <button
            className={"workspace-link " + (workspace ? "active" : "")}
            onClick={() => {
              setWorkspaceOpen(true);
              if (workspace) void loadWorkspaceFiles(workspace);
            }}
          >
            <HardDrive size={17} />
            <span>
              <b>{workspace?.name || "Выбрать workspace"}</b>
              {workspace && (
                <small title={workspace.root_path}>{workspace.root_path}</small>
              )}
            </span>
          </button>
          <div className="history-heading">ИСТОРИЯ ЧАТОВ</div>
          <label className="search">
            <Search size={15} />
            <input
              aria-label="Поиск чатов"
              placeholder="Поиск диалога"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </label>
          <nav>
            {chats
              .filter((c) =>
                c.title
                  .toLocaleLowerCase()
                  .includes(search.toLocaleLowerCase()),
              )
              .map((c) => (
                <div
                  className={"chat-row " + (active === c.id ? "selected" : "")}
                  key={c.id}
                >
                  <button
                    className="chat-title"
                    title={c.title}
                    disabled={locked}
                    onClick={() => open(c)}
                  >
                    {c.title}
                  </button>
                  <Menu.Root>
                    <Menu.Trigger asChild>
                      <button
                        className="dots icon"
                        aria-label={"Действия: " + c.title}
                        disabled={locked}
                      >
                        <MoreVertical size={17} />
                      </button>
                    </Menu.Trigger>
                    <Menu.Portal>
                      <Menu.Content
                        className="dropdown"
                        side="right"
                        align="start"
                        sideOffset={8}
                      >
                        <Menu.Item
                          onSelect={() => {
                            setRename(c);
                            setRenameText(c.title);
                          }}
                        >
                          <Pencil size={15} />
                          Переименовать
                        </Menu.Item>
                        <Menu.Separator />
                        <Menu.Item
                          className="danger"
                          onSelect={() => setDeleting(c)}
                        >
                          <Trash2 size={15} />
                          Удалить чат
                        </Menu.Item>
                      </Menu.Content>
                    </Menu.Portal>
                  </Menu.Root>
                </div>
              ))}
            {hasMore && <button disabled={loadingMore || locked} onClick={() => void loadMore()}>
              {loadingMore ? "Загружаю…" : "Ещё диалоги"}
            </button>}
            {loaded && !chats.length && (
              <p className="empty-history">Ваши диалоги появятся здесь</p>
            )}
          </nav>
          <div className="sidebar-foot">
            <span className="status-dot" /> Личное пространство
          </div>
        </aside>
      )}
      <main>
        <header>
          <div className="header-left">
            {!sidebar && (
              <>
                <button
                  className="icon"
                  aria-label="Открыть панель"
                  onClick={() => setSidebar(true)}
                >
                  <MenuIcon />
                </button>
                <span className="mini-brand">Фортуна</span>
              </>
            )}
            <span className="current-title">
              {chat?.title || "Новый диалог"}
            </span>
          </div>
          <div className="header-actions">
            <small
              role="status"
              title={modelStatus}
              style={{
                maxWidth: "20vw",
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
              }}
            >
              {modelStatus}
            </small>
            <select
              aria-label="Локальная модель"
              value={selectedChatModel}
              onChange={(e) => void switchChatModel(e.target.value).catch(report)}
              disabled={locked || liveRunning || modelLoading || !chatModels.length}
            >
              {!chatModels.length && <option value="">Модели загружаются...</option>}
              {chatModels.map((choice) => <option key={choice.id} value={choice.id} disabled={!choice.available}>
                {choice.label}{!choice.available ? ' · не установлена' : ''}
              </option>)}
            </select>
            <select
              aria-label="Профиль ответа"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              disabled={locked}
            >
              <option value="FAST">FAST</option>
              <option value="BALANCED">BALANCED</option>
              <option value="REASONING">REASONING</option>
            </select>
            <button
              className="icon"
              aria-label="Скачать переписку"
              disabled={!chat}
              onClick={download}
            >
              <Download size={18} />
            </button>
          </div>
        </header>
        <div className="conversation">
          {!chat && (
            <div className="empty-chat">
              <div className="empty-symbol">
                <FileText size={25} />
              </div>
              <p>Место для ваших идей и документов</p>
              <span>Напишите сообщение или прикрепите файл</span>
            </div>
          )}
          {chat?.messages.map((m, i) => (
            <article className={"message " + m.role} key={i}>
              <div className="message-meta">
                <strong>{m.role === "user" ? "Вы" : "Фортуна"}</strong>
                <time>{date(m.timestamp)}</time>
              </div>
              <SafeMarkdown>{m.content}</SafeMarkdown>
              {m.role === "assistant" && m.strategy && (
                <span className="strategy">
                  {m.strategy.replaceAll("_", " ")}
                </span>
              )}
              {m.verification && (
                <div className="verification">
                  <div>
                    <strong>
                      {m.verification.review_mode === 'CONTEXTUAL'
                        ? "Сверка с источниками"
                        : m.strategy === "WORKSPACE_QA"
                        ? "Опора на источники"
                        : m.verification.confidence != null
                        ? `Уверенность: ${Math.round(m.verification.confidence * 100)}%`
                        : "Уверенность: недоступна"}
                    </strong>
                    <span>
                        Проверка:{" "}
                        {m.verification.review_mode === 'CONTEXTUAL' ? ({
                          SUPPORTED: 'смысловых замечаний не выявлено',
                          PARTIALLY_SUPPORTED: 'есть интерпретации или неподтверждённые сведения',
                          UNSUPPORTED: 'есть неподтверждённые утверждения',
                          CONFLICTING: 'найдены противоречия источникам',
                          INSUFFICIENT_INFORMATION: 'часть сведений источники не раскрывают',
                          UNAVAILABLE: 'не завершена; ответ сохранён без подтверждения',
                        } as Record<string, string>)[m.verification.status] || m.verification.status :
                        m.verification.answer_scope === 'LIMITED' && ['SUPPORTED', 'PARTIALLY_SUPPORTED'].includes(m.verification.status) ? 'подтверждены отдельные сведения; описание темы ограничено' : (
                        {
                          SUPPORTED: m.strategy === "WORKSPACE_QA"
                            ? "сведения и оговорённые трактовки сверены с источниками"
                            : "подтверждено источниками",
                          PARTIALLY_SUPPORTED: "подтверждено частично",
                          UNSUPPORTED: "не подтверждено",
                          CONFLICTING: "источники противоречат",
                          INSUFFICIENT_INFORMATION: "недостаточно информации",
                          UNAVAILABLE: "недоступна",
                        } as Record<string, string>
                      )[m.verification.status] || m.verification.status}
                    </span>
                  </div>
                  {m.verification.warning && <p>{m.verification.warning}</p>}
                  {m.verification.checked_answer && m.verification.checked_answer.trim() !== m.content.trim() && (
                    <p>Эта проверка относится к другой версии ответа.</p>
                  )}
                  <details>
                    <summary>Детали проверки</summary>
                    {m.verification.claims.map((c, n) => (
                      <p key={n}>
                        <b>{c.status}</b>
                        {c.basis && ` · ${({
                          EXPLICIT: "прямо в тексте",
                          INFERENCE: "трактовка",
                          UNANSWERED: "в источнике не раскрыто",
                        } as Record<string, string>)[c.basis] || c.basis}`}
                        {` — ${c.text}`}
                        {c.source_chunk_ids.length
                          ? ` [${c.source_chunk_ids.join(", ")}]`
                          : ""}
                        {c.evidence_quote && <><br />Цитата: «{c.evidence_quote}»</>}
                        {c.note && <><br />Замечание: {c.note}</>}
                        {c.original_text && <><br />До исправления: {c.original_text}</>}
                      </p>
                    ))}
                    {!!m.verification.source_chunk_ids.length && (
                      <p>
                        <b>Использованы chunks:</b>{" "}
                        {m.verification.source_chunk_ids.join(", ")}
                      </p>
                    )}
                    {m.verification.components && (
                      <p>
                        <b>Компоненты:</b> retrieval{" "}
                        {Math.round(m.verification.components.retrieval * 100)}
                        %, claims{" "}
                        {Math.round(
                          m.verification.components.claim_support * 100,
                        )}
                        %, agreement{" "}
                        {Math.round(
                          m.verification.components.source_agreement * 100,
                        )}
                        %
                      </p>
                    )}
                  </details>
                </div>
              )}
              {m.context && (
                <details>
                  <summary>Источники ответа</summary>
                  <pre>{m.context}</pre>
                </details>
              )}
              {m.steps?.draft && (
                <details>
                  <summary>Как создавался ответ</summary>
                  <p>Черновик до проверки источников; может содержать ошибки.</p>
                  <SafeMarkdown>{m.steps.draft}</SafeMarkdown>
                  <SafeMarkdown>{m.steps.edited || ""}</SafeMarkdown>
                  {m.steps.enhanced?.map((v, n) => (
                    <SafeMarkdown key={n}>{v}</SafeMarkdown>
                  ))}
                </details>
              )}
              <CopyButton text={m.content} />
            </article>
          ))}
          {busy && (
            <div className="thinking" role="status">
              <span />
              {voiceState || "Фортуна работает…"}
            </div>
          )}
          {recording && input.trim() && (
            <article className="message user live-transcript">
              <div className="message-meta">
                <strong>Вы · Live</strong>
                <span>распознаётся…</span>
              </div>
              <p>{input}</p>
            </article>
          )}
          <div ref={bottom} />
        </div>
        <div className="composer-wrap">
          {error && (
            <div className="error" role="alert">
              {error}
              <button
                className="icon"
                aria-label="Закрыть ошибку"
                onClick={() => setError("")}
              >
                <X size={16} />
              </button>
              {retry && !busy && (
                <button
                  onClick={() => {
                    setBusy(true);
                    setError("");
                    void answer(retry, retryLive);
                  }}
                >
                  Повторить ответ
                </button>
              )}
              {!loaded && (
                <button onClick={() => void load()}>Переподключиться</button>
              )}
            </div>
          )}
          {voiceDraft && (
            <div className="voice-review" role="status">
              <b>Речь распознана.</b> Проверьте текст перед отправкой.
              <button
                onClick={() => {
                  setInput("");
                  setVoiceDraft(false);
                  setVoiceState("");
                }}
              >
                Отменить
              </button>
            </div>
          )}
          {live && (
            <button
              className="stop-session"
              type="button"
              onClick={() => void toggleLive()}
            >
              <Square size={13} /> Завершить Live-разговор
            </button>
          )}
          <form
            className={"composer " + (live ? "live-on" : "")}
            onSubmit={send}
          >
            {files.length > 0 && (
              <div className="attachments">
                {files.map((f, i) => (
                  <span key={i}>
                    {f.name}
                    <button
                      type="button"
                      className="icon"
                      disabled={busy}
                      aria-label={"Убрать " + f.name}
                      onClick={() => setFiles(files.filter((_, n) => i !== n))}
                    >
                      <X size={13} />
                    </button>
                  </span>
                ))}
              </div>
            )}
            <textarea
              aria-label="Сообщение"
              placeholder={
                live
                  ? "Live: текст и голос будут отправлены с потоковым ответом и озвучиванием"
                  : "Напишите сообщение…"
              }
              value={input}
              disabled={locked}
              onChange={(e) => {
                setInput(e.target.value);
                if (voiceDraft) setVoiceDraft(false);
              }}
              onKeyDown={(e) => {
                if (
                  e.key === "Enter" &&
                  !e.shiftKey &&
                  !e.nativeEvent.isComposing
                ) {
                  e.preventDefault();
                  e.currentTarget.form?.requestSubmit();
                }
              }}
            />
            <div className="composer-bottom">
              <div className="composer-tools">
                <input
                  ref={fileInput}
                  type="file"
                  multiple
                  accept=".pdf,.docx,.html,.txt"
                  hidden
                  onChange={(e) => {
                    setFiles((prev) => [
                      ...prev,
                      ...Array.from(e.target.files || []),
                    ]);
                    e.target.value = "";
                  }}
                />
                <button
                  type="button"
                  className="icon"
                  aria-label="Прикрепить файлы"
                  disabled={locked}
                  onClick={() => fileInput.current?.click()}
                >
                  <Paperclip size={19} />
                </button>
                <button
                  type="button"
                  title="Потоковый ответ с озвучиванием для текста и голоса"
                  className={"live-toggle " + (live ? "active" : "")}
                  aria-label="Потоковый Live-режим с озвучиванием"
                  aria-pressed={live}
                  disabled={busy || recording}
                  onClick={() => {
                    setLive((v) => !v);
                    setVoiceState("");
                    stopAudio();
                  }}
                >
                  LIVE <span>{live ? "ВКЛ" : "ВЫКЛ"}</span>
                </button>
                <button
                  type="button"
                  className={"icon " + (recording ? "recording" : "")}
                  aria-label={
                    recording ? "Остановить запись" : "Записать голос"
                  }
                  disabled={busy}
                  onClick={() => void record()}
                >
                  {recording ? <Square size={18} /> : <Mic size={19} />}
                </button>
              </div>
              {liveRunning ? (
                <button
                  type="button"
                  className="stop-live"
                  onClick={() => void cancelLive()}
                  aria-label={live ? "Остановить Live-ответ" : "Остановить ответ"}
                >
                  <Square size={17} /> Стоп
                </button>
              ) : (
                <button
                  className="send"
                  disabled={
                    locked || !loaded || (!input.trim() && !files.length)
                  }
                  aria-label={live ? "Отправить в Live-режиме" : "Отправить"}
                >
                  <ArrowUp size={20} />
                </button>
              )}
            </div>
          </form>
          <div className="composer-hint" role="status">
            {voiceState ||
              `${live ? "Live включён · " : ""}${voiceProvider} · Enter — отправить · Shift + Enter — новая строка`}
          </div>
        </div>
      </main>
      <Dialog.Root open={!!rename} onOpenChange={(o) => !o && setRename(null)}>
        <Dialog.Portal>
          <Dialog.Overlay className="overlay" />
          <Dialog.Content className="modal small">
            <Dialog.Title>Название диалога</Dialog.Title>
            <Dialog.Description>
              Выберите короткое и понятное название.
            </Dialog.Description>
            <form
              onSubmit={async (e) => {
                e.preventDefault();
                if (!rename || !renameText.trim()) return;
                try {
                  await save({ ...rename, title: renameText.trim() });
                  setRename(null);
                } catch (e) {
                  report(e);
                }
              }}
            >
              <input
                autoFocus
                maxLength={120}
                value={renameText}
                onChange={(e) => setRenameText(e.target.value)}
              />
              <div className="modal-actions">
                <Dialog.Close type="button">Отмена</Dialog.Close>
                <button className="primary" disabled={!renameText.trim()}>
                  Сохранить
                </button>
              </div>
            </form>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
      <Dialog.Root
        open={!!deleting}
        onOpenChange={(o) => !o && setDeleting(null)}
      >
        <Dialog.Portal>
          <Dialog.Overlay className="overlay" />
          <Dialog.Content className="modal small">
            <Dialog.Title>Удалить диалог?</Dialog.Title>
            <Dialog.Description>
              «{deleting?.title}» будет удалён из истории. Это действие нельзя
              отменить.
            </Dialog.Description>
            <div className="modal-actions">
              <Dialog.Close>Отмена</Dialog.Close>
              <button
                className="delete"
                onClick={async () => {
                  if (!deleting) return;
                  try {
                    await api("/chats/" + deleting.id, { method: "DELETE" });
                    setChats((c) => c.filter((x) => x.id !== deleting.id));
                    if (active === deleting.id) fresh();
                    setDeleting(null);
                  } catch (e) {
                    report(e);
                  }
                }}
              >
                Удалить
              </button>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
      <Dialog.Root open={documentsOpen} onOpenChange={setDocumentsOpen}>
        <Dialog.Portal>
          <Dialog.Overlay className="overlay" />
          <Dialog.Content className="modal">
            <div className="modal-top">
              <Dialog.Title>Документы</Dialog.Title>
              <Dialog.Close className="icon" aria-label="Закрыть">
                <X />
              </Dialog.Close>
            </div>
            <Dialog.Description>
              Материалы для ответов во всех ваших диалогах.
            </Dialog.Description>
            <button disabled={busy} onClick={async () => {
              setBusy(true);
              try {
                await post('/rebuild-search-index', {});
                await refreshDocs();
              } catch (error) { report(error); }
              finally { setBusy(false); }
            }}>Обновить поисковый индекс</button>
            <label className="upload-zone">
              <Paperclip /> Добавить PDF, DOCX, HTML или TXT
              <input
                type="file"
                multiple
                accept=".pdf,.docx,.html,.txt"
                disabled={busy}
                onChange={async (e) => {
                  const selected = Array.from(e.target.files || []);
                  e.target.value = "";
                  setBusy(true);
                  try {
                    for (const f of selected) await upload("/upload-doc", f);
                    await refreshDocs();
                  } catch (e) {
                    report(e);
                  } finally {
                    setBusy(false);
                  }
                }}
              />
            </label>
            <form
              className="drive-form"
              onSubmit={async (e) => {
                e.preventDefault();
                setBusy(true);
                try {
                  const result = await post<{status:string;indexed_files:number;limited:boolean;failed:{name:string;error:string}[]}>("/upload-gdrive", { folder_url: drive });
                  if (result.status === 'partial') {
                    setError(`Импортировано документов: ${result.indexed_files}. ` +
                      (result.limited ? 'Достигнут лимит импорта. ' : '') +
                      (result.failed?.length ? 'Не обработаны: ' + result.failed.map(file => file.name).join(', ') : ''));
                  }
                  setDrive("");
                  await refreshDocs();
                } catch (e) {
                  report(e);
                } finally {
                  setBusy(false);
                }
              }}
            >
              <input
                aria-label="Ссылка Google Drive"
                placeholder="Ссылка на папку Google Drive"
                value={drive}
                onChange={(e) => setDrive(e.target.value)}
              />
              <button disabled={busy || !drive.trim()}>Импортировать</button>
            </form>
            <details>
              <summary>Как подключить Google Drive</summary>
              <p>
                Сохраните credentials.json в папку api и выполните из папки
                проекта:
              </p>
              <code>.venv\Scripts\python.exe api/gdrive_auth.py</code>
            </details>
            {error && <p className="error">{error}</p>}
            {busy && <p role="status">Обрабатываю материалы…</p>}
            <div className="document-list">
              {docs.map((d) => (
                <div className="document-row" key={d.id}>
                  <FileText size={18} />
                  <div>
                    {d.filename}
                    <small>{date(d.upload_timestamp)}</small>
                  </div>
                  <button
                    className="icon"
                    title={
                      d.can_reindex
                        ? "Переиндексировать с поддержкой формул"
                        : "Исходник недоступен — загрузите документ заново"
                    }
                    aria-label={"Переиндексировать " + d.filename}
                    disabled={busy || !d.can_reindex}
                    onClick={async () => {
                      if (
                        !window.confirm(
                          `Переиндексировать «${d.filename}»? Старые chunks и связи будут заменены.`,
                        )
                      )
                        return;
                      setBusy(true);
                      try {
                        await post("/reindex-docs", { file_ids: [d.id] });
                        await refreshDocs();
                      } catch (e) {
                        report(e);
                      } finally {
                        setBusy(false);
                      }
                    }}
                  >
                    <RotateCcw size={16} />
                  </button>
                  <button
                    className="icon"
                    aria-label={"Удалить " + d.filename}
                    disabled={busy}
                    onClick={async () => {
                      if (
                        !window.confirm(
                          "Удалить документ из поиска? Исходный файл сохранится.",
                        )
                      )
                        return;
                      try {
                        await post("/delete-doc", { file_id: d.id });
                        await refreshDocs();
                      } catch (e) {
                        report(e);
                      }
                    }}
                  >
                    <Trash2 size={17} />
                  </button>
                </div>
              ))}
              {!docs.length && (
                <p className="empty-history">Библиотека пока пуста</p>
              )}
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
      <Dialog.Root
        open={graphBrowse}
        onOpenChange={setGraphBrowse}
      >
        <Dialog.Portal>
          <Dialog.Overlay className="overlay" />
          <Dialog.Content
            className="modal graph-modal"
            aria-describedby={undefined}
          >
            <Dialog.Title className="sr-only">Граф знаний</Dialog.Title>
            <Suspense fallback={<p role="status">Загружаю граф…</p>}>
              <GraphExplorer workspace={workspace} onClose={() => setGraphBrowse(false)} />
            </Suspense>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
      <Dialog.Root open={workspaceOpen} onOpenChange={setWorkspaceOpen}>
        <Dialog.Portal>
          <Dialog.Overlay className="overlay" />
          <Dialog.Content className="modal workspace-modal">
            <div className="modal-top">
              <div>
                <Dialog.Title>Workspace</Dialog.Title>
                <Dialog.Description>
                  {workspace
                    ? "Активная рабочая папка доступна ассистенту только для чтения."
                    : "Выберите папку для текущего рабочего контекста."}
                </Dialog.Description>
              </div>
              <Dialog.Close className="icon" aria-label="Закрыть">
                <X />
              </Dialog.Close>
            </div>
            <div className="workspace-toolbar">
              <button
                className="primary"
                disabled={workspaceBusy || workspaceIndexBusy}
                onClick={() => void chooseWorkspace()}
              >
                <FolderOpen size={16} />
                {workspace ? "Сменить папку" : "Выбрать папку"}
              </button>
              {workspace && <button disabled={workspaceBusy || workspaceIndexBusy || locked}
                onClick={() => void indexWorkspace()}>
                {workspaceIndexBusy ? 'Индексирую папку…' : 'Индексировать папку'}
              </button>}
              {workspaceFile ? (
                <button
                  disabled={workspaceBusy}
                  onClick={() => setWorkspaceFile(null)}
                >
                  <ChevronLeft size={16} />К списку
                </button>
              ) : (
                workspaceListing?.parent != null && (
                  <button
                    disabled={workspaceBusy}
                    onClick={() =>
                      workspace &&
                      void loadWorkspaceFiles(
                        workspace,
                        workspaceListing.parent || "",
                      )
                    }
                  >
                    <ChevronLeft size={16} />
                    Назад
                  </button>
                )
              )}
              <span>
                {workspaceFile?.path ||
                  workspaceListing?.path ||
                  workspace?.name ||
                  "Папка не выбрана"}
              </span>
            </div>
            {workspace && <p className="workspace-index-status" role="status">
              {workspaceIndexBusy ? 'Создаю поисковый индекс для файлов и вложенных папок…' :
                workspaceIndexStatus ? `Проиндексировано ${workspaceIndexStatus.indexed_documents} из ${workspaceIndexStatus.readable_documents} доступных документов.` :
                  'Проверяю поисковый индекс…'}
              {!!workspaceIndexStatus?.failed_documents && ` Ошибок индексации: ${workspaceIndexStatus.failed_documents}.`}
              {workspaceIndexStatus?.truncated && ' Список файлов ограничен; обход папки неполный.'}
              {!!workspaceIndexStatus?.scan_errors && ' Некоторые файлы или папки недоступны.'}
            </p>}
            {workspaceIndexMessage && <p role="status">{workspaceIndexMessage}</p>}
            {error && <p className="error">{error}</p>}
            {workspaceBusy && <p role="status">Читаю содержимое папки…</p>}
            {workspaceFile ? (
              <pre className="workspace-preview">{workspaceFile.content}</pre>
            ) : (
              <div className="workspace-files">
                {workspaceListing?.entries.map((item) => (
                  <button
                    key={item.relative_path}
                    className={"workspace-entry " + item.kind}
                    disabled={
                      workspaceBusy || (!item.navigable && !item.readable)
                    }
                    onClick={() => {
                      if (!workspace) return;
                      if (item.navigable)
                        void loadWorkspaceFiles(workspace, item.relative_path);
                      else if (item.readable)
                        void readWorkspaceFile(workspace, item.relative_path);
                    }}
                    title={
                      item.kind === "symlink"
                        ? "Символические ссылки недоступны"
                        : !item.navigable && !item.readable
                          ? "Формат пока не поддерживается"
                          : item.name
                    }
                  >
                    {item.kind === "directory" ? (
                      <Folder size={18} />
                    ) : (
                      <File size={18} />
                    )}
                    <span>{item.name}</span>
                    {item.kind === "file" && item.size != null && (
                      <small>
                        {item.size < 1024
                          ? `${item.size} Б`
                          : `${Math.ceil(item.size / 1024)} КБ`}
                      </small>
                    )}
                  </button>
                ))}
                {workspaceListing && !workspaceListing.entries.length && (
                  <p className="empty-history">Папка пуста</p>
                )}
                {workspaceListing?.truncated && (
                  <p className="empty-history">Показаны первые 500 элементов</p>
                )}
              </div>
            )}
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </div>
  );
}
