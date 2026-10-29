export type WorkspaceIndexStatus = {
  readable_documents: number; indexed_documents: number; failed_documents: number;
  pending_documents?: number; changed_documents?: number; unavailable_documents?: number;
  truncated: boolean; scan_errors: number; checked_at?: string;
  issues?: {path: string; state: string}[]; issues_truncated?: boolean;
  text_index?: {enabled: boolean; available: boolean; indexed_documents: number;
    pending_documents: number; failed_documents: number};
};

export type WorkspaceIndexResult = {
  indexed: number; unchanged: number; failed: number; unavailable: boolean;
  skipped?: number; deleted_chunks?: number; scan_truncated?: boolean; scan_errors?: number;
  cleanup_deferred?: boolean; issues?: {path: string; code: string}[]; issues_truncated?: boolean;
  text_index?: {indexed: number; unchanged: number; failed: number; omitted: number; unavailable: boolean};
};

const states: Record<string, string> = {
  PENDING: 'Новый файл — ещё не индексировался',
  STALE: 'Файл изменён — индекс требует обновления',
  FAILED: 'Последняя обработка завершилась ошибкой',
  UNAVAILABLE: 'Не удалось получить стабильную версию файла при обходе',
};
const errors: Record<string, string> = {
  SOURCE_CHANGED: 'Файл изменился во время обработки. Повторите обновление.',
  EXTRACTION_FAILED: 'Не удалось прочитать или извлечь текст. Проверьте доступность и формат файла.',
  EMBEDDING_FAILED: 'Не удалось построить или сохранить поисковый индекс. Проверьте embedding-модель и журнал backend.',
};

export function WorkspaceIndexDetails({status, result, busy, checking, disabled, onRefresh, onRetry}: {
  status: WorkspaceIndexStatus | null; result: WorkspaceIndexResult | null;
  busy: boolean; checking: boolean; disabled: boolean; onRefresh: () => void; onRetry: () => void;
}) {
  return <section className="workspace-index" aria-label="Поисковый индекс папки">
    <div className="workspace-index-bar">
      <p className="workspace-index-status" role="status">
        {busy ? 'Индексирую…' : checking ? 'Проверяю…' :
          status ? `Индекс: ${status.indexed_documents}/${status.readable_documents}` : 'Индекс недоступен'}
        {status?.text_index?.enabled && <span>
          {status.text_index.available ? ` · Текст: ${status.text_index.indexed_documents}/${status.readable_documents}` :
            ' · Текстовый индекс недоступен'}
        </span>}
      </p>
      <div className="workspace-index-actions">
        <button type="button" onClick={onRefresh} disabled={busy || checking || disabled}>Проверить изменения</button>
        {!!(status?.failed_documents || status?.text_index?.failed_documents) && <button type="button" onClick={onRetry} disabled={busy || checking || disabled}>
          Повторить ошибки
        </button>}
      </div>
    </div>
    <p className="workspace-index-warning" role="status">
      {!!status?.pending_documents && `Новых: ${status.pending_documents}. `}
      {!!status?.changed_documents && `Изменённых: ${status.changed_documents}. `}
      {!!status?.failed_documents && `С ошибками: ${status.failed_documents}. `}
      {!!status?.unavailable_documents && `Нестабильных при чтении: ${status.unavailable_documents}. `}
      {!!status?.text_index?.failed_documents && `Ошибок извлечения: ${status.text_index.failed_documents}. `}
      {!!result?.text_index?.omitted && `Текст не сохранён: ${result.text_index.omitted}. `}
      {(status?.truncated || result?.scan_truncated) && 'Обход ограничен. '}
      {!!(status?.scan_errors || result?.scan_errors) && 'Есть недоступные файлы или папки.'}
    </p>
    <details className="workspace-help">
      <summary>Об индексе{status?.checked_at && ` · проверено ${new Date(status.checked_at).toLocaleTimeString('ru-RU')}`}</summary>
      <p>Индекс относится ко всей рабочей папке, независимо от выбранных источников ответа.
      Обновление не обрабатывает неизменённые документы заново. Без актуальных embeddings остаётся поиск по словам.
      Это снимок на момент проверки, не постоянное наблюдение за диском.
      При недоступном текстовом индексе используется ограниченное чтение файлов во время поиска.
      Ограниченный обход не подтверждает полный охват папки.</p>
    </details>
    {!!status?.issues?.length && <details className="workspace-index-issues">
      <summary>Требуют внимания · {status.readable_documents - status.indexed_documents}</summary>
      <ul>{status.issues.map(item => <li key={item.path}><b>{item.path}</b> — {states[item.state] || item.state}</li>)}</ul>
      {status.issues_truncated && <p>Показаны первые 50 документов. Общий счётчик учитывает остальные.</p>}
    </details>}
    {!!result?.issues?.length && <details className="workspace-index-issues">
      <summary>Ошибки последнего запуска · {result.failed}</summary>
      <ul>{result.issues.map(item => <li key={item.path}><b>{item.path}</b> — {errors[item.code] || 'Обработка не завершена.'}</li>)}</ul>
      {result.issues_truncated && <p>Показаны первые 50 ошибок.</p>}
    </details>}
    {result?.cleanup_deferred && <p>Из-за неполного обхода очистка старых записей индекса отложена.
      Исходные файлы не удаляются.</p>}
  </section>;
}
