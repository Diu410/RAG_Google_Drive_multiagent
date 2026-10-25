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
    <p className="workspace-index-status" role="status">
      {busy ? 'Обрабатываю новые и изменённые файлы…' : checking ? 'Проверяю изменения в папке…' :
        status ? `Проиндексировано ${status.indexed_documents} из ${status.readable_documents} доступных документов.` :
          'Статус индекса недоступен. Повторите проверку.'}
      {!!status?.pending_documents && ` Новых: ${status.pending_documents}.`}
      {!!status?.changed_documents && ` Изменённых: ${status.changed_documents}.`}
      {!!status?.failed_documents && ` С ошибками: ${status.failed_documents}.`}
      {!!status?.unavailable_documents && ` Нестабильных при чтении: ${status.unavailable_documents}.`}
      {(status?.truncated || result?.scan_truncated) && ' Обход ограничен; полный охват папки не подтверждён.'}
      {!!(status?.scan_errors || result?.scan_errors) && ' Некоторые файлы или папки недоступны.'}
    </p>
    {status?.text_index?.enabled && <p>
      {status.text_index.available ? `Текстовый индекс: ${status.text_index.indexed_documents} из ${status.readable_documents}.` :
        'Текстовый индекс недоступен — используется чтение файлов при поиске.'}
      {!!status.text_index.failed_documents && ` Ошибок извлечения: ${status.text_index.failed_documents}.`}
      {!!result?.text_index?.omitted && ` Не сохранено в текстовом индексе: ${result.text_index.omitted}. Используется чтение при поиске.`}
    </p>}
    <div className="workspace-index-actions">
      <button type="button" onClick={onRefresh} disabled={busy || checking || disabled}>Проверить изменения</button>
      {!!(status?.failed_documents || status?.text_index?.failed_documents) && <button type="button" onClick={onRetry} disabled={busy || checking || disabled}>
        Повторить ошибки
      </button>}
    </div>
    <small>Индекс относится ко всей рабочей папке, независимо от выбранных источников ответа.
      Обновление не обрабатывает неизменённые документы заново. Без актуальных embeddings остаётся поиск по словам.
      Это состояние на момент проверки, не постоянное наблюдение за диском.
      {status?.checked_at && ` Проверено: ${new Date(status.checked_at).toLocaleTimeString('ru-RU')}.`}
    </small>
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
