"""Real local compaction and recall; isolated synthetic archive, no originals changed."""
import asyncio
from contextlib import closing
import hashlib
import json
from pathlib import Path
import time


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


async def run():
    import evaluate_user_queries as h
    from pydantic_models import QueryInput
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    main,root,out=h.main,h.ROOT,h.RUN
    code_hashes={p.name:digest(p) for p in (root/'api').glob('*.py')}
    protected={p:digest(p) for p in (root/'api/rag_app.db',root/'api/chat-model-selection.json')}
    provider=h.get_llm_provider()
    session=main.assistant_repository.ensure_session()
    print('OUTPUT',out,flush=True)
    with closing(main.assistant_repository.connection_factory()) as c,c:
        samples=[('Google Drive пока не убирай. Работай только в локальной папке.','Принято.'),
            ('Проект Нарвал получил контрольный код 4817.','Код был обсуждён.')]
        samples += [(f'Обсуждаем дополнительную тему {i}.','Постороннее старое объяснение. '*100) for i in range(36)]
        for user,answer in samples:
            row=c.execute('INSERT INTO application_logs(session_id,user_query,gpt_response,model,knowledge_epoch) VALUES(?,?,?,?,0)',
                (session.id,user,answer,'FAST'))
            c.execute('INSERT INTO assistant_history_scopes(log_id,scope_key) VALUES(?,?)',(row.lastrowid,'LIBRARY'))
        originals=[tuple(r) for r in c.execute('SELECT * FROM application_logs ORDER BY id')]
    rows=[]
    try:
        await asyncio.to_thread(provider.manager.ensure_ready)
        print('MODEL READY',flush=True)
        for case,question in [('constraint','Что я просил насчёт Google Drive?'),
                              ('old-fact','Что я говорил о коде проекта Нарвал?')]:
            start=time.perf_counter()
            response,_=await main.assistant_core.process_input(QueryInput(question=question,model='FAST',session_id=session.id))
            elapsed=time.perf_counter()-start
            (out/(case+'-response.json')).write_text(response.model_dump_json(indent=2),encoding='utf-8')
            answer=response.answer.casefold()
            check=(('drive' in answer and any(x in answer for x in ('не убира','не удал','не отключ','сохран','остав')))
                   if case=='constraint' else '4817' in answer)
            info=response.context_memory or {}
            good=(response.strategy=='GENERAL_CHAT' and check and info.get('after_tokens',10**9)<=info.get('history_budget_tokens',0))
            row={'case':case,'seconds':round(elapsed,3),'passed':good,'context':info,'strategy':response.strategy}
            rows.append(row)
            print(json.dumps(row,ensure_ascii=False),flush=True)
        # Reload the durable checkpoint as a fresh memory manager.
        from conversation_memory import ConversationMemory
        reloaded=ConversationMemory(main.assistant_repository,main.get_chat_archive,main._conversation_epoch)
        view=await asyncio.to_thread(reloaded.prepare,session.id,None,'Что я просил насчёт Google Drive?','FAST')
        restart_ok='Google Drive пока не убирай.' in str(view.history)
        other=await asyncio.to_thread(reloaded.prepare,session.id,'another-workspace','Что я просил?','FAST')
        isolation_ok='Google Drive' not in str(other.history) and '4817' not in str(other.history)
        with closing(main.assistant_repository.connection_factory()) as c:
            originals_ok=originals==[tuple(r) for r in c.execute('SELECT * FROM application_logs ORDER BY id LIMIT ?',(len(originals),))]
        result={'cases':rows,'restart_preserves_constraint':restart_ok,'workspace_isolated':isolation_ok,
            'original_archive_rows_unchanged':originals_ok,'working_files_unchanged':all(digest(p)==d for p,d in protected.items()),
            'code_hashes':code_hashes,'code_unchanged':code_hashes=={p.name:digest(p) for p in (root/'api').glob('*.py')}}
        result['passed']=all(r['passed'] for r in rows) and all(result[k] for k in
            ('restart_preserves_constraint','workspace_isolated','original_archive_rows_unchanged','working_files_unchanged','code_unchanged'))
        (out/'context-memory-report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print('RESULT',result['passed'],flush=True)
        return result['passed']
    finally:
        provider.close();provider.manager.stop()
        from chroma_utils import vectorstore,workspace_vectorstore
        for system in {s._client._system for s in (vectorstore,workspace_vectorstore)}:system.stop()


if __name__=='__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    sys.exit(0 if asyncio.run(run()) else 1)
