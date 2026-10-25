"""Real model replay of a named private chat and natural document tools.

Sources, chat and working database remain read-only. All mutations are confined
to copies in ignored evaluation storage. Expectations come from a private spec.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import statistics
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


async def evaluate(args):
    if args.reuse_running:
        os.environ['EVALUATION_CHAT_SERVER_URL']='http://127.0.0.1:8081'
        os.environ['EVALUATION_EMBEDDING_SERVER_URL']='http://127.0.0.1:8082'
    import evaluate_user_queries as harness
    from pydantic_models import QueryInput
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    main,run,root=harness.main,harness.RUN,harness.ROOT
    code_hashes={p.name:digest(p) for p in (root/'api').glob('*.py')}
    spec=json.loads(args.spec.read_text(encoding='utf-8'))
    database=root/'api/rag_app.db'
    protected={p:digest(p) for p in (database,root/'api/chat-model-selection.json')}
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as conn:
        chats=[json.loads(row[0]) for row in conn.execute('SELECT data FROM workspace_chats')]
        chat=next(item for item in chats if item.get('title','').casefold()==args.chat.casefold())
        workspace_id,original_root=conn.execute('SELECT s.workspace_id,w.root_path FROM assistant_sessions s '
            'JOIN assistant_workspaces w ON w.id=s.workspace_id WHERE s.id=?',
            (chat['session_id'],)).fetchone()
        paths=[row[0] for row in conn.execute('SELECT relative_path FROM assistant_workspace_files WHERE workspace_id=? AND readable=1',
            (workspace_id,))]
    original_root=Path(original_root)
    copied=run/'documents'
    copied.mkdir()
    originals={}
    for relative in paths:
        source=original_root/relative
        if not source.resolve().is_relative_to(original_root.resolve()) or source.is_symlink():
            raise ValueError('Unsafe source path')
        if any(part.startswith('.') for part in Path(relative).parts) or source.stat().st_size>10*1024*1024:continue
        originals[source]=digest(source)
        target=copied/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
    (run/'original-chat.json').write_text(json.dumps(chat,ensure_ascii=False,indent=2),encoding='utf-8')
    provider,embedding=harness.get_llm_provider(),harness.get_embedding_provider()
    rows=[]
    def save():
        (run/'agent-documents.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    async def ask(case,question,workspace,session=None,checks=None):
        start=time.perf_counter()
        row={'id':case,'question':question}
        try:
            response,_=await main.assistant_core.process_input(QueryInput(question=question,model=args.profile,
                workspace_id=workspace.id,session_id=session,use_multi_agent=True))
            actual={'completed':response.task_status.value=='COMPLETED'}
            if checks:actual.update(checks(response))
            row.update(response=response.model_dump(mode='json'),checks=actual,passed=all(actual.values()))
        except Exception as exc:
            response=None;row.update(passed=False,error=type(exc).__name__)
        row['seconds']=time.perf_counter()-start;rows.append(row);save()
        print(case, 'PASS' if row['passed'] else 'FAIL',round(row['seconds'],2),'s',flush=True)
        return response
    def saved_path(response):
        for action in reversed(response.actions):
            if action.action_type=='APPLY_DOCUMENT_CHANGE':
                return (action.output or {}).get('data',{}).get('path')
        return None
    def text_checks(response,case):
        import re
        return {**{'expected:'+p:bool(re.search(p,response.answer,re.I)) for p in spec.get(case,{}).get('expected',[])},
            **{'forbidden:'+p:not re.search(p,response.answer,re.I) for p in spec.get(case,{}).get('forbidden',[])}}
    try:
        print('Output:',run,flush=True)
        await asyncio.gather(asyncio.to_thread(provider.manager.ensure_ready),asyncio.to_thread(embedding.manager.ensure_ready))
        await asyncio.to_thread(provider.generate,[{'role':'user','content':'Ответь: готов.'}],profile='FAST',thinking=False,max_tokens=20)
        workspace=main.workspace_repository.select(str(copied))
        main.filesystem_provider.opener=lambda path:(_ for _ in ()).throw(AssertionError('Unexpected system open'))
        main.workspace_vector_index.graph_extractor_factory=None
        main.workspace_vector_index.graph_repository=None
        await asyncio.to_thread(main.workspace_vector_index.sync,workspace.id,extract_relations=False)
        questions=[m['content'] for m in chat['messages'] if m['role']=='user']
        first=await ask('identity',questions[1],workspace,checks=lambda r:{**text_checks(r,'identity'),
            'supported':bool(r.verification and r.verification.status=='SUPPORTED' and r.verification.checked_answer==r.answer)})
        session=first.session_id if first else None
        await ask('narrow-followup',questions[2],workspace,session,lambda r:{**text_checks(r,'followup'),
            'supported':bool(r.verification and r.verification.status=='SUPPORTED' and r.verification.checked_answer==r.answer)})
        creation=await ask('create-description',questions[3],workspace,session,lambda r:{
            'saved':bool(saved_path(r) and (copied/saved_path(r)).is_file()),'no_extra_confirmation':r.plan_status=='COMPLETED'})
        if creation and saved_path(creation):
            path=saved_path(creation)
            content=main.filesystem_provider.read_text_file(workspace.id,path)[2]
            import re
            checks={p:bool(re.search(p,content,re.I)) for p in spec.get('document',{}).get('expected',[])}
            rows[-1]['checks'].update(checks);rows[-1]['passed']=all(rows[-1]['checks'].values());save()
            await ask('implicit-pdf','Мне нужен PDF этого документа, чтобы отправить его коллеге',workspace,session,
                lambda r:{'saved_pdf':bool(saved_path(r) and (copied/saved_path(r)).read_bytes().startswith(b'%PDF-')),
                    'same_content':bool(saved_path(r) and all(re.search(p,main.filesystem_provider.read_text_file(
                        workspace.id,saved_path(r))[2],re.I) for p in spec.get('document',{}).get('expected',[]))),
                    'no_extra_confirmation':r.plan_status=='COMPLETED'})
        tools=run/'mutations';tools.mkdir()
        fixture=main.workspace_repository.select(str(tools))
        response=await ask('literal-create','Создай файл Заметка.md с текстом «Код: 4817»',fixture,
            checks=lambda r:{'saved':(tools/'Заметка.md').read_text(encoding='utf-8')=='Код: 4817'})
        tool_session=response.session_id if response else None
        await ask('natural-edit','Можешь в документе Заметка.md поменять «4817» на «6248»?',fixture,tool_session,
            lambda r:{'changed':(tools/'Заметка.md').read_text(encoding='utf-8')=='Код: 6248',
                'backup':any((tools/'.fortuna-backups').glob('*'))})
        await ask('natural-search','Где встречается фраза «Код: 6248»?',fixture,tool_session,
            lambda r:{'located':'Заметка.md:1' in r.answer})
        await ask('project-structure','Покажи структуру проекта',fixture,tool_session,
            lambda r:{'located':'Заметка.md' in r.answer})
        summary={'passed':sum(row['passed'] for row in rows),'total':len(rows),
            'reused_inference_servers':args.reuse_running,
            'protected_files_unchanged':all(digest(p)==h for p,h in protected.items()),
            'original_sources_unchanged':all(digest(p)==h for p,h in originals.items()),
            'code_hashes':code_hashes,
            'code_unchanged_during_run':code_hashes=={p.name:digest(p) for p in (root/'api').glob('*.py')}}
        (run/'agent-documents-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        print('RESULT',summary['passed'],summary['total'],'protected:',summary['protected_files_unchanged'],
            'sources:',summary['original_sources_unchanged'],flush=True)
        return all(row['passed'] for row in rows) and summary['protected_files_unchanged'] and summary['original_sources_unchanged']
    finally:
        provider.close();provider.manager.stop();embedding.manager.stop()
        from chroma_utils import vectorstore,workspace_vectorstore
        for system in {store._client._system for store in (vectorstore,workspace_vectorstore)}:system.stop()


if __name__=='__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chat',required=True)
    parser.add_argument('--spec',required=True,type=Path)
    parser.add_argument('--profile',choices=('FAST','BALANCED','REASONING'),default='BALANCED')
    parser.add_argument('--reuse-running',action='store_true',help='Use existing local model servers without starting or stopping them')
    args=parser.parse_args()
    sys.exit(0 if asyncio.run(evaluate(args)) else 1)
