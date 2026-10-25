"""Replay newest private creation goal on copies, with real local inference.

All detailed artifacts are ignored. Does not launch desktop applications or
write to the original workspace, database, history or model selection.
"""
import argparse
import asyncio
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


async def evaluate(args):
    if args.resume_run:
        target=args.resume_run.resolve()
        allowed=Path(__file__).resolve().parents[1]/'.user-evaluation'
        if not target.is_relative_to(allowed) or not (target/'evaluation.db').is_file():raise ValueError('Invalid isolated resume directory')
        os.environ['EVALUATION_RUN_DIR']=str(target)
    import evaluate_user_queries as harness
    from pydantic_models import QueryInput
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    main,run,root=harness.main,harness.RUN,harness.ROOT
    code_hashes={p.name:digest(p) for p in (root/'api').glob('*.py')}
    database=root/'api/rag_app.db'
    protected={p:digest(p) for p in [database,root/'api/chat-model-selection.json'] if p.is_file()}
    with closing(sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True)) as conn:
        chats=[json.loads(r[0]) for r in conn.execute('SELECT data FROM workspace_chats')]
        chat=next(c for c in chats if c['title']==args.chat)
        workspace_id,original=conn.execute('SELECT s.workspace_id,w.root_path FROM assistant_sessions s '
            'JOIN assistant_workspaces w ON w.id=s.workspace_id WHERE s.id=?',(chat['session_id'],)).fetchone()
        paths=[r[0] for r in conn.execute('SELECT relative_path FROM assistant_workspace_files WHERE workspace_id=? AND readable=1',
            (workspace_id,))]
    original=Path(original).resolve()
    copies=run/'summary-sources'
    copies.mkdir(exist_ok=bool(args.resume_run))
    originals={}
    for relative in paths:
        source=original/relative
        if source.is_symlink() or not source.resolve().is_relative_to(original):raise ValueError('Unsafe source')
        originals[source]=digest(source)
        dest=copies/relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        if not args.resume_run:shutil.copyfile(source,dest)
    from local_tool_intent import directory_summary_request
    question=next(m['content'] for m in reversed(chat['messages'])
        if m['role']=='user' and directory_summary_request(m['content']))
    workspace=main.workspace_repository.select(str(copies))
    session=None
    label='practical-summary'
    if args.resume_run:
        prior=json.loads((run/'practical-summary-response.json').read_text(encoding='utf-8'))
        session=prior['session_id']
        question='Это не пересказ, это огрызок. Слишком коротко.'
        label='practical-summary-resumed'
    provider=harness.get_llm_provider()
    opened=[]
    main.filesystem_provider.opener=opened.append
    try:
        print('OUTPUT',run,flush=True)
        await asyncio.to_thread(provider.manager.ensure_ready)
        print('MODEL READY',flush=True)
        start=time.perf_counter()
        response,_=await main.assistant_core.process_input(QueryInput(question=question,model=args.profile,
            workspace_id=workspace.id,session_id=session,workspace_directory='',use_multi_agent=True))
        seconds=time.perf_counter()-start
        (run/(label+'-response.json')).write_text(response.model_dump_json(indent=2),encoding='utf-8')
        saved=[a.output['data']['path'] for a in response.actions if a.action_type=='APPLY_DOCUMENT_CHANGE']
        coverage=[a.output['data'] for a in response.actions if a.action_type=='INVENTORY_SUMMARY']
        reviews=[a.output['data'] for a in response.actions if a.action_type=='VERIFY_DOCUMENT_PART' and a.output]
        summaries=[a.output['data'] for a in response.actions if a.action_type=='SUMMARIZE_DOCUMENT_PART' and a.output]
        reused=[a for a in response.actions if a.action_type=='REUSE_DOCUMENT_SUMMARY']
        evidence=next((a.output['data'].get('document_verification') for a in response.actions if a.action_type=='CREATE_DOCUMENT'),{}) or {}
        full=(copies/saved[-1]).read_text(encoding='utf-8') if saved else ''
        report={'seconds':seconds,'profile':args.profile,'task_status':response.task_status.value,
            'plan_status':response.plan_status,'saved':bool(saved),'saved_characters':len(full),
            'source_files':len(paths),'inventoried_files':len(coverage[0]['sources']) if coverage else 0,
            'drafts':len(summaries),'reviewed_parts':len(reviews),
            'reused_files':len(reused),'covered_files':evidence.get('covered_files'),
            'supported_parts':sum(r['status']=='SUPPORTED' for r in reviews),
            'protected_files_unchanged':all(digest(p)==h for p,h in protected.items()),
            'original_sources_unchanged':all(digest(p)==h for p,h in originals.items()),
            'opened_files':len(opened),'code_hashes':code_hashes,
            'code_unchanged_during_run':code_hashes=={p.name:digest(p) for p in (root/'api').glob('*.py')}}
        report['passed']=(report['plan_status']=='COMPLETED' and report['saved']
            and report['inventoried_files']==len(paths) and report['saved_characters']>2500
            and report['protected_files_unchanged'] and report['original_sources_unchanged'] and not opened
            and report['code_unchanged_during_run'] and report['covered_files']==len(paths))
        (run/(label+'-report.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in report.items() if k!='code_hashes'},ensure_ascii=False),flush=True)
        return report['passed']
    finally:
        provider.close()
        provider.manager.stop()
        from chroma_utils import vectorstore,workspace_vectorstore
        for system in {s._client._system for s in (vectorstore,workspace_vectorstore)}:system.stop()


if __name__=='__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chat',required=True)
    parser.add_argument('--profile',choices=['FAST','BALANCED','REASONING'],default='BALANCED')
    parser.add_argument('--resume-run',type=Path)
    args=parser.parse_args()
    sys.exit(0 if asyncio.run(evaluate(args)) else 1)
