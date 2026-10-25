"""Read-only real GGUF evaluation against a chosen local workspace.

Cases are a private JSON array: id, question, required, forbidden, source (optional),
require_supported (default true). Keep case files and output outside tracked docs.
"""
import argparse
import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


async def evaluate(args):
    source_root = Path(args.workspace).resolve(strict=True)
    cases = json.loads(Path(args.cases).read_text(encoding='utf-8'))
    if not isinstance(cases,list) or not 1 <= len(cases) <= 40:
        raise ValueError('Supply between 1 and 40 private evaluation cases')
    protected = [Path('api/rag_app.db'), Path('api/chat-model-selection.json')]
    before = {path: digest(path) for path in protected}
    import evaluate_user_queries as harness
    from pydantic_models import QueryInput
    from model_profiles import get_profile
    logging.getLogger().setLevel(logging.ERROR)
    main, provider = harness.main, harness.get_llm_provider()
    workspace = main.workspace_repository.select(str(source_root))
    scan = main.filesystem_provider.sync_index(workspace.id)
    indexed = [item for item in main.workspace_repository.indexed_files(workspace.id) if item.readable]
    original_files = {item.relative_path: digest(source_root/item.relative_path) for item in indexed}
    rows=[]
    print(f'Output: {harness.RUN}; readable files: {len(indexed)}; scan limited: {scan.truncated}',flush=True)
    try:
        started=time.perf_counter()
        await asyncio.to_thread(provider.manager.ensure_ready)
        startup=time.perf_counter()-started
        await asyncio.to_thread(provider.generate,[{'role':'user','content':'Ответь: готов.'}],
                                profile='FAST',thinking=False,max_tokens=20)
        for repeat in range(args.repeats):
            for case in cases:
                start=time.perf_counter()
                row={'id':case['id'],'repeat':repeat+1,'profile':args.profile}
                try:
                    response,_=await main.assistant_core.process_input(QueryInput(
                        question=case['question'],workspace_id=workspace.id,model=args.profile,use_multi_agent=True))
                    checks={f'contains:{word}':word.casefold() in response.answer.casefold() for word in case.get('required',[])}
                    checks.update({f'excludes:{word}':word.casefold() not in response.answer.casefold() for word in case.get('forbidden',[])})
                    for alternatives in case.get('required_any',[]):
                        checks['any:'+repr(alternatives)]=any(word.casefold() in response.answer.casefold() for word in alternatives)
                    checks['completed']=response.task_status.value=='COMPLETED'
                    if case.get('require_supported',True):
                        checks['supported']=bool(response.verification and response.verification.status=='SUPPORTED'
                            and response.verification.checked_answer==response.answer)
                    if case.get('source'):
                        checks['source']=bool(response.results) and all(
                            result.metadata.get('source')==case['source'] for result in response.results)
                    for evidence in case.get('required_evidence',[]):
                        checks['evidence:'+evidence]=bool(response.verification and any(
                            evidence.casefold() in claim.evidence_quote.casefold() for claim in response.verification.claims))
                    if case.get('missing_fact'):
                        checks['admits_missing']=any(word in response.answer.casefold() for word in
                            ('не указан','не содерж','нет информац','недостаточно','не найден','не предостав'))
                        checks['review_completed']=bool(response.verification and response.verification.status!='UNAVAILABLE')
                        checks['no_positive_facts']=bool(response.verification and all(
                            claim.basis=='UNANSWERED' for claim in response.verification.claims))
                    row.update(checks=checks,passed=all(checks.values()),response=response.model_dump(mode='json'))
                except Exception as error:
                    row.update(passed=False,error=type(error).__name__)
                row['seconds']=time.perf_counter()-start;rows.append(row)
                (harness.RUN/'local-workspace.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
                print(f"{row['id']} repeat={repeat+1}: {'PASS' if row['passed'] else 'FAIL'} {row['seconds']:.2f}s",flush=True)
        summary={'passed':sum(row['passed'] for row in rows),'total':len(rows),'startup_seconds':startup,
            'model':str(get_profile().model_path),'profile':args.profile,'readable_files':len(indexed),
            'protected_files_unchanged':all(digest(path)==value for path,value in before.items()),
            'source_documents_unchanged':all(digest(source_root/name)==value for name,value in original_files.items())}
        (harness.RUN/'local-workspace-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        print(f'RESULT {summary}',flush=True)
        return summary['passed']==summary['total'] and summary['protected_files_unchanged'] and summary['source_documents_unchanged']
    finally:
        provider.close();provider.manager.stop()
        harness.get_embedding_provider().manager.stop()
        from chroma_utils import vectorstore,workspace_vectorstore
        for system in {store._client._system for store in (vectorstore,workspace_vectorstore)}:
            system.stop()


if __name__=='__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',required=True)
    parser.add_argument('--cases',required=True)
    parser.add_argument('--profile',choices=('FAST','BALANCED','REASONING'),default='BALANCED')
    parser.add_argument('--repeats',type=int,default=2)
    args=parser.parse_args()
    if args.repeats<1:parser.error('--repeats must be positive')
    sys.exit(0 if asyncio.run(evaluate(args)) else 1)
