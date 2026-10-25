"""Opt-in real GGUF benchmark on the project's hard, synthetic source folder."""
import argparse
import asyncio
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import re
import shutil
import statistics
import time

from complex_workspace_cases import CASES, REVIEW_CASES, EXAMPLE_ROOT


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def answer_checks(case, response):
    answer = response.answer.replace('`','').replace('**','')
    checks = {name:any(re.search(pattern, answer, re.I | re.S) for pattern in patterns)
              for name,patterns in case['facts'].items()}
    checks.update({'forbidden:'+pattern:not re.search(pattern, answer, re.I | re.S)
                   for pattern in case.get('forbidden',[])})
    checks['completed'] = response.task_status.value == 'COMPLETED'
    check = response.verification
    checks['review_completed'] = bool(check and check.status != 'UNAVAILABLE')
    supported = bool(check and check.status == 'SUPPORTED')
    # A requested absent fact is a correct answer to that subquestion. The
    # production contract marks mixed known/missing answers PARTIALLY_SUPPORTED;
    # accepting that requires every positive claim S and only the independently
    # expected missing fact UNANSWERED. No inference or unsupported claim passes.
    if check and check.status == 'PARTIALLY_SUPPORTED' and case.get('missing_fact'):
        claims=getattr(check,'claims',[])
        positive=[c for c in claims if c.status=='SUPPORTED']
        missing=[c for c in claims if c.status=='INSUFFICIENT_INFORMATION' and c.basis=='UNANSWERED'
            and any(re.search(p,c.text,re.I) for p in case['facts'][case['missing_fact']])]
        supported=bool(positive and missing and len(positive)+len(missing)==len(claims))
    checks['answer_supported'] = bool(supported and check.checked_answer == response.answer)
    paths = {row.metadata.get('source') for row in response.results}
    checks.update({'source:'+path:path in paths for path in case['required_sources']})
    return checks


async def benchmark(args):
    import evaluate_user_queries as harness
    import settings
    from pydantic_models import QueryInput
    from llm_contract import current_action
    from model_profiles import get_profile
    from answer_review import review_answer
    logging.getLogger().setLevel(logging.ERROR)
    main, run = harness.main, harness.RUN
    provider, embeddings = harness.get_llm_provider(), harness.get_embedding_provider()
    text_index = main.filesystem_provider.text_index
    if args.text_index_ablation and text_index is None:
        raise ValueError('Text-index ablation requires WORKSPACE_TEXT_INDEX_ENABLED=true')
    protected = {path:hash_file(path) for path in
        (harness.ROOT/'api/rag_app.db',harness.ROOT/'api/chat-model-selection.json')}
    source_hashes = {path:hash_file(path) for path in EXAMPLE_ROOT.iterdir() if path.is_file()}
    code_hashes = {path.name:hash_file(path) for path in (harness.ROOT/'api').glob('*.py')}
    calls, rows = [], []
    stream = provider.stream
    structured = provider.structured_generate
    def experimental_structured(**options):
        schema=options.get('schema') or {};props=schema.get('properties',{})
        is_audit='cs' in props or 'claim_subject' in props or props.get('assessments',{}).get('type')=='array'
        if args.audit_reasoning and is_audit:
            options['thinking']=True
            options['reasoning_budget_tokens']=args.audit_reasoning
            options['max_tokens']=min(get_profile(options.get('profile')).max_tokens,
                                      options.get('max_tokens',640)+args.audit_reasoning)
        return structured(**options)
    provider.structured_generate=experimental_structured
    def observed(messages, **options):
        start=time.perf_counter()
        schema=options.get('schema') or {}
        row={'action_id':current_action.get(),'stage':schema.get('title') or ','.join(schema.get('properties',{})) or 'text',
              'thinking':options.get('thinking'),'max_tokens':options.get('max_tokens'),
              'messages_sha256':hashlib.sha256(json.dumps(messages,ensure_ascii=False,sort_keys=True).encode('utf-8')).hexdigest(),
             'input_characters':sum(len(m.get('content','')) for m in messages),'output_characters':0,'reasoning_characters':0}
        try:
            for delta in stream(messages,**options):
                row['output_characters']+=len(delta.text);row['reasoning_characters']+=len(delta.reasoning)
                if delta.metrics:row.update(asdict(delta.metrics))
                yield delta
        except BaseException as error:
            row['error']=type(error).__name__;raise
        finally:
            row['seconds']=time.perf_counter()-start;calls.append(row)
    provider.stream=observed

    def save(row, start, first_call):
        row.update(seconds=time.perf_counter()-start,calls=calls[first_call:]);rows.append(row)
        (run/'complex-results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
        failed=[key for key,value in row.get('checks',{}).items() if not value]
        mode = (' text' if row.get('text_index_enabled') else ' scan') if args.text_index_ablation else ''
        print(f"{row['phase']} {row['id']}{mode} repeat={row['repeat']}: {'PASS' if row['passed'] else 'FAIL'} "
              f"{row['seconds']:.2f}s, calls={len(row['calls'])}"+(f' failed={failed}' if failed else ''),flush=True)

    try:
        print(f'Output: {run}',flush=True)
        started=time.perf_counter()
        await asyncio.to_thread(provider.manager.ensure_ready)
        startup=time.perf_counter()-started
        await asyncio.to_thread(provider.generate,[{'role':'user','content':'Ответь одним словом: готов.'}],
                                profile='FAST',thinking=False,max_tokens=20,temperature=0)
        print(f'Model ready: {startup:.2f}s; warmup excluded',flush=True)
        if args.phase in ('answers','both'):
            root=run/'documents'
            shutil.copytree(EXAMPLE_ROOT,root)
            workspace=main.workspace_repository.select(str(root))
            main.filesystem_provider.opener=lambda path:(_ for _ in ()).throw(AssertionError('Unexpected open'))
            main.workspace_vector_index.graph_extractor_factory=None
            main.workspace_vector_index.graph_repository=None
            await asyncio.to_thread(embeddings.manager.ensure_ready)
            started=time.perf_counter()
            sync=await asyncio.to_thread(main.workspace_vector_index.sync,workspace.id,extract_relations=False)
            print(f'Index: {time.perf_counter()-started:.2f}s; {sync}',flush=True)
            for repeat in range(1,args.repeats+1):
                for case in CASES:
                    if args.case and case['id'] not in args.case:continue
                    modes = ((False,True) if repeat % 2 else (True,False)) if args.text_index_ablation else (text_index is not None,)
                    for enabled in modes:
                        main.filesystem_provider.text_index = text_index if enabled else None
                        start,first=time.perf_counter(),len(calls)
                        row={'phase':'answers','id':case['id'],'repeat':repeat,'profile':args.profile,
                             'text_index_enabled':enabled}
                        try:
                            response,_=await main.assistant_core.process_input(QueryInput(
                                question=case['question'],model=args.profile,workspace_id=workspace.id,use_multi_agent=True))
                            checks=answer_checks(case,response)
                            action_names={a.id:a.action_type for a in response.actions}
                            row.update(checks=checks,passed=all(checks.values()),response=response.model_dump(mode='json'),
                                action_times=[{'action':a.action_type,'seconds':(datetime.fromisoformat(a.completed_at)-
                                    datetime.fromisoformat(a.started_at)).total_seconds()} for a in response.actions if a.started_at and a.completed_at])
                            for call in calls[first:]:call['action']=action_names.get(call['action_id'])
                        except Exception as error:row.update(passed=False,error=type(error).__name__)
                        save(row,start,first)
            main.filesystem_provider.text_index = text_index
            assert all(hash_file(root/path.name)==digest for path,digest in source_hashes.items()),'Fixture was modified'
        if args.phase in ('review','both'):
            for repeat in range(1,args.repeats+1):
                for case in REVIEW_CASES:
                    if args.case and case['id'] not in args.case:continue
                    sources=[{'chunk_id':f'complex:{case["id"]}:{i}','text':text,'metadata':{'source':f'Source-{i}.md'}}
                             for i,text in enumerate(case['sources'])]
                    for variant in ('valid','invalid'):
                        start,first=time.perf_counter(),len(calls)
                        draft=case[variant];row={'phase':'review','id':case['id']+':'+variant,'repeat':repeat,'profile':args.profile}
                        try:
                            answer,check=await asyncio.to_thread(review_answer,case['question'],draft,sources,
                                'WORKSPACE_QA',args.profile,client=provider)
                            supported_original=any(c.status=='SUPPORTED' and c.text.strip()==draft.strip() for c in check.claims)
                            checks={'review_completed':check.status!='UNAVAILABLE','no_false_supported_original':variant=='valid' or not supported_original}
                            if variant=='valid':checks.update(valid_supported=check.status=='SUPPORTED',valid_preserved=answer==draft)
                            row.update(checks=checks,passed=all(checks.values()),answer=answer,draft=draft,verification=check.model_dump(mode='json'))
                        except Exception as error:row.update(passed=False,error=type(error).__name__)
                        save(row,start,first)
        groups=defaultdict(list)
        for row in rows:
            suffix = (':text' if row.get('text_index_enabled') else ':scan') if args.text_index_ablation else ''
            groups[row['phase']+':'+row['id']+suffix].append(row)
        summary={'passed':sum(row['passed'] for row in rows),'total':len(rows),'tag':args.tag,
            'model':str(get_profile().model_path),'profile':args.profile,'startup_seconds':startup,
            'groups':{key:{'median_seconds':statistics.median(row['seconds'] for row in group),
                         'passed':sum(row['passed'] for row in group),'total':len(group),
                         'llm_calls':[len(row['calls']) for row in group]} for key,group in groups.items()},
            'code_hashes':code_hashes,'compact_audit':settings.EVIDENCE_AUDIT_COMPACT,
            'experimental_audit_reasoning':args.audit_reasoning,
            'protected_files_unchanged':all(hash_file(path)==value for path,value in protected.items()),
            'source_examples_unchanged':all(hash_file(path)==value for path,value in source_hashes.items())}
        (run/'complex-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        print(f"RESULT {summary['passed']}/{summary['total']}; protected files: {summary['protected_files_unchanged']}; examples: {summary['source_examples_unchanged']}; {run}",flush=True)
        return summary['passed']==summary['total'] and summary['protected_files_unchanged'] and summary['source_examples_unchanged']
    finally:
        main.filesystem_provider.text_index = text_index
        provider.stream=stream
        provider.structured_generate=structured
        try:provider.close()
        finally:
            provider.manager.stop();embeddings.manager.stop()
            from chroma_utils import vectorstore,workspace_vectorstore
            for system in {store._client._system for store in (vectorstore,workspace_vectorstore)}:system.stop()


if __name__=='__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('answers','review','both'),default='both')
    parser.add_argument('--profile',choices=('FAST','BALANCED','REASONING'),default='BALANCED')
    parser.add_argument('--case',action='append',choices=[case['id'] for case in CASES+REVIEW_CASES])
    parser.add_argument('--repeats',type=int,default=1)
    parser.add_argument('--tag',default='complex')
    parser.add_argument('--text-index-ablation',action='store_true',
                        help='Alternate live scans/persistent text on the same model, fixture and fresh sessions')
    parser.add_argument('--audit-reasoning',type=int,choices=(0,128,256),default=0,
                        help='Experiment only: bounded thinking for semantic audits')
    args=parser.parse_args()
    if args.repeats<1:parser.error('--repeats must be positive')
    if args.text_index_ablation and args.phase=='review':parser.error('Text-index ablation requires answers')
    sys.exit(0 if asyncio.run(benchmark(args)) else 1)
