"""Compare saved real runs under the same current, independent criteria.

Original artifacts are never overwritten. No model, database or private source
is accessed. Content matching accepts equivalent wording; completion and exact
checked-answer requirements remain mandatory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
from types import SimpleNamespace

from benchmark_complex_workspace import answer_checks
from complex_workspace_cases import CASES


def read_run(directory):
    rows=json.loads((directory/'complex-results.json').read_text(encoding='utf-8'))
    cases={case['id']:case for case in CASES}
    checked=[]
    for row in rows:
        checks=row.get('checks',{})
        if row['phase']=='answers' and 'response' in row:
            response=dict(row['response'])
            response['task_status']=SimpleNamespace(value=response['task_status'])
            if response.get('verification'):
                verification=dict(response['verification'])
                verification['claims']=[SimpleNamespace(**c) for c in verification.get('claims',[])]
                response['verification']=SimpleNamespace(**verification)
            response['results']=[SimpleNamespace(**item) for item in response['results']]
            checks=answer_checks(cases[row['id']],SimpleNamespace(**response))
        checked.append({'phase':row['phase'],'id':row['id'],'repeat':row['repeat'],
            'passed':bool(checks) and all(checks.values()),'checks':checks,
            'original_passed':row['passed'],'seconds':row['seconds'],'calls':len(row['calls'])})
    return checked


def summary(rows):
    groups={}
    for phase in ('answers','review'):
        selected=[row for row in rows if row['phase']==phase]
        groups[phase]={'passed':sum(row['passed'] for row in selected),'total':len(selected)}
    groups['all']={'passed':sum(row['passed'] for row in rows),'total':len(rows)}
    return groups


def compare(baseline,candidate):
    before,after=read_run(baseline),read_run(candidate)
    comparisons=[]
    for key in dict.fromkeys((r['phase'],r['id']) for r in before):
        a=[row for row in before if (row['phase'],row['id'])==key]
        b=[row for row in after if (row['phase'],row['id'])==key]
        if not b:continue
        comparisons.append({'phase':key[0],'id':key[1],
            'baseline_passed':sum(r['passed'] for r in a),'baseline_total':len(a),
            'candidate_passed':sum(r['passed'] for r in b),'candidate_total':len(b),
            'baseline_median_seconds':statistics.median(r['seconds'] for r in a),
            'candidate_median_seconds':statistics.median(r['seconds'] for r in b),
            'baseline_calls':[r['calls'] for r in a],'candidate_calls':[r['calls'] for r in b]})
    return {'baseline':str(baseline),'candidate':str(candidate),
        'criteria_sha256':hashlib.sha256(Path(__file__).with_name('complex_workspace_cases.py').read_bytes()).hexdigest(),
        'checker_sha256':hashlib.sha256(Path(__file__).with_name('benchmark_complex_workspace.py').read_bytes()).hexdigest(),
        'baseline_summary':summary(before),'candidate_summary':summary(after),
        'comparison':comparisons,'baseline_rechecked':before,'candidate_rechecked':after}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=compare(args.baseline,args.candidate)
    output=args.output or args.candidate/'complex-comparison.json'
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({key:result[key] for key in ('baseline_summary','candidate_summary')}))
    print(output)
