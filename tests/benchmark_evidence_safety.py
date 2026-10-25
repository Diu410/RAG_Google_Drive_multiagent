"""Real GGUF paired comparison on synthetic adversarial evidence; isolated data."""
import argparse
import asyncio
from dataclasses import asdict
import json
import logging
from pathlib import Path
import statistics
import time


CASES = [
    ('subject', 'Анна хранит карты северного архива.',
     'Анна хранит карты северного архива.', 'Борис хранит карты северного архива.', 'Кто хранит карты северного архива?'),
    ('direction', 'Анна обучает Бориса работе с архивом.',
     'Анна обучает Бориса работе с архивом.', 'Борис обучает Анну работе с архивом.', 'Кто кого обучает работе с архивом?'),
    ('negation', 'Мирон не открывает южный архив.',
     'Мирон не открывает южный архив.', 'Мирон открывает южный архив.', 'Открывает ли Мирон южный архив?'),
    ('intention', 'Анна намерена открыть северный архив.',
     'Анна намерена открыть северный архив.', 'Анна открыла северный архив.', 'Каково намерение Анны?'),
    ('condition', 'Если получено разрешение, Мирон открывает архив.',
     'Если получено разрешение, Мирон открывает архив.', 'Мирон всегда открывает архив.', 'При каком условии Мирон открывает архив?'),
    ('self-description', 'В записке Мирон пишет: «Я хранитель северного архива».',
     'В записке Мирон называет себя хранителем северного архива.',
     'Мирон действительно является хранителем северного архива.', 'Как Мирон описывает себя в записке?'),
]


async def benchmark(repeats, selected_cases=None):
    import evaluate_user_queries as harness
    logging.getLogger().setLevel(logging.ERROR)
    import settings
    from evidence_audit import EvidenceAuditor
    from pydantic_models import VerificationClaim
    from verifier import Verifier
    provider = harness.get_llm_provider()
    original_compact, original_batch = settings.EVIDENCE_AUDIT_COMPACT, settings.EVIDENCE_AUDIT_BATCH_SIZE
    rows = []
    try:
        await asyncio.to_thread(provider.manager.ensure_ready)
        await asyncio.to_thread(provider.generate, [{'role':'user','content':'Ответь: готов.'}],
                                profile='FAST', thinking=False, max_tokens=20)
        settings.EVIDENCE_AUDIT_BATCH_SIZE = 2
        for repeat in range(repeats):
            for name, quote, valid, invalid, question in CASES:
                if selected_cases and name not in selected_cases:
                    continue
                claims = [VerificationClaim(text=text, status='SUPPORTED', basis='EXPLICIT',
                    evidence_quote=quote, source_chunk_ids=['safety:'+name]) for text in (valid, invalid)]
                source = [{'chunk_id':'safety:'+name, 'text':quote}]
                for compact in ((False, True) if repeat % 2 == 0 else (True, False)):
                    settings.EVIDENCE_AUDIT_COMPACT = compact
                    start = time.perf_counter()
                    row = {'case':name, 'compact':compact, 'repeat':repeat+1}
                    try:
                        outcomes = await asyncio.to_thread(EvidenceAuditor(provider, Verifier(provider).bounded_sources).supports_many,
                            claims, source, 'FAST', question=question, think=False)
                        row['actual'] = [bool(supported) for supported,_ in outcomes]
                        row['assessments'] = [auditor.last_assessment.model_dump() for _,auditor in outcomes]
                        row['guards'] = [getattr(auditor,'guard_rejection',None) for _,auditor in outcomes]
                        row['passed'] = row['actual'] == [True, False]
                    except Exception as error:
                        row.update(passed=False,error=type(error).__name__)
                    row['seconds'] = time.perf_counter()-start
                    rows.append(row)
                    print(f"{name} {'compact' if compact else 'legacy'} repeat {repeat+1}: {row.get('actual',row.get('error'))} {row['seconds']:.2f}s",flush=True)
                    (harness.RUN/'evidence-safety.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
        summary = {'passed':sum(row['passed'] for row in rows),'total':len(rows),
            'formats':{name:{'passed':sum(row['passed'] for row in rows if row['compact']==compact),
                'total':sum(row['compact']==compact for row in rows),
                'median_seconds':statistics.median(row['seconds'] for row in rows if row['compact']==compact)}
                for name,compact in [('legacy',False),('compact',True)]}}
        (harness.RUN/'evidence-safety-summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print(f'RESULT {summary}; {harness.RUN}',flush=True)
        return summary['passed']==summary['total']
    finally:
        settings.EVIDENCE_AUDIT_COMPACT, settings.EVIDENCE_AUDIT_BATCH_SIZE = original_compact, original_batch
        provider.close();provider.manager.stop()
        from chroma_utils import vectorstore,workspace_vectorstore
        for system in {store._client._system for store in (vectorstore,workspace_vectorstore)}:
            system.stop()


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--case',action='append',choices=[case[0] for case in CASES])
    args=parser.parse_args()
    if args.repeats<1:parser.error('--repeats must be positive')
    sys.exit(0 if asyncio.run(benchmark(args.repeats,args.case)) else 1)
