"""Real evidence audits near the context limit, with independent negative controls."""
import argparse
import asyncio
from dataclasses import asdict
import json
import logging
import os
import time


async def benchmark(args):
    if args.reuse_running:
        os.environ['EVALUATION_CHAT_SERVER_URL'] = 'http://127.0.0.1:8081'
        os.environ['EVALUATION_EMBEDDING_SERVER_URL'] = 'http://127.0.0.1:8082'
    import evaluate_user_queries as harness
    from evidence_audit import EvidenceAuditor
    from pydantic_models import VerificationClaim
    from verifier import Verifier
    logging.getLogger().setLevel(logging.ERROR)
    provider = harness.get_llm_provider()
    question = 'Что Ольга сообщает о себе и о задании Бориса?'
    letter = ('Борис, ты должен подготовить карты до пятницы. '
              'Если ты восстановишь карты, то я намерена открыть северный архив. '
              'Я называю себя хранителем северного архива. Автор письма — Ольга.')
    padding = ('Служебная запись о хранении бумаг: номер коробки не является должностью '
               'автора или адресата. Запись не устанавливает факта назначения на должность.\n') * 40
    cases = [
        ('self-report', 'В письме Ольга называет себя хранителем северного архива.', True),
        ('wrong-person', 'В письме Борис называет себя хранителем северного архива.', False),
        ('recipient-duty', 'Ольга должна подготовить карты до пятницы.', False),
        ('lost-condition', 'Ольга уже открыла северный архив.', False),
    ]
    rows, calls = [], []
    original_stream = provider.stream

    def observe(messages, **options):
        call = {'input_characters': sum(len(m['content']) for m in messages),
                'quote_references': '"quote_ids"' in messages[-1]['content']}
        calls.append(call)
        for delta in original_stream(messages, **options):
            if delta.metrics:
                call.update(asdict(delta.metrics))
            yield delta

    provider.stream = observe
    try:
        await asyncio.to_thread(provider.manager.ensure_ready)
        await asyncio.to_thread(provider.generate, [{'role':'user','content':'Ответь: готов.'}],
                                profile='FAST', thinking=False, max_tokens=20)
        for label, source in [('short',letter),('long',padding + letter)]:
            for name, answer, expected in cases:
                start, first_call = time.perf_counter(), len(calls)
                claim = VerificationClaim(text=answer,status='SUPPORTED',basis='EXPLICIT',
                    source_chunk_ids=['context:'+label],evidence_quote=source)
                auditor = EvidenceAuditor(provider,Verifier(provider).bounded_sources)
                row = {'case':name,'context':label,'expected_supported':expected}
                try:
                    actual = await asyncio.to_thread(auditor.supports,claim,
                        [{'chunk_id':'context:'+label,'text':source}],args.profile,
                        question=question,think=False)
                    row.update(actual_supported=actual,passed=actual==expected,
                               comparison=auditor.last_assessment.model_dump(mode='json'),
                               guard_rejection=auditor.guard_rejection)
                except Exception as exc:
                    row.update(passed=False,error=type(exc).__name__,detail=str(exc))
                row.update(seconds=time.perf_counter()-start,calls=calls[first_call:])
                rows.append(row)
                (harness.RUN/'evidence-context.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
                print(label,name,'PASS' if row['passed'] else 'FAIL',round(row['seconds'],2),'s',flush=True)
        summary = {'passed':sum(r['passed'] for r in rows),'total':len(rows),
            'reference_calls':sum(c['quote_references'] for r in rows for c in r['calls']),
            'false_acceptances':sum(r.get('actual_supported') is True and not r['expected_supported'] for r in rows),
            'unavailable':sum('error' in r for r in rows)}
        (harness.RUN/'evidence-context-summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print('RESULT',summary,harness.RUN,flush=True)
        return all(r['passed'] for r in rows) and summary['reference_calls'] > 0
    finally:
        provider.stream = original_stream
        provider.close();provider.manager.stop();harness.get_embedding_provider().manager.stop()
        from chroma_utils import vectorstore,workspace_vectorstore
        for system in {v._client._system for v in (vectorstore,workspace_vectorstore)}:
            system.stop()


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-running',action='store_true')
    parser.add_argument('--profile',choices=('FAST','BALANCED','REASONING'),default='BALANCED')
    sys.exit(0 if asyncio.run(benchmark(parser.parse_args())) else 1)
