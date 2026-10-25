"""Real local model, repository edits and checks on immutable fixture copies."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


async def evaluate(args):
    if args.reuse_running:
        os.environ['EVALUATION_CHAT_SERVER_URL'] = 'http://127.0.0.1:8081'
        os.environ['EVALUATION_EMBEDDING_SERVER_URL'] = 'http://127.0.0.1:8082'
    import evaluate_user_queries as harness
    from pydantic_models import QueryInput
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    main, root, run = harness.main, harness.ROOT, harness.RUN
    protected = {p:digest(p) for p in (root/'api/rag_app.db', root/'api/chat-model-selection.json')}
    originals = {p:digest(p) for p in (root/'examples/agent_workspace').rglob('*') if p.is_file()}
    code = {p.name:digest(p) for p in (root/'api').glob('*.py')}
    rows, calls = [], []
    provider = harness.get_llm_provider()
    original_generate = provider.structured_generate

    def measured(*a, **kw):
        started = time.perf_counter()
        result = original_generate(*a, **kw)
        calls.append({'seconds': time.perf_counter()-started, 'text': result.text,
            'metrics': vars(result.metrics)})
        (run/'repository-model-calls.json').write_text(json.dumps(calls, ensure_ascii=False, indent=2), encoding='utf-8')
        return result

    provider.structured_generate = measured
    definitions = [
        ('python_reporting', 'В проекте расчёт выручки включает неоплаченные счета. Разбери и исправь код, '
         'затем запусти тесты. Учитывай только строки со статусом paid. Тесты не меняй.',
         ('report.py',), [sys.executable, '-c', 'from report import total_paid; '
         'assert total_paid([])==0; assert total_paid([dict(amount=5,status="paid"),'
         'dict(amount=90,status="refunded"),dict(amount=30,status="pending")])==5']),
        ('node_ranges', 'Функция clamp неверно ограничивает числа заданным диапазоном. Найди и исправь '
         'ошибку в коде проекта, затем запусти тесты npm. Тесты не меняй.',
         ('range.js',), [shutil.which('node'), '--input-type=module', '-e',
         'import {clamp} from "./range.js"; '
         'for(const [v,lo,hi,want] of [[-20,-5,8,-5],[3,-5,8,3],[30,-5,8,8]])'
         '{if(clamp(v,lo,hi)!==want)process.exit(1)}']),
        ('python_pipeline', 'В проекте неверный счёт по заказу: скидки заданы процентами от 0 до 100, '
         'а денежный итог должен округляться до двух знаков после суммирования всех строк. '
         'Разбери связь pricing.py и orders.py, исправь код и запусти тесты. Тесты не меняй.',
         ('pricing.py', 'orders.py'), [sys.executable, '-c',
         'from pricing import line_total;from orders import bill_total;'
         'assert line_total(4,10,25)==30;assert bill_total([])==0;'
         'assert bill_total([dict(quantity=3,price=1.23,discount=0)])==3.69']),
    ]
    try:
        print('Output:', run, flush=True)
        await asyncio.to_thread(provider.manager.ensure_ready)
        for name, question, source_names, independent in definitions:
            if args.case and name != args.case:
                continue
            target = run/name
            shutil.copytree(root/'examples/agent_workspace'/name, target)
            test_hashes = {p:digest(p) for p in target.rglob('*') if p.is_file() and p.name not in source_names}
            workspace = main.workspace_repository.select(str(target))
            start, initial_calls = time.perf_counter(), len(calls)
            row = {'id': name, 'question': question}
            try:
                response, _ = await main.assistant_core.process_input(QueryInput(question=question,
                    model=args.profile, workspace_id=workspace.id))
                actions = response.actions
                effects = [a.output.get('data', {}) for a in actions if a.action_type in {'REPOSITORY_EDIT','REPOSITORY_REWRITE'}
                           and a.status.value == 'COMPLETED' and a.output.get('data', {}).get('applied') is True]
                verified = await asyncio.to_thread(main.assistant_core.repository_agent_skill.checks._run,
                    target, independent, 'independent-behavior')
                checks = {'repository_route': response.strategy == 'REPOSITORY_AGENT',
                    'completed': response.plan_status == 'COMPLETED',
                    'source_edited': all(any(e['path']==path for e in effects) for path in source_names),
                    'actual_check': any(a.action_type=='REPOSITORY_CHECK' and a.output.get('data',{}).get('passed') for a in actions),
                    'behavior': verified['passed'], 'tests_and_config_unchanged': all(digest(p)==h for p,h in test_hashes.items()),
                    'backup': all(e.get('backup_path') and (target/e['backup_path']).is_file() for e in effects)}
                row.update(response=response.model_dump(mode='json'), checks=checks,
                           independent_check=verified, passed=all(checks.values()))
            except Exception as exc:
                row.update(passed=False, error=type(exc).__name__+': '+str(exc))
            row.update(seconds=time.perf_counter()-start, model_calls=len(calls)-initial_calls)
            rows.append(row)
            (run/'repository-agent.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
            print(name, 'PASS' if row['passed'] else 'FAIL', round(row['seconds'], 2),
                  's', row['model_calls'], 'model calls', flush=True)
        summary = {'passed': sum(r['passed'] for r in rows), 'total': len(rows),
            'protected_unchanged': all(digest(p)==h for p,h in protected.items()),
            'fixtures_unchanged': all(digest(p)==h for p,h in originals.items()),
            'code_unchanged': code == {p.name:digest(p) for p in (root/'api').glob('*.py')}, 'code_hashes': code}
        (run/'repository-agent-summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        print('RESULT', summary['passed'], '/', summary['total'], flush=True)
        return summary['total']>0 and summary['passed']==summary['total'] and all(summary[k] for k in (
            'protected_unchanged', 'fixtures_unchanged', 'code_unchanged'))
    finally:
        provider.structured_generate = original_generate
        provider.close()
        provider.manager.stop()  # Stops only the subprocess owned by this harness.
        from chroma_utils import vectorstore, workspace_vectorstore
        for system in {s._client._system for s in (vectorstore, workspace_vectorstore)}:
            system.stop()


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('FAST', 'BALANCED', 'REASONING'), default='BALANCED')
    parser.add_argument('--case', choices=('python_reporting', 'node_ranges', 'python_pipeline'))
    parser.add_argument('--reuse-running', action='store_true', help='Reuse existing 8081/8082 without starting another model')
    sys.exit(0 if asyncio.run(evaluate(parser.parse_args())) else 1)
