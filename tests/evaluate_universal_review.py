"""Opt-in GGUF checks on neutral documents; never writes to the working DB."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


CASES = [
    ('valid-fact', 'Кто изготовил датчик Q17?',
     ['Инженер Лада изготовила датчик модели Q17.'],
     'Лада изготовила датчик модели Q17.', True),
    ('valid-renamed-domain', 'Кто настроил сервер Z9?',
     ['Программист Нури настроил сервер Z9.'],
     'Нури настроил сервер Z9.', True),
    ('valid-condition', 'При каком условии откроется цех?',
     ['Если согласуют бюджет, Инга откроет цех.'],
     'При согласовании бюджета Инга откроет цех.', True),
    ('valid-english', 'Who designed bridge M3?',
     ['Architect Hana designed bridge M3.'], 'Hana designed bridge M3.', True),
    ('english-advice-as-fact', 'What did Simon do?',
     ['Simon, please inspect the valve tomorrow.'], 'Simon inspected the valve.', False),
    ('valid-conflict-report', 'Что известно о температуре образца в 09:00?',
     ['Журнал А: температура образца в 09:00 составляет 20 градусов.',
      'Журнал Б: температура того же образца в 09:00 составляет 35 градусов.'],
     'Журналы расходятся: в журнале А указано 20 градусов, в журнале Б — 35 градусов.', True),
    ('wrong-actor', 'Кто изготовил датчик Q17?',
     ['Инженер Лада изготовила датчик модели Q17.'],
     'Инженер Олег изготовил датчик модели Q17.', False),
    ('wrong-object', 'Что изготовила Лада?',
     ['Инженер Лада изготовила датчик модели Q17.'],
     'Лада изготовила двигатель модели Q17.', False),
    ('reversed-transfer', 'Кто передал чертежи?',
     ['Тамара передала чертежи Денису.'],
     'Денис передал чертежи Тамаре.', False),
    ('example-as-biography', 'Кто такой Арсен?',
     ['Для упражнения повторите: «Я пилот Арсен и перевожу грузы». Это вымышленный пример.'],
     'Арсен работает пилотом и перевозит грузы.', False),
    ('intention-as-event', 'Что сделала Анна?',
     ['Анна планирует построить мост в следующем году. Работы ещё не начались.'],
     'Анна построила мост.', False),
    ('advice-as-duty', 'Что известно о работе Киры?',
     ['Кира, измерь температуру образца утром, если успеешь.'],
     'Кира обязана ежедневно измерять температуру образца.', False),
    ('author-as-recipient', 'Кто заведует лабораторией?',
     ['Саша, ты заведуешь лабораторией. Я перевожу журналы. Автор письма — Нина.'],
     'Нина заведует лабораторией.', False),
    ('reader-as-user', 'Кому поручен отчёт?',
     ['Лев, тебе поручено подготовить отчёт. Автор письма — Зоя.'],
     'Пользователю чата поручено подготовить отчёт.', False),
    ('role-as-organization-rule', 'Какие правила у мастерской Парус?',
     ['Координатор мастерской Парус собирается изучить английский язык.'],
     'Все работники мастерской Парус обязаны изучать английский язык.', False),
    ('metaphor-as-mechanism', 'Что делает Вера?',
     ['В дневнике Вера пишет: «Мне нужно разжечь огонь идеи». Это образное выражение.'],
     'Вера разжигает настоящий огонь.', False),
    ('conflicting-sources', 'Какая температура образца в 09:00?',
     ['Журнал А: температура образца в 09:00 составляет 20 градусов.',
      'Журнал Б: температура того же образца в 09:00 составляет 35 градусов.'],
     'Температура образца в 09:00 составляет 20 градусов.', False),
]


def evaluate(repeats, selected=None):
    # This harness sets isolated storage and free ports BEFORE application imports.
    import evaluate_user_queries as harness
    from answer_review import review_answer
    from model_runtime import get_llm_provider
    provider = get_llm_provider()
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    protected = [harness.ROOT / 'api' / name for name in ('rag_app.db', 'chat-model-selection.json')]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
              for path in protected}
    records = []
    print(f'Output: {harness.RUN}', flush=True)
    try:
        provider.manager.ensure_ready()
        provider.generate([{'role': 'user', 'content': 'Ответь одним словом: готов.'}],
                          profile='FAST', max_tokens=16, temperature=0)
        for repeat in range(repeats):
            for label, question, texts, draft, expected in CASES:
                if selected and label not in selected:
                    continue
                sources = [{'chunk_id': f'source:{i}', 'text': text,
                            'metadata': {'source': f'Источник-{i + 1}.txt'}}
                           for i, text in enumerate(texts)]
                started = time.perf_counter()
                answer, check = review_answer(question, draft, sources, 'WORKSPACE_QA', 'FAST', client=provider)
                passed = (check.status == 'SUPPORTED' if expected else
                          check.status in {'UNSUPPORTED', 'CONFLICTING', 'PARTIALLY_SUPPORTED'}
                          or check.status == 'SUPPORTED' and answer != draft)
                row = {'id': label, 'repeat': repeat + 1, 'passed': passed,
                       'seconds': time.perf_counter() - started, 'sources': sources,
                       'question': question, 'draft': draft, 'answer': answer,
                       'verification': check.model_dump(mode='json')}
                records.append(row)
                (harness.RUN / 'universal-review.json').write_text(
                    json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
                print(f'{label}: {"PASS" if passed else "FAIL"} {check.status} {row["seconds"]:.2f}s', flush=True)
        unchanged = all((hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None) == digest
                        for path, digest in before.items())
        summary = {'passed': sum(r['passed'] for r in records), 'total': len(records),
                   'protected_files_unchanged': unchanged}
        (harness.RUN / 'universal-review-summary.json').write_text(json.dumps(summary), encoding='utf-8')
        print(f'RESULT {summary}', flush=True)
        return all(r['passed'] for r in records) and unchanged
    finally:
        provider.close()
        provider.manager.stop()
        from chroma_utils import vectorstore, workspace_vectorstore
        for system in {store._client._system for store in (vectorstore, workspace_vectorstore)}:
            system.stop()


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--case', action='append', choices=[row[0] for row in CASES])
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    sys.exit(0 if evaluate(args.repeats, args.case) else 1)
