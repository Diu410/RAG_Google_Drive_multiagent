"""Reduced model calls retain evidence isolation, guards and file freshness."""
import asyncio
from contextlib import asynccontextmanager, closing
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from support import UserEnvironment
from evidence_audit import EvidenceAuditor, EvidenceRelation
from graph_rag import GraphAugmentedRetriever
from graph_service import GraphSearchResult
from graph_extraction import GraphExtractor, GraphExtractionResult, ExtractedRelation
from langchain_core.documents import Document
from hybrid_context_extractor import SearchResult
from knowledge_skill import KnowledgeSkill, KnowledgeServices
from langchain_core.runnables import RunnableLambda
import langchain_utils
import multi_agent_chain
from agent_planner import is_single_information_request
from llm_contract import GenerationCancelled, StructuredOutputTruncated
from overview_facts import FactFirstOverview
from pydantic_models import VerificationClaim
from query_router import RouteDecision, QueryStrategy
from test_overview_facts import Provider, SOURCE
from verifier import Verifier
from workspace import workspace_scan_cache
from prompt_budget import fit_prompt


class SingleQuestionPlanningTests(unittest.TestCase):
    def test_core_skips_model_planner_and_router_for_information_question_in_workspace(self):
        from assistant_models import ToolResult
        from workspace_router import WorkspaceRouteDecision
        environment = UserEnvironment()
        self.addCleanup(environment.close)
        planner = Mock()
        environment.core.agent_planner = planner
        decision = WorkspaceRouteDecision(route='WORKSPACE_CONTENT', operation='answer_question', confidence=1, reason='fact')
        result = ToolResult(success=True, tool='workspace', summary='ok', data={'final': 'CHECKED_WORKSPACE_ANSWER'})
        with patch.object(environment.skill, 'route_ambiguous', return_value=decision) as route, \
             patch.object(environment.skill, 'execute', return_value=result) as execute:
            response = asyncio.run(environment.ask('Как зовут хранителя северного архива?'))
        planner.plan.assert_not_called()
        route.assert_not_called()
        execute.assert_called_once()
        self.assertEqual(response.answer, 'CHECKED_WORKSPACE_ANSWER')
        self.assertNotIn('BUILD_AGENT_PLAN', [a.action_type for a in response.actions])

    def test_information_question_keeps_retrieval_without_workflow_planning(self):
        for question in ('Как зовут хранителя северного архива?', 'Кто такой Мирон?',
                         'К кому обращается Мирон?', 'С кем работает хранитель?',
                         'Для кого хранитель готовит карты?',
                         'По документам кто такой Мирон и что он хранит?',
                         'Какой рост и допустимая нагрузка у смотрителя?',
                         'Как зовут хранителя и какой у него телефон?'):
            with self.subTest(question=question):
                self.assertTrue(is_single_information_request(question))

    def test_operations_sequences_and_ambiguous_requests_keep_planner(self):
        for question in ('Найди документы и сравни их', 'Кто такой Мирон? Затем открой файл',
                         'Какая последняя версия отчёта, найди и открой её?',
                         'Что в документах? Сохрани ответ в файл', 'Сделай обзор всех документов',
                         'Посмотри проект', 'Кто такой Мирон? Что он хранит?'):
            with self.subTest(question=question):
                self.assertFalse(is_single_information_request(question))

    def test_simple_fact_budget_avoids_llm_and_explicit_analysis_keeps_priority(self):
        from response_budget import ResponseBudgetPlanner
        with patch('response_budget.model_runtime.get_llm_provider') as provider:
            for question in ('Как зовут хранителя?', 'Какой телефон у хранителя?',
                             'Какой рост и допустимая нагрузка указаны в файле Стандарт.md?'):
                self.assertEqual(ResponseBudgetPlanner().decide(question, 'FAST').detail.value, 'BRIEF')
            self.assertEqual(ResponseBudgetPlanner().decide('Подробно проанализируй рост и нагрузку.', 'FAST').detail.value,
                             'DETAILED')
            self.assertEqual(ResponseBudgetPlanner().decide('Кто такой Мирон?', 'FAST').detail.value, 'NORMAL')
            self.assertEqual(ResponseBudgetPlanner().decide('К кому обращается Мирон?', 'FAST').detail.value, 'NORMAL')
            provider.assert_not_called()

    def test_declined_role_is_explicit_but_anonymous_speaker_is_not_bound(self):
        def assess(quote):
            claim = VerificationClaim(text='Рост смотрителя — 184 см.', status='SUPPORTED', basis='EXPLICIT',
                                      evidence_quote=quote, source_chunk_ids=['source'])
            auditor = EvidenceAuditor(SimpleNamespace(), lambda *args: None)
            relation = EvidenceRelation(**dict(
                claim_subject='смотритель', source_subject='смотритель', claim_predicate='рост 184 см',
                source_predicate='рост 184 см', claim_modality='FACT', source_modality='FACT',
                source_subject_kind='PERSON', relevant_to_question=True,
                same_subject=True, same_predicate=True, same_modality=True))
            return auditor.evaluate(claim, [{'chunk_id': 'source', 'text': quote}], relation), auditor
        for quote in ('Рост смотрителя — 184 см.', 'Смотрителю приписан рост 184 см.'):
            self.assertTrue(assess(quote)[0])
        rejected, auditor = assess('Мой рост — 184 см.')
        self.assertFalse(rejected)
        self.assertEqual(auditor.guard_rejection, 'speaker_binding')
        self.assertFalse(assess('Рост смотрительности — 184 см.')[0])


class GraphEvidenceProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [Document(page_content='Проект Орион курирует Мирон.', metadata={'chunk_id': 'first'}),
                       Document(page_content='Мирон хранит карты архива.', metadata={'chunk_id': 'second'})]

    def test_unknown_or_wrong_id_resolves_only_to_actual_unique_quote(self):
        for chunk_id in ('invented', 'first'):
            relation = ExtractedRelation(source='Мирон', relation='STORES', target='карты',
                evidence='Мирон хранит карты архива.', chunk_id=chunk_id)
            self.assertEqual(GraphExtractor._evidence_chunk(relation, self.chunks), 'second')

    def test_missing_invented_or_ambiguous_quote_cannot_pick_a_default_chunk(self):
        for quote in ('', 'Мирон хранит деньги.'):
            relation = ExtractedRelation(source='Мирон', relation='STORES', target='карты',
                evidence=quote, chunk_id='first')
            self.assertIsNone(GraphExtractor._evidence_chunk(relation, self.chunks))
        duplicate = [self.chunks[0], Document(page_content=self.chunks[0].page_content, metadata={'chunk_id': 'copy'})]
        relation = ExtractedRelation(source='Мирон', relation='CURATES', target='Орион',
                evidence=self.chunks[0].page_content, chunk_id='unknown')
        self.assertIsNone(GraphExtractor._evidence_chunk(relation, duplicate))
        relation.chunk_id = 'copy'
        self.assertEqual(GraphExtractor._evidence_chunk(relation, duplicate), 'copy')

    def test_persistence_excludes_unsupported_relation_and_keeps_literal_provenance(self):
        repository = Mock()
        extractor = GraphExtractor(repository)
        result = GraphExtractionResult(relations=[
            ExtractedRelation(source='Мирон', relation='STORES', target='карты',
                              evidence=self.chunks[1].page_content, chunk_id='unknown'),
            ExtractedRelation(source='Мирон', relation='STORES', target='деньги',
                              evidence='Мирон хранит деньги.', chunk_id='first')])
        with patch.object(extractor, '_extract_batch_adaptive', return_value=[(self.chunks, result)]):
            self.assertTrue(extractor.extract_document(self.chunks, 1, 'test.txt'))
        relations = repository.replace_document_graph.call_args.args[2]
        self.assertEqual(len(relations), 1)
        self.assertEqual(relations[0]['chunk_id'], 'second')
        self.assertEqual(extractor.rejected_relations, 1)



def relation(**updates):
    values = dict(claim_subject='Архив', source_subject='Архив', claim_predicate='хранит карты',
        source_predicate='хранит карты', claim_modality='FACT', source_modality='FACT',
        source_subject_kind='ORGANIZATION', relevant_to_question=True,
        same_subject=True, same_predicate=True, same_modality=True)
    return {**values, **updates}


class BatchProvider:
    supports_batch_evidence = True

    def __init__(self, failure=None):
        self.calls = []
        self.failure = failure

    def count_prompt(self, *args):
        return 100

    def structured_generate(self, **options):
        self.calls.append(options)
        if 'assessments' in options['schema']['properties']:
            if isinstance(self.failure, BaseException):
                raise self.failure
            packet = json.loads(options['messages'][-1]['content'])
            rows = [dict(id=item['id'], **relation(claim_predicate=item['claim'],
                source_predicate=item['claim'])) for item in packet['items']]
            if self.failure == 'duplicate':
                rows[1]['id'] = '0'
            if self.failure == 'missing':
                rows[1].pop('same_modality')
            # Reverse response order to exercise ID matching, not zip order.
            payload = {'assessments': rows[::-1]}
        else:
            payload = relation()
        return SimpleNamespace(text=json.dumps(payload))


class BatchAuditTests(unittest.TestCase):
    def setUp(self):
        self.sources = [{'chunk_id': 'a', 'text': 'Архив хранит карты.'},
                        {'chunk_id': 'b', 'text': 'Архив основан в 2001 году.'}]
        self.claims = [VerificationClaim(text=s['text'], status='SUPPORTED', basis='EXPLICIT',
            evidence_quote=s['text'], source_chunk_ids=[s['chunk_id']]) for s in self.sources]

    def audit(self, provider, claims=None):
        return EvidenceAuditor(provider, lambda prefix, sources, *args, **kw: sources).supports_many(
            claims or self.claims, self.sources, 'BALANCED', question='Что за Архив?')

    def test_two_claims_use_one_call_with_isolated_evidence(self):
        provider = BatchProvider()
        outcomes = self.audit(provider)
        self.assertEqual(len(provider.calls), 1)
        self.assertTrue(all(supported for supported, _ in outcomes))
        packet = json.loads(provider.calls[0]['messages'][-1]['content'])
        self.assertEqual([s['chunk_id'] for s in packet['items'][0]['sources']], ['a'])
        self.assertEqual([s['chunk_id'] for s in packet['items'][1]['sources']], ['b'])
        self.assertEqual(len(outcomes), 2)

    def test_reversed_response_order_keeps_rejection_attached_to_its_claim(self):
        provider = BatchProvider()
        def generate(**options):
            provider.calls.append(options)
            return SimpleNamespace(text=json.dumps({'assessments': [
                dict(id='1', **relation(same_predicate=False)), dict(id='0', **relation())]}))
        provider.structured_generate = generate
        self.assertEqual([supported for supported, _ in self.audit(provider)], [True, False])

    def test_invalid_or_truncated_batch_falls_back_to_complete_single_checks(self):
        for failure in ('duplicate', 'missing', StructuredOutputTruncated('length')):
            with self.subTest(failure=failure):
                provider = BatchProvider(failure)
                self.assertTrue(all(supported for supported, _ in self.audit(provider)))
                self.assertEqual(len(provider.calls), 3)

    def test_cancelled_batch_never_retries(self):
        provider = BatchProvider(GenerationCancelled())
        with self.assertRaises(GenerationCancelled):
            self.audit(provider)
        self.assertEqual(len(provider.calls), 1)

    def test_over_budget_batch_skips_generation_and_uses_single_checks(self):
        provider = BatchProvider()
        provider.count_prompt = lambda *args: 100000
        self.audit(provider)
        self.assertEqual(len(provider.calls), 2)
        self.assertTrue(all('assessments' not in c['schema']['properties'] for c in provider.calls))


    def test_batch_verdict_keeps_modality_guard(self):
        claim = self.claims[0]
        auditor = EvidenceAuditor(None, None)
        assessment = EvidenceRelation.model_validate({**relation(), 'source_modality': 'INTENTION'})
        self.assertFalse(auditor.evaluate(claim, self.sources[:1], assessment))
        self.assertEqual(auditor.guard_rejection, 'modality_strengthening')

    def test_disabled_batching_uses_only_individual_checks(self):
        provider = BatchProvider()
        with patch('settings.EVIDENCE_AUDIT_BATCH_SIZE', 1):
            self.audit(provider)
        self.assertEqual(len(provider.calls), 2)
        self.assertTrue(all('assessments' not in c['schema']['properties'] for c in provider.calls))


class PromptPreparationTests(unittest.TestCase):
    def test_large_prompt_reaches_exact_budget_without_repeated_halving(self):
        calls = []
        def count(messages):
            size = sum(len(item['content']) for item in messages)
            calls.append(size)
            return size
        messages = [{'role': 'system', 'content': 'RULE'},
                    {'role': 'user', 'content': 'QUESTION\nКонтекст:\n' + 'x' * 100000}]
        fitted, output = fit_prompt(messages, 512, 128, count)
        self.assertLessEqual(sum(len(m['content']) for m in fitted) + output + 32, 512)
        self.assertLessEqual(len(calls), 3)
        self.assertTrue(fitted[-1]['content'].startswith('QUESTION\nКонтекст:\n'))
        self.assertEqual(len(messages[-1]['content']), 100019)


class FastWorkflowTests(unittest.TestCase):
    def test_fact_preparation_batches_candidates_without_merging_their_proof(self):
        provider = Provider()
        provider.supports_batch_evidence = True
        provider.count_prompt = lambda *args: 100
        generate = provider.structured_generate
        def batched(**kw):
            if 'assessments' not in kw['schema']['properties']:
                return generate(**kw)
            provider.calls.append(kw)
            packet = json.loads(kw['messages'][-1]['content'])
            return SimpleNamespace(text=json.dumps({'assessments': [dict(id=item['id'],
                **relation(claim_predicate=item['claim'], source_predicate=item['claim']))
                for item in packet['items']]}))
        provider.structured_generate = batched
        pipeline = FactFirstOverview(Verifier(provider))
        facts = pipeline.prepare('Что за Северный архив?', [SOURCE], 'BALANCED')
        self.assertEqual(len(facts), 2)
        self.assertEqual(len(provider.calls), 2, 'One extraction and one batch audit')
        self.assertEqual([f.evidence_quote for f in facts], [p['evidence_quote'] for p in provider.proposals])

    def test_fast_overview_keeps_checked_facts_without_writer_or_retelling(self):
        provider = Provider()
        pipeline = FactFirstOverview(Verifier(provider))
        facts = pipeline.prepare('Что за Северный архив?', [SOURCE], 'FAST')
        calls = len(provider.calls)
        answer, verification = pipeline.render('Что за Северный архив?', facts, [SOURCE], 'FAST')
        self.assertEqual(len(provider.calls), calls)
        self.assertEqual(answer, ' '.join(f.statement for f in facts))
        self.assertEqual(verification.checked_answer, answer)
        self.assertEqual(len(verification.claims), len(facts))
        self.assertEqual(pipeline.report['writer_attempts'], 0)

    def test_fast_edit_skips_alternatives_but_balanced_preserves_them(self):
        for model, expected in [('FAST', ['draft', 'edit']),
                                ('BALANCED', ['draft', 'edit', 'enhance', 'merge'])]:
            calls = []
            def stage(name, output):
                def run(state):
                    calls.append(name)
                    return output
                return run
            with patch.object(multi_agent_chain, 'draft_agent', stage('draft', {'draft': 'D', 'results': [SOURCE]})), \
                 patch.object(multi_agent_chain, 'edit_agent', stage('edit', {'edited': 'E'})), \
                 patch.object(multi_agent_chain, 'enhance_agent', stage('enhance', {'enhanced': ['A', 'B']})), \
                 patch.object(multi_agent_chain, 'merge_agent', stage('merge', {'final': 'M'})):
                result = multi_agent_chain.run_multi_agent('question', [], model=model)
            self.assertEqual(calls, expected)
            self.assertEqual(result['final'], 'E' if model == 'FAST' else 'M')

    def test_empty_retrieval_never_runs_editor(self):
        with patch.object(multi_agent_chain, 'draft_agent', return_value={'draft': 'No evidence', 'results': []}), \
             patch.object(multi_agent_chain, 'edit_agent') as edit:
            result = multi_agent_chain.run_multi_agent('question', [], model='BALANCED')
        edit.assert_not_called()
        self.assertEqual(result['final'], 'No evidence')


class WorkspaceScanTests(unittest.TestCase):
    def setUp(self):
        self.env = UserEnvironment()
        self.addCleanup(self.env.close)

    def test_unchanged_scan_does_not_rewrite_file_rows(self):
        fs = self.env.filesystem
        fs.sync_index(self.env.workspace.id)
        with closing(self.env.connection()) as conn:
            conn.execute("UPDATE assistant_workspace_files SET indexed_at='unchanged-marker'")
            conn.commit()
        result = fs.sync_index(self.env.workspace.id)
        self.assertGreater(result.unchanged, 0)
        with closing(self.env.connection()) as conn:
            self.assertTrue(all(r[0] == 'unchanged-marker' for r in conn.execute(
                'SELECT indexed_at FROM assistant_workspace_files')))

    def test_snapshot_is_shared_with_threads_and_next_request_observes_new_files(self):
        async def request():
            token = workspace_scan_cache.set({})
            try:
                with patch.object(self.env.filesystem, '_sync_index', wraps=self.env.filesystem._sync_index) as scan:
                    await asyncio.to_thread(self.env.filesystem.sync_index, self.env.workspace.id)
                    await asyncio.to_thread(self.env.filesystem.sync_index, self.env.workspace.id)
                    self.assertEqual(scan.call_count, 1)
            finally:
                workspace_scan_cache.reset(token)
        asyncio.run(request())
        from pathlib import Path
        root = Path(self.env.workspace.root_path)
        (root / 'new.txt').write_text('NEW', encoding='utf-8')
        result = self.env.filesystem.sync_index(self.env.workspace.id)
        self.assertEqual(result.created, 1)

    def test_long_request_refreshes_its_expired_snapshot(self):
        token = workspace_scan_cache.set({})
        try:
            with patch('workspace.time.monotonic', return_value=0):
                self.env.filesystem.sync_index(self.env.workspace.id)
            with patch('workspace.time.monotonic', return_value=2), \
                 patch.object(self.env.filesystem, '_sync_index', wraps=self.env.filesystem._sync_index) as scan:
                self.env.filesystem.sync_index(self.env.workspace.id)
                self.assertEqual(scan.call_count, 1)
        finally:
            workspace_scan_cache.reset(token)


class GraphReuseTests(unittest.TestCase):
    def test_multi_entity_route_is_bounded_and_preserves_all_seeds(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        import graph_routes
        app = FastAPI()
        app.include_router(graph_routes.router)
        with TestClient(app) as client, patch.object(graph_routes, 'GraphService') as service, \
             patch.object(graph_routes, '_store', return_value=None):
            service.return_value.entities_subgraph.return_value = {'nodes': [], 'edges': []}
            result = client.get('/graph/subgraph?entity_ids=1&entity_ids=2&depth=1')
            self.assertEqual(result.status_code, 200)
            service.return_value.entities_subgraph.assert_called_once_with([1, 2], 1, 20, None)
            for url in ('/graph/subgraph', '/graph/subgraph?entity_ids=no',
                        '/graph/subgraph?entity_ids=1&depth=3',
                        '/graph/subgraph?' + '&'.join('entity_ids=1' for _ in range(21))):
                self.assertEqual(client.get(url).status_code, 422)

    def test_multi_entity_service_includes_isolated_seeds_and_uses_one_traversal(self):
        from graph_service import GraphService
        entities = {i: {'id': i, 'display_name': str(i), 'entity_type': 'PROJECT'} for i in (1, 2)}
        service = GraphService(SimpleNamespace(get_entity=entities.get))
        with patch.object(service, 'traverse', return_value=GraphSearchResult()) as traverse:
            graph = service.entities_subgraph([1, 2, 1, 999])
        traverse.assert_called_once_with([1, 2], 1, 20)
        self.assertEqual(graph['center_ids'], ['entity:1', 'entity:2'])
        self.assertEqual(len(graph['nodes']), 2)

    def test_knowledge_returns_retrieved_graph_without_a_second_search(self):
        row = {'id': 1, 'source_entity_id': 1, 'target_entity_id': 2, 'document_id': 9,
               'document_name': 'Архив.md', 'chunk_id': 'g', 'relation_type': 'ХРАНИТ'}
        graph = GraphSearchResult(entities=[{'id': 1}, {'id': 2}], relations=[row],
                                  document_ids=[9], chunk_ids=['g'])
        store = SimpleNamespace(get=lambda **kw: {'ids': ['g'], 'documents': ['Архив хранит карты.'],
            'metadatas': [{'document_id': 9, 'source': 'Архив.md'}]})
        hybrid = SimpleNamespace(result_k=3, search=lambda *args: [])
        @asynccontextmanager
        async def action(*args, **kw):
            yield SimpleNamespace(output={})
        execution = SimpleNamespace(task=SimpleNamespace(session_id='s'), action=action)
        with patch.object(langchain_utils, 'vectorstore', store), \
             patch.object(langchain_utils, 'current_retriever', object()), \
             patch.object(langchain_utils, 'HybridRetriever', return_value=hybrid), \
             patch.object(langchain_utils, 'create_chat_model', return_value=RunnableLambda(lambda value: 'ANSWER')), \
             patch('graph_rag.GraphService') as service:
            service.return_value.search.return_value = graph
            chain = langchain_utils.get_rag_chain(use_graph=True)
            services = KnowledgeServices(route=lambda *args: RouteDecision(strategy=QueryStrategy.GRAPH_QUERY,
                confidence=1, entities=[]), history=lambda *args: [], general_chat=None, multi_chain=None,
                rag_chain=lambda **kw: chain, document_result=lambda d: {'chunk_id': d.metadata['chunk_id']},
                verify=lambda q, a, *args: (a, None))
            result = asyncio.run(KnowledgeSkill(services).execute({'question': 'Связи архива?',
                '_chat_history': []}, execution))
            self.assertEqual(service.return_value.search.call_count, 1)
        self.assertEqual(result.data['graph']['retrieval']['graph_added_source_count'], 1)

    def test_unselected_graph_edges_are_not_presented_as_answer_sources(self):
        adapter = GraphAugmentedRetriever(None, None, '', [], 1)
        adapter.graph = GraphSearchResult(entities=[{'id': 1}, {'id': 2}, {'id': 3}], relations=[
            {'id': 1, 'source_entity_id': 1, 'target_entity_id': 2, 'document_id': 1, 'chunk_id': 'kept'},
            {'id': 2, 'source_entity_id': 1, 'target_entity_id': 3, 'document_id': 2, 'chunk_id': 'omitted'}])
        adapter.last_results = [SearchResult(1, 'kept', 'text', {})]
        result = adapter.selected_graph()
        self.assertEqual([r['id'] for r in result['relations']], [1])
        self.assertEqual([r['id'] for r in result['entities']], [1, 2])
