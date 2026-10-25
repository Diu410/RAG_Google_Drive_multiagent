"""Source selection, hybrid retrieval and revision-bound evidence on real files."""
import asyncio
from contextlib import closing
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from support import UserEnvironment
from assistant_models import AssistantError
from assistant_repository import AssistantRepository
from directory_summary import DirectorySummary
from file_revision import file_revision
from langchain_core.documents import Document
from pydantic import ValidationError
from pydantic_models import QueryInput
from workspace_document_reader import WorkspaceDocumentReader
from workspace_context import file_passages
from workspace_retrieval import fuse_files
from workspace_source_scope import WorkspaceSourceScope
from workspace_vector_index import WorkspaceVectorIndex
from local_tools_skill import LocalToolsSkill
from workspace_documents import WorkspaceDocuments
from workspace_repository_tools import RepositoryTools


class SourceScopeTests(unittest.TestCase):
    def test_invalid_combinations_and_paths_are_not_silently_accepted(self):
        invalid = [dict(kind='workspace', files=['a.md']), dict(kind='directory'), dict(kind='files'),
                   dict(kind='files', files=['a.md', 'a.md']), dict(kind='files', files=['a.md'], extra=True)]
        for path in ('../a.md', '/a.md', 'C:/a.md', 'a\\b.md', 'a//b.md', './a.md', 'a/../b.md', 'a\x00.md'):
            invalid.append(dict(kind='files', files=[path]))
        for data in invalid:
            with self.subTest(data=data), self.assertRaises(ValidationError):
                WorkspaceSourceScope.model_validate(data)

    def test_directory_matches_components_not_prefixes(self):
        scope = WorkspaceSourceScope(kind='directory', directory='Sources/One')
        self.assertTrue(scope.contains('Sources/One/child/a.md'))
        self.assertFalse(scope.contains('Sources/OneMore/a.md'))
        self.assertFalse(scope.contains('Sources/Other/a.md'))

    def test_request_requires_workspace_and_rejects_ambiguous_graph_selection(self):
        with self.assertRaises(ValidationError):
            QueryInput(question='Question', source_scope={'kind':'workspace'})
        with self.assertRaises(ValidationError):
            QueryInput(question='Question', workspace_id='w', source_scope={'kind':'workspace'},
                       graph_selection={'workspace_id':'w','relation_ids':[1]})

    def test_retry_fingerprint_includes_scope_and_destination_not_file_order(self):
        def request(**kwargs):
            return QueryInput(question='Question', workspace_id='w', **kwargs)
        first = request(source_scope={'kind':'files','files':['a.md','b.md']})
        reordered = request(source_scope={'kind':'files','files':['b.md','a.md']})
        fingerprint = AssistantRepository._request_fingerprint
        self.assertEqual(fingerprint(first), fingerprint(reordered))
        self.assertNotEqual(fingerprint(first), fingerprint(request(source_scope={'kind':'workspace'})))
        self.assertNotEqual(fingerprint(request(workspace_directory='a')), fingerprint(request(workspace_directory='b')))
        self.assertEqual(json.loads(fingerprint(request())),
                         {'question':'Question','workspace_id':'w','graph_selection':None})


class WorkspaceSourceTests(unittest.TestCase):
    def setUp(self):
        self.env = UserEnvironment()
        self.addCleanup(self.env.close)
        self.id = self.env.workspace.id
        (self.env.root / 'Sources/Sub').mkdir(parents=True)
        self.write('Sources/Chosen.md', 'Каппа: код ALPHA.\nВторая строка: условие сохранено.\n')
        self.write('Sources/Sub/Other.md', 'Каппа: код BETA.\n')
        self.write('Outside.md', 'Каппа: код OUTSIDE.\n')

    def write(self, path, text):
        (self.env.root / path).write_text(text, encoding='utf-8', newline='')

    def search(self, scope, question='Какой код каппа?', **kwargs):
        return self.env.skill.relevant_context(question, self.id, source_scope=scope, **kwargs)

    def indexed_vector(self, path):
        self.env.filesystem.sync_index(self.id)
        record = next(r for r in self.env.repository.indexed_files(self.id) if r.relative_path == path)
        self.env.repository.update_vector_state(self.id, path, content_hash=record.content_hash, status='INDEXED')
        return record

    def test_selected_file_does_not_mix_lexical_neighbors(self):
        rows = self.search({'kind':'files','files':['Sources/Chosen.md']})
        self.assertTrue(rows)
        self.assertEqual({r['path'] for r in rows}, {'Sources/Chosen.md'})
        self.assertNotIn('OUTSIDE', str(rows))

    def test_subfolder_includes_nested_files_but_not_sibling_prefix(self):
        (self.env.root / 'SourcesMore').mkdir()
        self.write('SourcesMore/False.md', 'Каппа: код WRONG.')
        rows = self.search({'kind':'directory','directory':'Sources'})
        self.assertEqual({r['path'] for r in rows}, {'Sources/Chosen.md','Sources/Sub/Other.md'})

    def test_explicit_filename_outside_selection_is_rejected_before_read(self):
        reader = Mock(wraps=self.env.filesystem.read_text_file)
        with patch.object(self.env.filesystem, 'read_text_file', reader):
            with self.assertRaisesRegex(AssistantError, 'вне выбранной области'):
                self.search({'kind':'files','files':['Sources/Chosen.md']}, 'Какой код в Outside.md?')
        self.assertFalse(any(call.args[1] == 'Outside.md' for call in reader.call_args_list))

    def test_active_file_does_not_widen_selected_sources(self):
        with self.assertRaises(AssistantError):
            self.search({'kind':'files','files':['Sources/Chosen.md']},
                        'Какой код в этом файле?', active_file='Outside.md')

    def test_ignored_selected_file_is_not_reported_as_no_information(self):
        self.write('.gitignore', 'Excluded.md\n')
        self.write('Excluded.md', 'Каппа: код IGNORED.')
        with self.assertRaisesRegex(AssistantError, 'исключена из индекса'):
            self.search({'kind':'files','files':['Excluded.md']})

    def test_missing_selected_file_has_explicit_unavailable_status(self):
        with self.assertRaisesRegex(AssistantError, 'больше недоступен') as caught:
            self.search({'kind':'files','files':['Missing.md']})
        self.assertEqual(caught.exception.status_code, 404)

    def test_semantic_search_runs_even_with_three_lexical_results(self):
        for index in range(3):
            self.write(f'Distractor{index}.md', 'Каппа упомянута, но искомого значения здесь нет.')
        text = 'Нейтральное вступление.\n' * 300 + 'Устройство маркируется значением OMEGA.'
        self.write('Meaning.md', text)
        self.indexed_vector('Meaning.md')
        search = Mock(return_value=[{'path':'Meaning.md','content':'Устройство маркируется значением OMEGA.',
                                    'found_by':['workspace_semantic']}])
        self.env.skill.vector_indexer = SimpleNamespace(graph_context=lambda *args: None,
            search=search, expand_context=lambda *args, **kwargs: [])
        rows = self.env.skill.relevant_context('Какова маркировка каппа?', self.id, max_characters=1500)
        search.assert_called_once()
        self.assertIn('Meaning.md', {r['path'] for r in rows})
        self.assertIn('OMEGA', str(rows))
        self.assertLessEqual(sum(len(r['content']) for r in rows), 1500)

    def test_hybrid_and_semantic_anchors_obey_small_context_budgets(self):
        content = 'Каппа — вводная строка.\n' * 100 + 'Буквальный смысловой якорь.'
        for limit in (1,100,239,240,719,720):
            with self.subTest(limit=limit):
                rows = file_passages('Что такое каппа?', 'Long.md', content, limit,
                                     preferred_text='Буквальный смысловой якорь.')
                self.assertLessEqual(sum(len(r['content']) for r in rows), limit)
        rows = self.search({'kind':'workspace'}, max_characters=1)
        self.assertLessEqual(sum(len(r['content']) for r in rows), 1)

    def test_semantic_neighbor_outside_scope_and_stale_vectors_are_rejected(self):
        self.indexed_vector('Outside.md')
        self.indexed_vector('Sources/Chosen.md')
        self.write('Sources/Chosen.md', 'Каппа: обновлённый код CURRENT.')
        self.env.skill.vector_indexer = SimpleNamespace(search=Mock(return_value=[
            {'path':'Outside.md','content':'Каппа: код OUTSIDE.'},
            {'path':'Sources/Chosen.md','content':'Каппа: код STALE_VECTOR.'}]))
        rows = self.search({'kind':'directory','directory':'Sources'})
        self.assertNotIn('OUTSIDE', str(rows))
        self.assertNotIn('STALE_VECTOR', str(rows))
        self.assertIn('CURRENT', str(rows))

    def test_scope_filter_is_applied_before_vector_selection(self):
        self.indexed_vector('Sources/Chosen.md')
        self.indexed_vector('Outside.md')
        store = SimpleNamespace(similarity_search=Mock(return_value=[]))
        index = WorkspaceVectorIndex(self.env.filesystem, lambda: store)
        index.search(self.id, 'Meaning', source_scope=WorkspaceSourceScope(kind='files',files=['Sources/Chosen.md']))
        self.assertEqual(store.similarity_search.call_args.kwargs['filter'],
            {'$and':[{'workspace_id':self.id},{'relative_path':{'$in':['Sources/Chosen.md']}}]})

    def test_excerpts_keep_literal_ranges_line_numbers_and_revision(self):
        rows = self.search({'kind':'files','files':['Sources/Chosen.md']})
        text = (self.env.root / 'Sources/Chosen.md').read_text(encoding='utf-8')
        for row in rows:
            start, end = row['source_range']
            self.assertEqual(text[start:end], row['content'])
            self.assertEqual(row['revision'], file_revision(self.env.root / row['path']))
            self.assertEqual(row['locations'][0]['line_start'], 1)
            excerpt = self.env.filesystem.source_excerpt(self.id, row['path'], row['revision'], start, end)
            self.assertEqual(excerpt['text'], row['content'])

    def test_external_edit_rejects_old_source_without_replacing_snapshot(self):
        row = self.search({'kind':'files','files':['Sources/Chosen.md']})[0]
        path = self.env.root / row['path']
        original_stat = path.stat()
        self.write(row['path'], 'Каппа: код ALTERED.')
        os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        with self.assertRaisesRegex(AssistantError, 'предыдущей версии') as caught:
            self.env.filesystem.source_excerpt(self.id, row['path'], row['revision'], *row['source_range'])
        self.assertEqual(caught.exception.status_code, 409)
        self.assertIn('ALPHA', row['content'])

    def test_changed_revision_cannot_be_attached_to_an_old_matching_excerpt(self):
        row = self.search({'kind':'files','files':['Sources/Chosen.md']})[0]
        self.write(row['path'], row['content'] + '\nДобавлено новое условие.')
        with self.assertRaisesRegex(AssistantError, 'изменился во время поиска'):
            self.env.filesystem.locate_passages(self.id, [row])

    def test_summary_inventory_uses_selection_and_detects_changes_within_it(self):
        summary = DirectorySummary(self.env.filesystem)
        scope = WorkspaceSourceScope(kind='files', files=['Sources/Chosen.md'])
        inventory = summary.inventory(self.id, '', source_scope=scope)
        self.assertEqual([r['path'] for r in inventory['sources']], ['Sources/Chosen.md'])
        self.write('Outside.md', 'Внешняя правка не относится к выбранным источникам.')
        self.assertTrue(summary.unchanged(self.id, inventory, []))
        self.write('Sources/Chosen.md', 'Изменён выбранный источник.')
        self.assertFalse(summary.unchanged(self.id, inventory, []))

    def test_named_summary_folder_narrows_workspace_scope_but_cannot_escape_selected_folder(self):
        summary = DirectorySummary(self.env.filesystem)
        inventory = summary.inventory(self.id, 'Sources/Sub', source_scope=WorkspaceSourceScope())
        self.assertEqual([r['path'] for r in inventory['sources']], ['Sources/Sub/Other.md'])
        with self.assertRaisesRegex(AssistantError, 'вне выбранной области'):
            summary.inventory(self.id, 'Архив', source_scope=WorkspaceSourceScope(kind='directory',directory='Sources'))

    def test_request_scope_reaches_search_and_is_saved_with_source_snapshots(self):
        with closing(self.env.connection()) as conn, conn:
            conn.execute('CREATE TABLE application_logs(id INTEGER PRIMARY KEY AUTOINCREMENT,session_id TEXT,'
                         'user_query TEXT,gpt_response TEXT,model TEXT,knowledge_epoch INTEGER)')
            conn.execute('CREATE TABLE application_state(key TEXT PRIMARY KEY,value TEXT)')
        self.env.core.response_saver = self.env.assistant.save_response
        generated = SimpleNamespace(text='Каппа: код ALPHA.', metrics=SimpleNamespace(finish_reason='stop'))
        scope = {'kind':'files','files':['Sources/Chosen.md']}
        request = QueryInput(question='Какой код каппа?', workspace_id=self.id, model='FAST',
                             source_scope=scope, workspace_directory='Sources',
                             chat_id='sources-chat', request_id='request-one')
        with patch('model_runtime.get_llm_provider', return_value=SimpleNamespace(generate=lambda *a, **k: generated)):
            response = asyncio.run(self.env.core.process_input(request))[0]
        self.assertTrue(response.results)
        self.assertEqual({r.metadata['source'] for r in response.results}, {'Sources/Chosen.md'})
        with closing(self.env.connection()) as conn:
            messages = json.loads(conn.execute('SELECT data FROM workspace_chats WHERE id=?', ('sources-chat',)).fetchone()[0])['messages']
        self.assertEqual(messages[0]['source_scope']['files'], scope['files'])
        self.assertEqual(messages[0]['workspace_directory'], 'Sources')
        self.assertEqual(messages[-1]['results'][0]['metadata']['revision'], response.results[0].metadata['revision'])
        changed = request.model_copy(update={'session_id':response.session_id,
            'source_scope':WorkspaceSourceScope(kind='files',files=['Outside.md'])})
        with self.assertRaisesRegex(AssistantError, 'Request id'):
            asyncio.run(self.env.core.process_input(changed))

    def confirm_creation(self, before, after):
        tools = LocalToolsSkill(WorkspaceDocuments(self.env.filesystem), RepositoryTools(self.env.filesystem), self.env.skill)
        self.env.core.local_tools_skill = tools
        self.env.core.skills.register(tools)
        first = QueryInput(question='Предпросмотр: Создай файл Confirm.md с текстом «Готовый текст.»',
                           workspace_id=self.id, model='FAST', source_scope=before)
        response = asyncio.run(self.env.core.process_input(first))[0]
        self.assertIn('Ответьте «да»', response.answer)
        self.assertFalse((self.env.root / 'Confirm.md').exists())
        confirmation = QueryInput(question='да', session_id=response.session_id, workspace_id=self.id,
                                  model='FAST', source_scope=after)
        return asyncio.run(self.env.core.process_input(confirmation))[0]

    def test_confirmation_keeps_json_selection_across_turns_and_ignores_file_order(self):
        before = {'kind':'files','files':['Sources/Chosen.md','Outside.md']}
        after = {'kind':'files','files':['Outside.md','Sources/Chosen.md']}
        response = self.confirm_creation(before, after)
        self.assertIn('Сохранён файл', response.answer)
        self.assertEqual((self.env.root / 'Confirm.md').read_text(encoding='utf-8'), 'Готовый текст.')

    def test_changed_scope_cannot_confirm_old_file_effect(self):
        self.confirm_creation({'kind':'files','files':['Sources/Chosen.md']}, {'kind':'files','files':['Outside.md']})
        self.assertFalse((self.env.root / 'Confirm.md').exists())


class StructuredExtractionTests(unittest.TestCase):
    def setUp(self):
        self.env = UserEnvironment()
        self.addCleanup(self.env.close)

    def document(self, suffix, documents):
        path = self.env.root / ('Structured' + suffix)
        path.write_bytes(b'controlled extraction fixture')
        with patch.object(WorkspaceDocumentReader, '_structured_documents', return_value=documents):
            result = WorkspaceDocumentReader().read_document(path)
        return path, result

    def test_pdf_keeps_pages_and_literal_offsets_and_distinguishes_repeated_blocks(self):
        path, result = self.document('.pdf', [Document(page_content='Повтор.', metadata={'page':0}),
            Document(page_content='Повтор.', metadata={'page':4,'extraction_method':'ocr'})])
        self.assertEqual(result.text, 'Повтор.\n\nПовтор.')
        self.assertEqual([b.page for b in result.blocks], [1,5])
        self.assertNotEqual(result.blocks[0].id, result.blocks[1].id)
        for block in result.blocks:
            self.assertEqual(result.text[block.start:block.end], 'Повтор.')
        self.assertEqual(result.locations(result.blocks[1].start, result.blocks[1].end)[0]['extraction_method'], 'ocr')
        self.assertEqual(result.revision, file_revision(path))

    def test_docx_retains_section_without_inventing_page(self):
        _, result = self.document('.docx', [Document(page_content='Условия.', metadata={'section':'Heading1','page':7})])
        location = result.locations(0,len(result.text))[0]
        self.assertEqual(location['section'], 'Heading1')
        self.assertNotIn('page', location)

    def test_real_two_page_pdf_excerpt_resolves_to_original_page(self):
        import pymupdf
        path = self.env.root / 'Pages.pdf'
        with pymupdf.open() as pdf:
            pdf.new_page().insert_text((72,72), 'Delivery code PAGE_ONE is on the first page.')
            pdf.new_page().insert_text((72,72), 'Delivery code PAGE_TWO is on the second page.')
            pdf.save(path)
        document = self.env.filesystem.document_reader.read_document(path)
        start = document.text.index('PAGE_TWO')
        excerpt = self.env.filesystem.source_excerpt(self.env.workspace.id, path.name,
            document.revision, start, start + len('PAGE_TWO'))
        self.assertEqual(excerpt['text'], 'PAGE_TWO')
        self.assertEqual([r['page'] for r in excerpt['locations']], [2])

    def test_legacy_text_read_is_exactly_the_structured_text_and_cache_keeps_identity(self):
        path = self.env.root / 'Lines.txt'
        path.write_bytes('Первая\r\nВторая\r\n'.encode('utf-8-sig'))
        reader = WorkspaceDocumentReader()
        document = reader.read_document(path)
        self.assertEqual(reader.read(path), document.text)
        self.assertIs(reader.read_document(path), document)
        self.assertEqual(document.locations(8,14)[0]['line_start'], 2)

    def test_fusion_uses_ranks_not_incomparable_raw_scores_and_deduplicates_channels(self):
        rows = fuse_files([{'path':'a','content':'A','score':999999}],
                          [{'path':'b','content':'B','score':-500}, {'path':'b','content':'B2'},
                           {'path':'a','content':'A2'}])
        self.assertEqual(rows[0]['path'], 'a')
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['found_by'], ['workspace_lexical','workspace_semantic'])
        self.assertEqual(rows[1]['preferred_text'], 'B')
