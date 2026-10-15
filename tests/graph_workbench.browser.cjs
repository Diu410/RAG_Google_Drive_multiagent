// Production UI, mocked APIs: no access to the user's files or database.
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE_PATH || 'playwright');

async function main() {
  const browser = await chromium.launch({channel: 'msedge', headless: true});
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
    const errors = [], questions = [], graphRequests = [], indexRequests = [], linkRequests = [];
    const savedChats = new Map();
    let indexed = false, linkUnavailable = false;
    page.on('pageerror', error => errors.push(error.message));
    const nodes = [
      {id: 'a', label: 'Резерв.txt', connections: 1, indexed: true},
      {id: 'b', label: 'Архив.md', connections: 1, indexed: true},
      {id: 'c', label: 'Рецепт.txt', connections: 0, indexed: true},
      {id: 'd', label: 'Новая заметка.md', connections: 0, indexed: false},
    ];
    const edges = [{id: 'ab', source: 'a', target: 'b', similarity: .9}];
    await page.route('**/*', async route => {
      const request = route.request(), url = new URL(request.url());
      // Dev has an /api proxy prefix; the production preview uses the root.
      const raw = url.pathname;
      const endpoint = raw.startsWith('/api/') && !['/api/model-status', '/api/chat-model'].includes(raw)
        ? raw.slice(4) : raw;
      if (!/^(\/access-token|\/chats|\/api\/(model-status|chat-model)|\/voice\/status|\/workspaces\/|\/graph\/(map|link)|\/assistant\/stream)/.test(endpoint)) {
        await route.continue(); return;
      }
      let body = {};
      if (endpoint === '/access-token') body = {token: 'browser-fixture-token'};
      else if (endpoint === '/chats/page') body = {items: [], next_cursor: null};
      else if (endpoint === '/chats') body = [];
      else if (endpoint.startsWith('/chats/')) {
        const id = endpoint.slice('/chats/'.length);
        if (request.method() === 'PUT') savedChats.set(id, {...request.postDataJSON(), version: 1});
        body = savedChats.get(id);
      }
      else if (endpoint === '/api/model-status') body = {ready: true, message: 'Готова', default_profile: 'FAST'};
      else if (endpoint === '/voice/status') body = {ready: false, configured: false};
      else if (endpoint === '/workspaces/active') body = {workspace: {id: 'demo', name: 'Тестовая папка', root_path: 'fixture'}};
      else if (endpoint === '/workspaces/demo/files') body = {workspace: {id: 'demo', name: 'Тестовая папка', root_path: 'fixture'}, entries: [], path: '', parent: null};
      else if (endpoint === '/workspaces/demo/index-status') body = {
        readable_documents: 4, indexed_documents: indexed ? 3 : 0, failed_documents: 0, truncated: false, scan_errors: 0,
      };
      else if (endpoint === '/workspaces/demo/vector-sync') {
        indexRequests.push(url); indexed = true;
        body = {indexed: 3, unchanged: 0, failed: 0, unavailable: false};
      }
      else if (endpoint === '/graph/map') {
        graphRequests.push(url);
        body = {nodes, edges, truncated: false, coverage: {documents: 4, indexed_documents: 3}};
      } else if (endpoint === '/graph/link') {
        linkRequests.push(url);
        if (linkUnavailable) {
          await route.fulfill({status: 409, contentType: 'application/json',
            body: JSON.stringify({detail: 'Источник связи изменился. Переиндексируйте документы.'})});
          return;
        }
        body = {similarity: .9, threshold: .82, passage_similarity: .95, sampled: false,
          passages: [
            {document_id: 'a', label: 'Резерв.txt', chunk_id: 'a1', text: 'Копия хранится отдельно.', truncated: false},
            {document_id: 'b', label: 'Архив.md', chunk_id: 'b1', text: 'Восстановление выполняется из архива.', truncated: false},
          ]};
      } else if (endpoint === '/assistant/stream') {
        questions.push(request.postDataJSON());
        body = {answer: 'Копия хранится в архиве.', session_id: 'fixture-session',
          context: 'Исходный файл и автоматически найденный сосед.', strategy: 'WORKSPACE_QA',
          chat_version: 2, results: [], draft: '', edited: '', enhanced: []};
        const input = request.postDataJSON(), chat = savedChats.get(input.chat_id);
        savedChats.set(input.chat_id, {...chat, version: 2, session_id: body.session_id,
          messages: [...chat.messages, {role: 'assistant', content: body.answer, timestamp: new Date().toISOString()}]});
        await route.fulfill({contentType: 'application/x-ndjson', body: JSON.stringify({type:'result',response:body}) + '\n'});
        return;
      }
      await route.fulfill({contentType: 'application/json', body: JSON.stringify(body)});
    });
    await page.goto(process.env.GRAPH_TEST_URL || 'http://127.0.0.1:5178/');
    await page.locator('.workspace-link').click();
    await page.getByText('Проиндексировано 0 из 4 доступных документов.', {exact: true}).waitFor();
    await page.getByRole('button', {name: 'Индексировать папку', exact: true}).click();
    await page.getByText('Проиндексировано 3 из 4 доступных документов.', {exact: true}).waitFor();
    assert.equal(indexRequests.length, 1);
    assert.equal(indexRequests[0].searchParams.get('extract_relations'), 'false');
    await page.locator('.workspace-modal').getByRole('button', {name: 'Закрыть', exact: true}).click();
    await page.getByRole('button', {name: 'Граф знаний', exact: true}).click();
    const map = page.locator('.graph-explorer');
    await map.locator('.graph-note').first().waitFor().catch(async error => {
      console.error('Graph diagnostics:', errors, graphRequests.map(String), await map.innerText());
      throw error;
    });
    assert.equal(await page.getByLabel('Источник графа').inputValue(), 'demo');
    assert.equal(graphRequests.at(-1).searchParams.get('workspace_id'), 'demo');
    assert.equal(await map.locator('textarea').count(), 0);
    assert.equal(await map.getByRole('button', {name: 'Спросить в чате'}).count(), 0);
    assert.equal(await map.getByRole('button', {name: 'Найти цепочку'}).count(), 0);
    assert.equal(await map.locator('.graph-note .graph-dot').count(), 4);
    assert.equal(await map.locator('.graph-link').count(), 1);
    assert.equal(await map.locator('.graph-note.unindexed').count(), 1);
    await page.getByLabel('Поиск документа на карте').fill('Архив');
    assert.equal(await map.locator('.graph-note:not(.dimmed)').count(), 1);
    await page.getByLabel('Поиск документа на карте').fill('');
    await map.locator('.graph-canvas[data-settled="true"]').waitFor();
    const edge = map.getByRole('button', {name: 'Связь: Резерв.txt ↔ Архив.md', exact: true});
    const cameraBeforeLink = await map.locator('.graph-camera').getAttribute('transform');
    await edge.locator('.graph-edge-hit').click();
    const explanation = map.getByLabel('Пояснение связи', {exact: true});
    await explanation.getByText('Восстановление выполняется из архива.', {exact: true}).waitFor();
    assert.equal(linkRequests.at(-1).searchParams.get('workspace_id'), 'demo');
    assert.equal(linkRequests.at(-1).searchParams.get('source_id'), 'a');
    assert.equal(linkRequests.at(-1).searchParams.get('target_id'), 'b');
    assert.match(await explanation.innerText(), /Близость документов: 0\.900/);
    assert.equal(await map.locator('.graph-camera').getAttribute('transform'), cameraBeforeLink);
    await explanation.getByRole('button', {name: 'Закрыть пояснение'}).focus();
    await page.keyboard.press('Escape');
    assert.equal(await explanation.count(), 0);
    linkUnavailable = true;
    await edge.focus(); await page.keyboard.press('Enter');
    await explanation.getByRole('alert').waitFor();
    assert.equal(await explanation.locator('blockquote').count(), 0, 'Unavailable link must not show previous passages');
    await explanation.getByRole('button', {name: 'Закрыть пояснение'}).click();
    linkUnavailable = false;
    const reserve = map.getByRole('button', {name: 'Резерв.txt', exact: true});
    await reserve.hover();
    assert.equal(await map.locator('.graph-note.dimmed').count(), 2);
    assert.equal(await map.getByRole('button', {name: 'Архив.md', exact: true}).evaluate(node => node.classList.contains('dimmed')), false);
    const nodePosition = await reserve.getAttribute('transform');
    const dot = await reserve.locator('.graph-dot').boundingBox();
    await page.mouse.move(dot.x + dot.width / 2, dot.y + dot.height / 2);
    await page.mouse.down(); await page.mouse.move(dot.x + dot.width / 2 + 60, dot.y + dot.height / 2 + 20); await page.mouse.up();
    assert.notEqual(await reserve.getAttribute('transform'), nodePosition, 'Node must move when dragged');
    const oldZoom = await map.locator('.graph-camera').getAttribute('transform');
    await map.getByRole('button', {name: 'Приблизить', exact: true}).click();
    assert.notEqual(await map.locator('.graph-camera').getAttribute('transform'), oldZoom);
    await map.getByRole('button', {name: 'Сбросить вид', exact: true}).click();
    const canvas = map.getByLabel('Карта связей документов'), box = await canvas.boundingBox();
    await page.mouse.move(box.x + 20, box.y + 20);
    await page.mouse.down(); await page.mouse.move(box.x + 60, box.y + 50); await page.mouse.up();
    assert.match(await map.locator('.graph-camera').getAttribute('transform'), /^translate\((?!0 0)/);
    await page.getByLabel('Источник графа').selectOption('');
    await page.waitForFunction(() => document.querySelectorAll('.graph-note').length === 4);
    assert.equal(graphRequests.at(-1).searchParams.get('workspace_id'), null);
    assert.equal(await explanation.count(), 0);
    await edge.focus(); await page.keyboard.press('Enter');
    await explanation.getByText('Копия хранится отдельно.', {exact: true}).waitFor();
    assert.equal(linkRequests.at(-1).searchParams.get('workspace_id'), null);
    await page.setViewportSize({width: 700, height: 900});
    assert.equal(await map.evaluate(element => element.scrollWidth > element.clientWidth + 2), false);
    assert.equal(await explanation.evaluate(element => element.scrollWidth > element.clientWidth + 2), false);
    await map.getByRole('button', {name: 'Закрыть', exact: true}).click();
    await page.getByLabel('Сообщение', {exact: true}).fill('Где хранится копия?');
    await page.getByRole('button', {name: 'Отправить', exact: true}).click();
    await page.getByText('Копия хранится в архиве.', {exact: true}).waitFor();
    assert.equal(questions.length, 1);
    assert.equal(questions[0].use_multi_agent, true, 'Editing runs internally without a manual workshop');
    assert.equal(await page.getByLabel('Режим ответа').count(), 0);
    assert.equal(questions[0].workspace_id, 'demo', 'Viewing library graph must not switch chat scope');
    assert.equal(questions[0].graph_selection, undefined, 'Ordinary chat needs no graph selection');
    await page.setViewportSize({width: 1440, height: 1000});
    await page.getByRole('button', {name: 'Граф знаний', exact: true}).click();
    await map.locator('.graph-note').first().waitFor();
    await edge.focus(); await page.keyboard.press('Enter');
    await explanation.getByText('Копия хранится отдельно.', {exact: true}).waitFor();
    const output = path.resolve('.user-evaluation/semantic-graph-ui.png');
    await fs.mkdir(path.dirname(output), {recursive: true});
    await page.screenshot({path: output});
    assert.deepEqual(errors, []);
    console.log('PASS: workspace indexing, semantic map, clickable link passages and scores, stale-link error, source scopes, keyboard, search, zoom/pan, narrow layout, ordinary chat.');
  } finally {await browser.close();}
}
main().catch(error => {console.error(error); process.exitCode = 1;});
