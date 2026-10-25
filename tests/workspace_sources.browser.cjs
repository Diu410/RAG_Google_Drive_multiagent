// Source selection and revision-bound evidence against real isolated API/files.
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const net = require('node:net');
const {spawn} = require('node:child_process');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE_PATH || 'playwright');

async function main() {
  const run = path.resolve('.user-evaluation/workspace-sources-browser-' + Date.now());
  await fs.mkdir(run, {recursive:true});
  const socket = net.createServer();
  await new Promise(resolve => socket.listen(0, '127.0.0.1', resolve));
  const port = socket.address().port;
  await new Promise(resolve => socket.close(resolve));
  const server = spawn(path.resolve('.venv/Scripts/python.exe'),
    ['tests/serve_local_tools_fixture.py','--port',String(port),'--run',run,'--workspace-sources','--workspace-index'],
    {stdio:['ignore','pipe','pipe'],windowsHide:true});
  const exited = new Promise(resolve => server.once('exit', resolve));
  let logs = '', browser, page;
  for (const stream of [server.stdout, server.stderr]) stream.on('data', data => {logs = (logs + data).slice(-64000);});
  const base = 'http://127.0.0.1:' + port;
  try {
    let ready = false;
    for (let attempt = 0; attempt < 120; attempt++) {
      if (server.exitCode !== null) throw Error('Fixture stopped: ' + logs);
      try {if ((await fetch(base + '/api/health')).ok) {ready = true; break;}} catch {}
      await new Promise(resolve => setTimeout(resolve, 250));
    }
    assert(ready);
    browser = await chromium.launch({channel:'msedge',headless:true});
    page = await browser.newPage({viewport:{width:1360,height:1000}});
    const errors = [], requests = [], indexRequests = [];
    page.on('pageerror', error => {errors.push(error.message);logs += '\nPAGE ERROR: ' + error.message;});
    page.on('console', message => {if (message.type() === 'error') logs += '\nCONSOLE: ' + message.text();});
    page.on('request', request => {
      if (request.url().endsWith('/assistant/stream')) requests.push(request.postDataJSON());
      if (new URL(request.url()).pathname.endsWith('/vector-sync')) indexRequests.push(new URL(request.url()));
    });
    await page.goto(base);
    await page.locator('.workspace-link.active').waitFor();
    assert.equal(await page.locator('.source-scope-bar').count(), 0);
    await page.locator('.workspace-link.active').click();
    const modal = page.locator('.workspace-modal');
    await modal.locator('.workspace-entry').getByText('Заметки.md',{exact:true}).click();
    await modal.getByRole('button',{name:'Только этот файл',exact:true}).click();
    await modal.getByRole('button',{name:'Исключить Заметки.md',exact:true}).waitFor();
    await modal.getByRole('button',{name:/К списку/}).click();
    await modal.locator('.workspace-entry').getByText('Документы прогона',{exact:true}).click();
    // Navigation never changes the selection.
    assert((await modal.locator('.source-scope-picker').innerText()).includes('Заметки.md'));
    await modal.getByRole('button',{name:'Использовать эту подпапку',exact:true}).click();
    await modal.locator('.workspace-entry').getByText('Тема 0.md',{exact:true}).click();
    await modal.getByRole('button',{name:'Только этот файл',exact:true}).click();
    await modal.getByRole('button',{name:/К списку/}).click();
    await modal.locator('.workspace-entry').getByText('Тема 1.md',{exact:true}).click();
    await modal.getByRole('button',{name:'Добавить файл к источникам',exact:true}).click();
    assert.equal(await modal.locator('.source-scope-picker li').count(), 2);
    await modal.getByRole('button',{name:'Исключить Документы прогона/Тема 1.md',exact:true}).click();
    assert.equal(await modal.locator('.source-scope-picker li').count(), 1);
    assert(await modal.getByRole('button',{name:'Исключить Документы прогона/Тема 0.md',exact:true}).isDisabled());
    await modal.getByRole('button',{name:/К списку/}).click();
    await modal.getByRole('button',{name:'Назад',exact:true}).click();
    await modal.locator('.workspace-entry').getByText('Заметки.md',{exact:true}).click();
    await modal.getByRole('button',{name:'Только этот файл',exact:true}).click();
    await modal.getByRole('button',{name:'Закрыть',exact:true}).click();
    await page.getByLabel('Сообщение',{exact:true}).fill('Какой код указан в документе?');
    await page.getByRole('button',{name:'Отправить',exact:true}).click();
    const message = page.locator('article.message.assistant').last();
    await message.getByText('Код: 4817.',{exact:true}).waitFor({timeout:30000});
    assert.deepEqual(requests.at(-1).source_scope, {kind:'files',files:['Заметки.md']});
    const used = message.locator('.workspace-sources');
    assert.equal(await message.locator('summary').filter({hasText:/^Источники ответа/}).count(), 1);
    await used.locator(':scope > summary').click();
    await used.locator('.workspace-source > summary').first().click();
    assert((await used.locator('pre').first().innerText()).includes('4817'));
    await used.getByRole('button',{name:'Проверить актуальность источника',exact:true}).first().click();
    await used.getByText(/Версия источника не изменилась/).waitFor();
    await fs.writeFile(path.join(run,'workspace/Заметки.md'), 'Код: 6248.\n', 'utf8');
    await used.getByRole('button',{name:'Проверить актуальность источника',exact:true}).first().click();
    await used.getByText(/Источник изменился/).waitFor();
    assert((await used.locator('pre').first().innerText()).includes('4817'));
    assert(!(await used.locator('pre').first().innerText()).includes('6248'));
    const selectedRequest = requests.at(-1);
    assert.equal((await fetch(base+'/workspaces/'+selectedRequest.workspace_id+'/source?path=x&revision=x&start=0&end=1')).status,401);
    await page.reload();
    await page.locator('article.message.assistant').last().locator('.workspace-sources > summary').waitFor();
    await page.locator('.workspace-link.active').click();
    await modal.getByRole('button',{name:'Исключить Заметки.md',exact:true}).waitFor();
    // Index maintenance uses real filesystem revisions and persistent metadata.
    const index = modal.getByRole('region',{name:'Поисковый индекс папки',exact:true});
    await index.getByRole('button',{name:'Проверить изменения',exact:true}).waitFor();
    await modal.getByRole('button',{name:'Индексировать папку',exact:true}).click();
    await modal.getByText(/Индексация завершена:.*Ошибок: 1/).waitFor();
    // Embedding failure does not discard successfully extracted text.
    await index.getByText(/^Текстовый индекс: (\d+) из \1\.$/).waitFor();
    await index.getByRole('button',{name:'Повторить ошибки',exact:true}).waitFor();
    await index.getByText('Ошибки последнего запуска · 1',{exact:true}).click();
    await index.getByText(/Не удалось построить или сохранить поисковый индекс/).waitFor();
    await fs.writeFile(path.join(run,'workspace/Новый документ.md'), 'Новый материал.\n', 'utf8');
    await fs.writeFile(path.join(run,'workspace/Заметки.md'), 'Ещё одна редакция: код 7359.\n', 'utf8');
    await index.getByRole('button',{name:'Проверить изменения',exact:true}).click();
    await index.getByText(/Новых: 1\. Изменённых: 1\. С ошибками: 1\./).waitFor();
    await index.getByRole('button',{name:'Повторить ошибки',exact:true}).click();
    await modal.getByText(/Повторная обработка завершена: обновлено 1/).waitFor();
    await index.getByText(/Новых: 1\. Изменённых: 1\./).waitFor();
    assert.equal(indexRequests.at(-1).searchParams.get('mode'), 'retry_failed');
    assert.equal(indexRequests.at(-1).searchParams.get('extract_relations'), 'false');
    await modal.getByRole('button',{name:'Индексировать папку',exact:true}).click();
    await modal.getByText(/Индексация завершена: обновлено 2/).waitFor();
    await index.getByText(/^Проиндексировано (\d+) из \1 доступных документов\.$/).waitFor();
    await index.getByText(/^Текстовый индекс: (\d+) из \1\.$/).waitFor();
    await modal.getByRole('button',{name:'Закрыть',exact:true}).click();
    // Old chat records and library evidence keep one source block as well.
    const compatibility = {id:'sources-compatibility',title:'Совместимость источников',updated_at:new Date().toISOString(),messages:[
      {role:'assistant',content:'Старый ответ.',timestamp:new Date().toISOString(),context:'LEGACY_SOURCE_TEXT'},
      {role:'assistant',content:'Ответ библиотеки.',timestamp:new Date().toISOString(),context:'FLATTENED_DUPLICATE_NOT_DISPLAYED',
        results:[0,1].map(() => ({document_id:7,chunk_id:'library-chunk',text:'LIBRARY_SOURCE_TEXT',
          metadata:{source:'Библиотечный документ.pdf'},score:1,found_by:['lexical']}))},
      {role:'assistant',content:'Ответ без документов.',timestamp:new Date().toISOString()},
    ]};
    assert(await page.evaluate(async chat => {
      const response = await fetch('/access-token',{headers:{'X-Fortuna-Client':'web'}});
      const {token} = await response.json();
      return (await fetch('/chats/'+chat.id,{method:'PUT',headers:{'X-Fortuna-Token':token,'Content-Type':'application/json'},body:JSON.stringify(chat)})).ok;
    }, compatibility));
    await page.reload();
    await page.getByRole('button',{name:'Совместимость источников',exact:true}).click();
    await page.getByText('Ответ без документов.',{exact:true}).waitFor();
    const compatibilityMessages = page.locator('article.message.assistant');
    assert.equal(await compatibilityMessages.nth(0).locator('summary').filter({hasText:/^Источники ответа/}).count(), 1);
    assert.equal(await compatibilityMessages.nth(1).locator('summary').filter({hasText:/^Источники ответа/}).count(), 1);
    assert.equal(await compatibilityMessages.nth(2).locator('.workspace-sources').count(), 0);
    await compatibilityMessages.nth(0).locator('.workspace-sources > summary').click();
    await page.getByText('LEGACY_SOURCE_TEXT',{exact:true}).waitFor();
    await compatibilityMessages.nth(1).locator('.workspace-sources > summary').click();
    assert.equal(await compatibilityMessages.nth(1).locator('.workspace-source').count(), 1);
    await compatibilityMessages.nth(1).locator('.workspace-source > summary').click();
    await page.getByText('LIBRARY_SOURCE_TEXT',{exact:true}).waitFor();
    assert.equal(await page.getByText('FLATTENED_DUPLICATE_NOT_DISPLAYED',{exact:true}).count(), 0);
    await page.screenshot({path:path.join(run,'sources.png'),fullPage:true});
    assert.deepEqual(errors, []);
    console.log('PASS: sidebar-only workspace, single evidence block, scoped sources, revision conflict, reload, index freshness, incremental update, failed-only retry and auth. '+run);
  } catch (error) {
    if (page) {
      logs += '\nBODY: ' + await page.locator('body').innerText().catch(() => 'unavailable');
      await page.screenshot({path:path.join(run,'failure.png'),fullPage:true}).catch(() => {});
    }
    throw error;
  } finally {
    if (browser) await browser.close();
    server.kill();
    await exited;
    await fs.writeFile(path.join(run,'server.log'),logs);
  }
}
main().catch(error => {console.error(error);process.exitCode = 1;});
