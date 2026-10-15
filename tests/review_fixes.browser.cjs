// Production UI against an isolated HTTP fixture: no personal API, files or models.
const assert = require('node:assert/strict');
const http = require('node:http');
const fs = require('node:fs/promises');
const path = require('node:path');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE_PATH || 'playwright');

async function main() {
  const dist = path.resolve('frontend/dist');
  const stamp = '2026-10-15T10:00:00Z', requests = [], errors = [], puts = [], cancellations = [];
  const chats = new Map([
    ['a', {id:'a',title:'Открытый диалог',session_id:'s-a',version:1,updated_at:stamp,
      messages:[{role:'user',content:'Первый вопрос',timestamp:stamp},{role:'assistant',content:'Сохранённый ответ',timestamp:stamp}]}],
    ['older', {id:'older',title:'Старый диалог',session_id:'s-older',version:1,updated_at:stamp,
      messages:[{role:'assistant',content:'Старая история',timestamp:stamp}]}],
  ]);
  let cancelledStream = false, validToken = 'fixture-token', bootstraps = 0;
  const server = http.createServer(async (req,res) => {
    try {
      const url = new URL(req.url, 'http://fixture');
      if (url.pathname === '/favicon.ico') {res.writeHead(204);res.end();return;}
      const json = (value,status=200) => {res.writeHead(status, {'content-type':'application/json'});res.end(JSON.stringify(value));};
      const body = async () => {let data='';for await (const part of req) data+=part;return JSON.parse(data||'{}');};
      if (url.pathname === '/access-token') {
        assert.equal(req.headers['x-fortuna-client'], 'web');bootstraps++;json({token:validToken});return;
      }
      const api = /^(\/chats|\/workspaces|\/assistant|\/voice|\/api)/.test(url.pathname);
      if (api && req.headers['x-fortuna-token'] !== validToken) {json({detail:'unauthorized'},401);return;}
      if (url.pathname === '/chats/page') {
        const list = url.searchParams.has('cursor') ? [chats.get('older')] : [chats.get('a')];
        json({items:list.map(chat=>({...chat,messages:[],messages_loaded:false})),next_cursor:url.searchParams.has('cursor')?null:'fixture-cursor'});return;
      }
      if (url.pathname.startsWith('/chats/')) {
        const id = url.pathname.slice(7);
        if (req.method === 'PUT') {const value=await body();puts.push(value);chats.set(id,{...value,version:(chats.get(id)?.version||0)+1});}
        json(chats.get(id));return;
      }
      if (url.pathname === '/workspaces/active') {json({workspace:{id:'demo',name:'Тестовая папка',root_path:'fixture'}});return;}
      if (url.pathname === '/voice/status') {json({ready:false,configured:false});return;}
      if (url.pathname === '/api/model-status') {json({ready:true,message:'Готова',default_profile:'FAST'});return;}
      if (/^\/assistant\/tasks\/.*\/cancel$/.test(url.pathname)) {cancellations.push(url.pathname);json({cancelled:true});return;}
      if (url.pathname === '/assistant/stream') {
        const input = await body();requests.push(input);
        assert.equal(input.use_multi_agent,true);
        assert.equal(input.workspace_id,'demo');
        res.writeHead(200,{'content-type':'application/x-ndjson','cache-control':'no-store'});
        res.write(JSON.stringify({type:'task',task_id:'fixture-task'})+'\n');
        if (input.question === 'Отмени этот ответ') {
          res.on('close',()=>{cancelledStream=true;});return;
        }
        const answer = input.question === 'Потеря ответа' ? 'Восстановленный ответ' : 'Готовый проверенный ответ';
        const chat = chats.get(input.chat_id);
        chat.messages.push({role:'assistant',content:answer,timestamp:stamp,reply_to:input.request_id});
        chat.version++;chat.session_id='s-a';
        if (input.question === 'Потеря ответа') {setTimeout(()=>res.destroy(),60);return;}
        setTimeout(()=>res.end(JSON.stringify({type:'result',response:{answer,session_id:chat.session_id,
          chat_version:chat.version,context:'Источник',draft:'Черновик',edited:answer,enhanced:[]}})+'\n'),80);
        return;
      }
      if (api) {json({detail:'Unknown fixture endpoint'},404);return;}
      const filename = path.resolve(dist, '.' + (url.pathname === '/' ? '/index.html' : decodeURIComponent(url.pathname)));
      if (!filename.startsWith(dist + path.sep)) {res.writeHead(403);res.end();return;}
      const mime = {'.html':'text/html','.js':'text/javascript','.css':'text/css','.woff':'font/woff','.woff2':'font/woff2','.ttf':'font/ttf'};
      res.writeHead(200,{'content-type':mime[path.extname(filename)]||'application/octet-stream'});
      res.end(await fs.readFile(filename));
    } catch (error) {errors.push(String(error));if(!res.headersSent)res.writeHead(500);res.end();}
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const browser = await chromium.launch({channel:'msedge',headless:true});
  try {
    const page = await browser.newPage({viewport:{width:1360,height:960}});
    page.on('pageerror',error=>errors.push(error.message));
    await page.goto('http://127.0.0.1:' + server.address().port);
    await page.getByRole('button',{name:'Открытый диалог',exact:true}).click();
    await page.getByText('Сохранённый ответ',{exact:true}).waitFor();
    assert.equal(await page.getByLabel('Режим ответа').count(),0);
    assert.equal(await page.getByLabel('Текст редактора').count(),0);
    await page.getByRole('button',{name:'Ещё диалоги',exact:true}).click();
    await page.getByRole('button',{name:'Старый диалог',exact:true}).click();
    await page.getByText('Старая история',{exact:true}).waitFor();
    await page.getByRole('button',{name:'Открытый диалог',exact:true}).click();
    const ask = async text => {
      await page.getByLabel('Сообщение',{exact:true}).fill(text);
      try {await page.getByRole('button',{name:'Отправить',exact:true}).click({timeout:5000});}
      catch (error) {console.error('UI fixture diagnostics:',await page.locator('body').innerText(),requests,puts);throw error;}
    };
    await ask('Подготовь результат');
    await page.getByText('Готовый проверенный ответ',{exact:true}).waitFor();
    assert.equal(puts.length,1,'Only the user turn is saved by the frontend');
    assert.equal(requests.length,1);
    validToken = 'rotated-fixture-token';
    await ask('Потеря ответа');
    await page.getByText('Восстановленный ответ',{exact:true}).waitFor();
    assert.equal(requests.length,2,'Disconnected completed response must be recovered without a second generation');
    assert.equal(bootstraps,2,'Dev restart renews access without reloading or losing the chat');
    await ask('Отмени этот ответ');
    await page.getByText('Готовлю ответ…',{exact:true}).first().waitFor();
    await page.getByRole('button',{name:'Остановить ответ',exact:true}).click();
    await page.getByRole('button',{name:'Отправить',exact:true}).waitFor({state:'visible'});
    for (let attempt=0;attempt<100 && (!cancelledStream || !cancellations.length);attempt++) await new Promise(resolve=>setTimeout(resolve,25));
    assert.equal(cancelledStream,true,'Stopping ordinary chat closes its HTTP stream');
    assert.equal(cancellations.length,1,'Stop cancels the running backend task');
    await page.reload();
    await page.getByText('Восстановленный ответ',{exact:true}).waitFor();
    assert.equal(await page.getByText('Готовый проверенный ответ',{exact:true}).count(),1);
    assert.deepEqual(errors,[]);
    console.log('PASS: token initialization, lazy chat history, pagination, internal editing, server persistence, disconnect recovery, ordinary response cancellation and reload.');
  } finally {
    await browser.close();server.closeAllConnections();await new Promise(resolve=>server.close(resolve));
  }
}
main().catch(error=>{console.error(error);process.exitCode=1;});
