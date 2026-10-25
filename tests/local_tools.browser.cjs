// Production chat, real isolated API/SQLite/files. No local model required.
const assert=require('node:assert/strict');
const fs=require('node:fs/promises');
const path=require('node:path');
const net=require('node:net');
const {spawn}=require('node:child_process');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE_PATH||'playwright');

async function main(){
  const run=path.resolve('.user-evaluation/agent-tools-browser-'+Date.now());
  await fs.mkdir(run,{recursive:true});
  const socket=net.createServer();await new Promise(resolve=>socket.listen(0,'127.0.0.1',resolve));
  const port=socket.address().port;await new Promise(resolve=>socket.close(resolve));
  const server=spawn(path.resolve('.venv/Scripts/python.exe'),[
    'tests/serve_local_tools_fixture.py','--port',String(port),'--run',run,'--repository-agent',
  ],{stdio:['ignore','pipe','pipe'],windowsHide:true});
  const exited=new Promise(resolve=>server.once('exit',resolve));
  let logs='',browser;for(const stream of [server.stdout,server.stderr])stream.on('data',d=>{logs=(logs+d).slice(-32000);});
  const base='http://127.0.0.1:'+port;
  try{
    let ready=false;
    for(let attempt=0;attempt<120;attempt++){
      if(server.exitCode!==null)throw Error('Fixture stopped: '+logs);
      try{if((await fetch(base+'/api/health')).ok){ready=true;break;}}catch{}
      await new Promise(resolve=>setTimeout(resolve,250));
    }
    assert(ready,'Isolated API must start');
    const {token}=await(await fetch(base+'/access-token',{headers:{'X-Fortuna-Client':'web'}})).json();
    const auth={'X-Fortuna-Token':token,'Content-Type':'application/json'};
    const {workspace}=await(await fetch(base+'/workspaces/active',{headers:auth})).json();
    const toolbase=base+'/workspaces/'+workspace.id;
    assert.equal((await fetch(toolbase+'/repository')).status,401);
    const overview=await(await fetch(toolbase+'/repository',{headers:auth})).json();
    assert(!overview.paths.some(p=>p.includes('ignored/')));
    browser=await chromium.launch({channel:'msedge',headless:true});
    const page=await browser.newPage({viewport:{width:1360,height:1000}});
    const errors=[];page.on('pageerror',e=>errors.push(e.message));
    await page.goto(base);
    await page.getByRole('button',{name:/workspace/}).click();
    await page.getByRole('heading',{name:'Workspace',exact:true}).waitFor();
    assert.equal(await page.getByLabel('Инструменты папки').count(),0);
    assert.equal(await page.getByRole('button',{name:'Создать документ',exact:true}).count(),0);
    await page.getByText(/Попросите в чате создать документ по теме/).waitFor();
    await page.locator('.workspace-modal').getByRole('button',{name:'Закрыть',exact:true}).click();
    async function send(text,reply){
      const previous=await page.locator('article.message.assistant').count();
      await page.getByLabel('Сообщение',{exact:true}).fill(text);
      await page.getByRole('button',{name:'Отправить',exact:true}).click();
      const message=page.locator('article.message.assistant').nth(previous);
      await message.getByText(reply).last().waitFor({timeout:30000});
      await page.getByRole('button',{name:'Отправить',exact:true}).waitFor({state:'visible',timeout:30000});
    }
    await send('Прочитай История прогона.md',/MEMORY_FILLER/);
    await send('Найди текст «4817»',/lib.rs:1/);
    await send('Создай файл Новый.md с текстом «Новый документ Фортуны.»',/Сохранён файл.*Новый.md/);
    assert.equal(await fs.readFile(path.join(run,'workspace/Новый.md'),'utf8'),'Новый документ Фортуны.');
    await send('открой его',/Файл.*Новый\.md.*открыт/);
    await send('открой Новый.md',/Файл.*Новый\.md.*открыт/);
    await send('открой Новый md',/Файл.*Новый\.md.*открыт/);
    await send('не то открыл',/Был открыт.*Новый.md.*Это не тот файл/);
    await send('ты чё открыл',/Последним был открыт файл.*Новый.md/);
    assert.deepEqual(JSON.parse(await fs.readFile(path.join(run,'opened.json'),'utf8')),['Новый.md','Новый.md','Новый.md']);
    await send('Прочитай файл Заметки.md',/Нагрузка: 27 кг/);
    await send('я тебя попросил открыть созданный тобой документы',/Файл.*Новый\.md.*открыт/);
    assert.deepEqual(JSON.parse(await fs.readFile(path.join(run,'opened.json'),'utf8')),['Новый.md','Новый.md','Новый.md','Новый.md']);
    assert.equal((await fs.readdir(path.join(run,'workspace'))).filter(p=>p.startsWith('Новый')).length,1);
    await send('Замени в файле Заметки.md текст «4817» на «6248»',/Сохранён файл.*Заметки.md/);
    assert((await fs.readFile(path.join(run,'workspace/Заметки.md'),'utf8')).includes('6248'));
    const backups=await fs.readdir(path.join(run,'workspace/.fortuna-backups'));
    assert.equal(backups.length,1);
    assert((await fs.readFile(path.join(run,'workspace/.fortuna-backups',backups[0]),'utf8')).includes('4817'));
    await send('Предпросмотр: Конвертируй Заметки.md в Копия.pdf',/Ответьте «да»/);
    await assert.rejects(fs.stat(path.join(run,'workspace/Копия.pdf')));
    await send('да',/Сохранён файл.*Копия.pdf/);
    assert((await fs.readFile(path.join(run,'workspace/Копия.pdf'))).subarray(0,5).toString()==='%PDF-');
    await send('Предпросмотр: Создай файл Отмена.md с текстом «Отменить»',/Подготовлен документ.*Отмена.md/);
    await send('отмена',/Изменение отменено/);
    await assert.rejects(fs.stat(path.join(run,'workspace/Отмена.md')));
    const preview=await(await fetch(toolbase+'/documents/preview',{method:'POST',headers:auth,
      body:JSON.stringify({operation:'edit',path:'Заметки.md',old:'6248',new:'0000'})})).json();
    await fs.writeFile(path.join(run,'workspace/Заметки.md'),'Внешняя правка','utf8');
    const stale=await fetch(toolbase+'/documents/apply',{method:'POST',headers:auth,
      body:JSON.stringify({proposal_id:preview.proposal_id})});assert.equal(stale.status,409);
    assert.equal(await fs.readFile(path.join(run,'workspace/Заметки.md'),'utf8'),'Внешняя правка');
    const escape=await fetch(toolbase+'/documents/preview',{method:'POST',headers:auth,
      body:JSON.stringify({operation:'create',path:'../escape.txt',text:'blocked'})});assert.equal(escape.status,400);
    await send('Создай файл с общим пересказом всего в папке «Документы прогона», назови сам',/Источников с проверенным пересказом: 6 из 6/);
    await page.getByText('Контекст диалога сжат. Полная история сохранена.',{exact:true}).last().waitFor();
    await page.getByText('Длительная задача · результаты шагов сохранены в журнале.',{exact:true}).last().waitFor();
    const summary=await fs.readFile(path.join(run,'workspace/Документы прогона/Общий пересказ.md'),'utf8');
    for(let index=0;index<6;index++)assert(summary.includes('отдельный проект '+index));
    await page.getByText('Действия, изменения и проверки',{exact:true}).last().click();
    await page.locator('.repository-task').last().getByText('Документов для пересказа: 6.',{exact:true}).waitFor();
    assert.equal(await page.locator('.repository-task').last().getByText('Пересказ этой части проверен.',{exact:true}).count(),6);
    assert.equal(await page.locator('.repository-task').last().getByText('Предложения и доказательства',{exact:true}).count(),6);
    await page.locator('.repository-task').last().getByText('Предложения и доказательства',{exact:true}).first().click();
    await page.locator('.repository-task').last().locator('blockquote').first().waitFor();
    assert((await page.locator('.repository-task').last().locator('blockquote').first().innerText()).includes('отдельный проект 0'));
    await send('Прочитай Лимит времени.md',/Достигнут лимит времени задачи/);
    await send('Исправь ошибку в коде проекта и запусти тесты',/Задача выполнена/);
    assert((await fs.readFile(path.join(run,'workspace/calc.py'),'utf8')).includes('return a + b'));
    await page.getByText('Действия, изменения и проверки',{exact:true}).last().click();
    const details=page.locator('.repository-task').last();
    await details.getByText('Правка файла',{exact:true}).waitFor();
    assert.equal(await details.getByText('Проверка проекта',{exact:true}).count(),2);
    await details.getByText(/код выхода 0/).waitFor();
    await details.getByText(/код выхода 1/).waitFor();
    await page.reload();
    await page.getByText('Действия, изменения и проверки',{exact:true}).last().click();
    await page.locator('.repository-task').last().getByText('Правка файла',{exact:true}).waitFor();
    await page.screenshot({path:path.join(run,'chat-tools.png'),fullPage:true});
    assert.deepEqual(errors,[]);
    console.log('PASS: chat document tools, repository repair after failed tests, rerun, diff and process output, journal after reload, auth and workspace boundary. '+run);
  }finally{
    if(browser)await browser.close();server.kill();await exited;
    await fs.writeFile(path.join(run,'server.log'),logs);
  }
}
main().catch(error=>{console.error(error);process.exitCode=1;});
