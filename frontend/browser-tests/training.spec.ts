import { test, expect } from './fixtures';
import { readdir, rm } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import type { APIRequestContext } from '@playwright/test';

const corpus='One day a little 🦙 found a café by the river. It learned to read, one character at a time.\n'.repeat(8);
async function cleanup(request: APIRequestContext, id: string) {
  await request.post(`/api/runs/${id}/stop`);
  await expect.poll(async()=> !['running','stopping'].includes((await (await request.get(`/api/runs/${id}/status`)).json()).state),{timeout:15000}).toBe(true);
  await rm(fileURLToPath(new URL(`../../out/${id}`,import.meta.url)),{recursive:true,force:true});
  const trash=fileURLToPath(new URL('../../out/.trash',import.meta.url));
  for(const name of await readdir(trash).catch(()=>[])) if(name.startsWith(id+'-')) await rm(`${trash}/${name}`,{recursive:true,force:true});
}

test('Training tab accepts pasted text and parameters, completes a real run, and opens its explorer',async({page,request})=>{
  let id='';
  try{
    await page.goto('/');
    await page.getByRole('tab',{name:'Training',exact:true}).click();
    await page.getByLabel('Training text',{exact:true}).fill(corpus);
    await page.getByLabel('Run name').fill('Browser training test');
    await page.getByLabel('Embedding dimension',{exact:true}).fill('3');
    await page.getByLabel('Layers',{exact:true}).fill('1');
    await page.getByLabel('Attention heads',{exact:true}).fill('1');
    await page.getByLabel('Query / key dimension per head',{exact:true}).fill('2');
    await page.getByLabel('Value dimension per head',{exact:true}).fill('3');
    await page.getByLabel('Feed-forward dimension',{exact:true}).fill('5');
    await page.getByLabel('Context length',{exact:true}).fill('4');
    await page.getByLabel('Training steps',{exact:true}).fill('8');
    await page.getByLabel('Batch size',{exact:true}).fill('2');
    await page.getByLabel('Record every N steps',{exact:true}).fill('3');
    await page.getByRole('tab',{name:'Explorer',exact:true}).click();
    await page.getByRole('tab',{name:'Training',exact:true}).click();
    await expect(page.getByLabel('Training text',{exact:true})).toHaveValue(corpus);
    await expect(page.getByLabel('Training steps',{exact:true})).toHaveValue('8');
    const started=page.waitForResponse(response=>response.url().endsWith('/api/runs')&&response.request().method()==='POST');
    await page.getByRole('button',{name:'Start training',exact:true}).click();
    const response=await started;expect(response.status()).toBe(202);id=(await response.json()).id;
    const open=page.getByRole('link',{name:'Open in explorer →'});
    await expect(open).toBeVisible({timeout:45000});
    const region=page.getByRole('region',{name:'Training progress'});
    await expect(region).toContainText('Complete');
    await expect(region.getByRole('progressbar')).toHaveAttribute('value','8');
    await expect(region.getByRole('img')).toBeVisible();
    await expect(region.locator('[data-loss-series="validation"] polyline')).not.toHaveAttribute('points','');
    await expect(region).toContainText('latest validation loss');
    const selectedRun = page.locator('.latte-run-list button').filter({hasText:'Browser training test'});
    await expect(selectedRun).toHaveAttribute('aria-pressed','true');
    const otherRun = page.locator('.latte-run-list button[aria-pressed="false"]').first();
    if (await otherRun.count()) {
      await otherRun.click();
      await expect(selectedRun).toHaveAttribute('aria-pressed','false');
      await selectedRun.click();
      await expect(selectedRun).toHaveAttribute('aria-pressed','true');
      await expect(page.locator('.latte-monitor-name')).toHaveText('Browser training test');
      await expect(page.getByLabel('Training text',{exact:true})).toHaveValue(corpus);
    }
    await expect(page.getByRole('link',{name:'Explore Browser training test',exact:true})).toHaveAttribute('href',`/?run=${id}`);
    const result=await (await request.get(`/api/runs/${id}/metadata`)).json();
    expect(result.architecture).toMatchObject({model_dim:3,qk_dim:2,value_dim:3,num_layers:1,num_heads:1,context_length:4});
    expect(result.corpus.text).toBe(corpus);
    expect(result.steps.map((step:any)=>step.step)).toEqual([1,4,7,8]);
    await page.reload();
    await expect(page.getByRole('tab',{name:'Training',exact:true})).toHaveAttribute('aria-selected','true');
    await expect(page.getByRole('link',{name:'Open in explorer →'})).toBeVisible();
    await page.getByRole('link',{name:'Open in explorer →'}).click();
    await expect(page).toHaveURL(new RegExp(`run=${id}`));
    await expect(page.getByRole('tab',{name:'Explorer',exact:true})).toHaveAttribute('aria-selected','true');
    const frame=page.frameLocator('iframe[title="Training forward pass"]');
    await expect(frame.locator('.matrix-value').first()).toBeVisible();
    const step=await (await request.get(`/api/runs/${id}/steps/0`)).json();
    const input=step.inputs[0].map((n:number)=>result.vocabulary[n]).join('');
    await expect(page.locator('.training-corpus mark')).toHaveText(input);
    await page.getByRole('checkbox',{name:'Corpus heatmap',exact:true}).check();
    const uses=await page.locator('.training-corpus [data-usage-count]').evaluateAll(elements=>elements.reduce((total,element)=>total+Array.from(element.textContent||'').length*Number((element as HTMLElement).dataset.usageCount),0));
    expect(uses).toBe(result.corpus_usage.total_uses); // Entire run, independent of selected step.
    await page.getByRole('tab',{name:'Training',exact:true}).click();
    await expect(page.getByLabel('Training text',{exact:true})).toHaveValue(corpus);
    await expect(page.getByLabel('Training steps',{exact:true})).toHaveValue('8');
    await page.getByRole('button',{name:'Delete Browser training test',exact:true}).click();
    await expect(page.getByRole('button',{name:'Delete Browser training test',exact:true})).toHaveCount(0);
    await expect.poll(async()=> (await request.get(`/api/runs/${id}/status`)).status()).toBe(404);
    await expect(page).toHaveURL(/tab=training/);
    await expect(page.getByLabel('Training text',{exact:true})).toHaveValue(corpus);

  }finally{await page.close();if(id)await cleanup(request,id);}
});

test('training can be stopped and validation prevents a corpus that is too short',async({page,request})=>{
  let id='';
  try{
    await page.goto('/?tab=training');
    await page.getByLabel('Training text',{exact:true}).fill('tiny');
    await expect(page.getByRole('button',{name:'Start training',exact:true})).toBeDisabled();
    await expect(page.locator('.latte-training-form')).toContainText('Add more text or reduce context length');
    await page.getByLabel('Training text',{exact:true}).fill(corpus);
    await page.getByLabel('Training steps',{exact:true}).fill('100000');
    await page.getByLabel('Record every N steps',{exact:true}).fill('10000');
    const started=page.waitForResponse(response=>response.url().endsWith('/api/runs')&&response.request().method()==='POST');
    await page.getByRole('button',{name:'Start training',exact:true}).click();
    id=(await (await started).json()).id;
    await page.getByRole('button',{name:'Stop training',exact:true}).click();
    await expect(page.getByRole('region',{name:'Training progress'})).toContainText('Stopped',{timeout:15000});
    await expect(page.getByRole('button',{name:'Start training',exact:true})).toBeEnabled();
    await expect(page.getByRole('link',{name:'Open in explorer →'})).toHaveCount(0);
  }finally{await page.close();if(id)await cleanup(request,id);}
});

test('starting a new run immediately clears the previous chart and focuses progress',async({page})=>{
  const old={id:'browser-check',name:'Old experiment',created_at:'2026-10-27T10:00:00Z',state:'complete',explorer_ready:false,log:'',progress:{step:900,total_steps:900,history:[{step:1,loss:4},{step:900,loss:1}]}};
  await page.addInitScript(()=>localStorage.setItem('latte-training-run','browser-check'));
  let release!:()=>void;
  const pending = new Promise<void>(resolve=>{release=resolve;});
  await page.route('**/api/runs',async route=>{
    if(route.request().method()==='POST'){
      await pending;
      await route.fulfill({status:202,json:{...old,id:'run-new-ui',name:'New experiment',progress:{step:1,total_steps:500,history:[{step:1,loss:4}]}}});
    }else await route.fulfill({json:[old]});
  });
  await page.route('**/api/runs/browser-check/status',route=>route.fulfill({json:old}));
  await page.route('**/api/runs/browser-check/corpus',route=>route.fulfill({json:{text:corpus}}));
  await page.route('**/api/runs/browser-check/settings',route=>route.fulfill({json:{
    model:{model_dim:3,qk_dim:2,value_dim:3,num_heads:1,num_layers:1,ffn_dim:5,context_length:4},
    training:{steps:900,batch_size:2,learning_rate:.003,weight_decay:0,seed:42,train_split:.9,record_every:10},
  }}));
  await page.goto('/?tab=training');
  const progress=page.getByRole('region',{name:'Training progress'});
  await expect(progress.getByRole('img')).toHaveAttribute('aria-label',/to 900/);
  await expect(page.getByRole('button',{name:'Delete Old experiment',exact:true})).toBeEnabled();
  await page.getByRole('button',{name:'Use current corpus',exact:true}).click();
  await page.getByLabel('Training steps',{exact:true}).fill('500');
  await page.getByRole('button',{name:'Start training',exact:true}).click();
  await expect(progress.getByRole('img')).toHaveAttribute('aria-label',/step 0 to 500/);
  await expect(progress.locator('[data-loss-series="training"] polyline')).toHaveAttribute('points','');
  await expect(progress.getByRole('progressbar')).toHaveAttribute('value','0');
  await expect(progress).toBeFocused();
  release();
  await expect(progress.getByRole('img')).toHaveAttribute('aria-label',/step 0 to 500/);
  await expect(page.locator('.latte-monitor-name')).toHaveText('New experiment');
  const firstX=await progress.locator('[data-loss-series="training"] polyline').getAttribute('points');
  expect(Number(firstX!.split(',')[0])).toBeCloseTo(44 + 492 / 500);

});

test('selecting a recent run loads its corpus and settings and ignores a stale earlier selection',async({page})=>{
  const runs=['a','b'].map((id,i)=>({id:`run-corpus-${id}`,name:`Corpus ${id}`,state:'complete',explorer_ready:false,log:'',progress:{step:i+1,total_steps:10,history:[{step:i+1,loss:3}]}}));
  await page.route('**/api/runs',route=>route.fulfill({json:runs}));
  let release!:()=>void;
  const delayed=new Promise<void>(resolve=>{release=resolve;});
  for(const run of runs){
    await page.route(`**/api/runs/${run.id}/status`,route=>route.fulfill({json:run}));
    const dimension=run.name==='Corpus a'?7:12;
    await page.route(`**/api/runs/${run.id}/settings`,route=>route.fulfill({json:{
      model:{model_dim:dimension,qk_dim:3,value_dim:4,num_heads:2,num_layers:3,ffn_dim:dimension*4,context_length:8},
      training:{steps:500,batch_size:8,learning_rate:.002,weight_decay:.01,seed:42,train_split:.8,record_every:20},
    }}));
    await page.route(`**/api/runs/${run.id}/corpus`,async route=>{
      if(run.name==='Corpus a')await delayed;
      await route.fulfill({json:{text:`Text belonging to ${run.name}. 🦙`}});
    });
  }
  await page.goto('/?tab=training');
  await page.getByRole('button',{name:/Corpus a.*View/}).click();
  await page.getByRole('button',{name:/Corpus b.*View/}).click();
  await expect(page.locator('.latte-monitor-name')).toHaveText('Corpus b');
  await expect(page.getByLabel('Training text',{exact:true})).toHaveValue('Text belonging to Corpus b. 🦙');
  await expect(page.getByLabel('Embedding dimension',{exact:true})).toHaveValue('12');
  await expect(page.getByLabel('Feed-forward dimension',{exact:true})).toHaveValue('48');
  await expect(page.getByLabel('Training steps',{exact:true})).toHaveValue('500');
  await expect(page.getByLabel('Batch size',{exact:true})).toHaveValue('8');
  await expect(page.getByLabel('Learning rate',{exact:true})).toHaveValue('0.002');
  await page.getByText('More settings',{exact:true}).click();
  await expect(page.getByLabel('Random seed',{exact:true})).toHaveValue('42');
  await expect(page.getByLabel('Training split',{exact:true})).toHaveValue('0.8');
  release();
  await expect(page.locator('.latte-monitor-name')).toHaveText('Corpus b');
  await expect(page.getByLabel('Training text',{exact:true})).toHaveValue('Text belonging to Corpus b. 🦙');
});

for(const running of [false,true]){
  test(`startup restores ${running?'the active run before newer saved runs':'the latest run instead of a draft or last viewed run'}`,async({page})=>{
    const runs=[
      {id:'run-latest-test',name:'Latest setup',created_at:'2026-10-27T12:00:00Z',state:'complete',explorer_ready:false,log:'',progress:{step:500,total_steps:500,history:[{step:1,loss:3}]}},
      {id:'run-older-test',name:'Older setup',created_at:'2026-10-26T12:00:00Z',state:running?'running':'complete',explorer_ready:false,log:'',progress:{step:100,total_steps:100,history:[{step:1,loss:3}]}},
    ];
    await page.addInitScript(()=>{
      localStorage.setItem('latte-training-run','run-older-test');
      localStorage.setItem('latte-training-draft',JSON.stringify({name:'Unsaved draft',corpus:'Wrong draft text',settings:{steps:9999}}));
    });
    await page.route('**/api/runs',route=>route.fulfill({json:runs}));
    for(const run of runs){
      const latest=run.id==='run-latest-test';
      await page.route(`**/api/runs/${run.id}/status`,route=>route.fulfill({json:run}));
      await page.route(`**/api/runs/${run.id}/corpus`,route=>route.fulfill({json:{text:`${run.name} corpus. `.repeat(10)}}));
      await page.route(`**/api/runs/${run.id}/settings`,route=>route.fulfill({json:{
        model:{model_dim:latest?6:3,qk_dim:2,value_dim:3,num_heads:2,num_layers:2,ffn_dim:latest?24:12,context_length:8},
        training:{steps:latest?500:100,batch_size:8,learning_rate:.002,weight_decay:0,seed:42,train_split:.9,record_every:20},
      }}));
    }
    const expected=running?'Older setup':'Latest setup';
    const verify=async()=>{
      await expect(page.locator('#latte-training-panel')).toHaveAttribute('aria-busy','false');
      await expect(page.locator('.latte-monitor-name')).toHaveText(expected);
      await expect(page.getByLabel('Run name')).toHaveValue(expected);
      await expect(page.getByLabel('Training text',{exact:true})).toHaveValue(`${expected} corpus. `.repeat(10));
      await expect(page.getByLabel('Training steps',{exact:true})).toHaveValue(running?'100':'500');
      await expect(page.getByLabel('Embedding dimension',{exact:true})).toHaveValue(running?'3':'6');
    };
    await page.goto('/?tab=training');await verify();
    if(!running){
      await page.getByRole('button',{name:/Older setup.*View/}).click();
      await expect(page.getByLabel('Training steps',{exact:true})).toHaveValue('100');
      await page.reload();await verify();
      await page.evaluate(()=>localStorage.clear());
      await page.reload();await verify();
    }
  });
}
