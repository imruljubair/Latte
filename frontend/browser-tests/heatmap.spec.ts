import { test, expect } from './fixtures';
import { fileURLToPath, pathToFileURL } from 'node:url';

const exported=pathToFileURL(fileURLToPath(new URL(`../../out/${process.env.LATTE_TEST_RUN||'default'}/latte.html`,import.meta.url))).href;
for(const url of ['/',exported]){
  test(`corpus heatmap follows all sampled inputs and preserves validation and model values: ${url==='/'?'app':'export'}`,async({page})=>{
    await page.goto(url);
    const corpus=page.locator('.training-corpus');
    await expect(page.locator('.training-predictions-section')).toHaveCount(0);
    const initialText=await corpus.textContent();
    const frame=page.frameLocator('iframe[title="Training forward pass"]');
    const matrix=frame.locator('.matrix-value').first();
    const values=()=>matrix.evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.matrixData().map(({name,row,column,value}:any)=>({name,row,column,value})));
    const before=await values();
    const toggle=page.getByRole('checkbox',{name:'Corpus heatmap',exact:true});
    await expect(toggle).toBeEnabled();await toggle.check();
    await expect(corpus).toHaveClass(/corpus-heatmap-enabled/);
    expect(await corpus.textContent()).toBe(initialText);
    expect(await values()).toEqual(before);
    const uses=()=>corpus.locator('[data-usage-count]').evaluateAll(elements=>elements.reduce((total,element)=>total+Array.from(element.textContent||'').length*Number((element as HTMLElement).dataset.usageCount),0));
    const total=await uses();
    expect(total).toBeGreaterThan(0);
    const validation=corpus.locator('.corpus-usage-span[data-usage-validation="true"]');
    await expect(validation).toHaveCount(1);
    await expect(validation).toHaveCSS('background-color','rgba(0, 0, 0, 0)');
    await expect(validation).toHaveAttribute('data-usage-count','0');
    await page.locator('[data-corpus-position="0"]').hover();
    await expect(page.getByRole('tooltip')).toBeHidden();
    await expect(page.locator('.latte-usage-tooltip')).toHaveCount(0);
    await page.locator('[data-corpus-position="0"]').click();
    await expect(page.getByRole('tooltip')).toContainText('P(target)');
    await expect(page.getByRole('tooltip')).toContainText('across the entire training run');
    await page.keyboard.press('Escape');
    await expect(page.getByRole('tooltip')).toBeHidden();
    await page.locator('[data-action="next"]').click();
    await expect.poll(uses).toBe(total);
    const next=await uses();
    expect(next).toBe(total);
    await page.locator('[data-action="sequence-next"]').click();expect(await uses()).toBe(next);
    await page.locator('[data-action="apply"]').click();expect(await uses()).toBe(next);
    const border=await corpus.locator('mark').evaluate(element=>getComputedStyle(element).borderTopStyle);
    expect(border).toBe('solid');
    // Let replay's scheduled corpus centering finish before manually scrolling.
    await page.evaluate(()=>new Promise<void>(resolve=>requestAnimationFrame(()=>requestAnimationFrame(()=>resolve()))));
    await validation.scrollIntoViewIfNeeded();
    // A multiline span's union-box center can be a blank paragraph. Hover ink.
    const ink=await validation.evaluate(element=>{
      const parent=element.parentElement!.getBoundingClientRect();
      const rect=[...element.getClientRects()].find(r=>r.width>2&&r.top>=parent.top&&r.bottom<=parent.bottom)!;
      return {x:rect.left+2,y:rect.top+rect.height/2};
    });
    await page.mouse.move(ink.x,ink.y);
    await expect(page.getByRole('tooltip')).toBeHidden();
    await page.mouse.click(ink.x,ink.y);
    await expect(page.getByRole('tooltip')).toContainText('Validation text · never used for training');
    await toggle.uncheck();await expect(corpus.locator('.corpus-usage-span')).toHaveCount(0);
  });
}

test('recordings without usage data clearly disable the heatmap',async({page,request})=>{
  const old=await (await request.get(`/api/runs/${process.env.LATTE_TEST_RUN||'default'}/metadata`)).json();
  delete old.corpus_usage;
  await page.route('**/api/runs/default/metadata',route=>route.fulfill({json:old}));
  await page.goto('/');
  await expect(page.getByRole('checkbox',{name:'Corpus heatmap',exact:true})).toBeDisabled();
  await expect(page.locator('.latte-usage-unavailable')).toContainText('Usage wasn’t recorded');
});
