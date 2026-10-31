import { test, expect } from './fixtures';
import { fileURLToPath, pathToFileURL } from 'node:url';

const exported=pathToFileURL(fileURLToPath(new URL(`../../out/${process.env.LATTE_TEST_RUN||'default'}/latte.html`,import.meta.url))).href;

for(const url of ['/',exported]){
  test(`block selection filters display while preserving math and replay: ${url==='/'?'app':'HTML export'}`,async({page})=>{
    await page.goto(url);
    const frame=page.frameLocator('iframe[title="Training forward pass"]');
    const nav=frame.locator('.stage-view-tabs');
    await expect(nav).toBeVisible();
    const matrix=frame.locator('.matrix-value').first();
    const values=()=>matrix.evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.matrixData().map(({name,row,column,value}:any)=>({name,row,column,value})));
    const initial=await values();
    const visible=()=>frame.locator('.stage-card').evaluateAll(elements=>elements.filter(element=>getComputedStyle(element).display!=='none').map(element=>(element as HTMLElement).dataset.operation));
    const total=(await visible()).length;
    await frame.locator('.steps-menu summary').click();
    await frame.getByRole('button',{name:'Choose from list…',exact:true}).click();
    const picker=frame.getByRole('region',{name:'Choose visualization blocks'});
    const choices=picker.locator('.stage-picker-list input[data-stage-leaf]');
    await expect(choices).toHaveCount(total);
    await expect(picker.getByRole('button',{name:'Show selected blocks'})).toBeDisabled();
    await choices.last().check();
    await picker.getByRole('button',{name:'Show selected blocks'}).click();
    expect((await visible())).toHaveLength(1);
    expect(await values()).toEqual(initial);
    await page.locator('[data-action="sequence-next"]').click();
    await page.locator('[data-action="next"]').click();
    await page.locator('[data-action="apply"]').click();
    expect((await visible())).toHaveLength(1);
    await page.locator('#latte-math').check();
    const loss=frame.locator('.matrix-value[data-matrix-name="Recorded loss before update"]').first();
    await loss.evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.setZoom(1));
    await page.waitForTimeout(450);
    await loss.evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.focusPresentationElements([element]));
    await loss.click();
    await expect(page.locator('.latte-hidden-input-note')).toBeVisible();
    await page.locator('.latte-variable-button[data-math-variable="b"]').click();
    expect(await visible()).toHaveLength(2);
    await expect(page.locator('.latte-math-panel')).toBeVisible();
    await expect(page.locator('.latte-hidden-input-note')).toBeHidden();
    await expect(frame.locator('.latte-ribbon')).toHaveCount(1);
    await nav.getByRole('button',{name:'Full model',exact:true}).click();
    expect(await visible()).toHaveLength(total);
    await expect(page.locator('.latte-math-panel')).toHaveCount(0);
    await frame.locator('.steps-menu summary').click();
    await frame.getByRole('button',{name:'Reset selected blocks'}).click();
    await frame.locator('.steps-menu summary').click();
    await frame.getByRole('button',{name:'Pick on diagram',exact:true}).click();
    await frame.locator('.stage-pick-control input').first().check();
    await nav.getByRole('button',{name:'Selected blocks',exact:true}).click();
    expect(await visible()).toHaveLength(1);
    await expect(frame.locator('.stage-pick-control:visible')).toHaveCount(0);
  });
}

for(const url of ['/',exported]){
  test(`hierarchical block choices select descendants and track partial selection: ${url==='/'?'app':'HTML export'}`,async({page})=>{
    await page.goto(url);
    const frame=page.frameLocator('iframe[title="Training forward pass"]');
    const values=()=>frame.locator('.matrix-value').first().evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.matrixData().map(({name,row,column,value}:any)=>({name,row,column,value})));
    const initial=await values();
    await frame.locator('.steps-menu summary').click();
    await frame.getByRole('button',{name:'Choose from list…',exact:true}).click();
    const picker=frame.getByRole('region',{name:'Choose visualization blocks'});
    await picker.getByRole('button',{name:'Clear',exact:true}).click();
    const layer=picker.getByRole('checkbox',{name:'Select Layer 0',exact:true});
    const root=layer.locator('xpath=../..');
    const leaves=root.locator('input[data-stage-leaf]');
    await layer.check();
    expect(await leaves.count()).toBeGreaterThan(10);
    await expect(root.locator('input[data-stage-leaf]:not(:checked)')).toHaveCount(0);
    const other=picker.getByRole('checkbox',{name:'Select Layer 1',exact:true});
    if(await other.count())await expect(other).not.toBeChecked();
    const head=picker.getByRole('checkbox',{name:'Select Layer 0 / Attention / Head 0',exact:true});
    const attention=picker.getByRole('checkbox',{name:'Select Layer 0 / Attention',exact:true});
    await head.uncheck();
    await expect(layer).toHaveJSProperty('indeterminate',true);
    await expect(attention).toHaveJSProperty('indeterminate',true);
    const headLeaves=head.locator('xpath=../..').locator('input[data-stage-leaf]');
    await expect(head.locator('xpath=../..').locator('input[data-stage-leaf]:checked')).toHaveCount(0);
    await headLeaves.first().check();
    await expect(head).toHaveJSProperty('indeterminate',true);
    await attention.check();
    await expect(layer).toBeChecked();
    await expect(head).toBeChecked();
    await layer.uncheck();
    await expect(root.locator('input:checked')).toHaveCount(0);
    await expect(picker.getByRole('button',{name:'Show selected blocks'})).toBeDisabled();
    await attention.check();
    const expected=await attention.locator('xpath=../..').locator('input[data-stage-leaf]').evaluateAll(elements=>elements.map(element=>(element as HTMLInputElement).value));
    await picker.getByRole('button',{name:'Show selected blocks'}).click();
    const visible=()=>frame.locator('.stage-card').evaluateAll(elements=>elements.filter(element=>getComputedStyle(element).display!=='none').map(element=>(element as HTMLElement).dataset.operation));
    expect(await visible()).toEqual(expected);
    expect(await values()).toEqual(initial);
    await page.locator('[data-action="next"]').click();
    expect(await visible()).toEqual(expected);
    await frame.getByRole('button',{name:'Full model',exact:true}).click();
    await frame.locator('.steps-menu summary').click();
    await frame.getByRole('button',{name:'Choose from list…',exact:true}).click();
    await expect(attention).toBeChecked();
    await expect(layer).toHaveJSProperty('indeterminate',true);
    await picker.getByRole('button',{name:'Select all',exact:true}).click();
    await expect(picker.locator('input:not(:checked)')).toHaveCount(0);
  });
}

for(const url of ['/',exported]){
  test(`diagram bracket groups select nested blocks and synchronize list checks: ${url==='/'?'app':'HTML export'}`,async({page})=>{
    await page.goto(url);
    const frame=page.frameLocator('iframe[title="Training forward pass"]');
    const nav=frame.locator('.stage-view-tabs');
    await frame.locator('.steps-menu summary').click();
    await frame.getByRole('button',{name:'Reset selected blocks',exact:true}).click();
    await frame.locator('.steps-menu summary').click();
    await frame.getByRole('button',{name:'Pick on diagram',exact:true}).click();
    const layer=frame.getByRole('checkbox',{name:'Select Layer 0 on diagram',exact:true});
    await expect(layer).toBeVisible();
    await layer.check();
    const state=()=>frame.locator('.matrix-value').first().evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.stageSelection());
    const chosen=await state();
    expect(chosen.stages.length).toBeGreaterThan(10);
    expect(chosen.stages.every((name:string)=>/layer 0(?:\s|$)/.test(name))).toBe(true);
    const head=frame.getByRole('checkbox',{name:'Select Layer 0 / Attention / Head 0 on diagram',exact:true});
    await expect(head).toBeVisible();
    await expect(head).toBeChecked();
    await head.uncheck();
    await expect(layer).toHaveJSProperty('indeterminate',true);
    expect((await state()).stages.some((name:string)=>name.includes('head 0'))).toBe(false);
    const attention=frame.getByRole('checkbox',{name:'Select Layer 0 / Attention on diagram',exact:true});
    await expect(attention).toHaveJSProperty('indeterminate',true);
    // Clicking the bracket text works even with canvas panning enabled.
    await frame.locator('.hierarchy-bracket-label[data-stage-band-label="Attention"]').first().click();
    await expect(attention).toBeChecked();await expect(head).toBeChecked();
    await frame.locator('.steps-menu summary').click();
    await frame.getByRole('button',{name:'Choose from list…',exact:true}).click();
    const picker=frame.getByRole('region',{name:'Choose visualization blocks'});
    await expect(picker.getByRole('checkbox',{name:'Select Layer 0',exact:true})).toBeChecked();
    await expect(picker.getByRole('checkbox',{name:'Select Layer 0 / Attention / Head 0',exact:true})).toBeChecked();
    await picker.getByRole('button',{name:'Show selected blocks',exact:true}).click();
    expect(await frame.locator('.stage-group-pick-control:visible').count()).toBe(0);
    await page.locator('[data-action="next"]').click();
    expect((await state()).stages).toEqual(chosen.stages);
    await nav.getByRole('button',{name:'Full model',exact:true}).click();
  });
}
