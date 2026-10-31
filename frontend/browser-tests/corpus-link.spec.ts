import { test, expect } from './fixtures';
import { fileURLToPath, pathToFileURL } from 'node:url';
const exported=pathToFileURL(fileURLToPath(new URL(`../../out/${process.env.LATTE_TEST_RUN||'default'}/latte.html`,import.meta.url))).href;

for(const url of ['/',exported]){
  test(`corpus click links one input column and offers navigation offscreen: ${url==='/'?'app':'export'}`,async({page})=>{
    await page.goto(url);
    const frame=page.frameLocator('iframe[title="Training forward pass"]');
    const headers=frame.locator('.stage-card .live-input-token[data-training-character="true"]');
    await expect(headers).toHaveCount(8);
    const values=()=>headers.first().evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.matrixData().map(({name,row,column,value}:any)=>({name,row,column,value})));
    const initial=await values();
    await headers.first().evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.focusPresentationElements([...element.ownerDocument.querySelectorAll('.stage-card .live-input-token[data-training-character="true"]')]));
    const wire=page.locator('.latte-corpus-connector');
    const go=page.locator('.latte-corpus-go-input');
    for(const position of [0,1]){
      await page.locator(`[data-corpus-position="${position}"]`).click();
      await expect(wire).toBeVisible();await expect(wire).toHaveAttribute('data-token-position',String(position));
      await expect(frame.locator('.latte-corpus-input-target')).toHaveCount(1);
      await expect(headers.nth(position)).toHaveClass(/latte-corpus-input-target/);
      await expect(page.locator('.latte-corpus-selected-token')).toHaveCount(1);
      await expect(go).toBeHidden();
      const popup=await page.locator('.latte-corpus-tooltip').boundingBox();
      const sidebar=await page.locator('.training-body>aside').boundingBox();
      expect(popup!.x).toBeGreaterThanOrEqual(sidebar!.x);
      expect(popup!.x+popup!.width).toBeLessThanOrEqual(sidebar!.x+sidebar!.width);
    }
    expect(await values()).toEqual(initial);
    // Real toolbar clicks must retain the selection, including when navigation
    // temporarily moves its destination outside the viewport.
    for(const action of ['zoom-out','zoom-in','zoom-reset']){
      await frame.locator(`[data-action="${action}"]`).click();
      await expect(page.locator('.latte-corpus-selected-token')).toHaveCount(1);
      await expect(headers.nth(1)).toHaveClass(/latte-corpus-input-target/);
      await expect(page.locator('.latte-corpus-tooltip')).toBeVisible();
    }
    await headers.nth(1).evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.focusPresentationElements([element]));
    await expect(wire).toBeVisible();await expect(go).toBeHidden();
    // Camera movement should update visibility without leaving a long wire.
    const loss=frame.locator('.matrix-value[data-matrix-name="Recorded loss before update"]').last();
    await loss.evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.focusPresentationElements([element]));
    await expect(wire).toBeHidden();await expect(go).toBeVisible();
    await go.click();
    await expect(wire).toBeVisible();await expect(go).toBeHidden();
    await expect(headers.nth(1)).toHaveClass(/latte-corpus-input-target/);
    await page.keyboard.press('Escape');await expect(wire).toBeHidden();
    await expect(frame.locator('.latte-corpus-input-target')).toHaveCount(0);
    await page.locator('[data-corpus-position="0"]').focus();await page.keyboard.press('Enter');
    await expect(wire).toBeVisible();
    await page.locator('[data-action="next"]').click();await expect(wire).toBeHidden();
    await expect(frame.locator('.latte-corpus-input-target')).toHaveCount(0);
    // Input can also be hidden by a block filter; navigation reveals that block.
    await frame.locator('.steps-menu summary').click();
    await frame.getByRole('button',{name:'Choose from list…',exact:true}).click();
    const picker=frame.getByRole('region',{name:'Choose visualization blocks'});
    await picker.getByRole('button',{name:'Clear',exact:true}).click();
    await picker.locator('input[data-stage-leaf]').last().check();
    await picker.getByRole('button',{name:'Show selected blocks',exact:true}).click();
    await page.locator('[data-corpus-position="0"]').click();
    await expect(go).toBeVisible();await expect(wire).toBeHidden();
    await go.click();await expect(wire).toBeVisible();
    await expect(headers.first()).toHaveClass(/latte-corpus-input-target/);
  });
}
