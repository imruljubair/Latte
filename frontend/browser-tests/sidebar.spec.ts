import { test, expect } from './fixtures';
import { fileURLToPath, pathToFileURL } from 'node:url';
const exported=pathToFileURL(fileURLToPath(new URL(`../../out/${process.env.LATTE_TEST_RUN||'default'}/latte.html`,import.meta.url))).href;
for(const url of ['/',exported])for(const width of [1440,390]){
  test(`corpus panel toggle and mobile bracket labels: ${url==='/'?'app':'export'} ${width}`,async({page})=>{
    await page.setViewportSize({width,height:1000});await page.goto(url);
    const sidebar=page.locator('#latte-corpus-sidebar'),button=page.locator('.latte-sidebar-toggle');
    const mobile=width<760;
    await expect(button).toHaveAttribute('aria-expanded',String(!mobile));
    if(mobile){
      await expect(page.locator('.training-corpus')).toBeHidden();
      expect((await sidebar.boundingBox())!.width).toBeLessThanOrEqual(36);
    }else await expect(page.locator('.training-corpus')).toBeVisible();
    await button.click();await expect(button).toHaveAttribute('aria-expanded',String(mobile));
    await button.click();await expect(button).toHaveAttribute('aria-expanded',String(!mobile));
    const frame=page.frameLocator('iframe[title="Training forward pass"]');
    const labels=frame.locator('[data-stage-band-label]');
    await expect(labels.first()).toBeVisible();
    if(mobile){
      await expect(labels.first()).toHaveAttribute('transform',/^rotate\(-90 /);
      const bounds=await labels.first().boundingBox();
      expect(bounds!.height).toBeGreaterThan(bounds!.width);
      const viewport=await frame.locator('.worksheet.active').evaluate(e=>(e.ownerDocument.defaultView as any).LittleLlamaWorkbook.presentationViewport());
      expect(viewport.left).toBeLessThanOrEqual(84);
      await button.focus();await page.keyboard.press('Enter');
      await expect(page.locator('.training-corpus')).toBeVisible();
      await button.click();
    }else await expect(labels.first()).not.toHaveAttribute('transform',/rotate/);
    await frame.getByRole('button',{name:'Go to output',exact:true}).click();
    await expect(button).toHaveAttribute('aria-expanded',String(!mobile));
  });
}
