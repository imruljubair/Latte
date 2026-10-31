import { type Page } from '@playwright/test';
import { test, expect } from './fixtures';
import { PNG } from 'pngjs';
import { fileURLToPath, pathToFileURL } from 'node:url';

const original = pathToFileURL(fileURLToPath(new URL(`../../out/${process.env.LATTE_TEST_RUN||'default'}/latte.html`, import.meta.url))).href;

async function settled(page: Page) {
  await expect(page.locator('.training-replay')).toBeVisible();
  await expect(page.frameLocator('iframe[title="Training forward pass"]').locator('.matrix-value').first()).toBeVisible();
  await page.evaluate(() => document.fonts.ready);
  // Training requires the local server; standalone exports share the explorer only.
  await page.addStyleTag({content:'.latte-app-tabs{visibility:hidden!important}'});
  // Chromium composites toolbar blur/shadows differently after an iframe
  // becomes visible. Remove those GPU effects for deterministic pixel checks.
  await page.frameLocator('iframe[title="Training forward pass"]').locator('body').evaluate(element => {
    const style=element.ownerDocument.createElement('style');
    style.textContent='.workbook-toolbar{backdrop-filter:none!important;box-shadow:none!important}';
    element.ownerDocument.head.append(style);
  });
  // Wait for the existing camera/annotation startup frames and cell transitions.
  await page.waitForTimeout(800);
}

test('server and HTML export share the new presentation at desktop and narrow widths', async ({ page }) => {
  for (const width of [1440, 760]) {
    await page.setViewportSize({ width, height: 1000 });
    await page.goto(original);
    await settled(page);
    const baseline = PNG.sync.read(await page.screenshot());
    const geometry = await page.locator('.training-replay').evaluate(element =>
      [':scope > header', '.training-body iframe', '.training-body > aside'].map(selector => element.querySelector(selector)!.getBoundingClientRect().toJSON()));
    await page.goto('/');
    await settled(page);
    const current = PNG.sync.read(await page.screenshot());
    expect(await page.locator('.training-replay').evaluate(element =>
      [':scope > header', '.training-body iframe', '.training-body > aside'].map(selector => element.querySelector(selector)!.getBoundingClientRect().toJSON()))).toEqual(geometry);
    expect(current.width).toBe(baseline.width);
    expect(current.height).toBe(baseline.height);
    let changed = 0;
    for (let i = 0; i < baseline.data.length; i += 4) {
      if (baseline.data[i] !== current.data[i] || baseline.data[i + 1] !== current.data[i + 1] || baseline.data[i + 2] !== current.data[i + 2]) changed++;
    }
    // Small font-rasterization differences are allowed; geometry must match.
    console.log(`Visual comparison at ${width}px: ${changed} changed pixels / ${current.width * current.height}`);
    expect(changed / (current.width * current.height)).toBeLessThan(.001);
  }
});

test('fetches steps on demand and preserves the diagram while controls update', async ({ page }) => {
  const errors: string[] = [];
  const fetched: string[] = [];
  page.on('pageerror', error => errors.push(String(error)));
  page.on('request', request => { if (request.url().includes('/steps/')) fetched.push(request.url()); });
  await page.goto('/');
  await settled(page);
  const frame = page.frameLocator('iframe[title="Training forward pass"]');
  const cell = await frame.locator('.matrix-value').first().elementHandle();
  const token = await page.locator('[data-corpus-position="0"]').elementHandle();
  await page.locator('[data-action="next"]').click();
  await expect(page.locator('.training-record')).toHaveValue('1');
  await page.locator('[data-action="apply"]').click();
  await expect(page.locator('.training-status')).toContainText('After update');
  await page.locator('[data-action="sequence-next"]').click();
  await expect(page.locator('.training-sequence')).toHaveValue('1');
  await expect(page.locator('.training-weight-section')).toBeHidden();
  expect(await cell!.evaluate(element => element.isConnected)).toBe(true);
  expect(await token!.evaluate(element => element.isConnected)).toBe(true);
  await page.locator('[data-action="previous"]').click();
  await expect(page.locator('.training-record')).toHaveValue('0');
  expect(fetched.filter(url => url.endsWith('/steps/0'))).toHaveLength(1);
  expect(fetched.filter(url => url.endsWith('/steps/1'))).toHaveLength(1);
  expect(errors).toEqual([]);
});

test('handles a failed step request without discarding the current view', async ({ page }) => {
  await page.goto('/');
  await settled(page);
  await page.route('**/api/runs/default/steps/1', route => route.fulfill({ status: 503, body: 'Temporarily unavailable' }));
  await page.locator('[data-action="next"]').click();
  await expect(page.getByRole('alert')).toContainText('Could not load recorded step');
  await expect(page.locator('.training-record')).toHaveValue('0');
  await page.unroute('**/api/runs/default/steps/1');
  await page.locator('[data-action="next"]').click();
  await expect(page.locator('.training-record')).toHaveValue('1');
});

test('HTML export uses the sidebar equation and clickable variable navigation',async({page})=>{
  await page.goto(original);
  await settled(page);
  await page.locator('#latte-math').check();
  const frame=page.frameLocator('iframe[title="Training forward pass"]');
  const cell=frame.locator('.matrix-value[data-matrix-name="Q · layer 0 · head 0"]').first();
  await cell.evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.setZoom(1));
  await page.waitForTimeout(350);
  await cell.evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.focusPresentationElements([element]));
  await cell.click();
  const panel=page.getByRole('region',{name:'Selected cell calculation'});
  await expect(panel).toBeVisible();
  await panel.locator('math [data-math-variable="c"]').click();
  await page.waitForTimeout(400);
  await expect(frame.locator('.latte-ribbon[data-operand="c"]')).toHaveClass(/latte-tracing-ribbon/);
  expect(await cell.evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.cameraZoom())).toBe(1);
  await expect(panel).toBeVisible();
  await panel.getByRole('button',{name:'Back to result a',exact:true}).click();
  await expect(cell).toHaveClass(/latte-math-destination/);
  await page.keyboard.press('Escape');
  await expect(panel).toHaveCount(0);
});

test('HTML export explains recorded weight origins across updates and training records',async({page})=>{
  await page.goto(original);
  await settled(page);
  await page.locator('#latte-math').check();
  const frame=page.frameLocator('iframe[title="Training forward pass"]');
  const cell=frame.locator('.matrix-value[data-matrix-name="Token embeddings"][data-matrix-row="2"][data-matrix-column="8"]');
  await cell.evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.setZoom(1));
  await page.waitForTimeout(350);
  await cell.evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.focusPresentationElements([element]));
  await cell.click();
  const panel=page.locator('.latte-math-panel');
  await expect(panel).toContainText('Parameter entering training step 1.');
  await page.locator('[data-action="apply"]').click();
  await expect(panel).toContainText('Updated by AdamW at training step 1');
  const snapshot=await cell.evaluate(element=>{
    const book=(element.ownerDocument.defaultView as any).LittleLlamaWorkbook;
    return {value:book.presentationCell(element).value,origin:book.weightOrigin(element)};
  });
  expect(snapshot.origin.after).toBeCloseTo(snapshot.value,7);
  await panel.locator('summary').click();
  await expect(panel).toContainText(`Batch-mean gradient: ${snapshot.origin.gradient.toPrecision(8)}`);
  await expect(frame.locator('.latte-ribbon')).toHaveCount(0);
  await expect(frame.locator('.latte-math-name-link')).toHaveCount(0);
  await page.locator('[data-action="next"]').click();
  await expect(panel).toContainText('Parameter entering training step 2.');
  await expect(panel.locator('details')).toHaveCount(0);
  await expect(page.locator('.training-weight-section')).toBeHidden();
  await panel.getByRole('button',{name:'Close calculation'}).click();
  await expect(panel).toHaveCount(0);
});
