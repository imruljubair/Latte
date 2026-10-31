import { test, expect } from './fixtures';

test('initial load and reload keep workbook content hidden until the app is ready',async({page})=>{
  let release!:()=>void;
  let pending=new Promise<void>(resolve=>{release=resolve;});
  await page.route('**/api/runs/**/metadata',async route=>{await pending;await route.fallback();});
  for(let pass=0;pass<2;pass++){
    if(pass)pending=new Promise<void>(resolve=>{release=resolve;});
    if(pass)await page.reload({waitUntil:'domcontentloaded'});else await page.goto('/',{waitUntil:'domcontentloaded'});
    await expect(page.locator('#latte-loading')).toBeVisible();
    await expect(page.locator('html')).toHaveAttribute('data-latte-loading','');
    await expect(page.locator('.worksheet').first()).toHaveCSS('visibility','hidden');
    const front=await page.evaluate(()=>document.elementFromPoint(innerWidth/2,innerHeight/2)?.id);
    expect(front).toBe('latte-loading');
    release();
    await expect(page.locator('#latte-loading')).toHaveCount(0);
    await expect(page.frameLocator('iframe[title="Training forward pass"]').locator('.matrix-value').first()).toBeVisible();
    await expect(page.locator('html')).not.toHaveAttribute('data-latte-loading','');
  }
});

test('startup errors replace loading with a readable error',async({page})=>{
  await page.route('**/api/runs/**/metadata',route=>route.fulfill({status:500,body:'Test data unavailable'}));
  await page.goto('/');
  await expect(page.getByRole('alert')).toContainText('Could not open Latte');
  await expect(page.locator('#latte-loading')).toHaveCount(0);
});
