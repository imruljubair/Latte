import { test, expect } from '@playwright/test';
const url=process.env.LATTE_PAGES_URL;
for(const width of [1440,390]){
  test(`published demo uses the local Training UI and guards edits at ${width}px`,async({page})=>{
    test.skip(!url,'Build and serve Pages, then set LATTE_PAGES_URL.');
    await page.setViewportSize({width,height:1000});
    const mutations:string[]=[];
    page.on('request',request=>{if(request.method()!=='GET')mutations.push(request.url());});
    await page.goto(`${url}?tab=training`);
    const panel=page.getByRole('tabpanel');
    await expect(panel).toHaveAttribute('aria-busy','false');
    await expect(page.getByRole('progressbar',{name:'Training step progress'})).toHaveAttribute('value','201');
    await expect(page.locator('[data-loss-series="validation"] polyline')).toHaveAttribute('stroke-dasharray','5 4');
    const corpus=page.getByRole('textbox',{name:'Training text'}),original=await corpus.inputValue();
    expect(original.length).toBeGreaterThan(100);
    const dialog=page.getByRole('dialog');
    for(const control of [corpus,page.getByRole('spinbutton',{name:'Embedding dimension',exact:true}),page.getByRole('button',{name:'Start training',exact:true}),page.getByRole('button',{name:/^Delete /})]){
      await control.click();await expect(dialog).toBeVisible();
      await expect(dialog.getByRole('link',{name:'Run locally →'})).toHaveAttribute('href','run-locally.html');
      await dialog.getByRole('button',{name:'Keep exploring'}).click();
    }
    await corpus.focus();await page.keyboard.press('x');await expect(dialog).toBeVisible();await page.keyboard.press('Escape');
    await expect(corpus).toHaveValue(original);
    await page.locator('.latte-advanced-settings summary').click();
    await expect(page.getByRole('spinbutton',{name:'Training split',exact:true})).toHaveValue('0.9');
    await page.getByRole('tab',{name:'Explorer',exact:true}).click();
    await page.locator('[data-action="next"]').click();
    await expect(page.locator('.training-record')).toHaveValue('1');
    await page.getByRole('tab',{name:'Training',exact:true}).click();
    await expect(corpus).toHaveValue(original);
    expect(mutations).toEqual([]);
  });
}
