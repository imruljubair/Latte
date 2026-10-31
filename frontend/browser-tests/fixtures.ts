import { test as base, expect } from '@playwright/test';

// Exercise a separate recording without overwriting the user's default run.
export const test=base.extend({
  page:async({page},use)=>{
    const run=process.env.LATTE_TEST_RUN;
    if(run)await page.route('**/api/runs/default/**',async route=>{
      const url=route.request().url().replace('/api/runs/default/',`/api/runs/${encodeURIComponent(run)}/`);
      await route.fulfill({response:await route.fetch({url})});
    });
    await use(page);
  },
});
export { expect };
