import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './browser-tests',
  workers: 1,
  use: {
    baseURL: 'http://127.0.0.1:8017',
    viewport: { width: 1440, height: 1000 },
    channel: process.env.LATTE_BROWSER_CHANNEL || undefined,
    screenshot: 'only-on-failure',
  },
  webServer: {
    command: process.platform === 'win32'
      ? '..\\.venv\\Scripts\\python.exe ..\\server.py --port 8017'
      : '../.venv/bin/python ../server.py --port 8017',
    url: 'http://127.0.0.1:8017/docs',
    reuseExistingServer: false,
  },
});
