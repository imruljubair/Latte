import { defineConfig } from 'vite';

export default defineConfig({
  base: '/ui/',
  build: {
    manifest: true,
    rollupOptions: { input: 'src/main.tsx' },
  },
});
