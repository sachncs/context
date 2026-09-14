// @ts-check
import { defineConfig } from 'astro/config';
import tailwindcss from '@tailwindcss/vite';

// Site lives at https://sachncs.github.io/context/
export default defineConfig({
  site: 'https://sachncs.github.io',
  base: '/context/',
  output: 'static',
  compressHTML: true,
  build: {
    inlineStylesheets: 'auto',
    inlineScripts: true,
  },
  vite: {
    plugins: [tailwindcss()],
  },
});