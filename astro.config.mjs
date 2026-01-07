import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import tailwind from '@astrojs/tailwind';
import react from '@astrojs/react';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

export default defineConfig({
  // IMPORTANT: This matches your github username
  site: 'https://fangxk2003.github.io',
  integrations: [mdx(), sitemap(), tailwind(), react()],
  markdown: {
    // This allows you to use $E=mc^2$ in your posts
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
  },
});