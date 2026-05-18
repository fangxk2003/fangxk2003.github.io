import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import tailwind from '@astrojs/tailwind';
import react from '@astrojs/react';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

function rehypeMermaid() {
  return (tree) => {
    const getText = (node) => {
      if (node.type === 'text') return node.value;
      return node.children?.map(getText).join('') ?? '';
    };

    const visit = (node) => {
      if (node.type === 'element' && node.tagName === 'pre') {
        const code = node.children?.[0];
        const preLanguage = node.properties?.dataLanguage;
        const className = code?.properties?.className ?? [];

        if (
          preLanguage === 'mermaid' ||
          (
            code?.type === 'element' &&
            code.tagName === 'code' &&
            className.includes('language-mermaid')
          )
        ) {
          node.tagName = 'div';
          node.properties = { className: ['mermaid'] };
          node.children = [{ type: 'text', value: getText(code ?? node) }];
        }
      }

      node.children?.forEach(visit);
    };

    visit(tree);
  };
}

export default defineConfig({
  // IMPORTANT: This matches your github username
  site: 'https://fangxk2003.github.io',
  integrations: [mdx(), sitemap(), tailwind(), react()],
  markdown: {
    // This allows you to use $E=mc^2$ in your posts
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex, rehypeMermaid],
  },
});
