import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const ORIGIN = 'https://calories-build.invalid';
const SHELL = '/calories/';
const FIXED_ASSETS = [
  '/calories/icon.svg', '/calories/icon-192.png', '/calories/icon-512.png',
  '/calories/apple-touch-icon.png', '/calories/manifest.webmanifest',
];
const ASSET_EXTENSION = /\.(?:m?js|css|png|jpe?g|gif|webp|avif|svg|ico|woff2?|ttf|otf|webmanifest)$/i;
const digest = bytes => createHash('sha256').update(bytes).digest();

function assetPath(reference, from) {
  if (!reference || /^(?:data|blob|javascript):/i.test(reference)) return null;
  const url = new URL(reference, new URL(from, ORIGIN));
  if (url.origin !== ORIGIN || url.search || url.pathname === '/calories/sw.js' || !ASSET_EXTENSION.test(url.pathname)) return null;
  return url.pathname;
}

/** Read the emitted HTML and Rollup dependency graph, never all unrelated site assets. */
export async function buildCaloriesWorker(directory, graph = new Map()) {
  const root = directory instanceof URL ? fileURLToPath(directory) : directory;
  const html = await readFile(path.join(root, 'calories/index.html'));
  const template = await readFile(new URL('./calories-service-worker.js', import.meta.url), 'utf8');
  const pending = [...FIXED_ASSETS];
  // Includes Astro island component/renderer URLs and inline module imports.
  for (const match of html.toString().matchAll(/["']([^"'\s<>]+)["']/g)) {
    const candidate = assetPath(match[1].replaceAll('&amp;', '&'), SHELL);
    if (candidate) pending.push(candidate);
  }
  const contents = new Map([[SHELL, html]]);
  while (pending.length) {
    const url = pending.pop();
    if (contents.has(url)) continue;
    const filename = path.resolve(root, `.${decodeURIComponent(url)}`);
    if (!filename.startsWith(`${path.resolve(root)}${path.sep}`)) throw new Error('Invalid Calories asset path.');
    const bytes = await readFile(filename);
    contents.set(url, bytes);
    for (const dependency of graph.get(url) ?? []) {
      const resolved = assetPath(dependency, '/');
      if (!resolved) throw new Error('Invalid Calories build dependency.');
      pending.push(resolved);
    }
    if (/\.css$/i.test(url)) {
      const css = bytes.toString();
      for (const match of css.matchAll(/url\(\s*["']?([^\s"')]+)["']?\s*\)|@import\s+["']([^"']+)["']/gi)) {
        const resolved = assetPath(match[1] ?? match[2], url);
        if (resolved) pending.push(resolved);
      }
    }
  }
  const resources = [...contents].sort(([a], [b]) => a.localeCompare(b)).map(([url, bytes]) => ({
    url, integrity: `sha256-${digest(bytes).toString('base64')}`,
  }));
  const version = digest(`${template}\n${JSON.stringify(resources)}`).toString('hex').slice(0, 24);
  const manifest = { version, resources };
  const worker = template.replace('/* CALORIES_PRECACHE_MANIFEST */ null', JSON.stringify(manifest));
  if (worker === template) throw new Error('Calories worker manifest marker is missing.');
  await writeFile(path.join(root, 'calories/sw.js'), worker);
  return manifest;
}

/** The normal Astro/GitHub Pages build always emits its matching offline shell. */
export default function caloriesPwa() {
  const graph = new Map();
  return {
    name: 'calories-offline-shell',
    hooks: {
      'astro:config:setup': ({ updateConfig }) => updateConfig({ vite: { plugins: [{
        name: 'calories-client-asset-graph',
        apply: 'build',
        // Astro finalizes chunk names during generateBundle; read the written names.
        writeBundle(_options, bundle) {
          for (const item of Object.values(bundle)) {
            if (item.type !== 'chunk') continue;
            const metadata = item.viteMetadata;
            graph.set(`/${item.fileName}`, [
              ...item.imports, ...item.dynamicImports,
              ...(metadata?.importedCss ?? []), ...(metadata?.importedAssets ?? []),
            ].map(filename => filename.startsWith('/') ? filename : `/${filename}`));
          }
        },
      }] } }),
      'astro:build:done': async ({ dir, logger }) => {
        const manifest = await buildCaloriesWorker(dir, graph);
        logger.info(`Generated Calories offline shell (${manifest.resources.length} static resources).`);
      },
    },
  };
}
