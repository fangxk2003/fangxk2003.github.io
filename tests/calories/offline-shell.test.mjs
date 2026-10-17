import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import vm from 'node:vm';
import caloriesPwa, { buildCaloriesWorker } from '../../scripts/calories-pwa.mjs';
import { registerCaloriesOfflineShell } from '../../src/lib/calories/offline-shell.ts';

const ORIGIN = 'https://example.test';
const fixed = ['/calories/icon.svg', '/calories/icon-192.png', '/calories/icon-512.png', '/calories/apple-touch-icon.png', '/calories/manifest.webmanifest'];
const graph = new Map([
  ['/_astro/app.a.js', ['/_astro/shared.a.js', '/_astro/lazy.a.js', '/_astro/imported.svg']],
  ['/_astro/renderer.a.js', ['/_astro/shared.a.js']],
  ['/_astro/register.a.js', []],
]);
async function fixture(t) {
  const directory = await mkdtemp('/private/tmp/calories-pwa-');
  t.after(() => rm(directory, { recursive: true, force: true }));
  const files = new Map([
    ['/calories/index.html', '<html><head><link href=\'/calories/manifest.webmanifest\' rel=\'manifest\'><link href=\'/_astro/calories.a.css\' rel=\'stylesheet\'></head><body><script src=\'/_astro/register.a.js\'></script><script>navigator.serviceWorker.register(\'/calories/sw.js\')</script><astro-island component-url=\'/_astro/app.a.js\' renderer-url=\'/_astro/renderer.a.js\'></astro-island><a href=\'/about/\'>Home</a></body></html>'],
    ['/_astro/app.a.js', 'app version a'], ['/_astro/shared.a.js', 'shared version a'],
    ['/_astro/lazy.a.js', 'lazy version a'], ['/_astro/renderer.a.js', 'renderer version a'],
    ['/_astro/register.a.js', 'registration version a'], ['/_astro/imported.svg', 'imported icon'],
    ['/_astro/calories.a.css', '@import "./theme.a.css";@font-face{src:url(../fonts/calories.woff2)}.x{background:url(data:image/png;base64,AA)}'],
    ['/_astro/theme.a.css', 'body{color:green}'], ['/fonts/calories.woff2', 'font bytes'],
    ['/about/index.html', 'not the app'], ['/_astro/blog.js', 'unrelated site chunk'],
    ['/calories/garmin-setup/index.html', 'online setup guide'], ['/calories/setup.sql', 'not an offline resource'],
    ...fixed.map(url => [url, `static ${url}`]),
  ]);
  for (const [url, bytes] of files) {
    const filename = path.join(directory, url);
    await mkdir(path.dirname(filename), { recursive: true });
    await writeFile(filename, bytes);
  }
  return { directory, files };
}

function cacheHarness(files) {
  const stores = new Map();
  let online = true;
  const fetched = [];
  const absolute = url => new URL(typeof url === 'string' ? url : url.url, ORIGIN).href;
  const network = async request => {
    const url = absolute(request);
    fetched.push(request);
    if (!online) throw new TypeError('Offline');
    const bytes = files.get(new URL(url).pathname);
    if (bytes === undefined) return new Response('Missing', { status: 404 });
    return new Response(bytes);
  };
  const caches = {
    async open(name) {
      if (!stores.has(name)) stores.set(name, new Map());
      const store = stores.get(name);
      return {
        async addAll(requests) {
          const additions = await Promise.all(requests.map(async request => {
            const response = await network(request);
            if (!response.ok) throw new TypeError('Precache failed');
            const integrity = `sha256-${createHash('sha256').update(Buffer.from(await response.clone().arrayBuffer())).digest('base64')}`;
            if (request.integrity !== integrity) throw new TypeError('Integrity mismatch');
            return [absolute(request), response];
          }));
          for (const [url, response] of additions) store.set(url, response);
        },
        async match(request) { return store.get(absolute(request))?.clone(); },
      };
    },
    async keys() { return [...stores.keys()]; },
    async delete(name) { return stores.delete(name); },
  };
  return { caches, stores, network, fetched, offline() { online = false; }, online() { online = true; } };
}

function worker(code, harness) {
  const handlers = new Map();
  let skipWaiting = 0, claim = 0;
  const self = {
    location: { origin: ORIGIN },
    addEventListener(type, callback) { handlers.set(type, callback); },
    skipWaiting() { skipWaiting++; }, clients: { claim() { claim++; } },
  };
  vm.runInNewContext(code, { self, caches: harness.caches, fetch: harness.network, Request, Response, URL, Set });
  return {
    async dispatch(type) {
      const pending = [];
      handlers.get(type)({ waitUntil(promise) { pending.push(promise); } });
      await Promise.all(pending);
    },
    async fetch(url, { mode = 'cors', method = 'GET', headers = {} } = {}) {
      let response;
      handlers.get('fetch')({ request: { url: new URL(url, ORIGIN).href, method, mode, headers: new Headers(headers) }, respondWith(promise) { response = promise; } });
      return response === undefined ? undefined : await response;
    },
    get takeoverCalls() { return { skipWaiting, claim }; },
  };
}

test('build manifest includes only Calories HTML, icons, manifest and transitive emitted assets', async t => {
  const { directory } = await fixture(t);
  const first = await buildCaloriesWorker(directory, graph);
  const urls = first.resources.map(resource => resource.url);
  for (const url of ['/calories/', ...fixed, ...graph.keys(), '/_astro/shared.a.js', '/_astro/lazy.a.js', '/_astro/imported.svg', '/_astro/calories.a.css', '/_astro/theme.a.css', '/fonts/calories.woff2']) assert.ok(urls.includes(url), url);
  for (const url of ['/about/', '/about/index.html', '/_astro/blog.js', '/calories/garmin-setup/index.html', '/calories/setup.sql', '/calories/sw.js']) assert.ok(!urls.includes(url), url);
  assert.ok(first.resources.every(resource => /^sha256-[A-Za-z0-9+/]{43}=$/.test(resource.integrity)));
  assert.equal(new Set(urls).size, urls.length);
  assert.deepEqual(await buildCaloriesWorker(directory, graph), first, 'identical output has one stable build version');
  await writeFile(path.join(directory, '_astro/lazy.a.js'), 'updated lazy chunk');
  const changed = await buildCaloriesWorker(directory, graph);
  assert.notEqual(changed.version, first.version, 'a transitive chunk change must invalidate the shell');
});

test('Astro integration captures real Rollup imports and emits its matching worker after the build', async t => {
  const { directory } = await fixture(t);
  const integration = caloriesPwa();
  let config;
  integration.hooks['astro:config:setup']({ updateConfig(value) { config = value; } });
  const plugin = config.vite.plugins[0];
  assert.equal(plugin.apply, 'build');
  assert.equal(plugin.generateBundle, undefined, 'pre-final Astro chunk names must not enter the cache manifest');
  plugin.writeBundle({}, {
    'app': { type: 'chunk', fileName: '_astro/app.a.js', imports: ['_astro/shared.a.js'], dynamicImports: ['_astro/lazy.a.js'], viteMetadata: { importedCss: new Set(['_astro/calories.a.css']), importedAssets: new Set(['_astro/imported.svg']) } },
  });
  await integration.hooks['astro:build:done']({ dir: pathToFileURL(`${directory}/`), logger: { info() {} } });
  const code = await readFile(path.join(directory, 'calories/sw.js'), 'utf8');
  assert.match(code, /_astro\/shared\.a\.js/);
  assert.match(code, /_astro\/lazy\.a\.js/);
  assert.match(code, /_astro\/imported\.svg/);
  assert.ok(!code.includes('/* CALORIES_PRECACHE_MANIFEST */ null'));
});

test('missing required build assets fail generation instead of publishing a partial offline shell', async t => {
  const { directory } = await fixture(t);
  await rm(path.join(directory, '_astro/shared.a.js'));
  await assert.rejects(buildCaloriesWorker(directory, graph), /ENOENT/);
});

test('successful install precaches the full version with integrity, then reopens the main page offline', async t => {
  const { directory, files } = await fixture(t);
  const manifest = await buildCaloriesWorker(directory, graph);
  const code = await readFile(path.join(directory, 'calories/sw.js'), 'utf8');
  const serverFiles = new Map(files); serverFiles.set('/calories/', files.get('/calories/index.html'));
  const harness = cacheHarness(serverFiles), sw = worker(code, harness);
  await sw.dispatch('install'); await sw.dispatch('activate');
  assert.equal(harness.stores.size, 1);
  assert.equal([...harness.stores.values()][0].size, manifest.resources.length);
  assert.ok(harness.fetched.every(request => request.cache === 'reload' && request.credentials === 'omit' && request.integrity.startsWith('sha256-')));
  harness.offline();
  const before = harness.fetched.length;
  for (const url of ['/calories/', '/calories/index.html', '/calories/?open=journal']) {
    assert.equal(await (await sw.fetch(url, { mode: 'navigate' })).text(), files.get('/calories/index.html'));
  }
  for (const url of ['/_astro/app.a.js', '/_astro/shared.a.js', '/_astro/lazy.a.js', '/_astro/calories.a.css', '/calories/icon-192.png']) {
    assert.equal(await (await sw.fetch(url)).text(), files.get(url));
  }
  assert.equal(harness.fetched.length, before, 'all shell requests are satisfied offline');
  assert.deepEqual(sw.takeoverCalls, { skipWaiting: 0, claim: 0 });
});

test('APIs, authentication, photo uploads, other pages and non-allowlisted requests bypass the worker', async t => {
  const { directory, files } = await fixture(t);
  await buildCaloriesWorker(directory, graph);
  const serverFiles = new Map(files); serverFiles.set('/calories/', files.get('/calories/index.html'));
  const harness = cacheHarness(serverFiles), sw = worker(await readFile(path.join(directory, 'calories/sw.js'), 'utf8'), harness);
  await sw.dispatch('install'); harness.offline();
  const before = harness.fetched.length;
  for (const [url, options] of [
    ['https://private.supabase.co/rest/v1/calories_data', {}],
    ['https://private.supabase.co/auth/v1/token', { method: 'POST' }],
    ['https://private.supabase.co/functions/v1/calories-food-photos', { method: 'POST' }],
    ['https://generativelanguage.googleapis.com/v1beta/models/gemini:generateContent', { method: 'POST' }],
    ['/calories/', { headers: { authorization: 'Bearer private-token' } }],
    ['/_astro/app.a.js', { headers: { range: 'bytes=0-100' } }],
    ['/_astro/app.a.js?token=private', {}], ['/_astro/blog.js', {}],
    ['/calories/private-photo.jpg', {}], ['/calories/setup.sql', {}],
    ['/calories/garmin-setup/', { mode: 'navigate' }], ['/about/', { mode: 'navigate' }],
    ['/calories/', { method: 'POST' }],
  ]) assert.equal(await sw.fetch(url, options), undefined, url);
  assert.equal(harness.fetched.length, before, 'the worker does not even fetch excluded traffic');
});

test('failed or integrity-mismatched updates leave the previous complete cache intact', async t => {
  const { directory, files } = await fixture(t);
  const oldManifest = await buildCaloriesWorker(directory, graph);
  const oldCode = await readFile(path.join(directory, 'calories/sw.js'), 'utf8');
  const serverFiles = new Map(files); serverFiles.set('/calories/', files.get('/calories/index.html'));
  const harness = cacheHarness(serverFiles), old = worker(oldCode, harness);
  await old.dispatch('install'); await old.dispatch('activate');
  await writeFile(path.join(directory, '_astro/lazy.a.js'), 'new lazy asset');
  const updated = await buildCaloriesWorker(directory, graph);
  const next = worker(await readFile(path.join(directory, 'calories/sw.js'), 'utf8'), harness);
  await assert.rejects(next.dispatch('install'), /Integrity mismatch/, 'an out-of-sync CDN cannot install mixed bytes');
  assert.deepEqual(await harness.caches.keys(), [`calories-shell-v1-${oldManifest.version}`]);
  assert.ok(!harness.stores.has(`calories-shell-v1-${updated.version}`));
  serverFiles.delete('/_astro/lazy.a.js');
  await assert.rejects(next.dispatch('install'), /Precache failed/, 'a missing resource also aborts the whole installation');
  assert.deepEqual(await harness.caches.keys(), [`calories-shell-v1-${oldManifest.version}`]);
  harness.offline();
  assert.equal(await (await old.fetch('/_astro/lazy.a.js')).text(), 'lazy version a');
});

test('new versions keep old clients coherent until activation and clean only old Calories caches', async t => {
  const { directory, files } = await fixture(t);
  const oldManifest = await buildCaloriesWorker(directory, graph);
  const serverFiles = new Map(files); serverFiles.set('/calories/', files.get('/calories/index.html'));
  const harness = cacheHarness(serverFiles), old = worker(await readFile(path.join(directory, 'calories/sw.js'), 'utf8'), harness);
  await old.dispatch('install'); await old.dispatch('activate');
  await harness.caches.open('another-app-cache');
  const newHtml = files.get('/calories/index.html').replace('Home', 'New build');
  await writeFile(path.join(directory, 'calories/index.html'), newHtml);
  await writeFile(path.join(directory, '_astro/app.a.js'), 'app version b');
  serverFiles.set('/calories/', newHtml); serverFiles.set('/_astro/app.a.js', 'app version b');
  const nextManifest = await buildCaloriesWorker(directory, graph);
  const next = worker(await readFile(path.join(directory, 'calories/sw.js'), 'utf8'), harness);
  await next.dispatch('install');
  assert.equal(await (await old.fetch('/calories/', { mode: 'navigate' })).text(), files.get('/calories/index.html'));
  assert.equal(await (await old.fetch('/_astro/app.a.js')).text(), 'app version a');
  assert.deepEqual(next.takeoverCalls, { skipWaiting: 0, claim: 0 });
  assert.ok(harness.stores.has(`calories-shell-v1-${oldManifest.version}`));
  await next.dispatch('activate'); // Browser dispatches this after all old clients close.
  assert.ok(!harness.stores.has(`calories-shell-v1-${oldManifest.version}`));
  assert.ok(harness.stores.has(`calories-shell-v1-${nextManifest.version}`));
  assert.ok(harness.stores.has('another-app-cache'));
  harness.offline();
  assert.equal(await (await next.fetch('/calories/', { mode: 'navigate' })).text(), newHtml);
  assert.equal(await (await next.fetch('/_astro/app.a.js')).text(), 'app version b');
});

test('registration is production-only, secure and scoped, with HTTP cache bypass for update checks', async () => {
  assert.equal(await registerCaloriesOfflineShell(false), undefined, 'development never touches the browser registration API');
  const calls = [];
  const registration = { active: {} };
  const browser = { location: { pathname: '/calories/' }, isSecureContext: true, navigator: { serviceWorker: { async register(...args) { calls.push(args); return registration; } } } };
  assert.equal(await registerCaloriesOfflineShell(true, browser), registration);
  assert.deepEqual(calls, [['/calories/sw.js', { scope: '/calories/', updateViaCache: 'none' }]]);
  for (const environment of [{ ...browser, isSecureContext: false }, { ...browser, location: { pathname: '/' } }, { ...browser, location: { pathname: '/calories-setup/' } }, { ...browser, navigator: {} }]) {
    assert.equal(await registerCaloriesOfflineShell(true, environment), undefined);
  }
  assert.equal(calls.length, 1);
  browser.navigator.serviceWorker.register = async () => { throw new Error('Browser storage unavailable'); };
  assert.equal(await registerCaloriesOfflineShell(true, browser), undefined, 'a PWA installation failure does not break online use');
});
