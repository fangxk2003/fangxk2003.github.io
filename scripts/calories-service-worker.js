'use strict';

// Replaced with exact URLs and SHA-256 integrity values from the completed build.
const MANIFEST = /* CALORIES_PRECACHE_MANIFEST */ null;
const CACHE_PREFIX = 'calories-shell-v1-';
const CACHE_NAME = CACHE_PREFIX + MANIFEST.version;
const SHELL_URL = new URL('/calories/', self.location.origin).href;
const ASSET_URLS = new Set(MANIFEST.resources.map(resource => new URL(resource.url, self.location.origin).href));

self.addEventListener('install', event => {
  event.waitUntil((async () => {
    try {
      const cache = await caches.open(CACHE_NAME);
      // addAll is atomic; integrity rejects a partial or mismatched deployment.
      await cache.addAll(MANIFEST.resources.map(resource => new Request(new URL(resource.url, self.location.origin), {
        cache: 'reload', credentials: 'omit', integrity: resource.integrity,
      })));
    } catch (error) {
      await caches.delete(CACHE_NAME);
      throw error;
    }
  })());
  // Let the normal waiting lifecycle keep existing windows on their current build.
});

self.addEventListener('activate', event => {
  event.waitUntil((async () => {
    const names = await caches.keys();
    await Promise.all(names.filter(name => name.startsWith(CACHE_PREFIX) && name !== CACHE_NAME).map(name => caches.delete(name)));
  })());
  // No clients.claim(): a loaded document and its assets must stay on one build.
});

self.addEventListener('fetch', event => {
  const request = event.request;
  if (request.method !== 'GET' || request.headers.has('authorization') || request.headers.has('range')) return;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;
  const navigation = request.mode === 'navigate' && (url.pathname === '/calories/' || url.pathname === '/calories/index.html');
  if (!navigation && (url.search || !ASSET_URLS.has(url.href))) return;
  event.respondWith((async () => {
    const cache = await caches.open(CACHE_NAME);
    const response = await cache.match(navigation ? SHELL_URL : url.href);
    // Never fill this cache at runtime, especially with account data or uploads.
    return response ?? fetch(request);
  })());
});
