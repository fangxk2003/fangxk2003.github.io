import test from 'node:test';
import assert from 'node:assert/strict';
import { createClient } from '../../src/lib/calories/client.ts';

const photo = { mimeType: 'image/png', data: 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aS4kAAAAASUVORK5CYII=' };
const photos = { product: photo, nutrition: photo };
const draft = { name: 'Oats', baseQuantity: 100, unit: 'g', state: 'As served', protein: 13, carbs: 60, fat: 7, fiber: null, estimatedFields: [], notes: [] };
function harness(handler) {
  const storage = () => { const values = new Map(); return { getItem: key => values.get(key) ?? null, setItem: (key, value) => values.set(key, value), removeItem: key => values.delete(key) }; };
  const calls = [];
  const client = createClient({ url: 'https://example.supabase.co', key: 'sb_publishable_photo_test_key', localStorage: storage(), sessionStorage: storage(), now: () => Date.UTC(2026, 9, 15), fetch: async (url, init) => {
    calls.push({ url, init });
    if (url.includes('grant_type=password')) return Response.json({ access_token: 'initial-token', refresh_token: 'refresh-token', expires_in: 3600, user: { id: 'owner', email: 'owner@example.test' } });
    return handler(url, init);
  } });
  return { client, calls };
}

test('photo transport sends only the two images to the authenticated function, then returns a nullable draft', async () => {
  const h = harness((url, init) => {
    assert.equal(url, 'https://example.supabase.co/functions/v1/calories-food-photos');
    const headers = new Headers(init.headers);
    assert.equal(headers.get('authorization'), 'Bearer initial-token');
    assert.equal(headers.get('apikey'), 'sb_publishable_photo_test_key');
    assert.equal(headers.get('content-type'), 'application/json');
    assert.deepEqual(JSON.parse(init.body), photos);
    assert.equal(init.cache, 'no-store');
    return Response.json({ ...draft, fat: 0, fiber: null, estimatedFields: [] });
  });
  await h.client.signIn('owner@example.test', 'password', false);
  const result = await h.client.readFoodPhotos(photos);
  assert.equal(result.fat, 0);
  assert.equal(result.fiber, null);
  assert.equal(h.calls.length, 2);
});

test('photo cancellation reaches the network request rather than being replaced by the storage timeout', async () => {
  let began;
  const started = new Promise(resolve => { began = resolve; });
  const h = harness((_url, init) => new Promise((_resolve, reject) => {
    began();
    init.signal.addEventListener('abort', () => reject(init.signal.reason), { once: true });
  }));
  await h.client.signIn('owner@example.test', 'password', false);
  const controller = new AbortController();
  const reading = h.client.readFoodPhotos(photos, controller.signal);
  await started;
  controller.abort();
  await assert.rejects(reading, error => error.name === 'AbortError');
});

test('provider usage errors stay actionable and do not trigger an auth refresh or second paid request', async () => {
  const h = harness(() => Response.json({ error: 'Photo reading has reached its usage limit.' }, { status: 429 }));
  await h.client.signIn('owner@example.test', 'password', false);
  await assert.rejects(h.client.readFoodPhotos(photos), /Photo reading has reached its usage limit/);
  assert.equal(h.calls.length, 2);
});

test('an expired caller token is refreshed once before retrying the protected photo endpoint', async () => {
  let attempts = 0, refreshes = 0;
  const h = harness((url, init) => {
    if (url.includes('grant_type=refresh_token')) {
      refreshes++;
      return Response.json({ access_token: 'rotated-token', refresh_token: 'rotated-refresh', expires_in: 3600, user: { id: 'owner', email: 'owner@example.test' } });
    }
    if (++attempts === 1) return Response.json({ error: 'Session expired' }, { status: 401 });
    assert.equal(new Headers(init.headers).get('authorization'), 'Bearer rotated-token');
    return Response.json(draft);
  });
  await h.client.signIn('owner@example.test', 'password', false);
  assert.equal((await h.client.readFoodPhotos(photos)).name, 'Oats');
  assert.equal(attempts, 2);
  assert.equal(refreshes, 1);
});

test('unsigned uploads and malformed image/draft payloads cannot silently fill the form', async () => {
  const h = harness(() => Response.json({ ...draft, fiber: -10 }));
  await assert.rejects(h.client.readFoodPhotos(photos), /session has ended/);
  assert.equal(h.calls.length, 0);
  await h.client.signIn('owner@example.test', 'password', false);
  await assert.rejects(h.client.readFoodPhotos({ product: { mimeType: 'image/svg+xml', data: 'PHN2Zz4=' }, nutrition: photo }), /JPEG/);
  assert.equal(h.calls.length, 1);
  await assert.rejects(h.client.readFoodPhotos(photos), /invalid quantity/);
});
