import test from 'node:test';
import assert from 'node:assert/strict';
import { createClient } from '../../src/lib/calories/client.ts';

const photo = { mimeType: 'image/png', data: 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aS4kAAAAASUVORK5CYII=' };
const input = { photo, description: 'Chicken and rice, one plate' };
const draft = { name: 'Chicken and rice', protein: 45, carbs: 65, fat: 18, fiber: 5, notes: ['Approximate amounts for the pictured portion.'] };
function harness(handler) {
  const storage = () => { const values = new Map(); return { getItem: key => values.get(key) ?? null, setItem: (key, value) => values.set(key, value), removeItem: key => values.delete(key) }; };
  const calls = [];
  const client = createClient({ url: 'https://example.supabase.co', key: 'sb_publishable_photo_test_key', localStorage: storage(), sessionStorage: storage(), now: () => Date.UTC(2026, 9, 16), fetch: async (url, init) => {
    calls.push({ url, init });
    if (url.includes('grant_type=password')) return Response.json({ access_token: 'initial-token', refresh_token: 'refresh-token', expires_in: 3600, user: { id: 'owner', email: 'owner@example.test' } });
    return handler(url, init);
  } });
  return { client, calls };
}

test('meal photo transport uses the existing protected endpoint with an explicit single-meal request', async () => {
  const h = harness((url, init) => {
    assert.equal(url, 'https://example.supabase.co/functions/v1/calories-food-photos');
    const headers = new Headers(init.headers);
    assert.equal(headers.get('authorization'), 'Bearer initial-token');
    assert.equal(headers.get('apikey'), 'sb_publishable_photo_test_key');
    assert.equal(headers.get('content-type'), 'application/json');
    assert.equal(init.cache, 'no-store');
    assert.deepEqual(JSON.parse(init.body), { mode: 'meal', ...input });
    return Response.json({ ...draft, fat: 0, fiber: null });
  });
  await h.client.signIn('owner@example.test', 'password', false);
  const result = await h.client.estimateMealPhoto(input);
  assert.equal(result.protein, 45);
  assert.equal(result.fat, 0);
  assert.equal(result.fiber, null);
  assert.equal(h.calls.length, 2, 'no journal write occurs while obtaining an estimate');
});

test('cancelling a meal estimate reaches the upload request and does not retry it', async () => {
  let begin;
  const started = new Promise(resolve => { begin = resolve; });
  const h = harness((_url, init) => new Promise((_resolve, reject) => {
    begin();
    init.signal.addEventListener('abort', () => reject(init.signal.reason), { once: true });
  }));
  await h.client.signIn('owner@example.test', 'password', false);
  const controller = new AbortController();
  const pending = h.client.estimateMealPhoto(input, controller.signal);
  await started;
  controller.abort();
  await assert.rejects(pending, error => error.name === 'AbortError');
  assert.equal(h.calls.length, 2);
});

test('a meal provider rate limit is actionable and cannot retry a paid request as a sign-in failure', async () => {
  const h = harness(() => Response.json({ error: 'Meal photo estimation has reached its usage limit.' }, { status: 429 }));
  await h.client.signIn('owner@example.test', 'password', false);
  await assert.rejects(h.client.estimateMealPhoto(input), /Meal photo estimation has reached its usage limit/);
  assert.equal(h.calls.length, 2);
});

test('an expired caller token refreshes once for a meal-photo request', async () => {
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
  assert.equal((await h.client.estimateMealPhoto(input)).carbs, 65);
  assert.equal(attempts, 2);
  assert.equal(refreshes, 1);
});

test('unsigned calls, invalid meal images and invalid nutrient outputs cannot fill the form', async () => {
  const h = harness(() => Response.json({ ...draft, protein: '45' }));
  await assert.rejects(h.client.estimateMealPhoto(input), /session has ended/);
  assert.equal(h.calls.length, 0);
  await h.client.signIn('owner@example.test', 'password', false);
  await assert.rejects(h.client.estimateMealPhoto({ photo: { mimeType: 'image/svg+xml', data: 'PHN2Zz4=' } }), /JPEG/);
  assert.equal(h.calls.length, 1);
  await assert.rejects(h.client.estimateMealPhoto(input));
});
