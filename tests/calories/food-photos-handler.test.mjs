import test from 'node:test';
import assert from 'node:assert/strict';
import { createFoodPhotoHandler } from '../../supabase/functions/calories-food-photos/handler.ts';
import { MAX_PHOTO_REQUEST_BYTES } from '../../src/lib/calories/food-photos.ts';

const ORIGIN = 'https://fangxk2003.github.io';
const SUPABASE_URL = 'https://storage.example.test';
const PUBLIC_KEY = 'sb_publishable_test_public_key';
const PROVIDER_KEY = 'test-provider-secret-must-not-appear';
const TOKEN = 'test-owner-session-token';
const MODEL = 'test-vision-model';
const PHOTO = {
  mimeType: 'image/png',
  data: 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aS4kAAAAASUVORK5CYII=',
};
const PHOTOS = { product: PHOTO, nutrition: PHOTO };
const DRAFT = {
  name: 'Beans', baseQuantity: 100, unit: 'g', state: 'As served',
  protein: 5, carbs: 12, fat: 0, fiber: null,
  estimatedFields: [], notes: [],
};

function upload(body = JSON.stringify(PHOTOS), extraHeaders = {}) {
  const headers = new Headers({ origin: ORIGIN, authorization: `Bearer ${TOKEN}`, 'Content-Type': 'application/json' });
  for (const [key, value] of Object.entries(extraHeaders)) {
    if (value === null) headers.delete(key);
    else headers.set(key, value);
  }
  return new Request('https://functions.example.test/calories-food-photos', {
    method: 'POST', headers, body,
    ...(body instanceof ReadableStream ? { duplex: 'half' } : {}),
  });
}

function unreadableBody(request) {
  const state = { reads: 0 };
  Object.defineProperty(request, 'body', { get() { state.reads++; throw new Error('Body must not be read before access is granted.'); } });
  return state;
}

function harness({ access = true, accessStatus = 200, authError, apiKey = PROVIDER_KEY, extract } = {}) {
  const calls = { auth: [], extract: [], order: [] };
  const handler = createFoodPhotoHandler({
    supabaseUrl: SUPABASE_URL, publicKey: PUBLIC_KEY, apiKey, model: MODEL,
    fetch: async (url, init) => {
      calls.order.push('auth');
      calls.auth.push({ url, init });
      assert.equal(url, `${SUPABASE_URL}/rest/v1/rpc/calories_access_allowed`, 'no journal table or other upstream is contacted');
      assert.equal(init.method, 'POST');
      assert.equal(init.body, '{}');
      const headers = new Headers(init.headers);
      assert.equal(headers.get('apikey'), PUBLIC_KEY);
      assert.equal(headers.get('authorization'), `Bearer ${TOKEN}`);
      assert.equal(init.cache, 'no-store');
      if (authError) throw authError;
      return Response.json(access, { status: accessStatus });
    },
    extract: async (input, options) => {
      calls.order.push('extract');
      calls.extract.push({ input, options });
      return extract ? await extract(input, options) : structuredClone(DRAFT);
    },
  });
  return { handler, calls };
}

async function assertFailure(response, status, code) {
  assert.equal(response.status, status);
  assert.equal(response.headers.get('cache-control'), 'no-store');
  const text = await response.text();
  assert.ok(!text.includes(PROVIDER_KEY));
  assert.ok(!text.includes(TOKEN));
  const body = JSON.parse(text);
  assert.equal(body.code, code);
  assert.equal(typeof body.error, 'string');
  return body;
}

test('missing or malformed caller credentials are denied without reading photos or contacting an upstream', async () => {
  for (const authorization of [null, '', 'Basic credentials', 'Bearer ', 'Bearer one two', `Bearer ${'a'.repeat(4096)}`]) {
    const { handler, calls } = harness();
    const request = upload(undefined, { authorization, apikey: PUBLIC_KEY });
    const body = unreadableBody(request);
    await assertFailure(await handler(request), 401, 'session_required');
    assert.equal(body.reads, 0);
    assert.deepEqual(calls.order, []);
  }
});

test('wrong owners and revoked or expired sessions cannot read photos or trigger inference', async () => {
  for (const [scenario, access] of [['wrong owner', false], ['revoked session', false], ['expired session', false]]) {
    const { handler, calls } = harness({ access });
    const request = upload();
    const body = unreadableBody(request);
    const response = await handler(request);
    await assertFailure(response, 403, 'access_denied');
    assert.equal(body.reads, 0, scenario);
    assert.equal(calls.auth.length, 1, scenario);
    assert.equal(calls.extract.length, 0, scenario);
  }
});

test('the owner access RPC must return the exact boolean true', async () => {
  for (const access of ['true', { allowed: true }, [true], null, 1]) {
    const { handler, calls } = harness({ access });
    const request = upload();
    const body = unreadableBody(request);
    await assertFailure(await handler(request), 403, 'access_denied');
    assert.equal(body.reads, 0);
    assert.equal(calls.extract.length, 0);
  }
});

test('auth RPC errors deny inference without exposing upstream details', async () => {
  for (const [accessStatus, expectedStatus, code] of [[401, 401, 'session_required'], [403, 403, 'session_check_failed'], [500, 503, 'session_check_failed']]) {
    const { handler, calls } = harness({ accessStatus, access: { message: `private details ${PROVIDER_KEY} ${TOKEN}` } });
    const request = upload();
    const body = unreadableBody(request);
    await assertFailure(await handler(request), expectedStatus, code);
    assert.equal(body.reads, 0);
    assert.equal(calls.extract.length, 0);
  }
});

test('an auth network failure does not fall through to the provider', async () => {
  const { handler, calls } = harness({ authError: new Error(`network details ${PROVIDER_KEY}`) });
  const response = await handler(upload());
  await assertFailure(response, 502, 'reader_failed');
  assert.equal(calls.extract.length, 0);
});

test('authorized photos produce a review draft, not a journal write or a caller-controlled provider request', async () => {
  const { handler, calls } = harness();
  const payload = { ...PHOTOS, apiKey: 'caller-key', model: 'caller-model', endpoint: 'https://attacker.example.test', userId: 'another-owner', journal: { entries: [] } };
  const before = structuredClone(payload);
  const response = await handler(upload(JSON.stringify(payload), { apikey: 'caller-controlled-public-key' }));
  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), DRAFT);
  assert.equal(response.headers.get('access-control-allow-origin'), ORIGIN);
  assert.equal(response.headers.get('cache-control'), 'no-store');
  assert.deepEqual(calls.order, ['auth', 'extract']);
  assert.deepEqual(calls.extract[0].input, PHOTOS);
  assert.equal(calls.extract[0].options.apiKey, PROVIDER_KEY);
  assert.equal(calls.extract[0].options.model, MODEL);
  assert.ok(calls.extract[0].options.signal instanceof AbortSignal);
  assert.equal(calls.extract[0].options.signal.aborted, false);
  assert.deepEqual(payload, before);
});

test('missing provider setup returns 503 only after checking owner access', async () => {
  const { handler, calls } = harness({ apiKey: '' });
  const request = upload();
  const body = unreadableBody(request);
  await assertFailure(await handler(request), 503, 'not_configured');
  assert.deepEqual(calls.order, ['auth']);
  assert.equal(body.reads, 0);
});

test('both photos, valid formats, and JSON request bodies are required before inference', async () => {
  for (const body of ['{broken json', 'null', '[]', '{}', JSON.stringify({ product: PHOTO }), JSON.stringify({ product: { ...PHOTO, mimeType: 'image/svg+xml' }, nutrition: PHOTO }), JSON.stringify({ product: { ...PHOTO, data: 'dGV4dA==' }, nutrition: PHOTO })]) {
    const { handler, calls } = harness();
    await assertFailure(await handler(upload(body)), 400, 'invalid_photos');
    assert.equal(calls.auth.length, 1);
    assert.equal(calls.extract.length, 0);
  }
  const { handler, calls } = harness();
  await assertFailure(await handler(upload(undefined, { 'Content-Type': 'text/plain' })), 415, 'invalid_content_type');
  assert.deepEqual(calls.order, ['auth']);
});

test('the declared body limit rejects oversized requests before reading photos', async () => {
  const { handler, calls } = harness();
  const request = upload(undefined, { 'Content-Length': String(MAX_PHOTO_REQUEST_BYTES + 1) });
  const body = unreadableBody(request);
  await assertFailure(await handler(request), 413, 'photos_too_large');
  assert.equal(body.reads, 0);
  assert.deepEqual(calls.order, ['auth']);
});

test('streaming byte limits cannot be bypassed by absent or false Content-Length', async () => {
  for (const declaredLength of [null, '1']) {
    const { handler, calls } = harness();
    let sent = 0, cancelled = false;
    const stream = new ReadableStream({
      pull(controller) {
        const size = Math.min(2 * 1024 * 1024, MAX_PHOTO_REQUEST_BYTES + 1 - sent);
        if (size <= 0) { controller.close(); return; }
        controller.enqueue(new Uint8Array(size).fill(32));
        sent += size;
      },
      cancel() { cancelled = true; },
    }, { highWaterMark: 0 });
    await assertFailure(await handler(upload(stream, { 'Content-Length': declaredLength })), 413, 'photos_too_large');
    assert.equal(cancelled, true);
    assert.equal(calls.extract.length, 0);
  }
});

test('CORS preflight supports only the production and explicit local QA origins without authentication', async () => {
  for (const origin of [ORIGIN, 'http://127.0.0.1:4321', 'http://localhost:4321']) {
    const { handler, calls } = harness();
    const response = await handler(new Request('https://functions.example.test/calories-food-photos', { method: 'OPTIONS', headers: { origin, 'Access-Control-Request-Method': 'POST', 'Access-Control-Request-Headers': 'authorization,apikey,content-type' } }));
    assert.equal(response.status, 204);
    assert.equal(response.headers.get('access-control-allow-origin'), origin);
    assert.match(response.headers.get('vary'), /\borigin\b/i);
    assert.match(response.headers.get('access-control-allow-methods'), /POST/);
    assert.match(response.headers.get('access-control-allow-methods'), /OPTIONS/);
    for (const header of ['authorization', 'apikey', 'content-type']) assert.ok(response.headers.get('access-control-allow-headers').includes(header));
    assert.equal(response.headers.get('access-control-allow-credentials'), null);
    assert.deepEqual(calls.order, []);
  }
});

test('lookalike origins and unsupported methods cannot trigger owner checks or provider calls', async () => {
  for (const origin of ['https://fangxk2003.github.io.attacker.example.test', 'http://fangxk2003.github.io', 'https://localhost:4321', 'null']) {
    const { handler, calls } = harness();
    const response = await handler(upload(undefined, { origin }));
    await assertFailure(response, 403, 'origin_denied');
    assert.equal(response.headers.get('access-control-allow-origin'), null);
    assert.deepEqual(calls.order, []);
  }
  for (const method of ['GET', 'PUT', 'DELETE']) {
    const { handler, calls } = harness();
    const response = await handler(new Request('https://functions.example.test/calories-food-photos', { method, headers: { origin: ORIGIN } }));
    await assertFailure(response, 405, 'method_not_allowed');
    assert.match(response.headers.get('allow'), /POST/);
    assert.deepEqual(calls.order, []);
  }
});

test('provider exceptions never become caller 401s or reveal secrets or raw upstream errors', async () => {
  const { handler } = harness({ extract: async () => { throw Object.assign(new Error(`raw provider authentication failure: ${PROVIDER_KEY} ${TOKEN}`), { status: 401 }); } });
  const body = await assertFailure(await handler(upload()), 502, 'reader_failed');
  assert.ok(!body.error.includes('raw provider authentication failure'));
});

test('provider output is normalized before returning a draft', async () => {
  const { handler } = harness({ extract: async () => ({ ...DRAFT, protein: -1 }) });
  await assertFailure(await handler(upload()), 502, 'reader_failed');
});

test('cancelling an in-flight extraction aborts its provider signal and returns no draft', async () => {
  let markStarted;
  const started = new Promise(resolve => { markStarted = resolve; });
  let providerSignal;
  const { handler } = harness({ extract: async (_input, { signal }) => {
    providerSignal = signal;
    return await new Promise((_resolve, reject) => {
      signal.addEventListener('abort', () => reject(new Error(`cancelled ${PROVIDER_KEY}`)), { once: true });
      markStarted();
    });
  } });
  const controller = new AbortController();
  const request = new Request(upload(), { signal: controller.signal });
  const pending = handler(request);
  await started;
  controller.abort();
  await assertFailure(await pending, 504, 'reader_timeout');
  assert.equal(providerSignal.aborted, true);
});
