import test from 'node:test';
import assert from 'node:assert/strict';
import { createFoodPhotoHandler } from '../../supabase/functions/calories-food-photos/handler.ts';
import { PhotoProviderError } from '../../supabase/functions/calories-food-photos/gemini.ts';
import { MAX_PHOTO_REQUEST_BYTES } from '../../src/lib/calories/food-photos.ts';
import { normalizeMealPhotoDraft, normalizeMealPhotoRequest } from '../../src/lib/calories/meal-photos.ts';

const ORIGIN = 'https://fangxk2003.github.io';
const TOKEN = 'meal-owner-token';
const API_KEY = 'private-meal-provider-key';
const PHOTO = { mimeType: 'image/png', data: 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aS4kAAAAASUVORK5CYII=' };
const MEAL = { name: 'Rice and chicken', protein: 25, carbs: 40, fat: 15, fiber: 3, notes: ['Portion size is an estimate.'] };
const LABEL = { name: 'Beans', baseQuantity: 100, unit: 'g', state: 'As served', protein: 5, carbs: 12, fat: 0, fiber: 4, estimatedFields: [], notes: [] };
const payload = () => ({ mode: 'meal', photo: PHOTO, description: 'Lunch plate' });

function upload(value = payload(), additionalHeaders = {}) {
  const headers = new Headers({ origin: ORIGIN, authorization: `Bearer ${TOKEN}`, 'Content-Type': 'application/json' });
  for (const [name, value] of Object.entries(additionalHeaders)) {
    if (value === null) headers.delete(name);
    else headers.set(name, value);
  }
  return new Request('https://edge.example.test/calories-food-photos', { method: 'POST', headers, body: JSON.stringify(value) });
}
function forbidBodyRead(request) {
  let reads = 0;
  Object.defineProperty(request, 'body', { get() { reads++; throw new Error('Photos must not be read yet.'); } });
  return () => reads;
}
function harness({ access = true, status = 200, apiKey = API_KEY, estimate } = {}) {
  const calls = { auth: 0, estimate: [], extract: [], order: [] };
  const handler = createFoodPhotoHandler({
    supabaseUrl: 'https://storage.example.test', publicKey: 'sb_publishable_test_meal_key', apiKey, model: 'meal-test-model',
    fetch: async (url, init) => {
      calls.auth++; calls.order.push('auth');
      assert.equal(url, 'https://storage.example.test/rest/v1/rpc/calories_access_allowed');
      assert.equal(init.method, 'POST');
      assert.equal(init.body, '{}');
      assert.equal(new Headers(init.headers).get('authorization'), `Bearer ${TOKEN}`);
      assert.equal(new Headers(init.headers).get('apikey'), 'sb_publishable_test_meal_key');
      return Response.json(access, { status });
    },
    extract: async (input, options) => {
      calls.extract.push({ input, options }); calls.order.push('extract');
      return structuredClone(LABEL);
    },
    estimate: async (input, options) => {
      calls.estimate.push({ input, options }); calls.order.push('estimate');
      return estimate ? await estimate(input, options) : structuredClone(MEAL);
    },
  });
  return { handler, calls };
}
async function errorBody(response, status, code) {
  assert.equal(response.status, status);
  assert.equal(response.headers.get('cache-control'), 'no-store');
  const text = await response.text();
  assert.ok(!text.includes(API_KEY));
  assert.ok(!text.includes(TOKEN));
  assert.ok(!text.includes(PHOTO.data));
  const body = JSON.parse(text);
  assert.equal(body.code, code);
  return body;
}

test('meal photo normalization trims description and returns only the allowed request fields', () => {
  assert.deepEqual(normalizeMealPhotoRequest({ ...payload(), description: '  Lunch plate  ', apiKey: 'caller-key', model: 'caller-model', endpoint: 'https://attacker.example.test' }), { photo: PHOTO, description: 'Lunch plate' });
  assert.deepEqual(normalizeMealPhotoRequest({ photo: PHOTO, description: '  ' }), { photo: PHOTO });
  assert.equal(normalizeMealPhotoRequest({ photo: PHOTO, description: 'x'.repeat(300) }).description.length, 300);
  for (const description of [null, false, 42, {}, 'x'.repeat(301)]) assert.throws(() => normalizeMealPhotoRequest({ photo: PHOTO, description }));
});

test('meal drafts keep null distinct from zero and apply strict text, nutrient and 4/4/9 bounds', () => {
  const partial = { name: null, protein: null, carbs: 0, fat: null, fiber: 0, notes: [] };
  assert.deepEqual(normalizeMealPhotoDraft(partial), partial);
  const maximum = { name: 'x'.repeat(160), protein: 10000, carbs: 3750, fat: 5000, fiber: 10000, notes: Array(3).fill('x'.repeat(200)) };
  assert.deepEqual(normalizeMealPhotoDraft(maximum), maximum, '100,000 kcal is inclusive and fibre adds no energy');
  const invalid = [
    { ...MEAL, unexpected: true }, { ...MEAL, protein: -1 }, { ...MEAL, protein: 10001 },
    { ...MEAL, carbs: '40' }, { ...MEAL, fat: NaN }, { ...MEAL, fiber: Infinity },
    { ...MEAL, name: ' ' }, { ...MEAL, name: 'x'.repeat(161) },
    { ...MEAL, notes: ['x'.repeat(201)] }, { ...MEAL, notes: Array(4).fill('note') },
    { ...MEAL, notes: Array(1) }, { ...MEAL, protein: 10000, carbs: 10000, fat: 10000 },
  ];
  const missingField = { ...MEAL }; delete missingField.fiber; invalid.push(missingField);
  for (const draft of invalid) assert.throws(() => normalizeMealPhotoDraft(draft));
  assert.deepEqual(normalizeMealPhotoDraft({ ...MEAL, name: '  Rice and chicken  ', notes: ['  Review portions.  '] }), { ...MEAL, notes: ['Review portions.'] });
});

test('missing, denied or revoked caller access prevents reading the meal photo or invoking either provider route', async () => {
  for (const check of [
    { authorization: null, access: true, status: 200, expected: 401, code: 'session_required', authCalls: 0 },
    { access: false, status: 200, expected: 403, code: 'access_denied', authCalls: 1 },
    { access: 'true', status: 200, expected: 403, code: 'access_denied', authCalls: 1 },
    { access: { details: API_KEY }, status: 401, expected: 401, code: 'session_required', authCalls: 1 },
  ]) {
    const { handler, calls } = harness(check);
    const request = upload(undefined, check.authorization === null ? { authorization: null } : {});
    const reads = forbidBodyRead(request);
    await errorBody(await handler(request), check.expected, check.code);
    assert.equal(reads(), 0);
    assert.equal(calls.auth, check.authCalls);
    assert.equal(calls.estimate.length, 0);
    assert.equal(calls.extract.length, 0);
  }
});

test('an authorized meal request invokes only the estimator and returns a normalized review draft', async () => {
  const { handler, calls } = harness();
  const response = await handler(upload({ ...payload(), description: '  Lunch plate  ', apiKey: 'caller-key', model: 'caller-model', userId: 'another-owner' }));
  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), MEAL);
  assert.equal(response.headers.get('access-control-allow-origin'), ORIGIN);
  assert.equal(response.headers.get('cache-control'), 'no-store');
  assert.deepEqual(calls.order, ['auth', 'estimate']);
  assert.deepEqual(calls.estimate[0].input, { photo: PHOTO, description: 'Lunch plate' });
  assert.equal(calls.estimate[0].options.apiKey, API_KEY);
  assert.equal(calls.estimate[0].options.model, 'meal-test-model');
  assert.ok(calls.estimate[0].options.signal instanceof AbortSignal);
  assert.equal(calls.extract.length, 0, 'there is no fallback to label extraction or journal mutation');
});

test('invalid meal modes or inputs cannot trigger any inference', async () => {
  for (const mode of ['labels', 'food', '', null, false, 1, {}]) {
    const { handler, calls } = harness();
    await errorBody(await handler(upload({ mode, product: PHOTO, nutrition: PHOTO, photo: PHOTO })), 400, 'invalid_mode');
    assert.deepEqual(calls.order, ['auth']);
  }
  for (const value of [
    { photo: PHOTO }, { mode: 'meal' }, { mode: 'meal', product: PHOTO, nutrition: PHOTO },
    { ...payload(), photo: { ...PHOTO, mimeType: 'image/svg+xml' } },
    { ...payload(), photo: { ...PHOTO, data: 'dGV4dA==' } },
    { ...payload(), description: 'x'.repeat(301) }, { ...payload(), description: null },
  ]) {
    const { handler, calls } = harness();
    await errorBody(await handler(upload(value)), 400, 'invalid_photos');
    assert.deepEqual(calls.order, ['auth']);
  }
});

test('the existing two-photo package reader stays on its original extraction path', async () => {
  const { handler, calls } = harness();
  const response = await handler(upload({ product: PHOTO, nutrition: PHOTO }));
  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), LABEL);
  assert.deepEqual(calls.order, ['auth', 'extract']);
  assert.deepEqual(calls.extract[0].input, { product: PHOTO, nutrition: PHOTO });
  assert.equal(calls.estimate.length, 0);
});

test('meal uploads retain the shared body limit and require provider setup after owner authorization', async () => {
  const large = harness();
  const request = upload(undefined, { 'Content-Length': String(MAX_PHOTO_REQUEST_BYTES + 1) });
  const reads = forbidBodyRead(request);
  await errorBody(await large.handler(request), 413, 'photos_too_large');
  assert.equal(reads(), 0);
  assert.deepEqual(large.calls.order, ['auth']);
  const missingKey = harness({ apiKey: '' });
  await errorBody(await missingKey.handler(upload()), 503, 'not_configured');
  assert.deepEqual(missingKey.calls.order, ['auth']);
});

test('unsafe estimator output is rejected instead of leaking fields or returning invalid nutrition', async () => {
  for (const output of [{ ...MEAL, apiKey: API_KEY }, { ...MEAL, protein: -1 }, { ...MEAL, protein: 10000, carbs: 10000, fat: 10000 }]) {
    const { handler, calls } = harness({ estimate: async () => output });
    await errorBody(await handler(upload()), 502, 'reader_failed');
    assert.equal(calls.estimate.length, 1);
    assert.equal(calls.extract.length, 0);
  }
});

test('meal provider failures are redacted and rate limits never retry or switch readers', async () => {
  for (const [error, status, code] of [
    [new PhotoProviderError('provider_rate_limited', 'Meal photo reading is busy. Try later.', 429), 429, 'provider_rate_limited'],
    [new PhotoProviderError('provider_not_configured', 'Meal photo reading is temporarily unavailable.', 401), 502, 'provider_not_configured'],
    [new Error(`raw upstream failure ${API_KEY} ${TOKEN}`), 502, 'reader_failed'],
  ]) {
    const { handler, calls } = harness({ estimate: async () => { throw error; } });
    const body = await errorBody(await handler(upload()), status, code);
    assert.ok(!body.error.includes('raw upstream failure'));
    assert.notEqual(status, 401, 'a provider failure cannot trigger browser authentication refresh/replay');
    assert.deepEqual(calls.order, ['auth', 'estimate']);
  }
});
