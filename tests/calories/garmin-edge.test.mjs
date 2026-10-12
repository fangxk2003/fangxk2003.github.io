import test from 'node:test';
import assert from 'node:assert/strict';
import { createHandler } from '../../backend/calories/garmin-edge/index.ts';

const credential = 'a'.repeat(64);
const request = value => new Request('https://example.test/worker', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(value) });
test('machine endpoint rejects missing credentials before touching privileged storage', async () => {
  const handler = createHandler({ url: 'https://storage.test', serviceKey: 'server-only', request: () => { throw new Error('must not call'); } });
  const response = await handler(request({ operation: 'acquire' }));
  assert.equal(response.status, 401);
  assert.deepEqual(await response.json(), { error: 'unauthorized' });
});
test('machine endpoint forwards scoped operations only using its server credential', async () => {
  const payload = { operation: 'acquire', credential };
  const handler = createHandler({ url: 'https://storage.test', serviceKey: 'server-only', request: async (url, init) => {
    assert.equal(url, 'https://storage.test/rest/v1/rpc/calories_garmin_worker');
    assert.equal(init.headers.Authorization, 'Bearer server-only');
    assert.deepEqual(JSON.parse(init.body), { p_request: payload });
    return Response.json({ lease_id: 'lease', encrypted_session: null });
  } });
  const response = await handler(request(payload));
  assert.deepEqual(await response.json(), { lease_id: 'lease', encrypted_session: null });
});
test('SQL details and private values never appear in failed endpoint responses', async () => {
  for (const [message, status, expected] of [
    ['worker_auth_failed', 401, 'unauthorized'], ['stale_lease', 409, 'stale_lease'],
    ['invalid_request', 400, 'invalid_request'], ['database error revealing session-token-value', 503, 'storage_unavailable'],
  ]) {
    const handler = createHandler({ url: 'https://storage.test', serviceKey: 'server-only', request: async () => Response.json({ message, hint: credential }, { status: 400 }) });
    const response = await handler(request({ operation: 'acquire', credential }));
    assert.equal(response.status, status);
    assert.deepEqual(await response.json(), { error: expected });
  }
});
test('stream size limit works even without Content-Length', async () => {
  const handler = createHandler({ url: 'https://storage.test', serviceKey: 'server-only', request: () => { throw new Error('must not call'); } });
  const response = await handler(request({ operation: 'checkpoint', credential, encrypted_session: 'x'.repeat(150000) }));
  assert.equal(response.status, 413);
});
