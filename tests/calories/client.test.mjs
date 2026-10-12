import test from 'node:test';
import assert from 'node:assert/strict';
import { createClient, REMEMBER_MS, validateConfiguration } from '../../src/lib/calories/client.ts';
import { emptyData } from '../../src/lib/calories/model.ts';

const url = 'https://calories-example.supabase.co';
const key = 'sb_publishable_test_key_for_calories';
const userId = 'd5640279-d78c-4537-a5d8-a31c3b0be53e';
const start = Date.UTC(2026, 9, 12, 17, 0, 0);
const json = (value, status = 200) => new Response(JSON.stringify(value), {
  status, headers: { 'Content-Type': 'application/json' },
});
const auth = (suffix = 'initial') => ({
  access_token: `access-${suffix}`, refresh_token: `refresh-${suffix}`,
  expires_in: 3600, user: { id: userId, email: 'owner@example.com' },
});

class MemoryStorage {
  values = new Map();
  getItem(name) { return this.values.get(name) ?? null; }
  setItem(name, value) { this.values.set(name, String(value)); }
  removeItem(name) { this.values.delete(name); }
}
const sessionIn = storage => [...storage.values.values()]
  .map(value => JSON.parse(value)).find(value => value.refreshToken);

function harness(handler, options = {}) {
  const localStorage = options.localStorage ?? new MemoryStorage();
  const sessionStorage = options.sessionStorage ?? new MemoryStorage();
  let time = start;
  const calls = [];
  const fetch = async (address, init) => {
    const call = {
      address: new URL(address), method: init.method ?? 'GET',
      headers: new Headers(init.headers), body: init.body ? JSON.parse(init.body) : undefined,
    };
    calls.push(call);
    if (call.address.pathname === '/auth/v1/token' && call.address.searchParams.get('grant_type') === 'password') return json(auth());
    if (call.address.pathname === '/auth/v1/user') return json({ id: userId });
    if (call.address.pathname === '/rest/v1/rpc/calories_access_allowed') return json(true);
    return handler(call);
  };
  const clientOptions = { url, key, localStorage, sessionStorage, fetch, now: () => time };
  return {
    client: createClient(clientOptions), calls, localStorage, sessionStorage, clientOptions,
    setTime(value) { time = value; },
  };
}

test('browser configuration rejects secret keys, service-role JWTs, and unsafe URLs', () => {
  const jwt = role => `header.${Buffer.from(JSON.stringify({ role })).toString('base64url')}.signature`;
  for (const unsafeKey of ['sb_secret_do_not_expose_this_key', jwt('service_role'), jwt('authenticated'), 'invalid-key']) {
    assert.throws(() => validateConfiguration({ url, key: unsafeKey }), /publishable|Secret|secret/);
  }
  for (const unsafeUrl of ['http://example.com', `${url}/rest/v1`, `${url}?key=secret`, `${url}#secret`, 'https://name:password@example.com']) {
    assert.throws(() => validateConfiguration({ url: unsafeUrl, key }));
  }
  assert.deepEqual(validateConfiguration({ url: `${url}/`, key: ` ${key} ` }), { url, key, email: '' });
  assert.equal(validateConfiguration({ url, key: jwt('anon') }).key, jwt('anon'));
});

test('remembered sign-in expires at its original 30-day deadline despite token rotation', async () => {
  const h = harness(call => {
    assert.equal(call.address.searchParams.get('grant_type'), 'refresh_token');
    return json(auth('rotated'));
  });
  await h.client.signIn(' owner@example.com ', 'password-value', true);
  const deadline = start + REMEMBER_MS;
  assert.equal(sessionIn(h.localStorage).deadline, deadline);
  assert.equal(sessionIn(h.sessionStorage), undefined);
  assert.equal(h.calls[0].body.email, 'owner@example.com');
  assert.equal(h.calls[0].headers.get('apikey'), key);
  h.setTime(start + 29 * 24 * 60 * 60 * 1000);
  assert.equal(await h.client.restoreSession(), true);
  assert.equal(sessionIn(h.localStorage).refreshToken, 'refresh-rotated');
  assert.equal(sessionIn(h.localStorage).deadline, deadline);
  const requestsBeforeExpiry = h.calls.length;
  h.setTime(deadline);
  assert.equal(await h.client.restoreSession(), false);
  await assert.rejects(h.client.load(), /session has ended/);
  assert.equal(h.calls.length, requestsBeforeExpiry, 'an expired device must not send or refresh credentials');
  assert.equal(sessionIn(h.localStorage), undefined);
});

test('without remember, credentials stay in the tab and cannot restore in a fresh tab', async () => {
  const h = harness(() => { throw new Error('Unexpected request'); });
  await h.client.signIn('owner@example.com', 'password-value', false);
  assert.equal(sessionIn(h.localStorage), undefined);
  assert.equal(sessionIn(h.sessionStorage).remember, false);
  const sameTab = createClient(h.clientOptions);
  assert.equal(await sameTab.restoreSession(), true);
  const newTab = createClient({ ...h.clientOptions, sessionStorage: new MemoryStorage() });
  const before = h.calls.length;
  assert.equal(await newTab.restoreSession(), false);
  assert.equal(h.calls.length, before);
});

test('restoring a session verifies the server user and rejects a mismatched identity', async () => {
  const localStorage = new MemoryStorage();
  const h = harness(() => { throw new Error('Unexpected request'); }, { localStorage });
  await h.client.signIn('owner@example.com', 'password-value', true);
  let calls = 0;
  const restored = createClient({
    ...h.clientOptions,
    fetch: async (address, init) => {
      calls++;
      assert.equal(new URL(address).pathname, '/auth/v1/user');
      assert.equal(new Headers(init.headers).get('Authorization'), 'Bearer access-initial');
      return json({ id: 'different-user' });
    },
  });
  assert.equal(await restored.restoreSession(), false);
  assert.equal(calls, 1);
  assert.equal(sessionIn(localStorage), undefined);
});

test('concurrent reads rotate an expiring refresh token once', async () => {
  let refreshes = 0;
  const h = harness(async call => {
    if (call.address.pathname === '/auth/v1/token') {
      refreshes++;
      assert.equal(call.body.refresh_token, 'refresh-initial');
      await new Promise(resolve => setImmediate(resolve));
      return json(auth('rotated'));
    }
    assert.equal(call.headers.get('Authorization'), 'Bearer access-rotated');
    return json([{ data: emptyData(), revision: 1 }]);
  });
  await h.client.signIn('owner@example.com', 'password-value', true);
  h.setTime(start + 3600 * 1000);
  await Promise.all([h.client.load(), h.client.load()]);
  assert.equal(refreshes, 1);
});

test('stale revisions block subsequent overwrites until data is reloaded', async () => {
  let loads = 0;
  let patches = 0;
  const h = harness(call => {
    assert.equal(call.address.pathname, '/rest/v1/calories_state');
    if (call.method === 'GET') return json([{ data: emptyData(), revision: ++loads === 1 ? 7 : 8 }]);
    assert.equal(call.method, 'PATCH');
    assert.deepEqual(Object.keys(call.body), ['data']);
    patches++;
    assert.equal(call.address.searchParams.get('revision'), patches === 1 ? 'eq.7' : 'eq.8');
    assert.equal(call.address.searchParams.get('user_id'), `eq.${userId}`);
    return json(patches === 1 ? [] : [{ revision: 9 }]);
  });
  await h.client.signIn('owner@example.com', 'password-value', true);
  await h.client.load();
  await assert.rejects(h.client.save(emptyData()), /Another device saved changes first/);
  const callsAtConflict = h.calls.length;
  await assert.rejects(h.client.save(emptyData()), /Reload your saved data/);
  assert.equal(h.calls.length, callsAtConflict);
  await h.client.load();
  await h.client.save(emptyData());
  assert.equal(patches, 2);
});

test('a first save requires a successful load and inserts only the authenticated user row', async () => {
  const h = harness(call => {
    if (call.method === 'GET') return json([]);
    assert.equal(call.method, 'POST');
    assert.deepEqual(call.body, { user_id: userId, data: emptyData() });
    return json([{ revision: 1 }], 201);
  });
  await h.client.signIn('owner@example.com', 'password-value', true);
  await assert.rejects(h.client.save(emptyData()), /Reload your saved data/);
  await h.client.load();
  await h.client.save(emptyData());
});

test('uncertain save failures require reload before retrying to avoid overwriting a completed request', async () => {
  const h = harness(call => {
    if (call.method === 'GET') return json([{ data: emptyData(), revision: 1 }]);
    throw new Error('The response was lost after the write reached the server');
  });
  await h.client.signIn('owner@example.com', 'password-value', true);
  await h.client.load();
  await assert.rejects(h.client.save(emptyData()), /Cannot reach your storage/);
  const beforeRetry = h.calls.length;
  await assert.rejects(h.client.save(emptyData()), /Reload your saved data/);
  assert.equal(h.calls.length, beforeRetry);
});

test('concurrent saves are rejected before sending a second request', async () => {
  let release;
  let writes = 0;
  const pending = new Promise(resolve => { release = resolve; });
  const h = harness(call => {
    if (call.method === 'GET') return json([{ data: emptyData(), revision: 1 }]);
    writes++;
    return writes === 1 ? pending : json([{ revision: 2 }]);
  });
  await h.client.signIn('owner@example.com', 'password-value', true);
  await h.client.load();
  const first = h.client.save(emptyData());
  const second = h.client.save(emptyData()).then(() => ({ ok: true }), error => ({ ok: false, error }));
  await new Promise(resolve => setImmediate(resolve));
  release(json([{ revision: 2 }]));
  await first;
  const outcome = await second;
  assert.equal(outcome.ok, false);
  assert.match(outcome.error.message, /save is already in progress/);
  assert.equal(writes, 1);
});

test('sign-out clears both storage scopes even when remote revocation fails', async () => {
  const h = harness(call => {
    assert.equal(call.address.pathname, '/auth/v1/logout');
    assert.equal(call.address.searchParams.get('scope'), 'local');
    throw new Error('offline');
  });
  await h.client.signIn('owner@example.com', 'password-value', true);
  await h.client.signOut();
  assert.equal(sessionIn(h.localStorage), undefined);
  assert.equal(sessionIn(h.sessionStorage), undefined);
  assert.equal(await h.client.restoreSession(), false);
});

test('another tab signing out prevents an in-flight refresh from restoring remembered credentials', async () => {
  let started;
  let release;
  const refreshing = new Promise(resolve => { started = resolve; });
  const response = new Promise(resolve => { release = resolve; });
  const h = harness(call => {
    if (call.address.pathname === '/auth/v1/token') {
      started();
      return response;
    }
    if (call.address.pathname === '/auth/v1/logout') return new Response(null, { status: 204 });
    throw new Error('Unexpected request');
  });
  await h.client.signIn('owner@example.com', 'password-value', true);
  const otherTab = createClient({ ...h.clientOptions, sessionStorage: new MemoryStorage() });
  h.setTime(start + 3600 * 1000);
  const restoring = h.client.restoreSession().then(value => value, () => false);
  await refreshing;
  await otherTab.signOut();
  assert.equal(sessionIn(h.localStorage), undefined);
  release(json(auth('late-refresh')));
  await restoring;
  assert.equal(sessionIn(h.localStorage), undefined, 'a late refresh must not undo sign-out in another tab');
  assert.equal(sessionIn(h.sessionStorage), undefined);
});

test('sign-out during refresh response parsing cannot restore remembered credentials', async () => {
  let parsing;
  let release;
  const parsingStarted = new Promise(resolve => { parsing = resolve; });
  const parsedBody = new Promise(resolve => { release = resolve; });
  const h = harness(call => {
    if (call.address.pathname === '/auth/v1/token') return {
      ok: true,
      async json() {
        parsing();
        return parsedBody;
      },
    };
    if (call.address.pathname === '/auth/v1/logout') return new Response(null, { status: 204 });
    throw new Error('Unexpected request');
  });
  await h.client.signIn('owner@example.com', 'password-value', true);
  const otherTab = createClient({ ...h.clientOptions, sessionStorage: new MemoryStorage() });
  h.setTime(start + 3600 * 1000);
  const restoring = h.client.restoreSession().then(value => value, () => false);
  await parsingStarted;
  await otherTab.signOut();
  release(auth('late-body'));
  await restoring;
  assert.equal(sessionIn(h.localStorage), undefined, 'parsing a response must not undo sign-out');
});

test('a refresh completed after the remember deadline cannot send an authenticated request', async () => {
  let started;
  let release;
  const refreshing = new Promise(resolve => { started = resolve; });
  const response = new Promise(resolve => { release = resolve; });
  const h = harness(call => {
    assert.equal(call.address.pathname, '/auth/v1/token');
    started();
    return response;
  });
  await h.client.signIn('owner@example.com', 'password-value', true);
  h.setTime(start + REMEMBER_MS - 1);
  const restoring = h.client.restoreSession().then(value => value, () => false);
  await refreshing;
  h.setTime(start + REMEMBER_MS);
  release(json(auth('after-deadline')));
  assert.equal(await restoring, false);
  assert.equal(h.calls.some(call => call.address.pathname === '/auth/v1/user'), false);
  assert.equal(sessionIn(h.localStorage), undefined);
});

test('Garmin reads use owner/session access and leave journal revision unchanged', async () => {
  const reading = { date: '2026-10-12', active: 500, resting: 1700, total: 2200, synced_at: '2026-10-12T18:00:00Z' };
  const status = { state: 'connected', last_attempt_at: reading.synced_at, last_sync_at: reading.synced_at, last_error_code: null };
  const h = harness(call => {
    if (call.address.pathname === '/rest/v1/calories_state') {
      if (call.method === 'GET') return json([{ data: emptyData(), revision: 7 }]);
      assert.equal(call.address.searchParams.get('revision'), 'eq.7');
      return json([{ revision: 8 }]);
    }
    assert.equal(call.headers.get('Authorization'), 'Bearer access-initial');
    assert.equal(call.address.searchParams.get('owner_id'), `eq.${userId}`);
    if (call.address.pathname === '/rest/v1/calories_garmin_days') return json([reading]);
    if (call.address.pathname === '/rest/v1/calories_garmin_status') return json([status]);
    throw new Error('Unexpected request');
  });
  await h.client.signIn('owner@example.com', 'password-value', true);
  await h.client.load();
  assert.deepEqual(await h.client.loadGarmin(), { days: [reading], status });
  await h.client.save(emptyData());
});

test('Garmin denied/expired access does not look like an empty disconnected account', async () => {
  const h = harness(() => json([]));
  await h.client.signIn('owner@example.com', 'password-value', true);
  await assert.rejects(h.client.loadGarmin(), /setup is incomplete|access has expired/);
  h.setTime(start + REMEMBER_MS);
  const before = h.calls.length;
  await assert.rejects(h.client.loadGarmin(), /session has ended/);
  assert.equal(h.calls.length, before);
});
