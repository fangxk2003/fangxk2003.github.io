import test from 'node:test';
import assert from 'node:assert/strict';
import { createClient, REMEMBER_MS } from '../../src/lib/calories/client.ts';
import { emptyData, manualMealEntry, normalizeData } from '../../src/lib/calories/model.ts';
import { consumeFoodBankPortion, prepareFoodBankPortions } from '../../src/lib/calories/food-bank.ts';
import { JournalConflictError, journalCacheKey, mergeJournals } from '../../src/lib/calories/offline.ts';

const url = 'https://offline-example.supabase.co', key = 'sb_publishable_offline_test_example_key';
const owner = 'd5640279-d78c-4537-a5d8-a31c3b0be53e', start = Date.UTC(2026, 9, 16, 12);
const clone = value => structuredClone(value);
const json = (value, status = 200) => new Response(JSON.stringify(value), { status, headers: { 'Content-Type': 'application/json' } });
class MemoryStorage {
  values = new Map();
  blocked = false;
  getItem(key) { return this.values.get(key) ?? null; }
  setItem(key, value) { if (this.blocked) throw new Error('quota'); this.values.set(key, value); }
  removeItem(key) { this.values.delete(key); }
}
const meal = (id, protein = 20) => manualMealEntry({ id, date: '2026-10-16', meal: 'Lunch', name: `Meal ${id}`, protein, carbs: 30, fat: 5, fiber: 2 });
const food = { id: 'rice', name: 'Rice', baseQuantity: 100, unit: 'g', state: 'Cooked', protein: 2, carbs: 25, fat: 1, fiber: 1, calories: 117 };
function bankData() {
  const data = { ...emptyData(), foods: [food] }; let id = 0;
  data.foodBank = prepareFoodBankPortions(data.foods, [{ foodId: food.id, quantity: 150 }, { foodId: food.id, quantity: 50 }], 1, 'Prepared lunch', () => `bank-${++id}`);
  return normalizeData(data);
}
function harness(initial = emptyData()) {
  const local = new MemoryStorage(), tab = new MemoryStorage();
  const state = { data: clone(initial), revision: 1, offline: false, access: true, userStatus: 200, lostWrite: false, holdWrite: null, time: start, writes: 0, reads: 0, requests: 0 };
  const fetch = async (address, init) => {
    state.requests++;
    if (state.offline) throw new Error('offline');
    const endpoint = new URL(address), body = init.body ? JSON.parse(init.body) : null;
    if (state.bodyFailure?.path === endpoint.pathname && (!init.method || init.method === 'GET' || endpoint.pathname.includes('/rpc/'))) return {
      ok: (state.bodyFailure.status ?? 200) < 400, status: state.bodyFailure.status ?? 200,
      async json() { throw state.bodyFailure.error; },
    };
    if (endpoint.pathname === '/auth/v1/token') return json({ access_token: 'test-access', refresh_token: 'test-refresh', expires_in: 3600, user: { id: owner, email: 'owner@example.com' } });
    if (endpoint.pathname === '/auth/v1/user') return json({ id: owner }, state.userStatus);
    if (endpoint.pathname === '/auth/v1/logout') return new Response(null, { status: 204 });
    if (endpoint.pathname === '/rest/v1/rpc/calories_access_allowed') return json(state.access);
    if (endpoint.pathname === '/rest/v1/calories_garmin_days') return json([{ date: '2026-10-15', active: 400, resting: 2000, total: 2400, synced_at: '2026-10-16T10:00:00Z' }]);
    if (endpoint.pathname === '/rest/v1/calories_garmin_status') return json([{ state: 'connected', last_attempt_at: '2026-10-16T10:00:00Z', last_sync_at: '2026-10-16T10:00:00Z', last_error_code: null }]);
    assert.equal(endpoint.pathname, '/rest/v1/calories_state');
    if (!init.method || init.method === 'GET') {
      state.reads++;
      if (state.holdRead) await state.holdRead;
      return json(state.revision === null ? [] : [{ data: state.data, revision: state.revision }]);
    }
    state.writes++;
    if (state.holdWrite) await state.holdWrite;
    if (state.revision === null) {
      assert.equal(init.method, 'POST'); assert.equal(body.user_id, owner);
    } else if (endpoint.searchParams.get('revision') !== `eq.${state.revision}`) return json([]);
    state.data = normalizeData(body.data); state.revision = (state.revision ?? 0) + 1;
    if (state.lostWrite) { state.lostWrite = false; throw new Error('response lost after commit'); }
    if (state.writeBodyFailure) { state.writeBodyFailure = false; return { ok: true, status: 200, async json() { throw new TypeError('connection interrupted during response body'); } }; }
    return json([{ revision: state.revision }]);
  };
  const options = { url, key, localStorage: local, sessionStorage: tab, fetch, now: () => state.time };
  return { state, local, tab, options, client: createClient(options), reload: () => createClient(options) };
}
async function open(h, remember = true) { await h.client.signIn('owner@example.com', 'never-cache-this-password', remember); return h.client.openJournal(); }
const append = (data, entry) => ({ ...data, entries: [...data.entries, entry] });

test('offline drafts survive reload, contain no auth secrets, and require a verified unexpired owner session', async () => {
  const h = harness(); const first = await open(h);
  h.state.offline = true;
  const next = await h.client.stageJournal(append(first.data, meal('local')), first.localVersion);
  assert.equal(next.pending, true);
  const offline = await h.client.syncJournal(); assert.equal(offline.offline, true);
  const encoded = h.local.getItem(journalCacheKey(url, owner));
  for (const secret of ['never-cache-this-password', 'test-access', 'test-refresh', key]) assert.equal(encoded.includes(secret), false);
  const restored = h.reload(); assert.equal(await restored.restoreSession(), true);
  assert.deepEqual((await restored.openJournal()).data.entries, [meal('local')]);
  h.state.time = start + REMEMBER_MS;
  assert.equal(await restored.restoreSession(), false);
  assert.equal(restored.getJournalSnapshot(), null);
  await assert.rejects(restored.stageJournal(first.data), /session has ended/);
  const unverified = harness(); await unverified.client.signIn('owner@example.com', 'password', true); unverified.state.offline = true;
  await assert.rejects(unverified.reload().restoreSession(), /Cannot reach/);
  assert.equal(unverified.client.getJournalSnapshot(), null);
});

test('tab-only credentials cannot unlock an offline cache in another tab and explicit signout clears owner caches', async () => {
  const h = harness(); const first = await open(h, false);
  await h.client.stageJournal(append(first.data, meal('draft')), first.localVersion);
  await h.client.loadGarmin();
  h.state.offline = true;
  const freshTab = createClient({ ...h.options, sessionStorage: new MemoryStorage() });
  assert.equal(await freshTab.restoreSession(), false);
  assert.equal(freshTab.getCachedGarmin(), null);
  assert.equal(freshTab.getJournalSnapshot(), null);
  assert.equal(await h.reload().restoreSession(), true);
  await h.client.signOut();
  assert.equal(h.local.getItem(journalCacheKey(url, owner)), null);
  assert.equal(h.local.getItem(`calories.garmin.v1:${url}:${owner}`), null);
  assert.equal(await h.reload().restoreSession(), false);
});

test('server user denial and revoked owner access never become an offline restore', async () => {
  const h = harness(); await open(h);
  h.state.userStatus = 403;
  assert.equal(await h.reload().restoreSession(), false);
  await h.client.signIn('owner@example.com', 'password', true);
  await h.client.openJournal();
  h.state.access = false;
  await assert.rejects(h.client.openJournal(), /session has ended|does not have access/);
  assert.equal(h.client.getJournalSnapshot(), null);
  h.state.offline = true;
  assert.equal(await h.reload().restoreSession(), false);
});

test('reconnect merges disjoint entries and targets without losing either device, and stops conflicting edits', async () => {
  const h = harness(); const first = await open(h);
  const staged = await h.client.stageJournal(append(first.data, meal('local')), first.localVersion);
  h.state.data = append(h.state.data, meal('remote'));
  h.state.data.days['2026-10-15'] = { ...h.state.data.defaults, deficit: 600 }; h.state.revision++;
  const synced = await h.client.syncJournal();
  assert.equal(synced.pending, false); assert.equal(synced.conflict, null);
  assert.deepEqual(new Set(synced.data.entries.map(e => e.id)), new Set(['local', 'remote']));
  assert.equal(synced.data.days['2026-10-15'].deficit, 600);
  assert.deepEqual(h.state.data, synced.data);
  const edit = { ...synced.data, entries: synced.data.entries.map(e => e.id === 'local' ? meal('local', 40) : e) };
  await h.client.stageJournal(edit, synced.localVersion);
  h.state.data.entries = h.state.data.entries.map(e => e.id === 'local' ? meal('local', 60) : e); h.state.revision++;
  const writes = h.state.writes, conflict = await h.client.syncJournal();
  assert.equal(conflict.pending, true); assert.match(conflict.conflict, /same journal entry/);
  assert.equal(h.state.writes, writes);
  assert.equal(conflict.data.entries.find(e => e.id === 'local').protein, 40);
  assert.equal(h.reload().getJournalSnapshot().conflict, conflict.conflict);
  assert.notEqual(staged.localVersion, synced.localVersion);
});

test('a committed-but-lost bank consumption response is recognized after reload and never duplicated', async () => {
  const h = harness(bankData()); const first = await open(h);
  const consumed = consumeFoodBankPortion(first.data, first.data.foodBank[0].id, '2026-10-16', 'Lunch');
  await h.client.stageJournal(consumed, first.localVersion); h.state.lostWrite = true;
  const uncertain = await h.client.syncJournal();
  assert.equal(uncertain.pending, true); assert.equal(uncertain.offline, true);
  assert.equal(h.state.data.foodBank.length, 0); assert.equal(h.state.data.entries.length, 2);
  const restored = h.reload(); assert.equal(await restored.restoreSession(), true);
  const resolved = await restored.syncJournal();
  assert.equal(resolved.pending, false); assert.equal(resolved.conflict, null);
  assert.equal(h.state.writes, 1); assert.equal(resolved.data.entries.length, 2);
});

test('the local snapshot is durable before a held write and newer edits during that write are drained', async () => {
  const h = harness(); const first = await open(h);
  const staged = await h.client.stageJournal(append(first.data, meal('first')), first.localVersion);
  let release; h.state.holdWrite = new Promise(resolve => { release = resolve; });
  const syncing = h.client.syncJournal();
  while (!h.state.writes) await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(h.reload().getJournalSnapshot().data.entries, [meal('first')]);
  const newer = await h.client.stageJournal(append(staged.data, meal('second')), staged.localVersion);
  assert.equal(newer.data.entries.length, 2);
  h.state.holdWrite = null; release();
  const result = await syncing;
  assert.equal(result.pending, false); assert.equal(result.data.entries.length, 2);
  assert.equal(h.state.data.entries.length, 2); assert.equal(h.state.writes, 2);
});

test('shared-tab local locks merge stale disjoint edits and reject a stale overlapping record instead of overwriting', async () => {
  const h = harness(); const first = await open(h);
  const other = createClient({ ...h.options, sessionStorage: new MemoryStorage() });
  const second = other.getJournalSnapshot();
  const [a, b] = await Promise.all([
    h.client.stageJournal(append(first.data, meal('one')), first.localVersion),
    other.stageJournal(append(second.data, meal('two')), second.localVersion),
  ]);
  assert.equal(b.data.entries.length, 2);
  const before = h.client.getJournalSnapshot(), beforeOther = other.getJournalSnapshot();
  await h.client.stageJournal({ ...before.data, entries: before.data.entries.map(e => e.id === 'one' ? meal('one', 30) : e) }, before.localVersion);
  await assert.rejects(other.stageJournal({ ...beforeOther.data, entries: beforeOther.data.entries.map(e => e.id === 'one' ? meal('one', 50) : e) }, beforeOther.localVersion), JournalConflictError);
  assert.equal(h.client.getJournalSnapshot().data.entries.find(e => e.id === 'one').protein, 30);
  assert.equal(a.data.entries.length, 1);
});

test('bank consumes/removes conflict as complete movements while unrelated changes and identical consumes merge', () => {
  const base = bankData(), id = base.foodBank[0].id;
  const lunch = consumeFoodBankPortion(base, id, '2026-10-16', 'Lunch');
  const dinner = consumeFoodBankPortion(base, id, '2026-10-16', 'Dinner');
  const removed = { ...base, foodBank: [] };
  assert.throws(() => mergeJournals(base, lunch, dinner), JournalConflictError);
  assert.throws(() => mergeJournals(base, lunch, removed), JournalConflictError);
  assert.deepEqual(mergeJournals(base, lunch, lunch), lunch);
  const disjoint = append(base, meal('unrelated'));
  const merged = mergeJournals(base, lunch, disjoint);
  assert.equal(merged.foodBank.length, 0); assert.equal(merged.entries.length, 3);
  assert.deepEqual(base, bankData(), 'merge cannot mutate any input');
});

test('discard requires an online read and quota failure cannot send or silently discard edits', async () => {
  const h = harness(); const first = await open(h);
  const staged = await h.client.stageJournal(append(first.data, meal('pending')), first.localVersion);
  h.state.offline = true;
  await assert.rejects(h.client.discardJournalChanges(), /Cannot reach/);
  assert.equal(h.client.getJournalSnapshot().data.entries.length, 1);
  h.state.offline = false; h.local.blocked = true;
  await assert.rejects(h.client.stageJournal(append(staged.data, meal('quota')), staged.localVersion), /could not keep/);
  assert.equal(h.state.writes, 0); assert.equal(h.client.getJournalSnapshot().data.entries.length, 1);
  h.local.blocked = false;
  const discarded = await h.client.discardJournalChanges();
  assert.equal(discarded.pending, false); assert.equal(discarded.data.entries.length, 0);
});

test('cached Garmin readings retain their original timestamps offline and disappear with expired credentials', async () => {
  const h = harness(); await open(h);
  const reading = await h.client.loadGarmin(); h.state.offline = true;
  assert.deepEqual(await h.reload().loadGarmin(), reading);
  assert.deepEqual(h.reload().getCachedGarmin(), reading);
  h.state.time = start + REMEMBER_MS;
  assert.equal(h.reload().getCachedGarmin(), null);
});

test('a first journal can be staged offline after an authorized empty load and inserts only the owner row', async () => {
  const h = harness(); h.state.revision = null;
  const first = await open(h);
  assert.equal(h.state.writes, 0);
  h.state.offline = true;
  await h.client.stageJournal(append(first.data, meal('first-row')), first.localVersion);
  assert.equal((await h.reload().openJournal()).data.entries.length, 1);
  h.state.offline = false;
  const result = await h.client.syncJournal();
  assert.equal(result.pending, false); assert.equal(h.state.revision, 1);
  assert.deepEqual(h.state.data.entries, [meal('first-row')]);
});

test('a lost response followed by newer local edits and unrelated remote additions rebases without a false conflict', async () => {
  const h = harness(bankData()), first = await open(h);
  const consumed = consumeFoodBankPortion(first.data, first.data.foodBank[0].id, '2026-10-16', 'Lunch');
  await h.client.stageJournal(consumed, first.localVersion); h.state.lostWrite = true;
  const uncertain = await h.client.syncJournal();
  const amended = { ...uncertain.data, entries: uncertain.data.entries.map((entry, index) => index ? entry : { ...entry, protein: entry.protein + 1, calories: entry.calories + 4 }) };
  await h.client.stageJournal(amended, uncertain.localVersion);
  h.state.data = append(h.state.data, meal('remote-after-consume')); h.state.revision++;
  const result = await h.client.syncJournal();
  assert.equal(result.conflict, null); assert.equal(result.pending, false);
  assert.equal(result.data.foodBank.length, 0); assert.equal(result.data.entries.length, 3);
  assert.equal(result.data.entries.find(entry => entry.id === amended.entries[0].id).protein, amended.entries[0].protein);
});

test('discarding cannot erase an edit another tab stages while the saved cloud copy is being read', async () => {
  const h = harness(), first = await open(h);
  let release; h.state.holdRead = new Promise(resolve => { release = resolve; });
  const reads = h.state.reads;
  const discarding = h.client.discardJournalChanges();
  while (h.state.reads === reads) await new Promise(resolve => setImmediate(resolve));
  await h.client.stageJournal(append(first.data, meal('during-discard')), first.localVersion);
  h.state.holdRead = null; release();
  await assert.rejects(discarding, /while saved data was being reloaded/);
  assert.deepEqual(h.client.getJournalSnapshot().data.entries, [meal('during-discard')]);
});

test('signout during a held save cannot resurrect a cleared journal or cached credentials', async () => {
  const h = harness(), first = await open(h);
  await h.client.stageJournal(append(first.data, meal('before-signout')), first.localVersion);
  let release; h.state.holdWrite = new Promise(resolve => { release = resolve; });
  const syncing = h.client.syncJournal();
  while (!h.state.writes) await new Promise(resolve => setImmediate(resolve));
  const result = syncing.catch(error => error);
  await h.client.signOut();
  h.state.holdWrite = null; release();
  assert.match((await result).message, /session has ended/);
  assert.equal(h.local.getItem(journalCacheKey(url, owner)), null);
  assert.equal(h.reload().getJournalSnapshot(), null);
  assert.equal(await h.reload().restoreSession(), false);
});

test('interrupted streamed response bodies fall back to verified cache, while invalid JSON and denied auth fail closed', async () => {
  const h = harness(); await open(h);
  const garmin = await h.client.loadGarmin();
  for (const error of [new TypeError('network body interrupted'), new DOMException('aborted body', 'AbortError')]) {
    h.state.bodyFailure = { path: '/auth/v1/user', error };
    assert.equal(await h.reload().restoreSession(), true);
    for (const path of ['/rest/v1/rpc/calories_access_allowed', '/rest/v1/calories_state']) {
      h.state.bodyFailure = { path, error };
      assert.equal((await h.reload().openJournal()).offline, true);
    }
    h.state.bodyFailure = { path: '/rest/v1/calories_garmin_days', error };
    assert.deepEqual(await h.reload().loadGarmin(), garmin);
  }
  h.state.bodyFailure = { path: '/auth/v1/user', error: new SyntaxError('malformed JSON') };
  await assert.rejects(h.reload().restoreSession(), /storage response was invalid/);
  h.state.bodyFailure = { path: '/rest/v1/calories_state', error: new SyntaxError('malformed JSON') };
  await assert.rejects(h.reload().openJournal(), /storage response was invalid/);
  h.state.bodyFailure = { path: '/auth/v1/user', status: 403, error: new TypeError('connection interrupted after rejection') };
  assert.equal(await h.reload().restoreSession(), false);
  assert.equal(h.client.getJournalSnapshot(), null);
});

test('an interrupted successful save body remains pending and resolves the committed write on the next read', async () => {
  const h = harness(), first = await open(h);
  await h.client.stageJournal(append(first.data, meal('streamed-write')), first.localVersion);
  h.state.writeBodyFailure = true;
  const uncertain = await h.client.syncJournal();
  assert.equal(uncertain.pending, true); assert.equal(uncertain.offline, true);
  assert.equal(h.state.data.entries.length, 1);
  const result = await h.reload().syncJournal();
  assert.equal(result.pending, false); assert.equal(result.conflict, null);
  assert.equal(h.state.writes, 1);
});
