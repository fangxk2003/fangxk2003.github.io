import test from 'node:test';
import assert from 'node:assert/strict';
import { createClient } from '../../src/lib/calories/client.ts';
import { emptyData, manualMealEntry, normalizeData } from '../../src/lib/calories/model.ts';
import { prepareFoodBankPortions, consumeFoodBankPortion } from '../../src/lib/calories/food-bank.ts';
import { createUndoAction, undoAction } from '../../src/lib/calories/undo.ts';
import { journalCacheKey, journalEqual } from '../../src/lib/calories/offline.ts';

const url = 'https://undo-offline.supabase.co', key = 'sb_publishable_undo_offline_test_key';
const owner = 'e49af72f-d699-4f30-a36b-b0619bc6d9a0', date = '2026-10-17';
const now = () => Date.UTC(2026, 9, 17, 12);
const clone = value => structuredClone(value);
const json = value => Response.json(value);
class MemoryStorage {
  rows = new Map();
  getItem(key) { return this.rows.get(key) ?? null; }
  setItem(key, value) { this.rows.set(key, value); }
  removeItem(key) { this.rows.delete(key); }
}
const manualMeal = id => manualMealEntry({ id, date, meal: 'Dinner', name: `Meal ${id}`, protein: 20, carbs: 30, fat: 5, fiber: 2 });
function initialData(existingDay = true) {
  const foods = [
    { id: 'rice', name: 'Rice', baseQuantity: 100, unit: 'g', state: 'Cooked', protein: 3, carbs: 28, fat: 1, fiber: 2, calories: 133 },
    { id: 'egg', name: 'Egg', baseQuantity: 2, unit: 'piece', state: 'Cooked', protein: 12, carbs: 1, fat: 10, fiber: 0, calories: 142 },
  ];
  let id = 0;
  const foodBank = prepareFoodBankPortions(foods, [{ foodId: 'rice', quantity: 150 }, { foodId: 'egg', quantity: 1 }], 2, 'Prepared lunch', () => `bank-${++id}`);
  foodBank[0].items[0].calories = 199.125; // Preserve the stored historical snapshot exactly.
  const defaults = { burned: 3200, burnedSource: 'manual', deficit: 500, protein: 175 };
  return normalizeData({ ...emptyData(), foods, foodBank, defaults, days: existingDay ? { [date]: { ...defaults, burned: 0, burnedSource: 'garmin' } } : {} });
}
function harness(initial = initialData()) {
  const local = new MemoryStorage(), tab = new MemoryStorage();
  const state = { data: clone(initial), revision: 1, offline: false, failure: null, holdResponse: null, writes: 0 };
  const fetch = async (address, init = {}) => {
    if (state.offline) throw new TypeError('No network');
    const endpoint = new URL(address), body = init.body ? JSON.parse(init.body) : null;
    if (endpoint.pathname === '/auth/v1/token') return json({ access_token: 'fake-access', refresh_token: 'fake-refresh', expires_in: 3600, user: { id: owner, email: 'test@example.com' } });
    if (endpoint.pathname === '/auth/v1/user') return json({ id: owner });
    if (endpoint.pathname === '/rest/v1/rpc/calories_access_allowed') return json(true);
    assert.equal(endpoint.pathname, '/rest/v1/calories_state');
    if (!init.method || init.method === 'GET') return json(state.revision === null ? [] : [{ data: state.data, revision: state.revision }]);
    state.writes++;
    if (state.failure === 'before') { state.failure = null; throw new TypeError('Write did not arrive'); }
    const expectedRevision = endpoint.searchParams.get('revision');
    const writeData = normalizeData(body.data);
    const apply = () => {
      if (init.method === 'POST') {
        assert.equal(body.user_id, owner);
        if (state.revision !== null) return false; // Unique owner row: a delayed INSERT cannot replace it.
      } else {
        assert.equal(init.method, 'PATCH');
        if (expectedRevision !== `eq.${state.revision}`) return false;
      }
      state.data = clone(writeData); state.revision = (state.revision ?? 0) + 1;
      return true;
    };
    if (state.failure === 'late') { state.failure = null; state.lateWrite = apply; throw new TypeError('Response lost before the server applies its CAS'); }
    if (state.beforeNextApply) { const before = state.beforeNextApply; state.beforeNextApply = null; before(); }
    if (!apply()) return init.method === 'POST' ? new Response(null, { status: 409 }) : json([]);
    if (state.holdResponse) await state.holdResponse;
    if (state.failure === 'after') { state.failure = null; throw new TypeError('Committed response lost'); }
    if (state.failure === 'body') { state.failure = null; return { ok: true, status: 200, async json() { throw new TypeError('Committed response body lost'); } }; }
    return json([{ revision: state.revision }]);
  };
  const options = { url, key, localStorage: local, sessionStorage: tab, fetch, now };
  return { state, local, client: createClient(options), reload: () => createClient(options) };
}
async function open(h) { await h.client.signIn('test@example.com', 'not-a-real-password', true); return h.client.openJournal(); }
function consumed(before) {
  const portion = before.foodBank[0];
  const after = consumeFoodBankPortion(before, portion.id, date, 'Lunch');
  return { after, receipt: createUndoAction(before, after, 'consume-bank', portion.id, () => 'undo-consume') };
}
function assertRestored(data, before, receipt) {
  assert.deepEqual(data.foodBank.find(portion => portion.id === receipt.recordId), before.foodBank[0]);
  assert.deepEqual(data.foodBank.find(portion => portion.id === before.foodBank[1].id), before.foodBank[1]);
  assert.equal(data.foodBank.length, 2);
  const ids = new Set(receipt.entries.map(entry => entry.id));
  assert.equal(data.entries.some(entry => ids.has(entry.id)), false, 'all generated ingredient IDs must leave the journal together');
}

test('committed consume with a lost response can be undone offline, then restored exactly while merging disjoint remote changes', async () => {
  for (const failure of ['after', 'body']) {
    const h = harness(), first = await open(h), { after, receipt } = consumed(first.data);
    await h.client.stageJournal(after, first.localVersion);
    h.state.failure = failure;
    const uncertain = await h.client.syncJournal();
    assert.equal(uncertain.pending, true);
    assert.equal(h.state.data.foodBank.length, 1);
    assert.equal(h.state.data.entries.length, 2);
    h.state.offline = true;
    const restored = await h.client.stageJournal(undoAction(uncertain.data, receipt), uncertain.localVersion);
    assert.ok(journalEqual(restored.data, first.data), 'the original working records can differ from an unconfirmed cloud commit');
    assert.equal(restored.pending, true, 'unresolved attempted writes must keep the Undo queued even when working equals the old baseline');
    assert.equal(JSON.parse(h.local.getItem(journalCacheKey(url, owner))).attempted.length, 1);
    assertRestored(restored.data, first.data, receipt);
    h.state.data = normalizeData({
      ...h.state.data,
      entries: [...h.state.data.entries, manualMeal('remote-dinner')],
      foods: h.state.data.foods.map(food => food.id === 'rice' ? { ...food, name: 'Updated library rice', protein: 4, calories: 137 } : food),
      days: { ...h.state.data.days, '2026-10-16': { ...h.state.data.defaults, deficit: 600 } },
    });
    h.state.revision++; h.state.offline = false;
    const reloaded = h.reload();
    assert.equal(await reloaded.restoreSession(), true);
    const result = await reloaded.syncJournal();
    assert.equal(result.pending, false); assert.equal(result.conflict, null);
    assertRestored(result.data, first.data, receipt);
    assert.deepEqual(result.data.entries, [manualMeal('remote-dinner')]);
    assert.equal(result.data.foods.find(food => food.id === 'rice').name, 'Updated library rice');
    assert.equal(result.data.days['2026-10-16'].deficit, 600);
    assert.deepEqual(h.state.data, result.data);
    assert.equal(h.state.writes, 2, 'one consume and one inverse write; never duplicate a generated ingredient');
  }
});

test('Undo staged while a consume response is in flight survives acknowledgement and drains as a second atomic write', async () => {
  const h = harness(), first = await open(h), { after, receipt } = consumed(first.data);
  const staged = await h.client.stageJournal(after, first.localVersion);
  let release;
  h.state.holdResponse = new Promise(resolve => { release = resolve; });
  const syncing = h.client.syncJournal();
  while (!h.state.writes) await new Promise(resolve => setImmediate(resolve));
  assert.equal(h.state.data.entries.length, 2, 'the consume is committed but its acknowledgement has not arrived');
  const restored = await h.client.stageJournal(undoAction(staged.data, receipt), staged.localVersion);
  assert.equal(restored.pending, true, 'the in-flight attempted commit still requires reconciliation');
  h.state.holdResponse = null; release();
  const result = await syncing;
  assert.equal(result.pending, false); assert.equal(result.conflict, null);
  assertRestored(result.data, first.data, receipt);
  assert.ok(journalEqual(h.state.data, first.data));
  assert.equal(h.state.writes, 2);
});

test('Undo matching unchanged cloud still sends a CAS fence before clearing its uncertain consume', async () => {
  const h = harness(), first = await open(h), { after, receipt } = consumed(first.data);
  await h.client.stageJournal(after, first.localVersion); h.state.failure = 'before';
  const uncertain = await h.client.syncJournal();
  assert.deepEqual(h.state.data, first.data);
  const restored = await h.client.stageJournal(undoAction(uncertain.data, receipt), uncertain.localVersion);
  assert.equal(restored.pending, true, 'only a successful read can settle whether the original request committed');
  const result = await h.client.syncJournal();
  assert.equal(result.pending, false); assert.equal(result.conflict, null);
  assert.deepEqual(result.data, first.data);
  assert.equal(h.state.writes, 2, 'the failed consume plus a confirming CAS fence protect against a late original commit');
});

test('a late original consume cannot overwrite Undo after its unchanged-cloud CAS fence advances the revision', async () => {
  const h = harness(), first = await open(h), { after, receipt } = consumed(first.data);
  await h.client.stageJournal(after, first.localVersion); h.state.failure = 'late';
  const uncertain = await h.client.syncJournal();
  assert.equal(h.state.revision, 1, 'the original request has not applied its CAS yet');
  const restored = await h.client.stageJournal(undoAction(uncertain.data, receipt), uncertain.localVersion);
  assert.equal(restored.pending, true);
  const result = await h.client.syncJournal();
  assert.equal(result.pending, false); assert.equal(result.conflict, null);
  assert.equal(h.state.revision, 2, 'even an equal desired snapshot must fence the old revision');
  assert.equal(h.state.lateWrite(), false, 'the delayed consume loses its stale compare-and-swap');
  assertRestored(h.state.data, first.data, receipt);
  assert.ok(journalEqual(result.data, h.state.data));
  assert.equal(h.state.writes, 2);
});

test('a missing-row Undo barrier inserts the owner row before a delayed first save can commit', async () => {
  const h = harness(emptyData()); h.state.revision = null;
  const first = await open(h);
  await h.client.stageJournal({ ...first.data, entries: [manualMeal('unconfirmed-first-row')] }, first.localVersion);
  h.state.failure = 'late';
  const uncertain = await h.client.syncJournal();
  assert.equal(h.state.revision, null);
  const reverted = await h.client.stageJournal(first.data, uncertain.localVersion);
  assert.equal(reverted.pending, true);
  const result = await h.client.syncJournal();
  assert.equal(result.pending, false); assert.equal(result.conflict, null);
  assert.equal(h.state.revision, 1, 'a known empty owner row fences any delayed insert');
  assert.equal(h.state.lateWrite(), false, 'the unique owner-row constraint rejects the original delayed POST');
  assert.deepEqual(h.state.data, first.data);
  assert.equal(h.state.writes, 2);
});

test('if the delayed consume wins just before the Undo fence, a fresh read retries the inverse without losing it', async () => {
  const h = harness(), first = await open(h), { after, receipt } = consumed(first.data);
  await h.client.stageJournal(after, first.localVersion); h.state.failure = 'late';
  const uncertain = await h.client.syncJournal();
  await h.client.stageJournal(undoAction(uncertain.data, receipt), uncertain.localVersion);
  h.state.beforeNextApply = () => assert.equal(h.state.lateWrite(), true, 'the original request commits after the read but before the first fence CAS');
  const result = await h.client.syncJournal();
  assert.equal(result.pending, false); assert.equal(result.conflict, null);
  assertRestored(result.data, first.data, receipt);
  assertRestored(h.state.data, first.data, receipt);
  assert.equal(h.state.revision, 3, 'the original consume and the retried inverse each commit once');
  assert.equal(h.state.writes, 3, 'late original attempt, stale fence, and one successful inverse');
});

test('a committed Undo with another lost response is recognized after reload without repeating either movement', async () => {
  const h = harness(), first = await open(h), { after, receipt } = consumed(first.data);
  await h.client.stageJournal(after, first.localVersion);
  const committed = await h.client.syncJournal();
  await h.client.stageJournal(undoAction(committed.data, receipt), committed.localVersion);
  h.state.failure = 'after';
  const uncertainUndo = await h.client.syncJournal();
  assert.equal(uncertainUndo.pending, true); assertRestored(h.state.data, first.data, receipt);
  h.state.data.entries.push(manualMeal('remote-after-undo')); h.state.revision++;
  const reloaded = h.reload(); assert.equal(await reloaded.restoreSession(), true);
  const result = await reloaded.syncJournal();
  assert.equal(result.pending, false); assert.equal(result.conflict, null);
  assertRestored(result.data, first.data, receipt);
  assert.deepEqual(result.data.entries, [manualMeal('remote-after-undo')]);
  assert.equal(h.state.writes, 2, 'the already committed inverse is not written again');
});

test('Undo preserves a consume-created Garmin target when an unseen remote meal survives on that same date', async () => {
  const h = harness(initialData(false)), first = await open(h), { after, receipt } = consumed(first.data);
  assert.equal(first.data.days[date], undefined);
  assert.equal(receipt.createdDay.burnedSource, 'garmin');
  await h.client.stageJournal(after, first.localVersion); h.state.failure = 'after';
  const uncertain = await h.client.syncJournal();
  await h.client.stageJournal(undoAction(uncertain.data, receipt), uncertain.localVersion);
  h.state.data.entries.push(manualMeal('remote-same-day')); h.state.revision++;
  const result = await h.client.syncJournal();
  assert.equal(result.pending, false); assert.equal(result.conflict, null);
  assertRestored(result.data, first.data, receipt);
  assert.deepEqual(result.data.entries, [manualMeal('remote-same-day')]);
  assert.deepEqual(result.data.days[date], receipt.createdDay, 'the remote meal must keep its frozen day settings rather than inherit legacy manual defaults');
  assert.equal(result.data.days[date].burned, 0);
  assert.equal(result.data.defaults.burned, 3200);
});

test('an unseen remote edit to a consumed ingredient conflicts with uncertain Undo instead of silently cancelling it', async () => {
  const h = harness(), first = await open(h), { after, receipt } = consumed(first.data);
  await h.client.stageJournal(after, first.localVersion); h.state.failure = 'after';
  const uncertain = await h.client.syncJournal();
  const restored = await h.client.stageJournal(undoAction(uncertain.data, receipt), uncertain.localVersion);
  const affectedId = receipt.entries[0].id;
  h.state.data.entries = h.state.data.entries.map(entry => entry.id === affectedId ? { ...entry, quantity: entry.quantity + 10, protein: entry.protein + 1, calories: entry.calories + 4 } : entry);
  h.state.revision++;
  const remote = clone(h.state.data), writes = h.state.writes;
  const result = await h.client.syncJournal();
  assert.equal(result.pending, true);
  assert.match(result.conflict, /food-bank portion|journal records|journal entry/);
  assertRestored(result.data, first.data, receipt);
  assert.deepEqual(h.state.data, remote, 'a remote edit must never be overwritten by an unseen Undo');
  assert.equal(h.state.writes, writes);
  assert.ok(journalEqual(restored.data, first.data));
});

test('a never-committed consume can still be amended locally without assuming the uncertain attempt succeeded', async () => {
  const h = harness(), first = await open(h), { after, receipt } = consumed(first.data);
  await h.client.stageJournal(after, first.localVersion); h.state.failure = 'before';
  const uncertain = await h.client.syncJournal();
  const affectedId = receipt.entries[0].id;
  const amended = { ...uncertain.data, entries: uncertain.data.entries.map(entry => entry.id === affectedId ? { ...entry, protein: entry.protein + 1, calories: entry.calories + 4 } : entry) };
  await h.client.stageJournal(amended, uncertain.localVersion);
  h.state.data.entries.push(manualMeal('remote-during-failed-consume')); h.state.revision++;
  const result = await h.client.syncJournal();
  assert.equal(result.pending, false); assert.equal(result.conflict, null);
  assert.equal(result.data.foodBank.length, 1);
  assert.equal(result.data.entries.length, 3);
  assert.equal(result.data.entries.find(entry => entry.id === affectedId).protein, amended.entries.find(entry => entry.id === affectedId).protein);
  assert.ok(result.data.entries.some(entry => entry.id === 'remote-during-failed-consume'));
  assert.equal(h.state.writes, 2, 'one uncommitted attempt and one successful amended movement');
});

test('a remote revert of another record cannot hide an uncertain committed consume and silently cancel its Undo', async () => {
  const h = harness(), first = await open(h);
  const edited = { ...first.data, foods: first.data.foods.map(food => food.id === 'rice' ? { ...food, name: 'Locally updated rice' } : food) };
  const { after, receipt } = consumed(edited);
  await h.client.stageJournal(after, first.localVersion); h.state.failure = 'after';
  const uncertain = await h.client.syncJournal();
  const restored = await h.client.stageJournal(undoAction(uncertain.data, receipt), uncertain.localVersion);
  assert.equal(restored.data.foods.find(food => food.id === 'rice').name, 'Locally updated rice', 'Undo affects only the consumption');
  h.state.data.foods = clone(first.data.foods); h.state.revision++;
  const remote = clone(h.state.data), writes = h.state.writes;
  const result = await h.client.syncJournal();
  assertRestored(result.data, first.data, receipt);
  if (result.conflict) {
    assert.equal(result.pending, true);
    assert.deepEqual(h.state.data, remote, 'an ambiguous record history may conflict, but must never overwrite it silently');
    assert.equal(h.state.writes, writes);
  } else {
    assert.equal(result.pending, false);
    assertRestored(h.state.data, first.data, receipt);
    assert.equal(h.state.writes, writes + 1);
  }
});
