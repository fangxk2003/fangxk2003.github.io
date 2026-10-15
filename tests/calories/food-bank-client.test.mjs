import test from 'node:test';
import assert from 'node:assert/strict';
import { createClient } from '../../src/lib/calories/client.ts';
import { emptyData, energyFromMacros, manualMealEntry } from '../../src/lib/calories/model.ts';
import { consumeFoodBankPortion, prepareFoodBankPortions } from '../../src/lib/calories/food-bank.ts';

const URL = 'https://food-bank-client-test.supabase.co';
const KEY = 'sb_publishable_food_bank_test_key';
const OWNER = 'd5640279-d78c-4537-a5d8-a31c3b0be53e';
const NOW = Date.UTC(2026, 9, 15, 12);
const json = value => Response.json(value);

class MemoryStorage {
  values = new Map();
  getItem(key) { return this.values.get(key) ?? null; }
  setItem(key, value) { this.values.set(key, String(value)); }
  removeItem(key) { this.values.delete(key); }
}

function fixture() {
  const data = emptyData();
  data.defaults = { burned: 3200, deficit: 500, protein: 175, burnedSource: 'manual' };
  data.foods = [
    { id: 'rice', name: 'Cooked rice', baseQuantity: 100, unit: 'g', state: 'Cooked', protein: 2.5, carbs: 28, fat: 0.3, fiber: 0.4 },
    { id: 'eggs', name: 'Egg', baseQuantity: 1, unit: 'piece', state: 'As served', protein: 6.3, carbs: 0.4, fat: 5.3, fiber: 0 },
  ].map(food => ({ ...food, calories: energyFromMacros(food) }));
  const historical = manualMealEntry({
    id: 'existing-meal', date: '2026-10-14', meal: 'Dinner', name: 'Existing dinner',
    protein: 20, carbs: 40, fat: 10, fiber: 5,
  });
  historical.calories = 331.125;
  data.entries = [historical];
  data.days['2026-10-14'] = { burned: 3000, deficit: 450, protein: 175, burnedSource: 'manual' };
  let id = 0;
  data.foodBank = prepareFoodBankPortions(data.foods, [
    { foodId: 'rice', quantity: 150 }, { foodId: 'eggs', quantity: 2 },
  ], 2, 'Rice and eggs', () => `stored-portion-id-${++id}`);
  return data;
}

/** Models the single-row conditional PATCH: one revision match updates the whole document. */
function server(initialData, { loseNextWriteResponse = false } = {}) {
  const state = { data: structuredClone(initialData), revision: 7, calls: [], writes: [] };
  const fetch = async (address, init = {}) => {
    const url = new globalThis.URL(address);
    const method = init.method ?? 'GET';
    const headers = new Headers(init.headers);
    const body = init.body ? JSON.parse(init.body) : undefined;
    state.calls.push({ url, method, body });
    assert.equal(url.origin, URL);
    assert.equal(headers.get('apikey'), KEY);
    if (url.pathname === '/auth/v1/token') {
      assert.equal(url.searchParams.get('grant_type'), 'password');
      return json({ access_token: 'access-bank-test', refresh_token: 'refresh-bank-test', expires_in: 3600, user: { id: OWNER, email: 'owner@example.test' } });
    }
    assert.equal(headers.get('authorization'), 'Bearer access-bank-test');
    if (url.pathname === '/rest/v1/rpc/calories_access_allowed') {
      assert.equal(method, 'POST');
      return json(true);
    }
    assert.equal(url.pathname, '/rest/v1/calories_state');
    assert.equal(url.searchParams.get('user_id'), `eq.${OWNER}`);
    if (method === 'GET') return json([{ data: structuredClone(state.data), revision: state.revision }]);
    assert.equal(method, 'PATCH');
    assert.equal(headers.get('prefer'), 'return=representation');
    assert.deepEqual(Object.keys(body), ['data'], 'the browser writes one whole data document, not revisions or separate bank/entry updates');
    const expected = url.searchParams.get('revision');
    assert.match(expected, /^eq\.\d+$/);
    if (expected !== `eq.${state.revision}`) return json([]);
    const before = structuredClone(state.data);
    state.data = structuredClone(body.data);
    state.revision++;
    state.writes.push({ before, after: structuredClone(state.data), revision: state.revision });
    if (loseNextWriteResponse) {
      loseNextWriteResponse = false;
      throw new Error('The server committed the write but the response was lost.');
    }
    return json([{ revision: state.revision }]);
  };
  const client = () => createClient({
    url: URL, key: KEY, fetch, now: () => NOW,
    localStorage: new MemoryStorage(), sessionStorage: new MemoryStorage(),
  });
  return { state, client };
}

async function signedIn(cloud) {
  const client = cloud.client();
  await client.signIn('owner@example.test', 'test-password', false);
  return client;
}

test('loading a legacy v1 journal migrates it in memory without writing or losing recorded snapshots', async () => {
  const original = fixture();
  const { foodBank: _bank, ...withoutBank } = original;
  const legacy = { ...withoutBank, version: 1 };
  const cloud = server(legacy);
  const client = await signedIn(cloud);
  const loaded = await client.load();
  assert.equal(loaded.version, 2);
  assert.deepEqual(loaded.foodBank, []);
  for (const field of ['foods', 'entries', 'days', 'defaults']) assert.deepEqual(loaded[field], legacy[field]);
  assert.equal(loaded.entries[0].calories, 331.125, 'historical energy is not recalculated during the format migration');
  assert.equal(cloud.state.data.version, 1, 'loading alone does not rewrite the server document');
  assert.equal(cloud.state.revision, 7);
  assert.equal(cloud.state.writes.length, 0);

  loaded.foodBank = structuredClone(original.foodBank);
  await client.save(loaded);
  assert.equal(cloud.state.data.version, 2);
  assert.deepEqual(cloud.state.data.foodBank, original.foodBank);
  assert.deepEqual(cloud.state.data.entries, legacy.entries);
  assert.equal(cloud.state.revision, 8);
  assert.equal(cloud.state.writes.length, 1);
});

test('a v2 bank survives whole-document saving and loading on another device', async () => {
  const initial = fixture();
  const cloud = server(initial);
  const first = await signedIn(cloud);
  const loaded = await first.load();
  loaded.foodBank[0].name = 'Lunch for tomorrow';
  await first.save(loaded);
  const second = await signedIn(cloud);
  const restored = await second.load();
  assert.deepEqual(restored, loaded);
  assert.deepEqual(restored.entries, initial.entries, 'preparing and saving food does not count it as eaten');
  assert.equal(restored.foodBank[0].items.length, 2);
  assert.equal(restored.foodBank[0].items[0].unit, 'g');
  assert.equal(restored.foodBank[0].items[1].unit, 'piece');
});

test('an unknown-format load does not adopt a writable revision that could overwrite a newer bank', async () => {
  const cloud = server({ ...fixture(), version: 3 });
  const client = await signedIn(cloud);
  await assert.rejects(client.load(), /supported Calories backup/);
  const requestsBeforeSave = cloud.state.calls.length;
  await assert.rejects(client.save(emptyData()), /Reload your saved data/);
  assert.equal(cloud.state.calls.length, requestsBeforeSave);
  assert.equal(cloud.state.writes.length, 0);
  assert.equal(cloud.state.data.version, 3);
});

test('a lost consumption response requires reload and cannot send a second consumption', async () => {
  const initial = fixture();
  const cloud = server(initial, { loseNextWriteResponse: true });
  const client = await signedIn(cloud);
  const loaded = await client.load();
  const portion = loaded.foodBank[0];
  const consumed = consumeFoodBankPortion(loaded, portion.id, '2026-10-15', 'Lunch');
  await assert.rejects(client.save(consumed), /Cannot reach your storage/);
  assert.equal(cloud.state.writes.length, 1);
  assert.deepEqual(cloud.state.data, consumed, 'removing the portion and adding every ingredient committed together');

  const retried = consumeFoodBankPortion(loaded, portion.id, '2026-10-15', 'Lunch');
  assert.deepEqual(retried, consumed, 'retrying the same stale portion has stable ingredient identities');
  const requestsAfterLoss = cloud.state.calls.length;
  await assert.rejects(client.save(retried), /Reload your saved data/);
  assert.equal(cloud.state.calls.length, requestsAfterLoss, 'an uncertain write is not blindly replayed');
  assert.equal(cloud.state.writes.length, 1);

  const fresh = await client.load();
  assert.deepEqual(fresh, consumed);
  assert.throws(() => consumeFoodBankPortion(fresh, portion.id, '2026-10-15', 'Lunch'), /no longer available/);
  for (const item of portion.items) assert.equal(fresh.entries.filter(entry => entry.id === item.id).length, 1);
});

test('two devices consuming the same mixed portion persist exactly one atomic winning state', async () => {
  const initial = fixture();
  const cloud = server(initial);
  const first = await signedIn(cloud);
  const second = await signedIn(cloud);
  const [firstData, secondData] = await Promise.all([first.load(), second.load()]);
  const portion = firstData.foodBank[0];
  const firstDraft = consumeFoodBankPortion(firstData, portion.id, '2026-10-15', 'Lunch');
  const secondDraft = consumeFoodBankPortion(secondData, portion.id, '2026-10-16', 'Dinner');
  const results = await Promise.allSettled([first.save(firstDraft), second.save(secondDraft)]);
  assert.equal(results.filter(result => result.status === 'fulfilled').length, 1);
  const winner = results.findIndex(result => result.status === 'fulfilled');
  const loser = 1 - winner;
  assert.match(results[loser].reason.message, /Another device saved changes first/);
  const drafts = [firstDraft, secondDraft];
  assert.deepEqual(cloud.state.data, drafts[winner]);
  assert.equal(cloud.state.writes.length, 1);
  assert.equal(cloud.state.revision, 8);
  assert.equal(cloud.state.data.foodBank.some(item => item.id === portion.id), false);
  assert.deepEqual(cloud.state.data.foodBank, initial.foodBank.slice(1));
  assert.deepEqual(cloud.state.data.entries.slice(0, initial.entries.length), initial.entries);
  assert.equal(cloud.state.data.entries.length, initial.entries.length + portion.items.length);
  for (const item of portion.items) {
    const matches = cloud.state.data.entries.filter(entry => entry.id === item.id);
    assert.equal(matches.length, 1);
    const { date, meal, ...snapshot } = matches[0];
    assert.deepEqual(snapshot, item, 'each mixed ingredient retains its own quantity, state and nutrient snapshot');
    assert.equal(date, winner === 0 ? '2026-10-15' : '2026-10-16');
    assert.equal(meal, winner === 0 ? 'Lunch' : 'Dinner');
  }
  const winningDate = winner === 0 ? '2026-10-15' : '2026-10-16';
  assert.equal(cloud.state.data.days[winningDate].burnedSource, 'garmin', 'the first meal does not inherit a legacy global manual burn');

  const clients = [first, second];
  const callsAfterConflict = cloud.state.calls.length;
  await assert.rejects(clients[loser].save(drafts[loser]), /Reload your saved data/);
  assert.equal(cloud.state.calls.length, callsAfterConflict);
  const refreshed = await clients[loser].load();
  assert.throws(() => consumeFoodBankPortion(refreshed, portion.id, winningDate, 'Lunch'), /no longer available/);
});
