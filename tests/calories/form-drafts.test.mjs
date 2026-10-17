import test from 'node:test';
import assert from 'node:assert/strict';
import { clearFormDrafts, listFormDrafts, readFormDraft, removeFormDraft, restoreFormDraft, writeFormDraft } from '../../src/lib/calories/form-drafts.ts';
import { prepareFoodBankPortions } from '../../src/lib/calories/food-bank.ts';

function storage() {
  const values = new Map();
  return {
    get length() { return values.size; }, key: index => [...values.keys()][index] ?? null,
    getItem: key => values.get(key) ?? null, setItem: (key, value) => values.set(key, value), removeItem: key => values.delete(key),
    values,
  };
}
const account = 'https://example.supabase.co:owner-a';
const descriptor = { kind: 'entry', label: 'Restaurant lunch', date: '2026-10-16' };

test('draft recovery preserves unfinished strings, blanks, zero text, and selections without numeric coercion', () => {
  const local = storage();
  const value = { mode: 'manual', meal: 'Custom', customMeal: 'Meal 2', date: '', quantity: '', name: 'Lunch', nutrients: { protein: '0', carbs: '12.50', fat: '', fiber: '' } };
  writeFormDraft(account, 'entry:new', value, descriptor, local, 1234);
  assert.deepEqual(readFormDraft(account, 'entry:new', local), { value, summary: { ...descriptor, key: 'entry:new', updatedAt: 1234 } });
  assert.deepEqual(restoreFormDraft(value, readFormDraft(account, 'entry:new', local).value), value);
});

test('project and owner scopes are isolated for reads, resume summaries, and sign-out clearing', () => {
  const local = storage();
  const otherOwner = 'https://example.supabase.co:owner-b';
  const otherProject = 'https://other.supabase.co:owner-a';
  const similarScope = `${account}:extra`;
  for (const scope of [account, otherOwner, otherProject, similarScope]) writeFormDraft(scope, 'entry:new', { name: scope }, descriptor, local, 100);
  assert.equal(listFormDrafts(account, local).length, 1);
  assert.equal(readFormDraft(otherOwner, 'entry:new', local).value.name, otherOwner);
  clearFormDrafts(account, local);
  assert.equal(readFormDraft(account, 'entry:new', local), null);
  for (const scope of [otherOwner, otherProject, similarScope]) assert.equal(listFormDrafts(scope, local).length, 1);
});

test('resume summaries expose only editor metadata, sorted by latest edit with stable ties', () => {
  const local = storage();
  writeFormDraft(account, 'targets:2026-10-16', { deficit: '850' }, { kind: 'targets', label: 'Daily targets', date: '2026-10-16' }, local, 100);
  writeFormDraft(account, 'entry:new', { name: 'Private typed text', nutrients: { fat: '44' } }, { ...descriptor, foodId: 'selected-food' }, local, 200);
  writeFormDraft(account, 'bank:prepare', { count: '4' }, { kind: 'bank-prepare', label: 'Prepared portions' }, local, 200);
  const summaries = listFormDrafts(account, local);
  assert.deepEqual(summaries.map(item => item.key), ['bank:prepare', 'entry:new', 'targets:2026-10-16']);
  assert.equal(summaries[1].foodId, 'selected-food');
  assert.ok(!JSON.stringify(summaries).includes('Private typed text'));
  assert.ok(!JSON.stringify(summaries).includes('850'));
});

test('discarding a single form keeps unrelated unfinished forms and replaces a form atomically', () => {
  const local = storage();
  writeFormDraft(account, 'entry:new', { name: 'First' }, descriptor, local, 1);
  writeFormDraft(account, 'food:new', { name: 'Oats' }, { kind: 'food', label: 'Oats' }, local, 2);
  writeFormDraft(account, 'entry:new', { name: 'Second' }, descriptor, local, 3);
  assert.equal(local.length, 2);
  assert.equal(readFormDraft(account, 'entry:new', local).value.name, 'Second');
  removeFormDraft(account, 'entry:new', local);
  assert.equal(readFormDraft(account, 'entry:new', local), null);
  assert.equal(readFormDraft(account, 'food:new', local).value.name, 'Oats');
});

test('photo payloads and live browser objects cannot enter form-draft storage', () => {
  const local = storage();
  for (const value of [{ photo: { data: 'abc' } }, { nested: { images: [] } }, { source: 'data:image/jpeg;base64,abc' }, { source: 'blob:https://example.test/id' }, { selected: new Blob(['photo']) }, { number: Infinity }]) {
    assert.throws(() => writeFormDraft(account, 'entry:new', value, descriptor, local));
  }
  assert.equal(local.length, 0);
  writeFormDraft(account, 'entry:new', { nutrients: { protein: '35', fiber: '' }, importedLabel: true }, descriptor, local);
  assert.equal(readFormDraft(account, 'entry:new', local).value.importedLabel, true);
});

test('malformed or oversized records do not crash recovery or appear in the resume list', () => {
  const local = storage();
  writeFormDraft(account, 'entry:new', { name: 'Good' }, descriptor, local, 1);
  const key = [...local.values.keys()][0];
  for (const encoded of ['{', '{}', 'x'.repeat(70000), JSON.stringify({ version: 1, summary: { ...descriptor, key: 'wrong-key', updatedAt: 1 }, value: { name: 'Wrong' } })]) {
    local.setItem(key, encoded);
    assert.equal(readFormDraft(account, 'entry:new', local), null);
    assert.deepEqual(listFormDrafts(account, local), []);
  }
});

test('restore uses the current editor shape, keeping new defaults and rejecting incompatible values', () => {
  const seed = { name: '', count: '1', ingredients: [{ key: 0, foodId: 'first', quantity: '', query: '' }], newOption: false };
  const raw = { name: 'Meal prep', count: '4', ingredients: [{ key: 2, foodId: 'rice', quantity: '100', query: 'ric' }, { key: 3, foodId: 'beans', quantity: '0.5', query: '' }], obsoleteField: 'ignore' };
  const restored = restoreFormDraft(seed, raw);
  assert.deepEqual(restored, { name: 'Meal prep', count: '4', ingredients: raw.ingredients, newOption: false });
  assert.equal(restoreFormDraft(seed, { ...raw, count: 4 }), null);
  assert.equal(restoreFormDraft(seed, { ...raw, ingredients: [{ key: 'bad', foodId: 'rice', quantity: '100', query: '' }] }), null);
  assert.deepEqual(seed.ingredients, [{ key: 0, foodId: 'first', quantity: '', query: '' }]);
});

test('write failures are explicit and cannot erase an older recoverable draft', () => {
  const local = storage();
  writeFormDraft(account, 'entry:new', { name: 'Recover me' }, descriptor, local, 1);
  const unavailable = { ...local, setItem() { throw new Error('QuotaExceededError'); } };
  assert.throws(() => writeFormDraft(account, 'entry:new', { name: 'New input' }, descriptor, unavailable), /could not be saved/);
  assert.equal(readFormDraft(account, 'entry:new', local).value.name, 'Recover me');
  const oversized = Object.fromEntries(Array.from({ length: 8 }, (_, index) => [`field${index}`, 'x'.repeat(10000)]));
  assert.throws(() => writeFormDraft(account, 'entry:new', oversized, descriptor, local), /could not be saved/);
  assert.equal(readFormDraft(account, 'entry:new', local).value.name, 'Recover me');
});

test('new entry and food drafts keep their creation identity across reloads without becoming edit descriptors', () => {
  const local = storage();
  for (const [key, field, kind] of [['entry:new', 'draftEntryId', 'entry'], ['food:new', 'draftFoodId', 'food']]) {
    const original = { [field]: `${kind}-stable-id`, name: 'Typed food', amount: '125' };
    writeFormDraft(account, key, original, { kind, label: 'Typed food' }, local, 1);
    const reloaded = restoreFormDraft({ ...original, [field]: 'new-id-from-a-new-mount', name: '', amount: '' }, readFormDraft(account, key, local).value);
    assert.equal(reloaded[field], `${kind}-stable-id`);
    assert.equal(reloaded.amount, '125');
    assert.equal(listFormDrafts(account, local).find(item => item.key === key).entryId, undefined);
    assert.equal(listFormDrafts(account, local).find(item => item.key === key).foodId, undefined);
  }
});

test('a recovered batch generates the same portion and ingredient IDs when a successful stage is retried', () => {
  const local = storage();
  const food = { id: 'rice', name: 'Rice', baseQuantity: 100, unit: 'g', state: 'Cooked', protein: 3, carbs: 28, fat: 1, fiber: 1, calories: 133 };
  const seed = { draftBatchId: 'batch-unique', name: 'Lunch', count: '3', ingredients: [{ key: 0, foodId: 'rice', quantity: '150', query: '' }] };
  writeFormDraft(account, 'bank:prepare', seed, { kind: 'bank-prepare', label: 'Lunch' }, local, 1);
  const recovered = restoreFormDraft({ ...seed, draftBatchId: 'different-new-mount' }, readFormDraft(account, 'bank:prepare', local).value);
  const prepare = draft => {
    let next = 0;
    return prepareFoodBankPortions([food], draft.ingredients.map(item => ({ foodId: item.foodId, quantity: Number(item.quantity) })), Number(draft.count), draft.name, () => `${draft.draftBatchId}:${next++}`);
  };
  const first = prepare(seed);
  const retry = prepare(recovered);
  assert.deepEqual(retry, first);
  const ids = retry.flatMap(portion => [portion.id, ...portion.items.map(item => item.id)]);
  assert.equal(new Set(ids).size, 6);
});

test('blocked draft storage cannot throw during sign-out cleanup', () => {
  const blocked = { get length() { throw new Error('SecurityError'); }, key() { throw new Error('SecurityError'); }, getItem() { throw new Error('SecurityError'); }, setItem() { throw new Error('SecurityError'); }, removeItem() { throw new Error('SecurityError'); } };
  assert.equal(clearFormDrafts(account, blocked), false);
  assert.equal(readFormDraft(account, 'entry:new', blocked), null);
  assert.deepEqual(listFormDrafts(account, blocked), []);
});

test('a recovered edit retains its original record and defaults baseline when the current record has changed', () => {
  const local = storage();
  const original = { id: 'existing', name: 'Original meal', quantity: 100 };
  const changed = { id: 'existing', name: 'Changed on Mac', quantity: 200 };
  const oldDefaults = { burned: 0, deficit: 500, protein: 175 };
  const newDefaults = { burned: 0, deficit: 700, protein: 175 };
  for (const kind of ['entry', 'food', 'targets']) {
    const key = `${kind}:existing`;
    const originalDraft = { name: 'My unfinished edit', originalRecordJson: JSON.stringify(original), originalDefaultsJson: JSON.stringify(oldDefaults) };
    writeFormDraft(account, key, originalDraft, { kind, label: 'Unfinished edit' }, local, 1);
    const currentSeed = { name: changed.name, originalRecordJson: JSON.stringify(changed), originalDefaultsJson: JSON.stringify(newDefaults) };
    const recovered = restoreFormDraft(currentSeed, readFormDraft(account, key, local).value);
    assert.equal(recovered.name, 'My unfinished edit');
    assert.deepEqual(JSON.parse(recovered.originalRecordJson), original);
    assert.deepEqual(JSON.parse(recovered.originalDefaultsJson), oldDefaults);
    assert.notEqual(recovered.originalRecordJson, currentSeed.originalRecordJson);
    removeFormDraft(account, key, local);
    assert.equal(readFormDraft(account, key, local), null);
    assert.deepEqual(JSON.parse(currentSeed.originalRecordJson), changed);
  }
});
