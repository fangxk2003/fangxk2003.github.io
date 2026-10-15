import test from 'node:test';
import assert from 'node:assert/strict';
import { consumeFoodBankPortion, prepareFoodBankPortions } from '../../src/lib/calories/food-bank.ts';
import { dayTotals, emptyData, normalizeData } from '../../src/lib/calories/model.ts';
import { calorieLedger } from '../../src/lib/calories/budget.ts';
import { sortFoodsByRecentUse } from '../../src/lib/calories/food-order.ts';

const rice = Object.freeze({ id: 'rice', name: 'Rice', baseQuantity: 100, unit: 'g', state: 'Cooked', protein: 3, carbs: 28, fat: 1, fiber: 2, calories: 999 });
const egg = Object.freeze({ id: 'egg', name: 'Eggs', baseQuantity: 2, unit: 'piece', state: 'Raw', protein: 12, carbs: 1, fat: 10, fiber: 0, calories: 999 });
const foods = Object.freeze([rice, egg]);
const ids = (prefix = 'id') => { let next = 0; return () => `${prefix}-${++next}`; };
const prepare = (count = 1) => prepareFoodBankPortions(foods, [{ foodId: 'rice', quantity: 150 }, { foodId: 'egg', quantity: 1 }], count, 'Rice and egg', ids());
const bankData = (count = 1) => ({ ...emptyData(), foods: foods.map(food => ({ ...food })), foodBank: prepare(count) });
const copy = value => structuredClone(value);
const freeze = value => {
  if (value && typeof value === 'object') { Object.freeze(value); Object.values(value).forEach(freeze); }
  return value;
};

test('new journals use version 2 with an empty food bank', () => {
  const data = emptyData();
  assert.equal(data.version, 2);
  assert.deepEqual(data.foodBank, []);
  assert.deepEqual(normalizeData(data), data);
});

test('preparation snapshots mixed units using quantities per portion and 4/4/9 energy', () => {
  const ingredients = freeze([{ foodId: 'rice', quantity: 150 }, { foodId: 'egg', quantity: 1 }]);
  const portions = prepareFoodBankPortions(foods, ingredients, 3, '  Rice and egg  ', ids());
  assert.equal(portions.length, 3);
  assert.equal(portions[0].name, 'Rice and egg');
  assert.deepEqual(portions[0].items[0], {
    id: 'id-2', foodId: 'rice', foodName: 'Rice', quantity: 150, unit: 'g', state: 'Cooked',
    protein: 4.5, carbs: 42, fat: 1.5, fiber: 3, calories: 199.5,
  });
  assert.deepEqual(portions[0].items[1], {
    id: 'id-3', foodId: 'egg', foodName: 'Eggs', quantity: 1, unit: 'piece', state: 'Raw',
    protein: 6, carbs: 0.5, fat: 5, fiber: 0, calories: 71,
  });
  for (const portion of portions) assert.deepEqual(portion.items.map(item => item.quantity), [150, 1]);
  assert.equal(portions.reduce((sum, portion) => sum + portion.items[0].quantity, 0), 450);
  assert.equal(rice.calories, 999, 'the original library is untouched');
  assert.deepEqual(ingredients, [{ foodId: 'rice', quantity: 150 }, { foodId: 'egg', quantity: 1 }]);
});

test('each prepared portion has independent ingredient objects and globally distinct IDs', () => {
  const portions = prepare(3);
  const allIds = portions.flatMap(portion => [portion.id, ...portion.items.map(item => item.id)]);
  assert.equal(new Set(allIds).size, 9);
  assert.notEqual(portions[0].items, portions[1].items);
  assert.notEqual(portions[0].items[0], portions[1].items[0]);
  portions[0].items[0].quantity = 1;
  portions[0].items[0].calories = 1;
  assert.equal(portions[1].items[0].quantity, 150);
  assert.equal(portions[1].items[0].calories, 199.5);
});

test('batch sizes and ingredient counts have inclusive bounds without dividing quantities', () => {
  const maximum = prepareFoodBankPortions(foods, Array.from({ length: 50 }, () => ({ foodId: 'egg', quantity: 0.5 })), 100, 'Batch', ids());
  assert.equal(maximum.length, 100);
  assert.equal(maximum[99].items.length, 50);
  assert.equal(maximum[99].items[49].quantity, 0.5);
  for (const count of [0, -1, 1.1, 101, NaN, Infinity, '2']) {
    assert.throws(() => prepareFoodBankPortions(foods, [{ foodId: 'egg', quantity: 1 }], count, 'Batch', ids()));
  }
  for (const ingredients of [[], Array(51).fill({ foodId: 'egg', quantity: 1 }), null, Array(1)]) {
    assert.throws(() => prepareFoodBankPortions(foods, ingredients, 1, 'Batch', ids()));
  }
});

test('preparation rejects invalid ingredients, names, libraries, generated IDs, and numeric overflow', () => {
  for (const quantity of [0, -1, NaN, Infinity, '1', null, 100_000_001]) {
    assert.throws(() => prepareFoodBankPortions(foods, [{ foodId: 'rice', quantity }], 1, 'Batch', ids()));
  }
  assert.throws(() => prepareFoodBankPortions(foods, [{ foodId: 'deleted', quantity: 1 }], 1, 'Batch', ids()), /existing food/);
  assert.throws(() => prepareFoodBankPortions(foods, [null], 1, 'Batch', ids()));
  for (const name of ['', '  ', null, 'x'.repeat(201)]) assert.throws(() => prepareFoodBankPortions(foods, [{ foodId: 'egg', quantity: 1 }], 1, name, ids()));
  assert.throws(() => prepareFoodBankPortions([rice, rice], [{ foodId: 'rice', quantity: 1 }], 1, 'Batch', ids()), /duplicate IDs/);
  assert.throws(() => prepareFoodBankPortions(foods, [{ foodId: 'rice', quantity: 1 }], 1, 'Batch', () => 'same'), /unique/);
  for (const id of ['', ' ', null, 'x'.repeat(201)]) assert.throws(() => prepareFoodBankPortions(foods, [{ foodId: 'rice', quantity: 1 }], 1, 'Batch', () => id));
  assert.throws(() => prepareFoodBankPortions([{ ...rice, baseQuantity: Number.MIN_VALUE }], [{ foodId: 'rice', quantity: 1 }], 1, 'Batch', ids()), /Invalid/);
});

test('consumption atomically moves exactly one portion into the journal with its snapshot IDs', () => {
  const initial = freeze(bankData(3));
  const before = copy(initial);
  const selected = initial.foodBank[1];
  const result = consumeFoodBankPortion(initial, selected.id, '2026-10-15', ' Lunch ');
  assert.equal(result.foodBank.length, 2);
  assert.deepEqual(result.foodBank.map(portion => portion.id), [initial.foodBank[0].id, initial.foodBank[2].id]);
  assert.deepEqual(result.entries, selected.items.map(item => ({ ...item, date: '2026-10-15', meal: 'Lunch' })));
  assert.equal(result.entries.every(entry => !Object.hasOwn(entry, 'source')), true);
  assert.deepEqual(initial, before);
  assert.notEqual(result.entries[0], selected.items[0]);
  assert.notEqual(result.foodBank[0].items[0], initial.foodBank[0].items[0]);
  assert.throws(() => consumeFoodBankPortion(result, selected.id, '2026-10-15', 'Dinner'), /no longer available/);
  assert.equal(result.entries.length, 2, 'a repeated attempt cannot duplicate intake');
});

test('consumption keeps explicit day targets and freezes untouched-day settings before logging', () => {
  const initial = bankData(2);
  initial.defaults = { burned: 3200, deficit: 450, protein: 175, burnedSource: 'manual' };
  initial.days['2026-10-14'] = { burned: 2800, deficit: 350, protein: 175, burnedSource: 'manual' };
  const withExplicitDay = consumeFoodBankPortion(initial, initial.foodBank[0].id, '2026-10-14', 'Dinner');
  assert.deepEqual(withExplicitDay.days['2026-10-14'], initial.days['2026-10-14']);
  const withNewDay = consumeFoodBankPortion(withExplicitDay, withExplicitDay.foodBank[0].id, '2026-10-15', 'Lunch');
  assert.deepEqual(withNewDay.days['2026-10-15'], { burned: 0, deficit: 450, protein: 175, burnedSource: 'garmin' });
  assert.deepEqual(withNewDay.defaults, initial.defaults);
});

test('legacy logged-day defaults stay historical when another portion is consumed that day', () => {
  const initial = bankData(2);
  initial.defaults = { burned: 3200, deficit: 400, protein: 175, burnedSource: 'manual' };
  initial.entries = [{ ...initial.foodBank[0].items[0], id: 'existing', date: '2026-10-14', meal: 'Breakfast' }];
  const result = consumeFoodBankPortion(initial, initial.foodBank[1].id, '2026-10-14', 'Dinner');
  assert.deepEqual(result.days['2026-10-14'], initial.defaults);
  assert.equal(initial.days['2026-10-14'], undefined);
});

test('library edits and deletion never change stored or consumed nutrient snapshots', () => {
  const initial = bankData(2);
  const snapshots = copy(initial.foodBank);
  initial.foods[0] = { ...initial.foods[0], name: 'New rice', carbs: 1, protein: 1, fat: 1, fiber: 1, calories: 17 };
  initial.foods = initial.foods.filter(food => food.id !== 'egg');
  const result = consumeFoodBankPortion(initial, initial.foodBank[0].id, '2026-10-15', 'Lunch');
  assert.deepEqual(result.entries.map(({ date, meal, ...item }) => item), snapshots[0].items);
  assert.deepEqual(result.foodBank[0], snapshots[1]);
  assert.equal(result.entries[0].foodName, 'Rice');
  assert.equal(result.entries[1].foodName, 'Eggs');
  result.foods = [];
  assert.deepEqual(normalizeData(result).entries, result.entries);
});

test('failed consumption leaves all portions, entries, and settings unchanged', () => {
  const initial = bankData();
  const before = copy(initial);
  for (const [id, date, meal] of [
    ['missing', '2026-10-15', 'Lunch'],
    [initial.foodBank[0].id, '2026-02-30', 'Lunch'],
    [initial.foodBank[0].id, '2026-10-1', 'Lunch'],
    [initial.foodBank[0].id, '2026-10-15', ' '],
    [initial.foodBank[0].id, '2026-10-15', 'x'.repeat(201)],
  ]) {
    assert.throws(() => consumeFoodBankPortion(initial, id, date, meal));
    assert.deepEqual(initial, before);
  }
  const stale = copy(initial);
  stale.entries.push({ ...stale.foodBank[0].items[0], date: '2026-10-14', meal: 'Already consumed' });
  assert.throws(() => consumeFoodBankPortion(stale, stale.foodBank[0].id, '2026-10-15', 'Lunch'), /duplicate IDs/);
  assert.equal(stale.foodBank.length, 1);
  assert.equal(stale.entries.length, 1);
});

test('legacy version 1 backups migrate with no food bank and preserve recorded energy', () => {
  const legacy = { ...emptyData(), version: 1, foods: [{ ...rice }], entries: [{ ...prepare()[0].items[0], id: 'historical', date: '2026-10-12', meal: 'Lunch', calories: 123.456 }] };
  delete legacy.foodBank;
  const result = normalizeData(legacy);
  assert.equal(result.version, 2);
  assert.deepEqual(result.foodBank, []);
  assert.equal(result.entries[0].calories, 123.456);
  assert.equal(result.foods[0].calories, 133);
  assert.equal(legacy.version, 1);
  assert.equal(Object.hasOwn(legacy, 'foodBank'), false);
});

test('version 2 JSON backup roundtrips preserve bank snapshots without recalculating their calories', () => {
  const initial = bankData(2);
  initial.foodBank[0].items[0].calories = 123.456;
  initial.foodBank[0].items[0].foodId = 'deleted-library-reference';
  const expected = normalizeData(initial);
  const result = normalizeData(JSON.parse(JSON.stringify(expected)));
  assert.deepEqual(result, expected);
  assert.equal(result.foodBank[0].items[0].calories, 123.456);
  assert.notEqual(result.foodBank[0].items[0], expected.foodBank[0].items[0]);
  const consumed = consumeFoodBankPortion(result, result.foodBank[0].id, '2026-10-15', 'Lunch');
  assert.equal(consumed.entries[0].calories, 123.456);
});

test('version 2 requires the food-bank array and rejects future backup versions', () => {
  for (const foodBank of [undefined, null, {}, 'bank', Array(10001)]) {
    assert.throws(() => normalizeData({ ...emptyData(), foodBank }));
  }
  for (const version of [0, 3, 99, '2', undefined]) assert.throws(() => normalizeData({ ...emptyData(), version }));
});

test('bank validation rejects missing and excessive ingredients, malformed records, and reserved journal fields', () => {
  for (const mutate of [
    data => { data.foodBank = Array(1); },
    data => { data.foodBank[0] = null; },
    data => { data.foodBank[0].items = []; },
    data => { data.foodBank[0].items = Array(51).fill(data.foodBank[0].items[0]); },
    data => { data.foodBank[0].items = Array(1); },
    data => { data.foodBank[0].items = null; },
    data => { data.foodBank[0].items[0] = null; },
    data => { data.foodBank[0].name = ' '; },
    data => { data.foodBank[0].items[0].date = '2026-10-15'; },
    data => { data.foodBank[0].items[0].meal = 'Lunch'; },
    data => { data.foodBank[0].items[0].source = 'manual'; },
  ]) {
    const data = bankData(); mutate(data); assert.throws(() => normalizeData(data));
  }
});

test('bank validation applies the same strict snapshot number, identity, unit, and state rules as the journal', () => {
  for (const field of ['protein', 'carbs', 'fat', 'fiber', 'calories', 'quantity']) {
    for (const value of [NaN, Infinity, -1, '2', null, 100_000_001]) {
      const data = bankData(); data.foodBank[0].items[0][field] = value;
      assert.throws(() => normalizeData(data), undefined, `${field}=${String(value)}`);
    }
  }
  for (const [field, value] of [['quantity', 0], ['foodId', ''], ['foodName', ' '], ['id', null], ['unit', 'kg'], ['state', 'Frozen']]) {
    const data = bankData(); data.foodBank[0].items[0][field] = value; assert.throws(() => normalizeData(data));
  }
});

test('bank ingredient identities must be globally unique and distinct from logged entries', () => {
  for (const mutate of [
    data => { data.foodBank[1].id = data.foodBank[0].id; },
    data => { data.foodBank[0].items[1].id = data.foodBank[0].items[0].id; },
    data => { data.foodBank[1].items[0].id = data.foodBank[0].items[0].id; },
    data => { data.entries.push({ ...data.foodBank[0].items[0], date: '2026-10-14', meal: 'Lunch' }); },
  ]) {
    const data = bankData(2); mutate(data); assert.throws(() => normalizeData(data), /duplicate IDs/);
  }
});

test('unconsumed preparation contributes no daily nutrients, recent food use, or calorie carry', () => {
  const initial = normalizeData({ ...emptyData(), foods: foods.map(food => ({ ...food })), defaults: { burned: 3000, deficit: 500, protein: 175, burnedSource: 'manual' } });
  const withBank = { ...initial, foodBank: prepare(3) };
  assert.deepEqual(dayTotals(withBank.entries), dayTotals(initial.entries));
  assert.deepEqual(sortFoodsByRecentUse(withBank.foods, withBank.entries, '2026-10-15'), sortFoodsByRecentUse(initial.foods, initial.entries, '2026-10-15'));
  assert.deepEqual(calorieLedger(withBank, [], '2026-10-15'), { days: {}, completed: [] });
  assert.deepEqual(withBank.days, {});
  const consumed = consumeFoodBankPortion(withBank, withBank.foodBank[0].id, '2026-10-14', 'Lunch');
  assert.equal(dayTotals(consumed.entries).calories, 270.5);
  assert.equal(consumed.foodBank.length, 2);
});
