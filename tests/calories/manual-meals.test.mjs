import test from 'node:test';
import assert from 'node:assert/strict';
import { dayTotals, emptyData, manualMealEntry, normalizeData } from '../../src/lib/calories/model.ts';

const estimate = (overrides = {}) => manualMealEntry({
  id: 'restaurant-1', date: '2026-10-13', meal: 'Dinner', name: 'Restaurant pasta',
  protein: 30, carbs: 80, fat: 20, fiber: 5, ...overrides,
});

test('manual meals store whole-meal nutrient totals and computed energy without a food-library item', () => {
  const entry = estimate();
  assert.equal(entry.source, 'manual');
  assert.equal(entry.foodId, 'manual:restaurant-1');
  assert.equal(entry.quantity, 1);
  assert.equal(entry.calories, 620);
  assert.deepEqual(dayTotals([entry]), { protein: 30, carbs: 80, fat: 20, fiber: 5, calories: 620 });
  assert.equal(estimate({ fiber: 50 }).calories, 620, 'fibre stays separate from energy');
  assert.equal(estimate({ protein: 12.5, carbs: 31.25, fat: 6.5 }).calories, 233.5);
});

test('manual and legacy library entries round-trip together through a JSON backup', () => {
  const manual = estimate();
  const library = { ...manual, id: 'legacy-entry', foodId: 'removed-food', foodName: 'Old library food', quantity: 150, unit: 'g', state: 'Cooked' };
  delete library.source;
  const data = { ...emptyData(), entries: [manual, library] };
  const loaded = normalizeData(JSON.parse(JSON.stringify(data)));
  assert.deepEqual(loaded.entries, [manual, library]);
  assert.deepEqual(loaded.foods, [], 'manual meals never create a library record');
  assert.deepEqual(dayTotals(loaded.entries), { protein: 60, carbs: 160, fat: 40, fiber: 10, calories: 1240 });
  assert.equal(loaded.entries[1].source, undefined);
});

test('editing a manual meal recalculates energy while retaining its journal identity', () => {
  const original = estimate();
  const edited = estimate({ id: original.id, protein: 35, carbs: 85, fat: 22 });
  assert.equal(edited.id, original.id);
  assert.equal(edited.foodId, original.foodId);
  assert.equal(edited.calories, 678);
  assert.equal(original.calories, 620);
  assert.equal(estimate({ protein: 0, carbs: 0, fat: 0, fiber: 0 }).calories, 0);
});

test('manual identity survives an older tab saving without the new source metadata', () => {
  const entry = estimate();
  delete entry.source;
  const loaded = normalizeData({ ...emptyData(), entries: [entry] });
  assert.equal(loaded.entries[0].source, 'manual');
  assert.equal(loaded.entries[0].calories, 620);
  assert.equal(loaded.entries[0].foodId, 'manual:restaurant-1');
});

test('manual estimates reject malformed values and excessive calculated energy', () => {
  for (const key of ['protein', 'carbs', 'fat', 'fiber']) {
    for (const value of [-1, NaN, Infinity, '30', null, 100_000_001]) {
      assert.throws(() => estimate({ [key]: value }), /Invalid/);
    }
  }
  assert.throws(() => estimate({ fat: 20_000_000 }), /Invalid calculated calories/);
  assert.throws(() => estimate({ name: '  ' }), /Invalid meal description/);
  assert.throws(() => estimate({ meal: '' }), /Invalid meal/);
  assert.throws(() => estimate({ date: '2026-02-30' }), /Invalid date/);
  assert.equal(estimate({ name: '  Pasta  ', meal: '  Dinner  ' }).foodName, 'Pasta');
});

test('backup import validates optional entry source metadata', () => {
  for (const source of ['library', 'unknown', true, null]) {
    const data = { ...emptyData(), entries: [{ ...estimate(), source }] };
    assert.throws(() => normalizeData(data), /Invalid entry source/);
  }
});
