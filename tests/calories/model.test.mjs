import test from 'node:test';
import assert from 'node:assert/strict';
import { addDays, dayTotals, emptyData, energyFromMacros, macrosForFood, normalizeData } from '../../src/lib/calories/model.ts';

const food = {
  id: 'food-1', name: 'Peas', baseQuantity: 100, unit: 'g', state: 'Cooked',
  carbs: 12, protein: 6, fat: 1, fiber: 5, calories: 85,
};
const entry = {
  id: 'entry-1', date: '2026-10-12', meal: 'Lunch', foodId: food.id,
  foodName: food.name, quantity: 150, unit: food.unit, state: food.state,
  ...macrosForFood(food, 150),
};
const backup = () => ({ ...emptyData(), foods: [{ ...food }], entries: [{ ...entry }] });

test('energy derives from protein, carbohydrate and fat while excluding fibre', () => {
  assert.equal(energyFromMacros({ protein: 30, carbs: 1, fat: 7, fiber: 0 }), 187);
  assert.equal(energyFromMacros({ protein: 30, carbs: 1, fat: 7, fiber: 10 }), 187);
  assert.equal(energyFromMacros({ protein: 0, carbs: 10, fat: 0, fiber: 5 }), 40);
  assert.equal(energyFromMacros({ protein: 0, carbs: 0, fat: 0, fiber: 5 }), 0);
});

test('derived energy retains precision when scaling decimal portions', () => {
  const nutrients = { protein: 2.5, carbs: 1.25, fat: 0.5, fiber: 1.5 };
  const reference = { ...food, ...nutrients, calories: energyFromMacros(nutrients) };
  assert.equal(reference.calories, 19.5);
  assert.equal(macrosForFood(reference, 37.5).calories, 7.3125);
});

test('quantity scales every nutrient from the food reference portion', () => {
  assert.deepEqual(macrosForFood(food, 150), {
    carbs: 18, protein: 9, fat: 1.5, fiber: 7.5, calories: 127.5,
  });
  assert.deepEqual(macrosForFood({ ...food, baseQuantity: 2, unit: 'piece' }, 1), {
    carbs: 6, protein: 3, fat: 0.5, fiber: 2.5, calories: 42.5,
  });
});

test('day totals sum nutrient snapshots without changing meal records', () => {
  const first = Object.freeze({ ...entry });
  const second = Object.freeze({ ...entry, id: 'entry-2', ...macrosForFood(food, 50) });
  assert.deepEqual(dayTotals([first, second]), {
    carbs: 24, protein: 12, fat: 2, fiber: 10, calories: 170,
  });
  assert.deepEqual(dayTotals([]), { carbs: 0, protein: 0, fat: 0, fiber: 0, calories: 0 });
  assert.equal(first.quantity, 150);
});

test('normalization preserves historical nutrient snapshots and day-specific targets', () => {
  const data = backup();
  data.foods[0].calories = 99;
  data.days['2026-10-12'] = { burned: 2900, deficit: 450, protein: 160 };
  const result = normalizeData(data);
  assert.equal(result.foods[0].calories, 81);
  assert.equal(result.entries[0].calories, 127.5);
  assert.equal(macrosForFood(result.foods[0], 150).calories, 121.5);
  const saved = result.entries[0];
  assert.equal(macrosForFood({ ...saved, id: saved.foodId, name: saved.foodName, baseQuantity: saved.quantity }, 75).calories, 63.75);
  assert.deepEqual(result.days['2026-10-12'], { burned: 2900, deficit: 450, protein: 160 });
  assert.notEqual(result.entries[0], data.entries[0]);
});

test('import rejects non-finite, negative, coerced, and excessively large nutrient values', () => {
  for (const value of [NaN, Infinity, -Infinity, -1, '12', null, 100_000_001]) {
    const data = backup();
    data.foods[0].protein = value;
    assert.throws(() => normalizeData(data), /Invalid protein/, `protein=${String(value)}`);
  }
  for (const [field, value] of [['baseQuantity', 0], ['baseQuantity', -1]]) {
    const data = backup();
    data.foods[0][field] = value;
    assert.throws(() => normalizeData(data), /Invalid base quantity/);
  }
  const zeroQuantity = backup();
  zeroQuantity.entries[0].quantity = 0;
  assert.throws(() => normalizeData(zeroQuantity), /Invalid quantity/);
});

test('import rejects impossible or ambiguous dates, including per-day setting keys', () => {
  for (const invalidDate of ['2026-02-29', '2026-02-30', '2026-04-31', '2026-13-01', '2026-10-1', '12/10/2026']) {
    const data = backup();
    data.entries[0].date = invalidDate;
    assert.throws(() => normalizeData(data), /Invalid date/, invalidDate);
  }
  const data = backup();
  data.days['2026-02-30'] = { burned: 3000, deficit: 500, protein: 175 };
  assert.throws(() => normalizeData(data), /Invalid date/);
  data.entries[0].date = '2024-02-29';
  data.days = {};
  assert.equal(normalizeData(data).entries[0].date, '2024-02-29');
});

test('import rejects duplicate identities and malformed records without silently dropping them', () => {
  const duplicateFood = backup();
  duplicateFood.foods.push({ ...food, name: 'Different food' });
  assert.throws(() => normalizeData(duplicateFood), /duplicate IDs/);
  const duplicateEntry = backup();
  duplicateEntry.entries.push({ ...entry, meal: 'Dinner' });
  assert.throws(() => normalizeData(duplicateEntry), /duplicate IDs/);
  for (const raw of [null, [], { ...emptyData(), version: 2 }, { ...emptyData(), foods: null }]) {
    assert.throws(() => normalizeData(raw));
  }
});

test('calendar navigation crosses month, leap year, and daylight saving boundaries', () => {
  assert.equal(addDays('2026-10-25', 1), '2026-10-26');
  assert.equal(addDays('2024-03-01', -1), '2024-02-29');
  assert.equal(addDays('2026-12-31', 1), '2027-01-01');
});
