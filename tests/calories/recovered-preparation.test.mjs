import test from 'node:test';
import assert from 'node:assert/strict';
import { emptyData } from '../../src/lib/calories/model.ts';
import { addPreparedPortions, consumeFoodBankPortion, prepareFoodBankPortions } from '../../src/lib/calories/food-bank.ts';

function batch() {
  const food = { id: 'rice', name: 'Rice', baseQuantity: 100, unit: 'g', state: 'Cooked', protein: 2, carbs: 25, fat: 1, fiber: 1, calories: 117 };
  let sequence = 0;
  return prepareFoodBankPortions([food], [{ foodId: food.id, quantity: 150 }], 2, 'Lunch', () => `recovered-batch-${sequence++}`);
}
test('a preparation recovered after its local save cannot create the same portions twice', () => {
  const portions = batch();
  const saved = addPreparedPortions(emptyData(), portions);
  const restored = addPreparedPortions(saved, batch());
  assert.deepEqual(restored, saved);
  assert.equal(restored.foodBank.length, 2);
  const changed = batch(); changed[0].items[0].quantity = 200;
  assert.throws(() => addPreparedPortions(saved, changed), /different amounts/);
  assert.equal(saved.foodBank[0].items[0].quantity, 150);
});
test('a stale preparation draft cannot recreate an already consumed portion', () => {
  const portions = batch();
  const saved = addPreparedPortions(emptyData(), portions);
  const consumed = consumeFoodBankPortion(saved, portions[0].id, '2026-10-16', 'Lunch');
  assert.throws(() => addPreparedPortions(consumed, batch()), /already been logged/);
  assert.equal(consumed.foodBank.length, 1); assert.equal(consumed.entries.length, 1);
});
