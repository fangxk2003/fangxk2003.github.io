import test from 'node:test';
import assert from 'node:assert/strict';
import { macroTargets, nutrientProgress } from '../../src/lib/calories/nutrition.ts';
import { emptyData, normalizeData } from '../../src/lib/calories/model.ts';
const close = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-9, `${actual} != ${expected}`);

test('macro targets allocate the food budget to 175 g protein, 25% fat energy and remaining carbs', () => {
  const targets = macroTargets(1940);
  assert.equal(targets.protein, 175);
  close(targets.fat, 485 / 9);
  assert.equal(targets.carbs, 188.75);
  close(targets.protein * 4 + targets.fat * 9 + targets.carbs * 4, 1940);
  assert.equal(targets.fat * 9 / 1940, 0.25);
  assert.equal(targets.budgetTooLow, false);
  close(nutrientProgress(100, targets.carbs).percent, 100 / 188.75 * 100);
});
test('targets scale with changed budgets while protein stays fixed', () => {
  for (const budget of [1000, 2700, 3333.33]) {
    const targets = macroTargets(budget);
    assert.equal(targets.protein, 175);
    close(targets.fat * 9, budget / 4);
    close(targets.protein * 4 + targets.fat * 9 + targets.carbs * 4, budget);
  }
});
test('low and zero budgets never create negative carb targets or divide by zero', () => {
  const low = macroTargets(800);
  assert.equal(low.carbs, 0);
  assert.equal(low.protein, 175);
  assert.equal(low.budgetTooLow, true);
  const zero = macroTargets(0);
  assert.equal(zero.fat, 0);
  assert.equal(zero.carbs, 0);
  assert.deepEqual(nutrientProgress(0, 0), { percent: null, barPercent: 0, reached: false });
  assert.deepEqual(nutrientProgress(10, 0), { percent: null, barPercent: 100, reached: false });
  assert.equal(nutrientProgress(1, Number.MIN_VALUE).percent, null);
});
test('the exact protein-and-fat-only budget gives a zero carb target without floating-point residue', () => {
  const targets = macroTargets(700 / 0.75);
  assert.equal(targets.carbs, 0);
  assert.equal(targets.budgetTooLow, false);
  assert.equal(nutrientProgress(1, targets.carbs).percent, null);
});
test('percentages can exceed 100% while the visual progress bar stops at full', () => {
  assert.deepEqual(nutrientProgress(210, 175), { percent: 120, barPercent: 100, reached: true });
  assert.deepEqual(nutrientProgress(87.5, 175), { percent: 50, barPercent: 50, reached: false });
});
test('legacy per-day and default protein targets both normalize to 175 without losing other settings', () => {
  const data = emptyData();
  data.defaults.protein = 200;
  data.days['2026-10-13'] = { burned: 2600, deficit: 450, protein: 150, burnedSource: 'manual' };
  const loaded = normalizeData(data);
  assert.equal(loaded.defaults.protein, 175);
  assert.deepEqual(loaded.days['2026-10-13'], { burned: 2600, deficit: 450, protein: 175, burnedSource: 'manual' });
  data.defaults.protein = -1;
  assert.throws(() => normalizeData(data), /Invalid protein target/);
});
test('invalid numeric inputs are rejected rather than producing broken targets', () => {
  for (const value of [NaN, Infinity, -1]) {
    assert.throws(() => macroTargets(value));
    assert.throws(() => nutrientProgress(value, 175));
    assert.throws(() => nutrientProgress(10, value));
  }
});
