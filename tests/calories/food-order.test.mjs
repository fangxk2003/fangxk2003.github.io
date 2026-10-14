import test from 'node:test';
import assert from 'node:assert/strict';
import { sortFoodsByRecentUse } from '../../src/lib/calories/food-order.ts';

const food = (id, name, overrides = {}) => ({
  id, name, baseQuantity: 100, unit: 'g', state: 'Cooked',
  protein: 6, carbs: 12, fat: 1, fiber: 5, calories: 81,
  ...overrides,
});
const entry = (id, foodId, date, overrides = {}) => ({
  id, foodId, date, meal: 'Lunch', foodName: `Recorded ${foodId}`,
  quantity: 100, unit: 'g', state: 'Cooked',
  protein: 6, carbs: 12, fat: 1, fiber: 5, calories: 81,
  ...overrides,
});
const ids = values => values.map(value => value.id);

test('the 14-date window includes today and 13 days ago, excluding older and future logs', () => {
  const foods = [food('older', 'Alpha'), food('future', 'Beta'), food('middle', 'Middle'), food('edges', 'Zulu')];
  const entries = [
    entry('first-edge', 'edges', '2026-10-01'),
    entry('last-edge', 'edges', '2026-10-14'),
    entry('middle', 'middle', '2026-10-07'),
    ...[1, 2, 3].map(id => entry(`old-${id}`, 'older', '2026-09-30')),
    ...[1, 2, 3].map(id => entry(`future-${id}`, 'future', '2026-10-15')),
  ];
  assert.deepEqual(ids(sortFoodsByRecentUse(foods, entries, '2026-10-14')), ['edges', 'middle', 'older', 'future']);
});

test('frequency counts individual entries, including repeated foods in one meal, rather than portion size or energy', () => {
  const foods = [food('large', 'Apple'), food('repeated', 'Zucchini')];
  const entries = [
    entry('large-portion', 'large', '2026-10-14', { quantity: 10000, calories: 100000 }),
    entry('first-bite', 'repeated', '2026-10-14', { quantity: 0.1, calories: 0.081 }),
    entry('second-bite', 'repeated', '2026-10-14', { quantity: 1, calories: 0.81 }),
  ];
  assert.deepEqual(ids(sortFoodsByRecentUse(foods, entries, '2026-10-14')), ['repeated', 'large']);
});

test('food IDs survive renamed foods and old snapshots; deleted food logs never become library choices', () => {
  const renamed = food('original-id', 'Zulu renamed food', { protein: 20, calories: 137 });
  const sameOldName = food('different-id', 'Old food name');
  const foods = [sameOldName, renamed];
  const entries = [
    entry('old-snapshot-1', renamed.id, '2026-10-13', { foodName: 'Old food name', protein: 1, calories: 61 }),
    entry('old-snapshot-2', renamed.id, '2026-10-14', { foodName: 'Old food name', protein: 2, calories: 65 }),
    ...[1, 2, 3].map(id => entry(`deleted-${id}`, 'deleted-id', '2026-10-14', { foodName: renamed.name })),
  ];
  assert.deepEqual(sortFoodsByRecentUse(foods, entries, '2026-10-14'), [renamed, sameOldName]);
});

test('manual source flags and canonical legacy manual IDs are excluded without excluding unrelated ID prefixes', () => {
  const foods = [
    food('regular', 'Alpha'),
    food('unused', 'Beta'),
    food('manual:unrelated', 'Xylophone'),
    food('manual:legacy-entry', 'Yam'),
    food('flagged', 'Zucchini'),
  ];
  const entries = [
    entry('regular-entry', 'regular', '2026-10-14'),
    entry('prefix-1', 'manual:unrelated', '2026-10-14'),
    entry('prefix-2', 'manual:unrelated', '2026-10-13'),
    // The source flag is absent in a legacy manual snapshot; its canonical ID still identifies it.
    entry('legacy-entry', 'manual:legacy-entry', '2026-10-14'),
    ...[1, 2, 3, 4].map(id => entry(`manual-${id}`, 'flagged', '2026-10-14', { source: 'manual' })),
  ];
  assert.deepEqual(ids(sortFoodsByRecentUse(foods, entries, '2026-10-14')), [
    'manual:unrelated', 'regular', 'unused', 'manual:legacy-entry', 'flagged',
  ]);
});

test('equal frequencies sort alphabetically, then by ID for duplicate names, independently of input order', () => {
  const foods = [food('rice-z', 'Rice'), food('pear', 'Pear'), food('rice-a', 'Rice'), food('apple', 'Apple')];
  const expected = ['apple', 'pear', 'rice-a', 'rice-z'];
  assert.deepEqual(ids(sortFoodsByRecentUse(foods, [], '2026-10-14')), expected, 'unused foods also receive stable alphabetical ordering');
  const entries = foods.flatMap(item => [
    entry(`${item.id}-1`, item.id, '2026-10-12'),
    entry(`${item.id}-2`, item.id, '2026-10-14'),
  ]);
  assert.deepEqual(ids(sortFoodsByRecentUse(foods, entries, '2026-10-14')), expected);
  assert.deepEqual(ids(sortFoodsByRecentUse([...foods].reverse(), [...entries].reverse(), '2026-10-14')), expected);
});

test('sorting leaves the supplied arrays, library records and recorded nutrient snapshots unchanged', () => {
  const foods = [food('apple', 'Apple'), food('peas', 'Peas')];
  const entries = [entry('peas-old-values', 'peas', '2026-10-14', { foodName: 'Original peas', protein: 2, calories: 65 })];
  const beforeFoods = structuredClone(foods);
  const beforeEntries = structuredClone(entries);
  for (const item of [...foods, ...entries]) Object.freeze(item);
  Object.freeze(foods);
  Object.freeze(entries);

  assert.deepEqual(sortFoodsByRecentUse(foods, entries, '2026-10-14'), [foods[1], foods[0]]);
  assert.deepEqual(foods, beforeFoods);
  assert.deepEqual(entries, beforeEntries);
  assert.deepEqual(sortFoodsByRecentUse([], entries, '2026-10-14'), []);
});

test('a new calendar-day anchor expires the old boundary and admits the newly current day across a year boundary', () => {
  const foods = [food('new-day', 'Apple'), food('recent', 'Banana'), food('old-boundary', 'Zucchini')];
  const entries = [
    ...[1, 2, 3].map(id => entry(`old-${id}`, 'old-boundary', '2025-12-21')),
    entry('recent', 'recent', '2026-01-03'),
    ...[1, 2].map(id => entry(`next-${id}`, 'new-day', '2026-01-04')),
  ];
  assert.deepEqual(ids(sortFoodsByRecentUse(foods, entries, '2026-01-03')), ['old-boundary', 'recent', 'new-day']);
  assert.deepEqual(ids(sortFoodsByRecentUse(foods, entries, '2026-01-04')), ['new-day', 'recent', 'old-boundary']);
});

test('calendar-window boundaries remain inclusive across leap day and the Stockholm daylight-saving change', () => {
  const foods = [food('outside', 'Apple'), food('inside', 'Zulu')];
  for (const [currentDay, firstDay, outsideDay] of [
    ['2024-03-01', '2024-02-17', '2024-02-16'],
    ['2026-10-26', '2026-10-13', '2026-10-12'],
  ]) {
    const entries = [
      entry('included', 'inside', firstDay),
      entry('excluded-1', 'outside', outsideDay),
      entry('excluded-2', 'outside', outsideDay),
    ];
    assert.deepEqual(ids(sortFoodsByRecentUse(foods, entries, currentDay)), ['inside', 'outside'], currentDay);
  }
});
