import test from 'node:test';
import assert from 'node:assert/strict';
import { adjustedBudget, calorieLedger, carryBefore } from '../../src/lib/calories/budget.ts';
import { emptyData, manualMealEntry } from '../../src/lib/calories/model.ts';

const today = '2026-10-12';
const settings = (burned = 3200, deficit = 500, burnedSource) => ({
  burned, deficit, protein: 175,
  ...(burnedSource ? { burnedSource } : {}),
});
const meal = (date, calories, id = `meal-${date}`) => manualMealEntry({
  id, date, meal: 'Dinner', name: 'Recorded meal',
  protein: calories / 4, carbs: 0, fat: 0, fiber: 0,
});
const dataWith = (entries, defaults = settings()) => ({ ...emptyData(), entries, defaults });
const reading = (date, active, resting, total = active + resting) => ({
  date, active, resting, total, synced_at: `${date}T20:00:00Z`,
});
const threeDays = () => dataWith([
  meal('2026-10-11', 2600),
  meal('2026-10-09', 2400),
  meal('2026-10-10', 2900),
]);

test('adjusted budget adds a signed carry and clamps only the displayed limit', () => {
  assert.deepEqual(adjustedBudget(3200, 500, 300), {
    baseBudget: 2700, carryIn: 300, rawBudget: 3000, budget: 3000,
  });
  assert.deepEqual(adjustedBudget(1000, 500, -900), {
    baseBudget: 500, carryIn: -900, rawBudget: -400, budget: 0,
  });
  assert.deepEqual(adjustedBudget(100, 500, 0), {
    baseBudget: 0, carryIn: 0, rawBudget: 0, budget: 0,
  });
  assert.deepEqual(adjustedBudget(0, 0), {
    baseBudget: 0, carryIn: 0, rawBudget: 0, budget: 0,
  });
});

test('unused and excess calories accumulate once across three completed days', () => {
  const ledger = calorieLedger(threeDays(), [], today);
  assert.deepEqual(ledger.completed.map(day => day.date), ['2026-10-09', '2026-10-10', '2026-10-11']);
  assert.deepEqual(ledger.completed.map(day => day.carryIn), [0, 300, 100]);
  assert.deepEqual(ledger.completed.map(day => day.rawBudget), [2700, 3000, 2800]);
  assert.deepEqual(ledger.completed.map(day => day.balance), [300, 100, 200]);
  assert.equal(carryBefore(today, ledger), 200, '300 - 200 + 100; previous carry is never counted twice');
  assert.ok(ledger.completed.every(day => day.settled));
});

test('every meal snapshot contributes to its day, independent of entry order', () => {
  const data = dataWith([
    meal('2026-10-10', 1800, 'late'),
    meal('2026-10-09', 1000, 'first'),
    meal('2026-10-10', 1200, 'early'),
    meal('2026-10-09', 1400, 'second'),
  ]);
  const ledger = calorieLedger(data, [], today);
  assert.equal(ledger.days['2026-10-09'].eaten, 2400);
  assert.equal(ledger.days['2026-10-10'].eaten, 3000);
  assert.equal(ledger.days['2026-10-10'].balance, 0);
});

test('gaps, settings-only days and Garmin-only dates do not create unused-calorie credit', () => {
  const data = dataWith([meal('2026-10-09', 2400), meal('2026-10-11', 2800)]);
  data.days['2026-10-10'] = settings(8000, 0);
  const ledger = calorieLedger(data, [reading('2026-10-08', 2000, 2500), reading('2026-10-10', 5000, 2500)], today);
  assert.deepEqual(Object.keys(ledger.days).sort(), ['2026-10-09', '2026-10-11']);
  assert.equal(ledger.days['2026-10-11'].carryIn, 300);
  assert.equal(carryBefore(today, ledger), 200);
  assert.equal(carryBefore('2026-10-10', ledger), 300);
});

test('today and future meals have a displayed balance but cannot fund later dates yet', () => {
  const data = threeDays();
  data.entries.push(meal(today, 500), meal('2026-10-14', 1000));
  const ledger = calorieLedger(data, [], today);
  assert.equal(ledger.days[today].carryIn, 200);
  assert.equal(ledger.days[today].balance, 2400);
  assert.equal(ledger.days[today].settled, false);
  assert.equal(ledger.days['2026-10-14'].carryIn, 200);
  assert.equal(ledger.days['2026-10-14'].balance, 1900);
  assert.equal(ledger.days['2026-10-14'].settled, false);
  assert.equal(ledger.completed.length, 3);
  assert.equal(carryBefore('2026-10-15', ledger), 200);
});

test('carry lookup uses the latest completed day strictly before any selected date', () => {
  const ledger = calorieLedger(threeDays(), [], today);
  assert.equal(carryBefore('2026-10-01', ledger), 0);
  assert.equal(carryBefore('2026-10-09', ledger), 0);
  assert.equal(carryBefore('2026-10-10', ledger), 300);
  assert.equal(carryBefore('2026-10-11', ledger), 100);
  assert.equal(carryBefore('2026-10-12', ledger), 200);
  assert.equal(carryBefore('2027-01-01', ledger), 200);
  assert.equal(carryBefore(today, calorieLedger(emptyData(), [], today)), 0);
});

test('each completed day applies its own saved deficit and burn target', () => {
  const data = dataWith([meal('2026-10-09', 2000), meal('2026-10-10', 2000), meal('2026-10-11', 2000)]);
  data.days['2026-10-09'] = settings(3000, 600);
  data.days['2026-10-10'] = settings(2500, 300);
  data.days['2026-10-11'] = settings(2000, 700);
  const ledger = calorieLedger(data, [], today);
  assert.deepEqual(ledger.completed.map(day => day.baseBudget), [2400, 2200, 1300]);
  assert.deepEqual(ledger.completed.map(day => day.balance), [400, 600, -100]);
  assert.equal(carryBefore(today, ledger), -100);
});

test('Garmin actual totals settle past days and explicit manual burn overrides still win', () => {
  const data = dataWith([meal('2026-10-09', 2000), meal('2026-10-10', 2000)]);
  data.days['2026-10-10'] = settings(2800, 500, 'manual');
  const ledger = calorieLedger(data, [reading('2026-10-09', 400, 1800), reading('2026-10-10', 900, 1900, 4000)], today);
  assert.equal(ledger.days['2026-10-09'].baseBudget, 1700);
  assert.equal(ledger.days['2026-10-09'].balance, -300);
  assert.equal(ledger.days['2026-10-10'].baseBudget, 2300);
  assert.equal(ledger.days['2026-10-10'].balance, 0, 'manual 2800, not Garmin 4000, controls this day');
});

test('today uses Garmin prediction without carrying it, then settles the recorded final total', () => {
  const journal = dataWith([meal('2026-10-12', 2500), meal('2026-10-13', 2200)]);
  const yesterday = reading('2026-10-12', 781, 2338, 3119);
  const partialToday = reading('2026-10-13', 10, 23, 33);
  const provisional = calorieLedger(journal, [yesterday, partialToday], '2026-10-13');
  assert.equal(provisional.days['2026-10-12'].balance, 119);
  assert.equal(provisional.days['2026-10-13'].baseBudget, 1848, '10 active plus 2338 yesterday resting minus deficit');
  assert.equal(provisional.days['2026-10-13'].rawBudget, 1967);
  assert.equal(provisional.days['2026-10-13'].settled, false);
  assert.equal(carryBefore('2026-10-14', provisional), 119);

  const finalToday = reading('2026-10-13', 962, 2488, 3450);
  const settled = calorieLedger(journal, [yesterday, finalToday], '2026-10-14');
  assert.equal(settled.days['2026-10-13'].baseBudget, 2950, 'completed day uses recorded total 3450');
  assert.equal(settled.days['2026-10-13'].balance, 869);
  assert.equal(carryBefore('2026-10-14', settled), 869);
});

test('debt survives a clamped zero budget and later unused calories repay it', () => {
  const data = dataWith([
    meal('2026-10-09', 2000),
    meal('2026-10-10', 100),
    meal('2026-10-11', 100),
  ], settings(1000, 200));
  const ledger = calorieLedger(data, [], today);
  assert.deepEqual(ledger.completed.map(day => day.rawBudget), [800, -400, 300]);
  assert.deepEqual(ledger.completed.map(day => day.budget), [800, 0, 300]);
  assert.deepEqual(ledger.completed.map(day => day.balance), [-1200, -500, 200]);
  assert.equal(carryBefore(today, ledger), 200);
});

test('a logged zero-calorie meal remains a logged day while an empty journal contributes nothing', () => {
  const data = dataWith([meal('2026-10-09', 1000), meal('2026-10-10', 0)], settings(800, 500));
  const ledger = calorieLedger(data, [], today);
  assert.equal(ledger.days['2026-10-09'].balance, -700);
  assert.equal(ledger.days['2026-10-10'].rawBudget, -400);
  assert.equal(ledger.days['2026-10-10'].budget, 0);
  assert.equal(ledger.days['2026-10-10'].balance, -400);
  assert.equal(ledger.completed.length, 2);
  assert.deepEqual(calorieLedger(emptyData(), [reading('2026-10-09', 500, 2000)], today).completed, []);
});

test('editing or deleting past meals recomputes the carry instead of adding another adjustment', () => {
  const data = threeDays();
  assert.equal(carryBefore(today, calorieLedger(data, [], today)), 200);
  data.entries = data.entries.map(entry => entry.date === '2026-10-10' ? meal(entry.date, 3100, entry.id) : entry);
  assert.equal(carryBefore(today, calorieLedger(data, [], today)), 0);
  data.entries = data.entries.filter(entry => entry.date !== '2026-10-10');
  const deletedDay = calorieLedger(data, [], today);
  assert.equal(carryBefore(today, deletedDay), 400);
  assert.equal(deletedDay.days['2026-10-10'], undefined);
});

test('target edits and delayed Garmin corrections recompute the same cumulative ledger', () => {
  const data = threeDays();
  data.days['2026-10-09'] = settings(3200, 600);
  assert.equal(carryBefore(today, calorieLedger(data, [], today)), 100);
  data.days['2026-10-09'] = settings();
  const corrected = calorieLedger(data, [reading('2026-10-09', 800, 2200)], today);
  assert.equal(carryBefore(today, corrected), 0);
  assert.deepEqual(corrected, calorieLedger(data, [reading('2026-10-09', 800, 2200)], today));
});

test('ledger derivation leaves the journal and Garmin snapshots unchanged', () => {
  const data = threeDays();
  const garmin = [reading('2026-10-09', 800, 2200), reading(today, 20, 40)];
  const beforeData = structuredClone(data);
  const beforeGarmin = structuredClone(garmin);
  const freeze = value => {
    for (const item of Object.values(value)) if (item && typeof item === 'object') freeze(item);
    return Object.freeze(value);
  };
  freeze(data);
  freeze(garmin);
  const first = calorieLedger(data, garmin, today);
  const second = calorieLedger(data, garmin, today);
  assert.deepEqual(first, second);
  assert.deepEqual(data, beforeData);
  assert.deepEqual(garmin, beforeGarmin);
});

test('nonfinite helper inputs are rejected while finite negative carry remains valid', () => {
  for (const value of [NaN, Infinity, -Infinity]) {
    assert.throws(() => adjustedBudget(value, 500, 0));
    assert.throws(() => adjustedBudget(3200, value, 0));
    assert.throws(() => adjustedBudget(3200, 500, value));
  }
  assert.equal(adjustedBudget(3200, 500, -300).budget, 2400);
});

test('invalid base targets and numeric coercion are rejected, including finite-input overflow', () => {
  for (const value of [-1, '3200', null, undefined, true]) {
    assert.throws(() => adjustedBudget(value, 500, 0));
    assert.throws(() => adjustedBudget(3200, value, 0));
  }
  for (const value of ['300', null, true]) assert.throws(() => adjustedBudget(3200, 500, value));
  assert.throws(() => adjustedBudget(Number.MAX_VALUE, 0, Number.MAX_VALUE));
});
