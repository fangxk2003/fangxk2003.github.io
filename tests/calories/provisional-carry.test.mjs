import test from 'node:test';
import assert from 'node:assert/strict';
import { adjustedBudget, calorieLedger, carryBefore } from '../../src/lib/calories/budget.ts';
import { effectiveBurned } from '../../src/lib/calories/garmin.ts';
import { emptyData, manualMealEntry, settingsForDay } from '../../src/lib/calories/model.ts';

const YESTERDAY = '2026-10-15', TODAY = '2026-10-16', TOMORROW = '2026-10-17', LATER = '2026-10-18';
const settings = (burned = 0, burnedSource = 'garmin', deficit = 500) => ({ burned, burnedSource, deficit, protein: 175 });
const meal = (date, calories, id = date) => manualMealEntry({ id, date, meal: 'Lunch', name: 'Recorded meal', protein: calories / 4, carbs: 0, fat: 0, fiber: 0 });
const reading = (date, active, resting, total = active + resting) => ({ date, active, resting, total, synced_at: `${date}T12:00:00Z` });
const close = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-8, `Expected ${expected}, received ${actual}`);
const journal = (entries, defaults = settings()) => ({ ...emptyData(), entries, defaults });
const fixture = () => ({
  data: journal([meal(YESTERDAY, 2904.6), meal(TODAY, 2232.1)]),
  garmin: [reading(YESTERDAY, 560, 2440), reading(TODAY, 521, 99)],
});

test('tomorrow reflects today’s provisional repayment without changing historical or today carry-in', () => {
  const { data, garmin } = fixture();
  const ledger = calorieLedger(data, garmin, TODAY);
  close(ledger.completed[0].balance, -404.6);
  close(ledger.days[TODAY].baseBudget, 2461, 'today uses active 521 plus resting 2440 minus deficit 500');
  close(ledger.provisional.balance, -175.7);
  assert.equal(ledger.provisional.date, TODAY);
  assert.equal(ledger.provisional.settled, false);
  assert.deepEqual(ledger.completed.map(row => row.date), [YESTERDAY]);
  assert.equal(carryBefore(YESTERDAY, ledger), 0);
  close(carryBefore(TODAY, ledger), -404.6);
  close(carryBefore(TOMORROW, ledger), -175.7);
  const futureBurn = effectiveBurned(TOMORROW, settingsForDay(data, TOMORROW), garmin, TODAY).burned;
  assert.equal(futureBurn, 2440);
  close(adjustedBudget(futureBurn, 500, carryBefore(TOMORROW, ledger)).budget, 1764.3);
});

test('positive provisional credit feeds every future row once; planned future meals never fund later previews', () => {
  const { data, garmin } = fixture();
  data.entries = [meal(YESTERDAY, 2904.6), meal(TODAY, 500), meal(TOMORROW, 100), meal(LATER, 9000)];
  const ledger = calorieLedger(data, garmin, TODAY);
  close(ledger.provisional.balance, 1556.4);
  for (const date of [TOMORROW, LATER]) {
    close(carryBefore(date, ledger), 1556.4);
    close(ledger.days[date].carryIn, carryBefore(date, ledger));
    close(ledger.days[date].budget, 3496.4);
    assert.equal(ledger.days[date].settled, false);
  }
  close(ledger.days[TOMORROW].balance, 3396.4);
  close(ledger.days[LATER].balance, -5503.6);
  close(carryBefore('2027-01-01', ledger), 1556.4);
  assert.deepEqual(ledger.completed.map(row => row.date), [YESTERDAY]);
});

test('empty or unavailable today adds no provisional allowance, while an available zero balance overrides old credit', () => {
  const { data, garmin } = fixture();
  const emptyToday = { ...data, entries: data.entries.filter(entry => entry.date !== TODAY) };
  const empty = calorieLedger(emptyToday, garmin, TODAY);
  assert.equal(empty.provisional, undefined);
  close(carryBefore(TOMORROW, empty), -404.6);

  const unavailableData = journal([meal(YESTERDAY, 2904.6), meal(TODAY, 500)]);
  unavailableData.days[YESTERDAY] = settings(3000, 'manual');
  const unavailable = calorieLedger(unavailableData, [], TODAY);
  assert.equal(unavailable.days[TODAY].available, false);
  assert.equal(unavailable.provisional, undefined);
  close(carryBefore(TOMORROW, unavailable), -404.6);

  const zeroData = journal([meal(YESTERDAY, 2400), meal(TODAY, 100)], settings(3000, 'manual'));
  zeroData.days[TODAY] = settings(0, 'manual', 0);
  const zero = calorieLedger(zeroData, [], TODAY);
  assert.equal(carryBefore(TODAY, zero), 100);
  assert.equal(zero.provisional.available, true);
  assert.equal(zero.provisional.balance, 0);
  assert.equal(carryBefore(TOMORROW, zero), 0, 'zero must not fall back to yesterday’s positive credit');
});

test('the 1,000 kcal deficit cap retains full provisional debt instead of treating the displayed budget as settlement', () => {
  const data = journal([meal(YESTERDAY, 4000), meal(TODAY, 2000), meal(TOMORROW, 100)], settings(3000, 'manual'));
  const ledger = calorieLedger(data, [], TODAY);
  const current = ledger.provisional;
  assert.equal(carryBefore(TODAY, ledger), -1500);
  assert.equal(current.rawBudget, 1000);
  assert.equal(current.budget, 2000);
  assert.equal(current.deferredDebt, 1000);
  assert.equal(current.budget - current.eaten, 0);
  assert.equal(current.balance, -1000, 'unpaid debt survives even when intake equals the displayed allowance');
  assert.equal(carryBefore(TOMORROW, ledger), -1000);
  assert.equal(ledger.days[TOMORROW].carryIn, -1000);
  assert.equal(ledger.days[TOMORROW].budget, 2000);
  assert.equal(ledger.days[TOMORROW].deferredDebt, 500);
  assert.equal(carryBefore(LATER, ledger), -1000, 'tomorrow’s planned meal does not settle today');
});

test('today’s added or edited meals, target changes, Garmin updates and deletions recompute the provisional preview', () => {
  const { data, garmin } = fixture();
  const futureCarry = () => carryBefore(TOMORROW, calorieLedger(data, garmin, TODAY));
  close(futureCarry(), -175.7);
  data.entries.push(meal(TODAY, 200, 'extra'));
  close(futureCarry(), -375.7);
  data.entries = data.entries.map(entry => entry.id === TODAY ? meal(TODAY, 2000) : entry);
  close(futureCarry(), -143.6);
  data.days[TODAY] = settings(0, 'garmin', 600);
  close(futureCarry(), -243.6);
  garmin[1] = { ...garmin[1], active: 621, total: 720 };
  close(futureCarry(), -143.6);
  data.entries = data.entries.filter(entry => entry.id !== 'extra');
  close(futureCarry(), 56.4);
  data.entries = data.entries.filter(entry => entry.date !== TODAY);
  const noToday = calorieLedger(data, garmin, TODAY);
  assert.equal(noToday.provisional, undefined);
  close(carryBefore(TOMORROW, noToday), -404.6);
});

test('midnight settles the final Garmin total once and starts a new provisional day independently', () => {
  const { data, garmin } = fixture();
  close(carryBefore(TOMORROW, calorieLedger(data, garmin, TODAY)), -175.7);
  const finalReadings = [garmin[0], reading(TODAY, 800, 2440)];
  const settled = calorieLedger(data, finalReadings, TOMORROW);
  assert.deepEqual(settled.completed.map(row => row.date), [YESTERDAY, TODAY]);
  assert.equal(settled.days[TODAY].settled, true);
  assert.equal(settled.provisional, undefined);
  close(settled.days[TODAY].baseBudget, 2740);
  close(carryBefore(TOMORROW, settled), 103.3, 'the earlier provisional amount is replaced, not added again');
  close(carryBefore(LATER, settled), 103.3);
  data.entries.push(meal(TOMORROW, 100));
  const newToday = calorieLedger(data, finalReadings, TOMORROW);
  close(carryBefore(TOMORROW, newToday), 103.3);
  assert.equal(newToday.provisional.date, TOMORROW);
  assert.equal(newToday.provisional.settled, false);
  close(carryBefore(LATER, newToday), 1943.3);
});

test('repeated provisional derivation and lookups do not mutate journal or Garmin snapshots', () => {
  const { data, garmin } = fixture();
  data.entries.push(meal(TOMORROW, 300));
  const beforeData = structuredClone(data), beforeGarmin = structuredClone(garmin);
  const freeze = value => {
    if (value && typeof value === 'object') { Object.values(value).forEach(freeze); Object.freeze(value); }
    return value;
  };
  freeze(data); freeze(garmin);
  const first = calorieLedger(data, garmin, TODAY);
  const second = calorieLedger(data, garmin, TODAY);
  assert.deepEqual(first, second);
  freeze(first);
  close(carryBefore(TODAY, first), -404.6);
  close(carryBefore(TOMORROW, first), -175.7);
  assert.deepEqual(data, beforeData);
  assert.deepEqual(garmin, beforeGarmin);
});
