import test from 'node:test';
import assert from 'node:assert/strict';
import { effectiveBurned, normalizeGarminSnapshot, emptyGarminSnapshot } from '../../src/lib/calories/garmin.ts';
import { normalizeData, emptyData, manualMealEntry, settingsForDay } from '../../src/lib/calories/model.ts';
const day = { date: '2026-10-12', active: 700, resting: 1700, total: 2400, synced_at: '2026-10-12T18:17:00Z' };
const settings = { burned: 3200, deficit: 500, protein: 175 };
const today = '2026-10-13';

test('Garmin total drives the day unless it has an explicit manual override', () => {
  assert.equal(effectiveBurned(day.date, settings, [day], today).burned, 2400);
  const override = effectiveBurned(day.date, { ...settings, burned: 2800, burnedSource: 'manual' }, [day], today);
  assert.equal(override.burned, 2800);
  assert.equal(override.source, 'manual');
  const missing = effectiveBurned('2026-10-11', settings, [day], today);
  assert.equal(missing.burned, 1700, 'missing date uses only the latest completed Garmin resting amount');
  assert.equal(missing.source, 'garmin');
  assert.equal(missing.fallback, true);
  assert.equal(missing.restingEstimate, true);
  assert.equal(missing.unavailable, false);
  assert.equal(missing.day, undefined);
  assert.equal(missing.restingDay.date, day.date);
  const zero = effectiveBurned(day.date, settings, [{ ...day, active: 0, resting: 0, total: 0 }], today);
  assert.equal(zero.burned, 0, 'a valid zero reading must not fall through to the estimate');
});

const current = { date: today, active: 10, resting: 23, total: 33, synced_at: '2026-10-12T22:25:00Z' };
const yesterday = { ...day, active: 781, resting: 2338, total: 3119 };
test('today predicts full-day burn using its active calories and exact yesterday resting calories', () => {
  const result = effectiveBurned(today, settings, [current, yesterday], today);
  assert.equal(result.burned, 2348);
  assert.equal(result.predicted, true);
  assert.equal(result.restingDay.date, yesterday.date);
  assert.equal(result.burned - settings.deficit, 1848, 'the daily budget uses the predicted total');
  assert.equal(current.total, 33, 'prediction does not mutate the imported reading');
});
test('manual overrides win even when a today prediction is available', () => {
  const result = effectiveBurned(today, { ...settings, burned: 2800, burnedSource: 'manual' }, [current, yesterday], today);
  assert.equal(result.burned, 2800);
  assert.equal(result.predicted, false);
});
test('ended days switch to Garmin recorded totals and future dates are not predicted', () => {
  const result = effectiveBurned(today, settings, [current, yesterday], '2026-10-14');
  assert.equal(result.burned, 33);
  assert.equal(result.predicted, false);
  assert.equal(result.predictionUnavailable, false);
  const future = effectiveBurned(today, settings, [current, yesterday], '2026-10-12');
  assert.equal(future.burned, 33);
  assert.equal(future.predicted, false);
});
test('missing yesterday leaves a labeled partial total; no other date substitutes for it', () => {
  const result = effectiveBurned(today, settings, [current, { ...yesterday, date: '2026-10-11' }], today);
  assert.equal(result.burned, 33);
  assert.equal(result.predicted, false);
  assert.equal(result.predictionUnavailable, true);
  const noCurrent = effectiveBurned(today, settings, [yesterday], today);
  assert.equal(noCurrent.burned, yesterday.resting);
  assert.equal(noCurrent.fallback, true);
  assert.equal(noCurrent.restingEstimate, true);
  assert.equal(noCurrent.unavailable, false);
});
test('zero calorie components stay valid and yesterday lookup crosses calendar boundaries', () => {
  const result = effectiveBurned('2027-01-01', settings, [
    { ...current, date: '2027-01-01', active: 0 },
    { ...yesterday, date: '2026-12-31', resting: 0 },
  ], '2027-01-01');
  assert.equal(result.burned, 0);
  assert.equal(result.predicted, true);
  const leap = effectiveBurned('2028-03-01', settings, [
    { ...current, date: '2028-03-01' },
    { ...yesterday, date: '2028-02-29' },
  ], '2028-03-01');
  assert.equal(leap.burned, 2348);
});
test('backups preserve manual source choices and legacy settings', () => {
  const data = emptyData();
  data.days[day.date] = { ...settings, burnedSource: 'manual' };
  assert.equal(normalizeData(data).days[day.date].burnedSource, 'manual');
  assert.equal(normalizeData(data).defaults.burnedSource, undefined);
  data.defaults.burnedSource = 'invalid';
  assert.throws(() => normalizeData(data));
});
test('imported Garmin history rejects malformed/incomplete values and duplicate dates', () => {
  const base = { days: [day], status: { ...emptyGarminSnapshot().status, state: 'connected' } };
  assert.deepEqual(normalizeGarminSnapshot(base), base);
  for (const bad of [{ ...day, active: null }, { ...day, total: NaN }, { ...day, resting: -1 }, { ...day, date: '2026-02-30' }, { ...day, synced_at: 'yesterday' }]) {
    assert.throws(() => normalizeGarminSnapshot({ ...base, days: [bad] }));
  }
  assert.throws(() => normalizeGarminSnapshot({ ...base, days: [day, day] }));
});

test('missing dates use the latest completed reading by date even when imports are unsorted', () => {
  const latest = { ...day, date: '2026-10-11', active: 900, resting: 2200, total: 3100 };
  const older = { ...day, date: '2026-10-08', active: 600, resting: 1800, total: 2400 };
  const currentPartial = { ...current, resting: 45 };
  const future = { ...day, date: '2026-10-15', resting: 4000, total: 5000 };
  for (const imports of [[older, future, latest, currentPartial], [latest, currentPartial, older, future]]) {
    const result = effectiveBurned('2026-10-10', settings, imports, today);
    assert.equal(result.burned, 2200, 'the donor active calories are not repeated on the missing day');
    assert.equal(result.restingDay.date, latest.date);
    assert.equal(result.day, undefined);
    assert.equal(result.source, 'garmin');
    assert.equal(result.requestedSource, 'garmin');
    assert.equal(result.restingEstimate, true);
    assert.equal(result.fallback, true);
    assert.equal(result.predicted, false);
    assert.equal(result.unavailable, false);
  }
});

test('today partials and future readings cannot supply a resting estimate', () => {
  const future = { ...day, date: '2026-10-15', resting: 4000 };
  const result = effectiveBurned('2026-10-10', settings, [current, future], today);
  assert.equal(result.burned, 0);
  assert.equal(result.source, 'garmin');
  assert.equal(result.unavailable, true);
  assert.equal(result.restingEstimate, false);
  assert.equal(result.restingDay, undefined);
  assert.equal(result.day, undefined);
});

test('legacy saved 3200 is not used by automatic mode when Garmin has no available data', () => {
  const result = effectiveBurned(today, settings, [], today);
  assert.equal(settings.burned, 3200);
  assert.equal(result.burned, 0);
  assert.equal(result.source, 'garmin');
  assert.equal(result.requestedSource, 'garmin');
  assert.equal(result.unavailable, true);
  assert.equal(emptyData().defaults.burned, 0, 'a new journal has no assumed burn amount');
});

test('an explicit manual override remains available without any imported Garmin reading', () => {
  const result = effectiveBurned(today, { ...settings, burned: 2800, burnedSource: 'manual' }, [], today);
  assert.equal(result.burned, 2800);
  assert.equal(result.source, 'manual');
  assert.equal(result.requestedSource, 'manual');
  assert.equal(result.fallback, false);
  assert.equal(result.unavailable, false);
  assert.equal(result.restingEstimate, false);
});

test('a latest completed zero resting amount is valid and does not select an older positive donor', () => {
  const zero = { ...day, date: '2026-10-11', active: 600, resting: 0, total: 600 };
  const older = { ...yesterday, date: '2026-10-10' };
  const result = effectiveBurned(today, settings, [older, zero], today);
  assert.equal(result.burned, 0);
  assert.equal(result.restingDay.date, zero.date);
  assert.equal(result.restingEstimate, true);
  assert.equal(result.unavailable, false);
});

test('fallback estimates and actual totals never rewrite imported history', () => {
  const imports = [
    Object.freeze({ ...day, date: '2026-10-09', active: 400, resting: 1800, total: 2200 }),
    Object.freeze({ ...yesterday }),
  ];
  Object.freeze(imports);
  const before = structuredClone(imports);
  const missing = effectiveBurned('2026-10-10', settings, imports, today);
  const actual = effectiveBurned('2026-10-09', settings, imports, today);
  assert.equal(missing.burned, yesterday.resting);
  assert.equal(actual.burned, 2200);
  assert.equal(actual.restingEstimate, false);
  assert.equal(actual.unavailable, false);
  assert.equal(actual.day.total, 2200);
  assert.deepEqual(imports, before);
});

test('all burn resolver outcomes provide explicit estimate and availability flags', () => {
  const outcomes = [
    effectiveBurned(day.date, settings, [day], today),
    effectiveBurned(today, settings, [current, yesterday], today),
    effectiveBurned(today, settings, [current], today),
    effectiveBurned(today, settings, [yesterday], today),
    effectiveBurned(today, settings, [], today),
    effectiveBurned(today, { ...settings, burnedSource: 'manual' }, [], today),
  ];
  for (const result of outcomes) {
    assert.equal(typeof result.restingEstimate, 'boolean');
    assert.equal(typeof result.unavailable, 'boolean');
  }
});

test('explicit manual settings remain selected even before a day has any meal entries', () => {
  const data = emptyData();
  data.days[today] = { burned: 2800, deficit: 450, protein: 175, burnedSource: 'manual' };
  const resolved = settingsForDay(data, today);
  assert.equal(data.entries.length, 0);
  assert.deepEqual(resolved, data.days[today]);
  const effective = effectiveBurned(today, resolved, [yesterday], today);
  assert.equal(effective.burned, 2800);
  assert.equal(effective.source, 'manual');
  assert.equal(effective.restingEstimate, false);
});

test('an untouched day starts automatic without inheriting legacy global manual burn', () => {
  const data = emptyData();
  data.defaults = Object.freeze({ burned: 3200, deficit: 450, protein: 175, burnedSource: 'manual' });
  const before = structuredClone(data);
  const resolved = settingsForDay(data, today);
  assert.deepEqual(resolved, { burned: 0, deficit: 450, protein: 175, burnedSource: 'garmin' });
  assert.notEqual(resolved, data.defaults, 'the untouched-day choice is an independent settings copy');
  assert.deepEqual(data, before);
  assert.equal(effectiveBurned(today, resolved, [yesterday], today).burned, yesterday.resting);
  assert.equal(effectiveBurned(today, resolved, [], today).unavailable, true);
});

test('freezing settings before the first meal keeps the new day automatic after logging', () => {
  const data = emptyData();
  data.defaults = { burned: 3200, deficit: 450, protein: 175, burnedSource: 'manual' };
  const initialSettings = { ...settingsForDay(data, today) };
  data.days[today] = initialSettings;
  data.entries.push(manualMealEntry({
    id: 'first-meal', date: today, meal: 'Lunch', name: 'First recorded meal',
    protein: 25, carbs: 0, fat: 0, fiber: 0,
  }));
  const resolved = settingsForDay(data, today);
  assert.equal(resolved.burned, 0);
  assert.equal(resolved.burnedSource, 'garmin');
  assert.equal(resolved.deficit, 450);
  assert.equal(effectiveBurned(today, resolved, [yesterday], today).burned, yesterday.resting);
  assert.equal(data.defaults.burned, 3200);
  assert.equal(data.defaults.burnedSource, 'manual');
});

test('a legacy logged day without saved day settings preserves its historical manual defaults', () => {
  const data = emptyData();
  data.defaults = { burned: 3200, deficit: 450, protein: 175, burnedSource: 'manual' };
  data.entries.push(manualMealEntry({
    id: 'legacy-meal', date: day.date, meal: 'Dinner', name: 'Existing recorded meal',
    protein: 25, carbs: 0, fat: 0, fiber: 0,
  }));
  const before = structuredClone(data);
  const resolved = settingsForDay(data, day.date);
  assert.deepEqual(resolved, data.defaults);
  assert.equal(effectiveBurned(day.date, resolved, [day], today).burned, 3200);
  assert.equal(effectiveBurned(day.date, resolved, [day], today).source, 'manual');
  assert.deepEqual(data, before);
});
