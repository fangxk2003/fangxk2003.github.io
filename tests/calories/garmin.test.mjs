import test from 'node:test';
import assert from 'node:assert/strict';
import { effectiveBurned, normalizeGarminSnapshot, emptyGarminSnapshot } from '../../src/lib/calories/garmin.ts';
import { normalizeData, emptyData } from '../../src/lib/calories/model.ts';
const day = { date: '2026-10-12', active: 700, resting: 1700, total: 2400, synced_at: '2026-10-12T18:17:00Z' };
const settings = { burned: 3200, deficit: 500, protein: 175 };
const today = '2026-10-13';

test('Garmin total drives the day unless it has an explicit manual override', () => {
  assert.equal(effectiveBurned(day.date, settings, [day], today).burned, 2400);
  const override = effectiveBurned(day.date, { ...settings, burned: 2800, burnedSource: 'manual' }, [day], today);
  assert.equal(override.burned, 2800);
  assert.equal(override.source, 'manual');
  const missing = effectiveBurned('2026-10-11', settings, [day], today);
  assert.equal(missing.burned, 3200);
  assert.equal(missing.fallback, true);
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
  assert.equal(noCurrent.burned, settings.burned);
  assert.equal(noCurrent.fallback, true);
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
