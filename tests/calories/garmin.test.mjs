import test from 'node:test';
import assert from 'node:assert/strict';
import { effectiveBurned, normalizeGarminSnapshot, emptyGarminSnapshot } from '../../src/lib/calories/garmin.ts';
import { normalizeData, emptyData } from '../../src/lib/calories/model.ts';
const day = { date: '2026-10-12', active: 700, resting: 1700, total: 2400, synced_at: '2026-10-12T18:17:00Z' };
const settings = { burned: 3200, deficit: 500, protein: 175 };

test('Garmin total drives the day unless it has an explicit manual override', () => {
  assert.equal(effectiveBurned(day.date, settings, [day]).burned, 2400);
  const override = effectiveBurned(day.date, { ...settings, burned: 2800, burnedSource: 'manual' }, [day]);
  assert.equal(override.burned, 2800);
  assert.equal(override.source, 'manual');
  const missing = effectiveBurned('2026-10-11', settings, [day]);
  assert.equal(missing.burned, 3200);
  assert.equal(missing.fallback, true);
  const zero = effectiveBurned(day.date, settings, [{ ...day, active: 0, resting: 0, total: 0 }]);
  assert.equal(zero.burned, 0, 'a valid zero reading must not fall through to the estimate');
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
