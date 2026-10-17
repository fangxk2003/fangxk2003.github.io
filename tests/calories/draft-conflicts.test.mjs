import test from 'node:test';
import assert from 'node:assert/strict';
import { assertCurrentRecord } from '../../src/lib/calories/draft-conflicts.ts';

test('an unchanged record can be edited, comparing content rather than property or stable-ID array order', () => {
  const original = { id: 'meal', protein: 20, rows: [{ id: 'b', quantity: 2 }, { id: 'a', quantity: 1 }] };
  const current = { rows: [{ quantity: 1, id: 'a' }, { quantity: 2, id: 'b' }], protein: 20, id: 'meal' };
  assert.doesNotThrow(() => assertCurrentRecord(current, JSON.stringify(original), { ...current, protein: 30 }, 'This entry'));
  assert.doesNotThrow(() => assertCurrentRecord(null, 'null', { id: 'new' }, 'This entry'));
});

test('an idempotent retry accepts the already saved proposed snapshot without creating another record', () => {
  const original = { id: 'food', name: 'Rice', quantity: 100 }, proposed = { ...original, quantity: 150 };
  assert.doesNotThrow(() => assertCurrentRecord(structuredClone(proposed), JSON.stringify(original), proposed, 'This food'));
  assert.doesNotThrow(() => assertCurrentRecord(null, JSON.stringify(original), null, 'This food'));
});

test('remote edits and deletions block an older draft and preserve its inputs', () => {
  const original = { id: 'entry', protein: 20, meal: 'Lunch' }, proposed = { ...original, meal: 'Dinner' };
  const before = structuredClone(proposed);
  for (const current of [{ ...original, protein: 30 }, null, undefined]) {
    assert.throws(() => assertCurrentRecord(current, JSON.stringify(original), proposed, 'This entry'), /changed since this draft was started.*draft is kept/);
  }
  assert.deepEqual(proposed, before);
});

test('malformed, nonfinite or unbounded original JSON is rejected even if current equals proposed', () => {
  const proposed = { id: 'entry', protein: 20 };
  for (const original of ['', ' ', '{', 'undefined', 'NaN', '{"protein":1e999}', ' '.repeat(65537), null, undefined, 42]) {
    assert.throws(() => assertCurrentRecord(proposed, original, proposed, 'This entry'), /original record.*cannot be read safely/);
  }
});
