import test from 'node:test';
import assert from 'node:assert/strict';
import { createUndoAction, undoAction, UndoConflictError } from '../../src/lib/calories/undo.ts';
import { consumeFoodBankPortion, prepareFoodBankPortions } from '../../src/lib/calories/food-bank.ts';
import { emptyData, manualMealEntry, normalizeData } from '../../src/lib/calories/model.ts';

const date = '2026-10-17';
const rice = { id: 'rice', name: 'Rice', baseQuantity: 100, unit: 'g', state: 'Cooked', protein: 3, carbs: 28, fat: 1, fiber: 2, calories: 133 };
const eggs = { id: 'eggs', name: 'Eggs', baseQuantity: 2, unit: 'piece', state: 'Raw', protein: 12, carbs: 1, fat: 10, fiber: 0, calories: 142 };
const manual = (id = 'manual') => manualMealEntry({ id, date, meal: 'Dinner', name: 'Restaurant meal', protein: 30, carbs: 40, fat: 15, fiber: 6 });
const libraryEntry = { id: 'library-entry', date, meal: 'Lunch', foodId: rice.id, foodName: 'Old rice snapshot', quantity: 150, unit: 'g', state: 'Cooked', protein: 7, carbs: 41, fat: 2, fiber: 3, calories: 123.456 };
function bankData(count = 2) {
  let id = 0;
  return normalizeData({ ...emptyData(), foods: [rice, eggs], foodBank: prepareFoodBankPortions([rice, eggs], [{ foodId: rice.id, quantity: 150 }, { foodId: eggs.id, quantity: 1 }], count, 'Rice and eggs', () => `bank-${++id}`) });
}
const freeze = value => {
  if (value && typeof value === 'object') { Object.freeze(value); Object.values(value).forEach(freeze); }
  return value;
};
const receiptFor = (before, after, kind, id) => createUndoAction(before, after, kind, id, () => 'undo-receipt');

test('deleted manual and library entries restore their exact snapshots while retaining later food, entries and targets', () => {
  for (const entry of [manual(), libraryEntry]) {
    const before = normalizeData({ ...emptyData(), foods: [rice], entries: [entry] });
    const after = { ...before, entries: [] };
    const receipt = receiptFor(before, after, 'delete-entry', entry.id);
    const later = normalizeData({ ...after, foods: [{ ...rice, name: 'New rice', protein: 1 }], entries: [manual('later')], days: { [date]: { burned: 2400, deficit: 650, protein: 175, burnedSource: 'manual' } }, defaults: { burned: 0, deficit: 400, protein: 175 } });
    const restored = undoAction(later, receipt);
    assert.deepEqual(restored.entries.find(item => item.id === entry.id), before.entries[0]);
    assert.deepEqual(restored.entries.find(item => item.id === 'later'), later.entries[0]);
    assert.deepEqual(restored.foods, later.foods); assert.deepEqual(restored.days, later.days); assert.deepEqual(restored.defaults, later.defaults);
    assert.deepEqual(undoAction(restored, receipt), restored, 'an exact already restored entry is idempotent');
    if (entry.id === libraryEntry.id) assert.equal(restored.entries[0].calories, 123.456, 'recorded energy is never recomputed from the library');
  }
});

test('Undo a library deletion restores only the library record and never rewrites logged or bank snapshots', () => {
  const before = bankData(); before.entries = [libraryEntry];
  const after = { ...before, foods: before.foods.filter(food => food.id !== rice.id) };
  const receipt = receiptFor(before, after, 'delete-food', rice.id);
  const later = { ...after, entries: [...after.entries, manual('later')], foodBank: after.foodBank.map((portion, index) => index ? portion : { ...portion, name: 'Renamed later' }) };
  const result = undoAction(later, receipt);
  assert.deepEqual(result.foods, before.foods);
  assert.deepEqual(result.entries, later.entries); assert.deepEqual(result.foodBank, later.foodBank);
  assert.deepEqual(undoAction(result, receipt), result);
  assert.equal(Object.hasOwn(receipt, 'foods'), false); assert.equal(Object.hasOwn(receipt, 'defaults'), false);
});

test('Undo an uneaten bank deletion restores the exact mixed portion and keeps unrelated later records', () => {
  const before = bankData(), selected = before.foodBank[0];
  const after = { ...before, foodBank: before.foodBank.slice(1) };
  const receipt = receiptFor(before, after, 'delete-bank', selected.id);
  const later = { ...after, entries: [manual('later')], foods: [] };
  const result = undoAction(later, receipt);
  assert.deepEqual(result.foodBank, before.foodBank); assert.deepEqual(result.entries, later.entries); assert.deepEqual(result.foods, []);
  assert.deepEqual(undoAction(result, receipt), result);
});

test('Undo a mixed bank consume atomically restores one portion and removes all generated entries while preserving the frozen day target', () => {
  const before = bankData(), selected = before.foodBank[1];
  const after = consumeFoodBankPortion(before, selected.id, date, 'Lunch');
  const receipt = receiptFor(before, after, 'consume-bank', selected.id);
  assert.equal(receipt.id, 'undo-receipt'); assert.equal(receipt.entries.length, 2); assert.deepEqual(receipt.record, selected);
  const restored = { ...before, days: after.days };
  assert.deepEqual(undoAction(after, receipt), restored);
  assert.deepEqual(undoAction(undoAction(after, receipt), receipt), restored, 'repeated Undo cannot add a portion or duplicate entries');
});

test('consume Undo preserves unrelated later entries, library edits, defaults and explicit targets', () => {
  const before = bankData(), selected = before.foodBank[0];
  before.days[date] = { burned: 2700, deficit: 450, protein: 175, burnedSource: 'manual' };
  const after = consumeFoodBankPortion(before, selected.id, date, 'Lunch');
  const receipt = receiptFor(before, after, 'consume-bank', selected.id);
  const later = normalizeData({ ...after, foods: [{ ...rice, name: 'Later rice' }], entries: [...after.entries, manual('same-day'), { ...manual('other-day'), date: '2026-10-18' }], defaults: { burned: 0, deficit: 700, protein: 175 }, days: { ...after.days, '2026-10-18': { burned: 2200, deficit: 300, protein: 175, burnedSource: 'manual' } } });
  const result = undoAction(later, receipt);
  assert.deepEqual(result.entries, later.entries.filter(entry => !selected.items.some(item => item.id === entry.id)));
  assert.deepEqual(result.days, later.days); assert.deepEqual(result.defaults, later.defaults); assert.deepEqual(result.foods, later.foods);
  assert.deepEqual(result.foodBank[0], selected);
});

test('auto-created day targets are kept for later remote meals, including later edits or same-day local entries', () => {
  const before = bankData(), selected = before.foodBank[0];
  const after = consumeFoodBankPortion(before, selected.id, date, 'Lunch');
  const receipt = receiptFor(before, after, 'consume-bank', selected.id);
  assert.deepEqual(undoAction(after, receipt).days, after.days);
  const editedTarget = { ...after, days: { ...after.days, [date]: { ...after.days[date], deficit: 850, burned: 2600, burnedSource: 'manual' } } };
  assert.deepEqual(undoAction(editedTarget, receipt).days[date], editedTarget.days[date]);
  const withLaterEntry = { ...after, entries: [...after.entries, manual('later')] };
  assert.deepEqual(undoAction(withLaterEntry, receipt).days[date], after.days[date]);
  const deletedTarget = { ...after, days: {} };
  assert.deepEqual(undoAction(deletedTarget, receipt).days, {}, 'a target explicitly removed later is not recreated');
});

test('an edited, moved, removed or reused generated entry blocks the entire consume inverse without mutation', () => {
  const before = bankData(), selected = before.foodBank[0];
  const after = consumeFoodBankPortion(before, selected.id, date, 'Lunch');
  const receipt = receiptFor(before, after, 'consume-bank', selected.id);
  for (const mutate of [
    data => { data.entries[0].protein++; },
    data => { data.entries[0].date = '2026-10-18'; },
    data => { data.entries[0].meal = 'Dinner'; },
    data => { data.entries.shift(); },
    data => { data.entries[0] = { ...manual(), id: data.entries[0].id }; },
  ]) {
    const changed = structuredClone(after); mutate(changed); const saved = structuredClone(changed);
    assert.throws(() => undoAction(changed, receipt), UndoConflictError);
    assert.deepEqual(changed, saved);
    assert.equal(changed.foodBank.some(portion => portion.id === selected.id), false);
  }
});

test('reused record or ingredient IDs block restoration instead of overwriting other records', () => {
  const before = bankData(); before.entries = [manual()];
  const entryReceipt = receiptFor(before, { ...before, entries: [] }, 'delete-entry', 'manual');
  assert.throws(() => undoAction({ ...before, entries: [{ ...manual(), foodName: 'Different meal' }] }, entryReceipt), UndoConflictError);
  const entryInBank = structuredClone(before); entryInBank.entries = []; entryInBank.foodBank[0].items[0].id = 'manual';
  assert.throws(() => undoAction(entryInBank, entryReceipt), UndoConflictError);
  const foodReceipt = receiptFor(before, { ...before, foods: [eggs] }, 'delete-food', rice.id);
  assert.throws(() => undoAction({ ...before, foods: [{ ...rice, name: 'Different food' }, eggs] }, foodReceipt), UndoConflictError);
  const selected = before.foodBank[0], removed = { ...before, foodBank: before.foodBank.slice(1) };
  const bankReceipt = receiptFor(before, removed, 'delete-bank', selected.id);
  assert.throws(() => undoAction({ ...removed, foodBank: [{ ...selected, name: 'Different portion' }, ...removed.foodBank] }, bankReceipt), UndoConflictError);
  assert.throws(() => undoAction({ ...removed, entries: [...removed.entries, { ...selected.items[0], date, meal: 'Reused entry ID' }] }, bankReceipt), UndoConflictError);
  const movedIngredient = structuredClone(removed); movedIngredient.foodBank[0].items[0].id = selected.items[0].id;
  assert.throws(() => undoAction(movedIngredient, bankReceipt), UndoConflictError);
});

test('receipts reject unrelated multi-action snapshots and invalid or mutated receipt records', () => {
  const before = bankData(); before.entries = [manual()];
  const after = { ...before, entries: [] };
  for (const changed of [{ ...after, foods: [] }, { ...after, days: { [date]: before.defaults } }, before]) {
    assert.throws(() => receiptFor(before, changed, 'delete-entry', 'manual'), /valid Undo action/);
  }
  assert.throws(() => receiptFor(before, after, 'delete-entry', 'missing'));
  assert.throws(() => receiptFor(before, after, 'unsupported', 'manual'));
  const receipt = receiptFor(before, after, 'delete-entry', 'manual');
  for (const changed of [null, { ...receipt, version: 2 }, { ...receipt, index: -1 }, { ...receipt, id: '' }, { ...receipt, recordId: 'other' }, { ...receipt, record: { ...receipt.record, protein: Infinity } }]) assert.throws(() => undoAction(after, changed));
});

test('normalization and successful or failed Undo leave every input snapshot and receipt untouched', () => {
  const before = freeze(bankData()), selected = before.foodBank[0];
  const beforeCopy = structuredClone(before);
  const after = freeze(consumeFoodBankPortion(before, selected.id, date, 'Lunch'));
  const afterCopy = structuredClone(after);
  const receipt = freeze(receiptFor(before, after, 'consume-bank', selected.id)), receiptCopy = structuredClone(receipt);
  const result = undoAction(after, receipt);
  assert.deepEqual(result, { ...before, days: after.days });
  assert.deepEqual(before, beforeCopy); assert.deepEqual(after, afterCopy); assert.deepEqual(receipt, receiptCopy);
  result.foodBank[0].items[0].protein++;
  assert.deepEqual(before, beforeCopy); assert.deepEqual(receipt, receiptCopy);
});
