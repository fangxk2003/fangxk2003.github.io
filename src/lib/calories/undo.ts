import { emptyData, normalizeData, type BankPortion, type DaySettings, type Entry, type Food, type TrackerData } from './model.ts';
import { consumeFoodBankPortion } from './food-bank.ts';
import { journalEqual } from './offline.ts';

export type UndoKind = 'delete-entry' | 'delete-food' | 'delete-bank' | 'consume-bank';
interface UndoBase { version: 1; id: string; recordId: string; index: number }
export type UndoAction =
  | (UndoBase & { kind: 'delete-entry'; record: Entry })
  | (UndoBase & { kind: 'delete-food'; record: Food })
  | (UndoBase & { kind: 'delete-bank'; record: BankPortion })
  | (UndoBase & { kind: 'consume-bank'; record: BankPortion; entries: Entry[]; date: string; createdDay?: DaySettings });

export class UndoConflictError extends Error {
  constructor() {
    super('This action can no longer be undone safely because the affected records changed. Your current journal has been kept.');
    this.name = 'UndoConflictError';
  }
}
const invalid = () => new Error('This change does not contain a valid Undo action.');
function validId(value: unknown): value is string { return typeof value === 'string' && Boolean(value.trim()) && value.length <= 200; }
function identity(): string { return globalThis.crypto?.randomUUID?.() ?? `undo-${Date.now()}-${Math.random().toString(36).slice(2)}`; }
function insert<T>(rows: readonly T[], record: T, index: number): T[] {
  const result = [...rows]; result.splice(Math.min(index, result.length), 0, record); return result;
}

/** Capture only the inverse's records. Unrelated journal data never enters a receipt. */
export function createUndoAction(beforeRaw: TrackerData, afterRaw: TrackerData, kind: UndoKind, recordId: string, makeId: () => string = identity): UndoAction {
  const before = normalizeData(beforeRaw), after = normalizeData(afterRaw);
  if (!validId(recordId) || typeof makeId !== 'function') throw invalid();
  const id = makeId();
  if (!validId(id)) throw invalid();
  const base: UndoBase = { version: 1, id, recordId, index: 0 };
  let receipt: UndoAction, expected: TrackerData;
  if (kind === 'delete-entry') {
    const index = before.entries.findIndex(entry => entry.id === recordId), record = before.entries[index];
    if (!record) throw invalid();
    receipt = { ...base, kind, index, record };
    expected = { ...before, entries: before.entries.filter(entry => entry.id !== recordId) };
  } else if (kind === 'delete-food') {
    const index = before.foods.findIndex(food => food.id === recordId), record = before.foods[index];
    if (!record) throw invalid();
    receipt = { ...base, kind, index, record };
    expected = { ...before, foods: before.foods.filter(food => food.id !== recordId) };
  } else if (kind === 'delete-bank' || kind === 'consume-bank') {
    const index = before.foodBank.findIndex(portion => portion.id === recordId), record = before.foodBank[index];
    if (!record) throw invalid();
    if (kind === 'delete-bank') {
      receipt = { ...base, kind, index, record };
      expected = { ...before, foodBank: before.foodBank.filter(portion => portion.id !== recordId) };
    } else {
      const itemIds = new Set(record.items.map(item => item.id));
      const entries = after.entries.filter(entry => itemIds.has(entry.id));
      if (entries.length !== record.items.length) throw invalid();
      const { date, meal } = entries[0];
      expected = consumeFoodBankPortion(before, recordId, date, meal);
      receipt = { ...base, kind, index, record, entries, date, ...(!Object.hasOwn(before.days, date) ? { createdDay: expected.days[date] } : {}) };
    }
  } else throw invalid();
  if (!journalEqual(expected, after)) throw invalid();
  return receipt;
}

function normalizeReceipt(raw: UndoAction): UndoAction {
  if (!raw || typeof raw !== 'object' || raw.version !== 1 || !validId(raw.id) || !validId(raw.recordId) || !Number.isInteger(raw.index) || raw.index < 0 || raw.index > 100000) throw invalid();
  const base: UndoBase = { version: 1, id: raw.id, recordId: raw.recordId, index: raw.index };
  let receipt: UndoAction;
  if (raw.kind === 'delete-entry') {
    receipt = { ...base, kind: raw.kind, record: normalizeData({ ...emptyData(), entries: [raw.record] }).entries[0] };
  } else if (raw.kind === 'delete-food') {
    receipt = { ...base, kind: raw.kind, record: normalizeData({ ...emptyData(), foods: [raw.record] }).foods[0] };
  } else if (raw.kind === 'delete-bank' || raw.kind === 'consume-bank') {
    const record = normalizeData({ ...emptyData(), foodBank: [raw.record] }).foodBank[0];
    if (raw.kind === 'delete-bank') receipt = { ...base, kind: raw.kind, record };
    else {
      const normalized = normalizeData({ ...emptyData(), entries: raw.entries, days: raw.createdDay === undefined ? {} : { [raw.date]: raw.createdDay } });
      if (normalized.entries.length !== record.items.length || !normalized.entries.length || normalized.entries[0].date !== raw.date) throw invalid();
      const expected = consumeFoodBankPortion({ ...emptyData(), foodBank: [record] }, record.id, raw.date, normalized.entries[0].meal);
      if (!journalEqual(expected.entries, normalized.entries)) throw invalid();
      receipt = { ...base, kind: raw.kind, record, entries: normalized.entries, date: raw.date, ...(raw.createdDay !== undefined ? { createdDay: normalized.days[raw.date] } : {}) };
    }
  } else throw invalid();
  if (receipt.record.id !== receipt.recordId) throw invalid();
  return receipt;
}

/** Reverse only unchanged affected records; normalize a fresh result so failures are atomic. */
export function undoAction(currentRaw: TrackerData, rawReceipt: UndoAction): TrackerData {
  const current = normalizeData(currentRaw), receipt = normalizeReceipt(rawReceipt);
  if (receipt.kind === 'delete-entry') {
    const existing = current.entries.find(entry => entry.id === receipt.recordId);
    if (existing) { if (journalEqual(existing, receipt.record)) return current; throw new UndoConflictError(); }
    if (current.foodBank.some(portion => portion.items.some(item => item.id === receipt.recordId))) throw new UndoConflictError();
    return normalizeData({ ...current, entries: insert(current.entries, receipt.record, receipt.index) });
  }
  if (receipt.kind === 'delete-food') {
    const existing = current.foods.find(food => food.id === receipt.recordId);
    if (existing) { if (journalEqual(existing, receipt.record)) return current; throw new UndoConflictError(); }
    return normalizeData({ ...current, foods: insert(current.foods, receipt.record, receipt.index) });
  }
  const existing = current.foodBank.find(portion => portion.id === receipt.recordId);
  if (existing) { if (journalEqual(existing, receipt.record)) return current; throw new UndoConflictError(); }
  const ids = new Set(receipt.record.items.map(item => item.id));
  if (current.foodBank.some(portion => portion.items.some(item => ids.has(item.id)))) throw new UndoConflictError();
  if (receipt.kind === 'delete-bank') {
    if (current.entries.some(entry => ids.has(entry.id))) throw new UndoConflictError();
    return normalizeData({ ...current, foodBank: insert(current.foodBank, receipt.record, receipt.index) });
  }
  const entriesById = new Map(current.entries.map(entry => [entry.id, entry]));
  if (receipt.entries.some(entry => !journalEqual(entriesById.get(entry.id), entry))) throw new UndoConflictError();
  const entries = current.entries.filter(entry => !ids.has(entry.id));
  // A remote meal on this date may not have arrived yet. Keeping its frozen targets
  // avoids reverting that meal to legacy defaults during a later three-way merge.
  return normalizeData({ ...current, entries, foodBank: insert(current.foodBank, receipt.record, receipt.index) });
}
