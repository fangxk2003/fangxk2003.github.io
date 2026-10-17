export interface FormDraftStorage {
  readonly length: number;
  key(index: number): string | null;
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}
export type FormDraftKind = 'entry' | 'food' | 'targets' | 'bank-prepare';
export interface FormDraftDescriptor {
  kind: FormDraftKind;
  label: string;
  date?: string;
  entryId?: string;
  foodId?: string;
}
export interface FormDraftSummary extends FormDraftDescriptor { key: string; updatedAt: number }
export interface StoredFormDraft { summary: FormDraftSummary; value: unknown }
const PREFIX = 'calories.form-drafts.v1:';
const MAX_RECORD_LENGTH = 64 * 1024;
const kinds = new Set<FormDraftKind>(['entry', 'food', 'targets', 'bank-prepare']);

function browserStorage(): FormDraftStorage | undefined {
  try { return typeof window === 'undefined' ? undefined : window.localStorage; } catch { return undefined; }
}
function prefix(scope: string): string {
  if (!scope.trim() || scope.length > 2000) throw new Error('A journal account is required for draft recovery.');
  return `${PREFIX}${encodeURIComponent(scope)}:`;
}
function storageKey(scope: string, key: string): string {
  if (!key.trim() || key.length > 400) throw new Error('Invalid form draft key.');
  return `${prefix(scope)}${encodeURIComponent(key)}`;
}
function safeValue(value: unknown, depth = 0): boolean {
  if (depth > 8) return false;
  if (value === null || typeof value === 'boolean') return true;
  if (typeof value === 'number') return Number.isFinite(value);
  if (typeof value === 'string') return value.length <= 10000 && !/^(?:data:image\/|blob:)/i.test(value);
  if (Array.isArray(value)) return value.length <= 1000 && value.every(item => safeValue(item, depth + 1));
  if (!value || typeof value !== 'object' || Object.getPrototypeOf(value) !== Object.prototype) return false;
  const entries = Object.entries(value);
  return entries.length <= 100 && entries.every(([key, item]) => !/^(?:photo|photos|image|images|base64|blob|preview)$/i.test(key) && safeValue(item, depth + 1));
}
function descriptor(raw: unknown): FormDraftDescriptor | null {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return null;
  const value = raw as Record<string, unknown>;
  if (!kinds.has(value.kind as FormDraftKind) || typeof value.label !== 'string' || !value.label.trim() || value.label.length > 200) return null;
  const result: FormDraftDescriptor = { kind: value.kind as FormDraftKind, label: value.label.trim() };
  for (const field of ['entryId', 'foodId'] as const) {
    if (value[field] !== undefined) {
      if (typeof value[field] !== 'string' || !value[field].trim() || value[field].length > 200) return null;
      result[field] = value[field];
    }
  }
  if (value.date !== undefined) {
    if (typeof value.date !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(value.date) || Number.isNaN(Date.parse(`${value.date}T12:00:00Z`)) || new Date(`${value.date}T12:00:00Z`).toISOString().slice(0, 10) !== value.date) return null;
    result.date = value.date;
  }
  return result;
}
function parse(encoded: string | null, expectedKey?: string): StoredFormDraft | null {
  if (!encoded || encoded.length > MAX_RECORD_LENGTH) return null;
  try {
    const record = JSON.parse(encoded);
    const summary = descriptor(record.summary);
    if (record.version !== 1 || !summary || typeof record.summary.key !== 'string' || !record.summary.key.trim() || record.summary.key.length > 400 || (expectedKey !== undefined && record.summary.key !== expectedKey) || !Number.isFinite(record.summary.updatedAt) || record.summary.updatedAt < 0 || !safeValue(record.value)) return null;
    return { summary: { ...summary, key: record.summary.key, updatedAt: record.summary.updatedAt }, value: record.value };
  } catch { return null; }
}

export function readFormDraft(scope: string, key: string, storage = browserStorage()): StoredFormDraft | null {
  try { return storage ? parse(storage.getItem(storageKey(scope, key)), key) : null; } catch { return null; }
}
export function listFormDrafts(scope: string, storage = browserStorage()): FormDraftSummary[] {
  if (!storage) return [];
  try {
    const accountPrefix = prefix(scope);
    const drafts: FormDraftSummary[] = [];
    for (let index = 0; index < storage.length; index++) {
      const key = storage.key(index);
      if (!key?.startsWith(accountPrefix)) continue;
      const draft = parse(storage.getItem(key));
      if (draft && key === storageKey(scope, draft.summary.key)) drafts.push(draft.summary);
    }
    return drafts.sort((a, b) => b.updatedAt - a.updatedAt || a.key.localeCompare(b.key));
  } catch { return []; }
}
export function writeFormDraft(scope: string, key: string, value: unknown, details: FormDraftDescriptor, storage = browserStorage(), now = Date.now()): void {
  const summary = descriptor(details);
  if (!summary || !safeValue(value) || !Number.isFinite(now) || now < 0) throw new Error('This form draft could not be saved on this device.');
  const encoded = JSON.stringify({ version: 1, summary: { ...summary, key, updatedAt: now }, value });
  if (!storage || encoded.length > MAX_RECORD_LENGTH) throw new Error('This form draft could not be saved on this device.');
  try { storage.setItem(storageKey(scope, key), encoded); } catch { throw new Error('This form draft could not be saved on this device.'); }
}
export function removeFormDraft(scope: string, key: string, storage = browserStorage()): void {
  if (!storage) return;
  storage.removeItem(storageKey(scope, key));
}
export function clearFormDrafts(scope: string, storage = browserStorage()): boolean {
  if (!storage) return false;
  try {
    const accountPrefix = prefix(scope);
    const keys: string[] = [];
    for (let index = 0; index < storage.length; index++) {
      const key = storage.key(index);
      if (key?.startsWith(accountPrefix)) keys.push(key);
    }
    for (const key of keys) storage.removeItem(key);
    return true;
  } catch { return false; }
}

/** Restore only fields present in this editor's seed; new fields retain their defaults. */
export function restoreFormDraft<T>(seed: T, raw: unknown): T | null {
  if (!safeValue(raw)) return null;
  function restore(template: unknown, value: unknown): unknown {
    if (template === null) return value === null ? null : undefined;
    if (Array.isArray(template)) {
      if (!Array.isArray(value) || !template.length) return undefined;
      const result = value.map(item => restore(template[0], item));
      return result.some(item => item === undefined) ? undefined : result;
    }
    if (template && typeof template === 'object') {
      if (!value || typeof value !== 'object' || Array.isArray(value)) return undefined;
      const result: Record<string, unknown> = {};
      for (const [key, fallback] of Object.entries(template)) {
        const restored = key in value ? restore(fallback, (value as Record<string, unknown>)[key]) : fallback;
        if (restored === undefined) return undefined;
        result[key] = restored;
      }
      return result;
    }
    return typeof value === typeof template ? value : undefined;
  }
  const restored = restore(seed, raw);
  return restored === undefined ? null : restored as T;
}
