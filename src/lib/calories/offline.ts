import { normalizeData, type TrackerData, type BankPortion, type Entry } from './model.ts';

export interface JournalSnapshot {
  data: TrackerData;
  pending: boolean;
  offline: boolean;
  conflict: string | null;
  localVersion: string;
}
export interface JournalCache {
  format: 1;
  project: string;
  owner: string;
  verifiedDeadline: number;
  baseline: { data: TrackerData; revision: number | null };
  working: TrackerData;
  localVersion: string;
  conflict: string | null;
  attempted?: { data: TrackerData; revision: number | null }[];
}

export class JournalConflictError extends Error {
  constructor(part = 'the same journal records') {
    super(`This device and another device changed ${part}. Your local changes are kept. Export a backup before choosing which version to keep.`);
    this.name = 'JournalConflictError';
  }
}
export class RecoverableNetworkError extends Error {
  constructor(message = 'Cannot reach your storage. Your local changes are kept on this device and will sync when you reconnect.') {
    super(message); this.name = 'RecoverableNetworkError';
  }
}
export function isRecoverableNetworkError(error: unknown): error is RecoverableNetworkError { return error instanceof RecoverableNetworkError; }

function stable(value: unknown): string {
  if (Array.isArray(value)) {
    // Journal collections are identified by stable IDs; their visual order is not an edit.
    const rows = value.every(item => item && typeof item === 'object' && typeof item.id === 'string')
      ? [...value].sort((a, b) => a.id.localeCompare(b.id)) : value;
    return `[${rows.map(stable).join(',')}]`;
  }
  if (value && typeof value === 'object') return `{${Object.entries(value).sort(([a], [b]) => a.localeCompare(b)).map(([key, item]) => `${JSON.stringify(key)}:${stable(item)}`).join(',')}}`;
  return JSON.stringify(value) ?? 'undefined';
}
export function journalEqual(a: unknown, b: unknown): boolean { return stable(a) === stable(b); }
function choose<T>(base: T, local: T, remote: T, label: string, uncertain: readonly T[] = []): T {
  // A lost response may hide a committed write. If this record changed again
  // locally (including Undo), preserve that later intent rather than mistaking
  // a return to the old baseline for "no local edit". An unchanged cloud record
  // still uses the original baseline: the attempted write may never have arrived.
  for (const attempt of uncertain) {
    if (!journalEqual(base, attempt) && !journalEqual(local, attempt) && !journalEqual(remote, base)) base = attempt;
  }
  if (journalEqual(local, remote) || journalEqual(base, remote)) return local;
  if (journalEqual(base, local)) return remote;
  throw new JournalConflictError(label);
}
function keyed<T extends { id: string }>(base: readonly T[], local: readonly T[], remote: readonly T[], label: string, uncertain: readonly (readonly T[])[] = []): T[] {
  const maps = [base, local, remote, ...uncertain].map(rows => new Map(rows.map(row => [row.id, row])));
  const ids = new Set([...remote, ...local, ...base].map(row => row.id));
  const result: T[] = [];
  for (const id of ids) {
    const value = choose(maps[0].get(id), maps[1].get(id), maps[2].get(id), label, maps.slice(3).map(map => map.get(id)));
    if (value !== undefined) result.push(value);
  }
  return result;
}

/** A portion and every entry made from its stable ingredient IDs move together.
 * Two different consumes/removes never merge into a partially consumed meal. */
function inventory(base: TrackerData, local: TrackerData, remote: TrackerData, uncertain: readonly TrackerData[]): { entries: Entry[]; foodBank: BankPortion[] } {
  const versions = [base, local, remote, ...uncertain];
  const portions = new Map<string, Set<string>>();
  const itemOwner = new Map<string, string>();
  for (const data of versions) for (const portion of data.foodBank) {
    const ids = portions.get(portion.id) ?? new Set<string>();
    for (const item of portion.items) {
      const previous = itemOwner.get(item.id);
      if (previous !== undefined && previous !== portion.id) throw new JournalConflictError('the same food-bank ingredients');
      itemOwner.set(item.id, portion.id); ids.add(item.id);
    }
    portions.set(portion.id, ids);
  }
  const allItems = new Set(itemOwner.keys());
  const entries = keyed(base.entries.filter(e => !allItems.has(e.id)), local.entries.filter(e => !allItems.has(e.id)), remote.entries.filter(e => !allItems.has(e.id)), 'the same journal entry', uncertain.map(data => data.entries.filter(e => !allItems.has(e.id))));
  const foodBank: BankPortion[] = [];
  const grouped = versions.map(data => {
    const byPortion = new Map(data.foodBank.map(portion => [portion.id, portion]));
    const byItems = new Map<string, Entry[]>();
    for (const entry of data.entries) {
      const portionId = itemOwner.get(entry.id);
      if (portionId !== undefined) {
        const rows = byItems.get(portionId) ?? [];
        rows.push(entry); byItems.set(portionId, rows);
      }
    }
    return { byPortion, byItems };
  });
  for (const id of portions.keys()) {
    const states = grouped.map(data => ({
      portion: data.byPortion.get(id),
      entries: data.byItems.get(id) ?? [],
    }));
    const selected = choose(states[0], states[1], states[2], 'the same food-bank portion', states.slice(3));
    if (selected.portion) foodBank.push(selected.portion);
    entries.push(...selected.entries);
  }
  return { entries, foodBank };
}

/** Three-way merge uses whole records: changed/deleted records conflict, disjoint IDs merge. */
export function mergeJournals(baseRaw: TrackerData, localRaw: TrackerData, remoteRaw: TrackerData, uncertainRaw: readonly TrackerData[] = []): TrackerData {
  const [base, local, remote, ...uncertain] = [baseRaw, localRaw, remoteRaw, ...uncertainRaw].map(normalizeData);
  const days: TrackerData['days'] = {};
  for (const date of new Set([...Object.keys(remote.days), ...Object.keys(local.days), ...Object.keys(base.days)])) {
    const day = choose(base.days[date], local.days[date], remote.days[date], `the targets for ${date}`, uncertain.map(data => data.days[date]));
    if (day !== undefined) days[date] = day;
  }
  const defaults = choose(base.defaults, local.defaults, remote.defaults, 'the default targets', uncertain.map(data => data.defaults));
  try {
    return normalizeData({ version: 2, foods: keyed(base.foods, local.foods, remote.foods, 'the same library food', uncertain.map(data => data.foods)), ...inventory(base, local, remote, uncertain), days, defaults });
  } catch (error) {
    if (error instanceof JournalConflictError) throw error;
    throw new JournalConflictError('overlapping journal or food-bank records');
  }
}

export function journalCacheKey(project: string, owner: string): string { return `calories.journal.v1:${project}:${owner}`; }
export function normalizeJournalCache(raw: unknown, project: string, owner: string): JournalCache {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) throw new Error('The local journal could not be read.');
  const value = raw as JournalCache;
  const validRevision = (revision: unknown) => revision === null || (typeof revision === 'number' && Number.isSafeInteger(revision) && revision >= 1);
  if (value.format !== 1 || value.project !== project || value.owner !== owner || !Number.isFinite(value.verifiedDeadline) || typeof value.localVersion !== 'string' || !value.localVersion || !value.baseline || !validRevision(value.baseline.revision) || (value.conflict !== null && typeof value.conflict !== 'string')) throw new Error('The local journal could not be read.');
  if (value.attempted && (!Array.isArray(value.attempted) || value.attempted.length > 20 || value.attempted.some(attempt => !attempt || !validRevision(attempt.revision)))) throw new Error('The local journal could not be read.');
  return {
    format: 1, project, owner, verifiedDeadline: value.verifiedDeadline,
    baseline: { data: normalizeData(value.baseline.data), revision: value.baseline.revision },
    working: normalizeData(value.working), localVersion: value.localVersion, conflict: value.conflict,
    ...(value.attempted ? { attempted: value.attempted.map(attempt => ({ data: normalizeData(attempt.data), revision: attempt.revision })) } : {}),
  };
}

const processLocks = new Map<string, Promise<unknown>>();
/** Web Locks serializes tabs. The process fallback is for non-browser test/runtime use. */
export async function withJournalLock<T>(name: string, run: () => Promise<T> | T): Promise<T> {
  const locks = typeof navigator !== 'undefined' ? navigator.locks : undefined;
  if (locks) return locks.request(name, async () => run());
  if (typeof window !== 'undefined') throw new Error('This browser cannot safely coordinate offline journal edits. Update Safari or use a current browser.');
  const previous = processLocks.get(name) ?? Promise.resolve();
  const result = previous.catch(() => {}).then(run);
  processLocks.set(name, result);
  try { return await result; } finally { if (processLocks.get(name) === result) processLocks.delete(name); }
}
