import type { DaySettings } from './model.ts';

/** Imported Garmin readings stay separate from the editable food journal. */
export interface GarminDay {
  date: string;
  active: number;
  resting: number;
  total: number;
  synced_at: string;
}

export interface GarminStatus {
  state: 'not_connected' | 'connected' | 'needs_login' | 'error';
  last_attempt_at: string | null;
  last_sync_at: string | null;
  last_error_code: string | null;
}

export interface GarminSnapshot { days: GarminDay[]; status: GarminStatus }
export interface EffectiveBurned {
  burned: number;
  source: 'manual' | 'garmin';
  requestedSource: 'manual' | 'garmin';
  day?: GarminDay;
  fallback: boolean;
}

export function emptyGarminSnapshot(): GarminSnapshot {
  return { days: [], status: { state: 'not_connected', last_attempt_at: null, last_sync_at: null, last_error_code: null } };
}

function record(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Garmin returned an invalid record.');
  return value as Record<string, unknown>;
}

function calories(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value > 1e8) throw new Error(`Garmin returned invalid ${label}.`);
  return value;
}

function date(value: unknown): string {
  if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(value)) throw new Error('Garmin returned an invalid date.');
  const parsed = new Date(`${value}T12:00:00Z`);
  if (Number.isNaN(parsed.getTime()) || parsed.toISOString().slice(0, 10) !== value) throw new Error('Garmin returned an invalid date.');
  return value;
}

function timestamp(value: unknown, nullable = false): string | null {
  if (nullable && value === null) return null;
  if (typeof value !== 'string' || value.length > 40 || !/^\d{4}-\d{2}-\d{2}T.*(?:Z|[+-]\d{2}:?\d{2})$/i.test(value) || Number.isNaN(Date.parse(value))) throw new Error('Garmin returned an invalid sync time.');
  return value;
}

export function normalizeGarminDay(raw: unknown): GarminDay {
  const value = record(raw);
  return { date: date(value.date), active: calories(value.active, 'active calories'), resting: calories(value.resting, 'resting calories'), total: calories(value.total, 'total calories'), synced_at: timestamp(value.synced_at) as string };
}

export function normalizeGarminStatus(raw: unknown): GarminStatus {
  const value = record(raw);
  if (value.state !== 'not_connected' && value.state !== 'connected' && value.state !== 'needs_login' && value.state !== 'error') throw new Error('Garmin returned an invalid connection state.');
  if (value.last_error_code !== null && (typeof value.last_error_code !== 'string' || !/^[a-zA-Z0-9_.:-]{1,80}$/.test(value.last_error_code))) throw new Error('Garmin returned an invalid error code.');
  return { state: value.state, last_attempt_at: timestamp(value.last_attempt_at, true), last_sync_at: timestamp(value.last_sync_at, true), last_error_code: value.last_error_code as string | null };
}

export function normalizeGarminSnapshot(raw: unknown): GarminSnapshot {
  const value = record(raw);
  if (!Array.isArray(value.days) || value.days.length > 10000) throw new Error('Garmin returned an invalid history.');
  const days = value.days.map(normalizeGarminDay);
  if (new Set(days.map(day => day.date)).size !== days.length) throw new Error('Garmin returned duplicate dates.');
  return { days, status: normalizeGarminStatus(value.status) };
}

/** Imported total = active + resting as recorded by Garmin. An explicit manual override wins. */
export function effectiveBurned(date: string, settings: DaySettings, days: readonly GarminDay[]): EffectiveBurned {
  const requestedSource = settings.burnedSource === 'manual' ? 'manual' : 'garmin';
  const day = days.find(item => item.date === date);
  if (requestedSource === 'garmin' && day) return { burned: day.total, source: 'garmin', requestedSource, day, fallback: false };
  return { burned: settings.burned, source: 'manual', requestedSource, day, fallback: requestedSource === 'garmin' };
}
