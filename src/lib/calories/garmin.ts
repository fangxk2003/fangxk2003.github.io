import { addDays, todayISO, type DaySettings } from './model.ts';

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
  predicted: boolean;
  restingDay?: GarminDay;
  predictionUnavailable: boolean;
  restingEstimate: boolean;
  unavailable: boolean;
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

/** The most recent full-day resting reading; today's partial and future values are excluded. */
export function latestCompletedGarminDay(days: readonly GarminDay[], today = todayISO()): GarminDay | undefined {
  let latest: GarminDay | undefined;
  for (const day of days) if (day.date < today && (!latest || day.date > latest.date)) latest = day;
  return latest;
}

/** Today uses active so far plus exactly yesterday's resting energy when both are imported.
 * A missing date uses the latest completed resting reading as an estimate.
 * Recorded totals and explicit manual overrides keep their existing meaning. */
export function effectiveBurned(date: string, settings: DaySettings, days: readonly GarminDay[], today = todayISO()): EffectiveBurned {
  const requestedSource = settings.burnedSource === 'manual' ? 'manual' : 'garmin';
  const day = days.find(item => item.date === date);
  if (requestedSource === 'garmin' && day) {
    const restingDay = date === today ? days.find(item => item.date === addDays(date, -1)) : undefined;
    if (restingDay) return { burned: day.active + restingDay.resting, source: 'garmin', requestedSource, day, restingDay, fallback: false, predicted: true, predictionUnavailable: false, restingEstimate: false, unavailable: false };
    return { burned: day.total, source: 'garmin', requestedSource, day, fallback: false, predicted: false, predictionUnavailable: date === today, restingEstimate: false, unavailable: false };
  }
  if (requestedSource === 'manual') return { burned: settings.burned, source: 'manual', requestedSource, day, fallback: false, predicted: false, predictionUnavailable: false, restingEstimate: false, unavailable: false };
  const restingDay = latestCompletedGarminDay(days, today);
  if (restingDay) return { burned: restingDay.resting, source: 'garmin', requestedSource, restingDay, fallback: true, predicted: false, predictionUnavailable: false, restingEstimate: true, unavailable: false };
  return { burned: 0, source: 'garmin', requestedSource, fallback: true, predicted: false, predictionUnavailable: false, restingEstimate: false, unavailable: true };
}
