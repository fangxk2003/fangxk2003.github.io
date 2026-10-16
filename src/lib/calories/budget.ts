import { addDays, todayISO, type TrackerData } from './model.ts';
import { effectiveBurned, latestCompletedGarminDay, type GarminDay } from './garmin.ts';

function finite(value: number, label: string, nonNegative = false): number {
  if (!Number.isFinite(value) || (nonNegative && value < 0)) throw new Error(`Invalid ${label}.`);
  return value;
}

export const MAX_DAILY_DEFICIT = 1000;

/** Limit the budgeted deficit; the raw allowance retains all unpaid debt. */
export function adjustedBudget(burned: number, deficit: number, carryIn = 0) {
  finite(burned, 'calories burned', true);
  finite(deficit, 'deficit target', true);
  finite(carryIn, 'carried calorie balance');
  const baseBudget = Math.max(0, burned - deficit);
  const rawBudget = finite(baseBudget + carryIn, 'adjusted calorie budget');
  const budget = Math.max(0, rawBudget, burned - MAX_DAILY_DEFICIT);
  const deferredDebt = finite(budget - rawBudget, 'deferred calorie debt');
  return { baseBudget, carryIn, rawBudget, budget, deferredDebt };
}

export interface BalanceDay extends ReturnType<typeof adjustedBudget> {
  date: string;
  eaten: number;
  balance: number;
  available: boolean;
  settled: boolean;
}
export interface CalorieLedger { days: Record<string, BalanceDay>; completed: BalanceDay[]; provisional?: BalanceDay }

/** Ended logged days settle carry; today’s logged balance also informs future previews.
 * Unlogged days and planned future meals never add carry.
 * Derive balances from recorded intake, each day's targets and the latest burned readings,
 * so edits and Garmin corrections propagate without storing or counting carry twice. */
export function calorieLedger(data: TrackerData, garminDays: readonly GarminDay[], currentDay = todayISO()): CalorieLedger {
  const eaten = new Map<string, number>();
  for (const entry of data.entries) eaten.set(entry.date, finite((eaten.get(entry.date) ?? 0) + finite(entry.calories, 'logged calories', true), 'daily logged calories'));
  const imported = new Map(garminDays.map(day => [day.date, day]));
  const latestResting = latestCompletedGarminDay(garminDays, currentDay);
  const days: Record<string, BalanceDay> = {};
  const completed: BalanceDay[] = [];
  let carry = 0;
  let provisional: BalanceDay | undefined;
  for (const date of [...eaten.keys()].sort()) {
    const settings = data.days[date] ?? data.defaults;
    const readings = [imported.get(date), imported.get(addDays(date, -1)), latestResting].filter((day): day is GarminDay => day !== undefined);
    const reading = effectiveBurned(date, settings, readings, currentDay);
    const amounts = adjustedBudget(reading.burned, settings.deficit, carry);
    // Settle against the full allowance, so the daily deficit cap never forgives debt.
    const row: BalanceDay = { date, ...amounts, eaten: eaten.get(date)!, balance: finite(amounts.rawBudget - eaten.get(date)!, 'running calorie balance'), available: !reading.unavailable, settled: date < currentDay && !reading.unavailable };
    days[date] = row;
    if (row.settled) { carry = row.balance; completed.push(row); }
    else if (date === currentDay && row.available) { carry = row.balance; provisional = row; }
  }
  return { days, completed, ...(provisional ? { provisional } : {}) };
}

/** Future previews include today’s provisional balance; earlier dates use settled carry. */
export function carryBefore(date: string, ledger: CalorieLedger): number {
  if (ledger.provisional && ledger.provisional.date < date) return ledger.provisional.balance;
  let low = 0, high = ledger.completed.length;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (ledger.completed[middle].date < date) low = middle + 1;
    else high = middle;
  }
  return low > 0 ? ledger.completed[low - 1].balance : 0;
}
