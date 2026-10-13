import { addDays, todayISO, type TrackerData } from './model.ts';
import { effectiveBurned, latestCompletedGarminDay, type GarminDay } from './garmin.ts';

function finite(value: number, label: string, nonNegative = false): number {
  if (!Number.isFinite(value) || (nonNegative && value < 0)) throw new Error(`Invalid ${label}.`);
  return value;
}

/** Clamp the available allowance only; retain any debt in the running balance. */
export function adjustedBudget(burned: number, deficit: number, carryIn = 0) {
  finite(burned, 'calories burned', true);
  finite(deficit, 'deficit target', true);
  finite(carryIn, 'carried calorie balance');
  const baseBudget = Math.max(0, burned - deficit);
  const rawBudget = finite(baseBudget + carryIn, 'adjusted calorie budget');
  return { baseBudget, carryIn, rawBudget, budget: Math.max(0, rawBudget) };
}

export interface BalanceDay extends ReturnType<typeof adjustedBudget> {
  date: string;
  eaten: number;
  balance: number;
  available: boolean;
  settled: boolean;
}
export interface CalorieLedger { days: Record<string, BalanceDay>; completed: BalanceDay[] }

/** Only ended days containing food entries contribute. Missing days never earn a budget.
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
  for (const date of [...eaten.keys()].sort()) {
    const settings = data.days[date] ?? data.defaults;
    const readings = [imported.get(date), imported.get(addDays(date, -1)), latestResting].filter((day): day is GarminDay => day !== undefined);
    const reading = effectiveBurned(date, settings, readings, currentDay);
    const amounts = adjustedBudget(reading.burned, settings.deficit, carry);
    const row: BalanceDay = { date, ...amounts, eaten: eaten.get(date)!, balance: finite(amounts.rawBudget - eaten.get(date)!, 'running calorie balance'), available: !reading.unavailable, settled: date < currentDay && !reading.unavailable };
    days[date] = row;
    if (row.settled) { carry = row.balance; completed.push(row); }
  }
  return { days, completed };
}

/** Find the prior completed balance for any selected date, including unlogged gaps. */
export function carryBefore(date: string, ledger: CalorieLedger): number {
  let low = 0, high = ledger.completed.length;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (ledger.completed[middle].date < date) low = middle + 1;
    else high = middle;
  }
  return low > 0 ? ledger.completed[low - 1].balance : 0;
}
