export type Unit = 'g' | 'ml' | 'piece';
export type FoodState = 'Raw' | 'Cooked' | 'As served';
export interface Macros { carbs: number; protein: number; fat: number; fiber: number; calories: number }
export interface Food extends Macros { id: string; name: string; baseQuantity: number; unit: Unit; state: FoodState }
export interface Entry extends Macros { id: string; date: string; meal: string; foodId: string; foodName: string; quantity: number; unit: Unit; state: FoodState }
export interface DaySettings { burned: number; deficit: number; protein: number }
export interface TrackerData { version: 1; foods: Food[]; entries: Entry[]; days: Record<string, DaySettings>; defaults: DaySettings }

export function emptyData(): TrackerData {
  return { version: 1, foods: [], entries: [], days: {}, defaults: { burned: 3200, deficit: 500, protein: 175 } };
}

export function todayISO(): string {
  const now = new Date();
  return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
}

export function addDays(date: string, offset: number): string {
  const d = new Date(`${date}T12:00:00Z`);
  d.setUTCDate(d.getUTCDate() + offset);
  return d.toISOString().slice(0, 10);
}

export function macrosForFood(food: Food, quantity: number): Macros {
  const ratio = quantity / food.baseQuantity;
  return { carbs: food.carbs * ratio, protein: food.protein * ratio, fat: food.fat * ratio, fiber: food.fiber * ratio, calories: food.calories * ratio };
}

export function dayTotals(entries: Entry[]): Macros {
  return entries.reduce((total, item) => ({ carbs: total.carbs + item.carbs, protein: total.protein + item.protein, fat: total.fat + item.fat, fiber: total.fiber + item.fiber, calories: total.calories + item.calories }), { carbs: 0, protein: 0, fat: 0, fiber: 0, calories: 0 });
}

function record(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('The backup contains an invalid record.');
  return value as Record<string, unknown>;
}
function number(value: unknown, label: string, positive = false): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value > 1e8 || (positive && value === 0)) throw new Error(`Invalid ${label}: enter a finite ${positive ? 'positive' : 'non-negative'} number.`);
  return value;
}
function text(value: unknown, label: string, max = 200): string {
  if (typeof value !== 'string' || !value.trim() || value.length > max) throw new Error(`Invalid ${label}.`);
  return value;
}
function unit(value: unknown): Unit {
  if (value !== 'g' && value !== 'ml' && value !== 'piece') throw new Error('Invalid food unit.');
  return value;
}
function state(value: unknown): FoodState {
  if (value !== 'Raw' && value !== 'Cooked' && value !== 'As served') throw new Error('Invalid food state.');
  return value;
}
function date(value: unknown): string {
  const s = text(value, 'date', 10);
  if (!/^\d{4}-\d{2}-\d{2}$/.test(s) || Number.isNaN(Date.parse(`${s}T12:00:00Z`)) || new Date(`${s}T12:00:00Z`).toISOString().slice(0, 10) !== s) throw new Error('Invalid date.');
  return s;
}
function macros(value: Record<string, unknown>): Macros {
  return { carbs: number(value.carbs, 'carbs'), protein: number(value.protein, 'protein'), fat: number(value.fat, 'fat'), fiber: number(value.fiber, 'fiber'), calories: number(value.calories, 'calories') };
}
function settings(raw: unknown): DaySettings {
  const v = record(raw);
  return { burned: number(v.burned, 'calories burned'), deficit: number(v.deficit, 'deficit target'), protein: number(v.protein, 'protein target') };
}

/** Strict validation for imported and remotely loaded data. No silent data loss. */
export function normalizeData(raw: unknown): TrackerData {
  const v = record(raw);
  if (v.version !== 1 || !Array.isArray(v.foods) || !Array.isArray(v.entries) || v.foods.length > 10000 || v.entries.length > 100000) throw new Error('This is not a supported Calories backup.');
  const foods: Food[] = v.foods.map(rawFood => {
    const f = record(rawFood);
    return { id: text(f.id, 'food ID'), name: text(f.name, 'food name'), baseQuantity: number(f.baseQuantity, 'base quantity', true), unit: unit(f.unit), state: state(f.state), ...macros(f) };
  });
  const entries: Entry[] = v.entries.map(rawEntry => {
    const e = record(rawEntry);
    return { id: text(e.id, 'entry ID'), date: date(e.date), meal: text(e.meal, 'meal'), foodId: text(e.foodId, 'food ID'), foodName: text(e.foodName, 'food name'), quantity: number(e.quantity, 'quantity', true), unit: unit(e.unit), state: state(e.state), ...macros(e) };
  });
  if (new Set(foods.map(f => f.id)).size !== foods.length || new Set(entries.map(e => e.id)).size !== entries.length) throw new Error('The backup contains duplicate IDs.');
  const days: Record<string, DaySettings> = {};
  for (const [key, value] of Object.entries(record(v.days))) days[date(key)] = settings(value);
  return { version: 1, foods, entries, days, defaults: settings(v.defaults) };
}
