import { DAILY_PROTEIN_GRAMS } from './nutrition.ts';

export type Unit = 'g' | 'ml' | 'piece';
export type FoodState = 'Raw' | 'Cooked' | 'As served';
export interface Macros { carbs: number; protein: number; fat: number; fiber: number; calories: number }
export interface Food extends Macros { id: string; name: string; baseQuantity: number; unit: Unit; state: FoodState }
export interface Entry extends Macros { id: string; date: string; meal: string; foodId: string; foodName: string; quantity: number; unit: Unit; state: FoodState; source?: 'manual' }
export type BankItem = Omit<Entry, 'date' | 'meal' | 'source'>;
export interface BankPortion { id: string; name: string; items: BankItem[] }
export interface ManualMealInput { id: string; date: string; meal: string; name: string; carbs: number; protein: number; fat: number; fiber: number }
export interface DaySettings { burned: number; deficit: number; protein: number; burnedSource?: 'manual' | 'garmin' }
export interface TrackerData { version: 2; foods: Food[]; entries: Entry[]; foodBank: BankPortion[]; days: Record<string, DaySettings>; defaults: DaySettings }

/** This journal uses 4/4/9; fibre is tracked separately and excluded by preference.
 * Keep full precision for portion calculations; round only when displaying. */
export function energyFromMacros(macros: Pick<Macros, 'protein' | 'carbs' | 'fat' | 'fiber'>): number {
  return 4 * macros.protein + 4 * macros.carbs + 9 * macros.fat;
}

export function emptyData(): TrackerData {
  return { version: 2, foods: [], entries: [], foodBank: [], days: {}, defaults: { burned: 0, deficit: 500, protein: DAILY_PROTEIN_GRAMS } };
}

/** An untouched day starts with Garmin; recorded days and explicit day overrides stay intact. */
export function settingsForDay(data: TrackerData, date: string): DaySettings {
  if (data.days[date]) return data.days[date];
  if (data.entries.some(entry => entry.date === date)) return data.defaults;
  return { ...data.defaults, burned: 0, burnedSource: 'garmin' };
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
  if (v.burnedSource !== undefined && v.burnedSource !== 'manual' && v.burnedSource !== 'garmin') throw new Error('Invalid calories burned source.');
  number(v.protein, 'protein target'); // Validate legacy backups before applying the fixed target.
  return { burned: number(v.burned, 'calories burned'), deficit: number(v.deficit, 'deficit target'), protein: DAILY_PROTEIN_GRAMS, ...(v.burnedSource !== undefined ? { burnedSource: v.burnedSource } : {}) };
}

/** Nutrition is the total for the eaten meal; no food-library record or portion scaling. */
export function manualMealEntry(input: ManualMealInput): Entry {
  const id = text(input.id, 'entry ID');
  const nutrients = {
    protein: number(input.protein, 'protein'), carbs: number(input.carbs, 'carbs'),
    fat: number(input.fat, 'fat'), fiber: number(input.fiber, 'fiber'),
  };
  return {
    id, date: date(input.date), meal: text(input.meal, 'meal').trim(),
    foodId: text(`manual:${id}`, 'food ID'), foodName: text(input.name, 'meal description').trim(),
    quantity: 1, unit: 'piece', state: 'As served', source: 'manual',
    ...nutrients, calories: number(energyFromMacros(nutrients), 'calculated calories'),
  };
}

/** Strict validation for imported and remotely loaded data. No silent data loss. */
export function normalizeData(raw: unknown): TrackerData {
  const v = record(raw);
  if ((v.version !== 1 && v.version !== 2) || !Array.isArray(v.foods) || !Array.isArray(v.entries) || v.foods.length > 10000 || v.entries.length > 100000 || (v.version === 2 && (!Array.isArray(v.foodBank) || v.foodBank.length > 10000))) throw new Error('This is not a supported Calories backup.');
  const foods: Food[] = Array.from(v.foods).map(rawFood => {
    const f = record(rawFood);
    const nutrients = macros(f);
    return { id: text(f.id, 'food ID'), name: text(f.name, 'food name'), baseQuantity: number(f.baseQuantity, 'base quantity', true), unit: unit(f.unit), state: state(f.state), ...nutrients, calories: number(energyFromMacros(nutrients), 'calculated calories') };
  });
  const entries: Entry[] = Array.from(v.entries).map(rawEntry => {
    const e = record(rawEntry);
    if (e.source !== undefined && e.source !== 'manual') throw new Error('Invalid entry source.');
    const id = text(e.id, 'entry ID');
    const foodId = text(e.foodId, 'food ID');
    // Older open tabs omit new metadata on save; the canonical identity preserves manual editing.
    const manual = e.source === 'manual' || foodId === `manual:${id}`;
    return { id, date: date(e.date), meal: text(e.meal, 'meal'), foodId, foodName: text(e.foodName, 'food name'), quantity: number(e.quantity, 'quantity', true), unit: unit(e.unit), state: state(e.state), ...macros(e), ...(manual ? { source: 'manual' as const } : {}) };
  });
  if (new Set(foods.map(f => f.id)).size !== foods.length || new Set(entries.map(e => e.id)).size !== entries.length) throw new Error('The backup contains duplicate IDs.');
  const itemIds = new Set(entries.map(entry => entry.id));
  const portionIds = new Set<string>();
  const foodBank: BankPortion[] = v.version === 1 ? [] : Array.from(v.foodBank as unknown[]).map(rawPortion => {
    const portion = record(rawPortion);
    const id = text(portion.id, 'food-bank portion ID');
    if (portionIds.has(id)) throw new Error('The backup contains duplicate IDs.');
    portionIds.add(id);
    if (!Array.isArray(portion.items) || portion.items.length < 1 || portion.items.length > 50) throw new Error('A food-bank portion must contain 1 to 50 ingredients.');
    const items: BankItem[] = Array.from(portion.items).map(rawItem => {
      const item = record(rawItem);
      if (item.date !== undefined || item.meal !== undefined || item.source !== undefined) throw new Error('Invalid food-bank ingredient metadata.');
      const itemId = text(item.id, 'food-bank ingredient ID');
      if (itemIds.has(itemId)) throw new Error('The backup contains duplicate IDs.');
      itemIds.add(itemId);
      return { id: itemId, foodId: text(item.foodId, 'food ID'), foodName: text(item.foodName, 'food name'), quantity: number(item.quantity, 'quantity', true), unit: unit(item.unit), state: state(item.state), ...macros(item) };
    });
    return { id, name: text(portion.name, 'food-bank portion name'), items };
  });
  const days: Record<string, DaySettings> = {};
  for (const [key, value] of Object.entries(record(v.days))) days[date(key)] = settings(value);
  return { version: 2, foods, entries, foodBank, days, defaults: settings(v.defaults) };
}
