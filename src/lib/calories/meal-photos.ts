import { normalizeFoodPhotoRequest, type FoodPhotoImage } from './food-photos.ts';
import { energyFromMacros } from './model.ts';

export interface MealPhotoRequest { photo: FoodPhotoImage; description?: string }
export interface MealPhotoDraft {
  name: string | null;
  protein: number | null;
  carbs: number | null;
  fat: number | null;
  fiber: number | null;
  notes: string[];
}

const DRAFT_FIELDS = new Set(['name', 'protein', 'carbs', 'fat', 'fiber', 'notes']);
const invalid = () => new Error('The meal photo estimate was invalid. Try a clearer photo.');
function record(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw invalid();
  return value as Record<string, unknown>;
}
function nutrient(value: unknown): number | null {
  if (value === null) return null;
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value > 10000) throw invalid();
  return value;
}

/** Reuse the package-photo byte/type checks; send only the meal image and optional description. */
export function normalizeMealPhotoRequest(raw: unknown): MealPhotoRequest {
  const value = record(raw);
  const photo = normalizeFoodPhotoRequest({ product: value.photo, nutrition: value.photo }).product;
  if (value.description !== undefined && typeof value.description !== 'string') throw invalid();
  const description = typeof value.description === 'string' ? value.description.trim() : '';
  if (description.length > 300) throw new Error('Keep the meal description to 300 characters or fewer.');
  return { photo, ...(description ? { description } : {}) };
}

/** A review draft only: null stays unknown, and fibre contributes no 4/4/9 energy. */
export function normalizeMealPhotoDraft(raw: unknown): MealPhotoDraft {
  const value = record(raw);
  const keys = Object.keys(value);
  if (keys.length !== DRAFT_FIELDS.size || keys.some(key => !DRAFT_FIELDS.has(key))) throw invalid();
  if (value.name !== null && (typeof value.name !== 'string' || !value.name.trim() || value.name.trim().length > 160)) throw invalid();
  if (!Array.isArray(value.notes) || value.notes.length > 3) throw invalid();
  const notes: string[] = [];
  for (const note of value.notes) {
    if (typeof note !== 'string' || !note.trim() || note.length > 200) throw invalid();
    notes.push(note.trim());
  }
  const draft: MealPhotoDraft = {
    name: typeof value.name === 'string' ? value.name.trim() : null,
    protein: nutrient(value.protein), carbs: nutrient(value.carbs),
    fat: nutrient(value.fat), fiber: nutrient(value.fiber), notes,
  };
  const energy = energyFromMacros({ protein: draft.protein ?? 0, carbs: draft.carbs ?? 0, fat: draft.fat ?? 0, fiber: draft.fiber ?? 0 });
  if (!Number.isFinite(energy) || energy > 100000) throw invalid();
  return draft;
}
