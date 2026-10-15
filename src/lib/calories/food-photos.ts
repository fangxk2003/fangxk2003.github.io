export const PHOTO_NUTRIENTS = ['protein', 'carbs', 'fat', 'fiber'] as const;
export type PhotoNutrient = typeof PHOTO_NUTRIENTS[number];
export const PHOTO_NUTRIENT_LABELS: Record<PhotoNutrient, string> = { protein: 'Protein', carbs: 'Carbohydrate', fat: 'Fat', fiber: 'Fibre' };
export const MAX_PHOTO_BYTES = 2 * 1024 * 1024;
export const MAX_PHOTO_REQUEST_BYTES = 6 * 1024 * 1024;

export interface FoodPhotoImage { mimeType: 'image/jpeg' | 'image/png' | 'image/webp'; data: string }
export interface FoodPhotoRequest { product: FoodPhotoImage; nutrition: FoodPhotoImage }
export interface FoodPhotoDraft {
  name: string | null;
  baseQuantity: number | null;
  unit: 'g' | 'ml' | 'piece' | null;
  state: 'Raw' | 'Cooked' | 'As served' | null;
  protein: number | null;
  carbs: number | null;
  fat: number | null;
  fiber: number | null;
  estimatedFields: PhotoNutrient[];
  notes: string[];
}

function record(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('The photo reader returned an invalid result. Try clearer photos.');
  return value as Record<string, unknown>;
}
function nullableNumber(value: unknown, positive = false): number | null {
  if (value === null) return null;
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value > 1e8 || (positive && value === 0)) throw new Error('The photo reader returned an invalid quantity. Try clearer photos.');
  return value;
}
function nullableEnum<T extends string>(value: unknown, options: readonly T[]): T | null {
  if (value === null) return null;
  if (typeof value !== 'string' || !options.includes(value as T)) throw new Error('The photo reader returned an invalid label basis. Try clearer photos.');
  return value as T;
}

/** An unresolved field stays null, distinct from a zero actually read on the label. */
export function normalizeFoodPhotoDraft(raw: unknown): FoodPhotoDraft {
  const value = record(raw);
  if (value.name !== null && (typeof value.name !== 'string' || !value.name.trim() || value.name.trim().length > 160)) throw new Error('The photo reader could not identify a valid food name.');
  if (!Array.isArray(value.estimatedFields) || value.estimatedFields.length > 4 || value.estimatedFields.some(field => !PHOTO_NUTRIENTS.includes(field)) || new Set(value.estimatedFields).size !== value.estimatedFields.length) throw new Error('The photo reader returned invalid estimate details.');
  if (!Array.isArray(value.notes) || value.notes.length > 10 || value.notes.some(note => typeof note !== 'string' || !note.trim() || note.length > 600)) throw new Error('The photo reader returned invalid review notes.');
  const result: FoodPhotoDraft = {
    name: typeof value.name === 'string' ? value.name.trim() : null,
    baseQuantity: nullableNumber(value.baseQuantity, true),
    unit: nullableEnum(value.unit, ['g', 'ml', 'piece']),
    state: nullableEnum(value.state, ['Raw', 'Cooked', 'As served']),
    protein: nullableNumber(value.protein), carbs: nullableNumber(value.carbs),
    fat: nullableNumber(value.fat), fiber: nullableNumber(value.fiber),
    estimatedFields: [...value.estimatedFields] as PhotoNutrient[],
    notes: value.notes.map(note => (note as string).trim()),
  };
  if (result.estimatedFields.some(field => result[field] === null)) throw new Error('The photo reader returned an estimate without a value.');
  return result;
}

function photo(raw: unknown): FoodPhotoImage {
  const value = record(raw);
  if (!['image/jpeg', 'image/png', 'image/webp'].includes(String(value.mimeType)) || typeof value.data !== 'string' || !value.data.length || value.data.length > Math.ceil(MAX_PHOTO_BYTES / 3) * 4 || value.data.length % 4 !== 0 || !/^[A-Za-z0-9+/]+={0,2}$/.test(value.data)) throw new Error('Use two JPEG, PNG or WebP photos, each no larger than 2 MB after preparation.');
  let decoded: string;
  try { decoded = atob(value.data); } catch { throw new Error('A photo could not be decoded. Choose it again.'); }
  if (decoded.length > MAX_PHOTO_BYTES) throw new Error('A prepared photo is too large. Choose a smaller image.');
  const jpeg = decoded.charCodeAt(0) === 0xff && decoded.charCodeAt(1) === 0xd8 && decoded.charCodeAt(2) === 0xff;
  const png = decoded.slice(0, 8) === '\x89PNG\r\n\x1a\n';
  const webp = decoded.slice(0, 4) === 'RIFF' && decoded.slice(8, 12) === 'WEBP';
  if (!(value.mimeType === 'image/jpeg' && jpeg || value.mimeType === 'image/png' && png || value.mimeType === 'image/webp' && webp)) throw new Error('A file does not match its photo format. Choose a JPEG, PNG or WebP image.');
  return { mimeType: value.mimeType as FoodPhotoImage['mimeType'], data: value.data };
}

export function normalizeFoodPhotoRequest(raw: unknown): FoodPhotoRequest {
  const value = record(raw);
  return { product: photo(value.product), nutrition: photo(value.nutrition) };
}
