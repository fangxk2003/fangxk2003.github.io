import { emptyData, energyFromMacros, macrosForFood, normalizeData, settingsForDay, type BankPortion, type Food, type TrackerData } from './model.ts';

function positiveQuantity(value: number): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0 || value > 1e8) throw new Error('Enter a finite positive ingredient quantity.');
  return value;
}

/** Ingredient quantities describe one portion; batching makes independent snapshots. */
export function prepareFoodBankPortions(
  foods: readonly Food[],
  ingredients: readonly { foodId: string; quantity: number }[],
  count: number,
  name: string,
  makeId: () => string,
): BankPortion[] {
  if (!Number.isInteger(count) || count < 1 || count > 100) throw new Error('Prepare between 1 and 100 portions.');
  if (typeof name !== 'string' || !name.trim() || name.length > 200) throw new Error('Enter a food-bank portion name.');
  if (!Array.isArray(foods) || !Array.isArray(ingredients) || ingredients.length < 1 || ingredients.length > 50) throw new Error('A food-bank portion must contain 1 to 50 ingredients.');
  if (typeof makeId !== 'function') throw new Error('A food-bank ID generator is required.');
  const library = new Map(normalizeData({ ...emptyData(), foods }).foods.map(food => [food.id, food]));
  const snapshots = ingredients.map(ingredient => {
    if (!ingredient || typeof ingredient !== 'object' || Array.isArray(ingredient)) throw new Error('Invalid food-bank ingredient.');
    const food = library.get(ingredient.foodId);
    if (!food) throw new Error('Choose an existing food for each ingredient.');
    const quantity = positiveQuantity(ingredient.quantity);
    const nutrients = macrosForFood(food, quantity);
    return { foodId: food.id, foodName: food.name, quantity, unit: food.unit, state: food.state, ...nutrients, calories: energyFromMacros(nutrients) };
  });
  const generatedIds = new Set<string>();
  const nextId = () => {
    const id = makeId();
    if (typeof id !== 'string' || !id.trim() || id.length > 200 || generatedIds.has(id)) throw new Error('Food-bank IDs must be valid and unique.');
    generatedIds.add(id);
    return id;
  };
  const portions = Array.from({ length: count }, () => ({
    id: nextId(), name: name.trim(), items: snapshots.map(snapshot => ({ ...snapshot, id: nextId() })),
  }));
  // Apply the same numerical bounds and collision checks used for backups.
  return normalizeData({ ...emptyData(), foodBank: portions }).foodBank;
}

/** Move one stored snapshot into the journal; its item IDs prevent duplicate logging. */
export function consumeFoodBankPortion(data: TrackerData, portionId: string, date: string, meal: string): TrackerData {
  const current = normalizeData(data);
  const portion = current.foodBank.find(item => item.id === portionId);
  if (!portion) throw new Error('This food-bank portion is no longer available.');
  if (typeof meal !== 'string' || !meal.trim() || meal.length > 200) throw new Error('Enter a meal name.');
  // Resolve before appending: a previously untouched date starts with Garmin settings.
  const daySettings = { ...settingsForDay(current, date) };
  return normalizeData({
    ...current,
    foodBank: current.foodBank.filter(item => item.id !== portionId),
    entries: [...current.entries, ...portion.items.map(item => ({ ...item, date, meal: meal.trim() }))],
    days: { ...current.days, [date]: daySettings },
  });
}
