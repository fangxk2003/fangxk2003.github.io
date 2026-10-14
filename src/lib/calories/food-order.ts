import { addDays, type Entry, type Food } from './model.ts';

/** Each library log counts once, across today and the previous 13 calendar days. */
export function sortFoodsByRecentUse(foods: readonly Food[], entries: readonly Entry[], currentDay: string): Food[] {
  const firstDay = addDays(currentDay, -13);
  const counts = new Map(foods.map(food => [food.id, 0]));
  for (const entry of entries) {
    if (entry.date < firstDay || entry.date > currentDay || entry.source === 'manual' || entry.foodId === `manual:${entry.id}`) continue;
    const count = counts.get(entry.foodId);
    if (count !== undefined) counts.set(entry.foodId, count + 1);
  }
  return [...foods].sort((a, b) =>
    counts.get(b.id)! - counts.get(a.id)! || a.name.localeCompare(b.name) || a.id.localeCompare(b.id),
  );
}
