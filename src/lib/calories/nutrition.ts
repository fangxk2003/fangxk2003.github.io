/** The owner's chosen daily macro allocation, based on the food calorie budget. */
export const DAILY_PROTEIN_GRAMS = 175;

function nonNegative(value: number, label: string): number {
  if (!Number.isFinite(value) || value < 0) throw new Error(`Invalid ${label}.`);
  return value;
}

export function macroTargets(budget: number) {
  nonNegative(budget, 'calorie budget');
  const proteinEnergy = DAILY_PROTEIN_GRAMS * 4;
  const fatEnergy = budget * 0.25;
  const carbEnergy = Math.max(0, budget * 0.75 - proteinEnergy);
  return {
    protein: DAILY_PROTEIN_GRAMS,
    fat: fatEnergy / 9,
    carbs: carbEnergy / 4,
    fiber: budget * 0.014,
    proteinEnergy, fatEnergy, carbEnergy,
    budgetTooLow: budget * 0.75 < proteinEnergy,
  };
}

/** Keep the numeric percentage uncapped; cap only the bar's visual width. */
export function nutrientProgress(consumed: number, target: number) {
  nonNegative(consumed, 'consumed nutrient');
  nonNegative(target, 'nutrient target');
  if (target === 0) return { percent: null, barPercent: consumed > 0 ? 100 : 0, reached: false };
  const rawPercent = consumed / target * 100;
  return {
    percent: Number.isFinite(rawPercent) ? rawPercent : null,
    barPercent: Math.min(100, rawPercent),
    reached: consumed >= target,
  };
}
