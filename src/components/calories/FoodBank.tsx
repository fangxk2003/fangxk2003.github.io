import { useId, useRef, useState, type FormEvent } from 'react';
import { macrosForFood, type BankPortion, type Food, type Macros } from '../../lib/calories/model';
import { prepareFoodBankPortions } from '../../lib/calories/food-bank';
import { DraftRestoredNotice, usePersistentFormDraft } from './FormDrafts';
import './food-bank.css';

const number = (value: number, decimals = 1) => new Intl.NumberFormat('en-GB', { maximumFractionDigits: decimals }).format(value);
const unitLabel = (unit: string, quantity: number) => unit === 'piece' ? quantity === 1 ? 'piece' : 'pieces' : unit;
const zeroMacros = (): Macros => ({ calories: 0, protein: 0, carbs: 0, fat: 0, fiber: 0 });
function sumNutrition(items: readonly Macros[]): Macros {
  return items.reduce((total, item) => ({ calories: total.calories + item.calories, protein: total.protein + item.protein, carbs: total.carbs + item.carbs, fat: total.fat + item.fat, fiber: total.fiber + item.fiber }), zeroMacros());
}
function BankIcon({ name, size = 19 }: { name: 'plus' | 'close' | 'trash' | 'search' | 'box' | 'check'; size?: number }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{name === 'plus' ? <path d="M12 5v14M5 12h14"/> : name === 'close' ? <path d="m6 6 12 12M6 18 18 6"/> : name === 'trash' ? <><path d="M3 6h18M9 6V3h6v3M6 6l1 15h10l1-15M10 10v7M14 10v7"/></> : name === 'search' ? <><circle cx="10" cy="10" r="6"/><path d="m15 15 5 5"/></> : name === 'check' ? <path d="m5 12 4 4L19 6"/> : <><path d="m3 7 9-4 9 4-9 4-9-4ZM3 7v10l9 4 9-4V7M12 11v10"/></>}</svg>;
}
function BankHeader({ title, onClose }: { title: string; onClose: () => void }) {
  return <div className="cal-modal-heading"><div><div className="cal-eyebrow">FOOD BANK</div><h2 id="cal-dialog-title">{title}</h2></div><button className="cal-icon-button cal-bank-touch" type="button" onClick={onClose} aria-label="Close dialog"><BankIcon name="close" size={20}/></button></div>;
}
function BankNutrition({ totals, label = 'Per portion' }: { totals: Macros; label?: string }) {
  return <div className="cal-bank-nutrition"><div className="cal-bank-energy"><span>{label}</span><strong>{number(totals.calories, 0)} <small>kcal</small></strong></div><div className="cal-bank-macros">{([['Protein', totals.protein], ['Carbs', totals.carbs], ['Fat', totals.fat], ['Fibre', totals.fiber]] as const).map(([label, value]) => <span key={label}>{label}<strong>{number(value)} g</strong></span>)}</div></div>;
}
function BankIngredients({ portion }: { portion: BankPortion }) {
  return <ul className="cal-bank-ingredients" aria-label="Ingredients in this portion">{portion.items.map((item, index) => <li key={`${item.id}-${index}`}><span>{item.foodName}<small>{item.state}</small></span><strong>{number(item.quantity, 3)} {unitLabel(item.unit, item.quantity)}</strong></li>)}</ul>;
}

export function FoodBankView({ portions, onPrepare, onEat, onRemove }: { portions: readonly BankPortion[]; onPrepare: () => void; onEat: (portion: BankPortion) => void; onRemove: (id: string) => void }) {
  const [search, setSearch] = useState('');
  const [removing, setRemoving] = useState<string | null>(null);
  const query = search.trim().toLowerCase();
  const visible = portions.filter(portion => portion.name.toLowerCase().includes(query) || portion.items.some(item => item.foodName.toLowerCase().includes(query)));
  return <div className="cal-bank-view">
    <div className="cal-bank-toolbar"><div className="cal-bank-search"><BankIcon name="search" size={17}/><input type="search" aria-label="Search prepared portions" value={search} onChange={event => setSearch(event.target.value)} placeholder="Search prepared food"/></div><button className="cal-button cal-button-primary" type="button" onClick={onPrepare}><BankIcon name="plus"/>Prepare portions</button></div>
    <p className="cal-bank-count" role="status">{query ? `${visible.length} matching ${visible.length === 1 ? 'portion' : 'portions'}` : `${portions.length} prepared ${portions.length === 1 ? 'portion' : 'portions'}`}</p>
    {visible.length > 0 ? <div className="cal-bank-grid">{visible.map(portion => <article className="cal-food-card cal-bank-card" key={portion.id}>
      <div className="cal-bank-card-heading"><span>1 prepared portion</span><BankIcon name="box" size={18}/></div><h2>{portion.name}</h2>
      <BankIngredients portion={portion}/><BankNutrition totals={sumNutrition(portion.items)}/>
      {removing === portion.id ? <div className="cal-bank-remove-confirm"><p>Remove this portion? It won’t be added to your journal.</p><div><button className="cal-text-button cal-bank-touch" type="button" onClick={() => setRemoving(null)}>Cancel</button><button className="cal-text-button cal-danger cal-bank-touch" type="button" onClick={() => { onRemove(portion.id); setRemoving(null); }}>Remove portion</button></div></div> : <div className="cal-bank-card-actions"><button className="cal-button cal-button-small cal-bank-eat" type="button" onClick={() => onEat(portion)}>Eat<BankIcon name="check" size={17}/></button><button className="cal-icon-button cal-bank-touch" type="button" aria-label={`Remove one portion of ${portion.name}`} onClick={() => setRemoving(portion.id)}><BankIcon name="trash" size={17}/></button></div>}
    </article>)}</div> : <div className="cal-empty-journal cal-bank-empty"><span className="cal-empty-art"><BankIcon name="box" size={36}/></span><h3>{query ? 'No portions found.' : 'A little preparation, ready for later.'}</h3><p>{query ? 'Try a food or portion name.' : 'Prepare individual portions from your food library. Eat one to move its ingredients into your journal.'}</p><button className="cal-text-button cal-bank-touch" type="button" onClick={onPrepare}>Prepare portions<BankIcon name="plus" size={17}/></button></div>}
  </div>;
}

interface IngredientDraft { key: number; foodId: string; query: string; quantity: string }
export function FoodBankEditor({ foods, onSave, onClose }: { foods: readonly Food[]; onSave: (portions: BankPortion[]) => void | Promise<void>; onClose: () => void }) {
  const id = useId();
  const form = usePersistentFormDraft('bank:prepare', { draftBatchId: crypto.randomUUID(), name: '', count: '1', ingredients: [{ key: 0, foodId: foods[0]?.id ?? '', query: '', quantity: '' }] as IngredientDraft[] }, value => ({ kind: 'bank-prepare', label: value.name.trim() || 'Prepared portions' }));
  const [name, setName] = form.field('name');
  const [count, setCount] = form.field('count');
  const [ingredients, setIngredients] = form.field('ingredients');
  const nextKey = useRef(Math.max(0, ...ingredients.map(ingredient => ingredient.key)) + 1);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');
  const matches = (food: Food, query: string) => food.name.toLowerCase().includes(query.trim().toLowerCase());
  const selectedFood = (ingredient: IngredientDraft) => foods.find(food => food.id === ingredient.foodId && matches(food, ingredient.query));
  const autoName = ingredients.map(selectedFood).filter((food): food is Food => Boolean(food)).map(food => food.name).join(' + ').slice(0, 160);
  const validPreviews = ingredients.flatMap(ingredient => {
    const food = selectedFood(ingredient);
    const quantity = Number(ingredient.quantity);
    if (!food || !ingredient.quantity.trim() || !Number.isFinite(quantity) || quantity <= 0 || quantity > 1e8) return [];
    const nutrients = macrosForFood(food, quantity);
    return Object.values(nutrients).every(value => Number.isFinite(value) && value >= 0 && value <= 1e8) ? [nutrients] : [];
  });
  const preview = sumNutrition(validPreviews);
  const previewComplete = validPreviews.length === ingredients.length;
  function updateIngredient(key: number, update: (ingredient: IngredientDraft) => IngredientDraft) {
    setIngredients(current => current.map(ingredient => ingredient.key === key ? update(ingredient) : ingredient));
    setError('');
  }
  function filterIngredient(key: number, query: string) {
    const first = foods.find(food => matches(food, query));
    updateIngredient(key, current => ({ ...current, query, foodId: first?.id ?? '', quantity: first?.id === current.foodId ? current.quantity : '' }));
  }
  function addIngredient() {
    if (ingredients.length >= 50) return;
    const key = nextKey.current++;
    setIngredients(current => [...current, { key, foodId: foods[0]?.id ?? '', query: '', quantity: '' }]);
    setError('');
  }
  function closeEditor() { form.clearDraft(); onClose(); }
  async function submit(event: FormEvent) {
    event.preventDefault();
    if (submitting) return;
    if (ingredients.some(ingredient => !selectedFood(ingredient) || !ingredient.quantity.trim())) { setError('Choose a food and enter an amount for every ingredient.'); return; }
    if (!Number.isInteger(Number(count)) || Number(count) < 1 || Number(count) > 100) { setError('Prepare between 1 and 100 whole portions.'); return; }
    try {
      let generatedId = 0;
      const prepared = prepareFoodBankPortions(foods, ingredients.map(ingredient => ({ foodId: ingredient.foodId, quantity: Number(ingredient.quantity) })), Number(count), name.trim() || autoName, () => `${form.draft.draftBatchId}:${generatedId++}`);
      setSubmitting(true);
      await onSave(prepared);
      form.clearDraft();
    } catch (caught) { setError(caught instanceof Error ? caught.message : 'These portions couldn’t be prepared. Check the ingredients and amounts.'); }
    finally { setSubmitting(false); }
  }
  return <>
    <BankHeader title="Prepare food for later" onClose={closeEditor}/>
    <form className="cal-form cal-bank-editor" onSubmit={submit}>
      <DraftRestoredNotice {...form} disabled={submitting} discardDraft={() => { form.discardDraft(); setError(''); }}/>
      <fieldset className="cal-form cal-food-editor-fields" disabled={submitting}>
      <div className="cal-form-grid cal-bank-name-grid"><label>Portion name <small className="cal-field-hint">Optional</small><input value={name} onChange={event => setName(event.target.value)} placeholder={autoName || 'e.g. Chicken and rice'} maxLength={160}/></label><label>Number of portions<input type="number" inputMode="numeric" value={count} onChange={event => { setCount(event.target.value); setError(''); }} min={1} max={100} step={1} required/></label></div>
      <div className="cal-bank-section-heading"><strong id={`${id}-amounts`}>Amounts per portion</strong><span>Each portion gets these amounts.</span></div>
      {foods.length === 0 && <p className="cal-bank-empty-library">Add a food to your Food library before preparing portions.</p>}
      <div className="cal-bank-ingredient-rows">{ingredients.map((ingredient, index) => {
        const filtered = foods.filter(food => matches(food, ingredient.query));
        const food = selectedFood(ingredient);
        const fieldId = `${id}-${ingredient.key}`;
        return <fieldset className="cal-bank-ingredient-row" key={ingredient.key}><legend>Ingredient {index + 1}</legend><div className="cal-bank-ingredient-fields">
          <div className="cal-food-picker"><label htmlFor={`${fieldId}-search`}>Food</label><div className="cal-food-picker-controls"><input id={`${fieldId}-search`} type="text" value={ingredient.query} onChange={event => filterIngredient(ingredient.key, event.target.value)} placeholder="Filter food names" autoComplete="off" autoCapitalize="none" spellCheck={false} aria-describedby={`${fieldId}-matches`}/><select aria-label={`Food for ingredient ${index + 1}`} value={food?.id ?? ''} onChange={event => updateIngredient(ingredient.key, current => ({ ...current, foodId: event.target.value, quantity: event.target.value === current.foodId ? current.quantity : '' }))} disabled={filtered.length === 0} required>{filtered.length === 0 && <option value="" disabled>No matching foods</option>}{filtered.map(food => <option key={food.id} value={food.id}>{food.name} · {food.state}</option>)}</select></div><small className="cal-bank-match-count" id={`${fieldId}-matches`} role="status">{filtered.length === 0 ? 'No matching foods' : ingredient.query.trim() ? `${filtered.length} matching ${filtered.length === 1 ? 'food' : 'foods'}` : ''}</small></div>
          <label>Amount per portion<span className="cal-unit-input"><input type="number" inputMode="decimal" value={ingredient.quantity} onChange={event => updateIngredient(ingredient.key, current => ({ ...current, quantity: event.target.value }))} placeholder="Amount" min={0.001} max={100000000} step="any" required aria-describedby={`${id}-amounts`}/><span>{food?.unit ?? '—'}</span></span></label>
          {ingredients.length > 1 && <button className="cal-icon-button cal-bank-touch cal-bank-remove-ingredient" type="button" aria-label={`Remove ingredient ${index + 1}`} onClick={() => { setIngredients(current => current.filter(item => item.key !== ingredient.key)); setError(''); }}><BankIcon name="trash" size={17}/></button>}
        </div></fieldset>;
      })}</div>
      <button className="cal-text-button cal-bank-add-ingredient cal-bank-touch" type="button" onClick={addIngredient} disabled={foods.length === 0 || ingredients.length >= 50}><BankIcon name="plus" size={17}/>{ingredients.length >= 50 ? '50 ingredients maximum' : 'Add ingredient'}</button>
      <div className="cal-bank-editor-preview" aria-live="polite"><BankNutrition totals={preview} label="Nutrition per portion"/>{!previewComplete && <p>Add every amount to complete the preview.</p>}</div>
      </fieldset>
      {error && <p className="cal-form-error" role="alert">{error}</p>}
      <div className="cal-modal-actions"><button className="cal-button" type="button" onClick={closeEditor} disabled={submitting}>Cancel</button><button className="cal-button cal-button-primary" type="submit" disabled={foods.length === 0 || submitting}>{submitting ? 'Saving portions…' : <>Prepare {Number.isInteger(Number(count)) && Number(count) > 0 && Number(count) <= 100 ? `${number(Number(count), 0)} ${Number(count) === 1 ? 'portion' : 'portions'}` : 'portions'}</>}<BankIcon name="check" size={17}/></button></div>
    </form>
  </>;
}

export function FoodBankConsume({ portion, date, onSave, onClose }: { portion: BankPortion; date: string; onSave: (date: string, meal: string) => void | Promise<void>; onClose: () => void }) {
  const [entryDate, setEntryDate] = useState(date);
  const [meal, setMeal] = useState('Lunch');
  const [customMeal, setCustomMeal] = useState('');
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);
  async function submit(event: FormEvent) {
    event.preventDefault();
    if (submitting) return;
    const selectedMeal = meal === 'Custom' ? customMeal.trim() : meal;
    if (!/^\d{4}-\d{2}-\d{2}$/.test(entryDate) || Number.isNaN(Date.parse(`${entryDate}T12:00:00Z`)) || new Date(`${entryDate}T12:00:00Z`).toISOString().slice(0, 10) !== entryDate || !selectedMeal) { setError('Choose a date and meal for this portion.'); return; }
    setSubmitting(true);
    try { await onSave(entryDate, selectedMeal); } catch (caught) { setError(caught instanceof Error ? caught.message : 'This portion couldn’t be logged. Try again.'); }
    finally { setSubmitting(false); }
  }
  return <>
    <BankHeader title="Eat a prepared portion" onClose={onClose}/>
    <div className="cal-bank-consume-summary"><h3>{portion.name}</h3><BankIngredients portion={portion}/><BankNutrition totals={sumNutrition(portion.items)}/></div>
    <form className="cal-form cal-bank-consume-form" onSubmit={submit}>
      <fieldset className="cal-form cal-food-editor-fields" disabled={submitting}>
      <div className="cal-form-grid"><label>Date<input type="date" value={entryDate} onChange={event => { setEntryDate(event.target.value); setError(''); }} required/></label><label>Meal<select value={meal} onChange={event => { setMeal(event.target.value); setError(''); }}>{['Breakfast', 'Lunch', 'Dinner', 'Snack', 'Custom'].map(meal => <option key={meal}>{meal}</option>)}</select></label></div>
      {meal === 'Custom' && <label>Meal name or number<input value={customMeal} onChange={event => { setCustomMeal(event.target.value); setError(''); }} placeholder="e.g. Meal 1" maxLength={80} required/></label>}
      </fieldset>
      {error && <p className="cal-form-error" role="alert">{error}</p>}
      <div className="cal-modal-actions"><button className="cal-button" type="button" onClick={onClose} disabled={submitting}>Cancel</button><button className="cal-button cal-button-primary" type="submit" disabled={submitting}>{submitting ? 'Logging…' : 'Eat and log'}<BankIcon name="check" size={17}/></button></div>
    </form>
  </>;
}
