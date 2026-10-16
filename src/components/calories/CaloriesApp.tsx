import { useEffect, useMemo, useRef, useState, type FormEvent, type ReactNode } from 'react';
import { addDays, dayTotals, emptyData, energyFromMacros, macrosForFood, manualMealEntry, normalizeData, settingsForDay, todayISO, type DaySettings, type Entry, type Food, type TrackerData, type BankPortion } from '../../lib/calories/model';
import { createClient } from '../../lib/calories/client';
import { effectiveBurned, emptyGarminSnapshot, type EffectiveBurned, type GarminDay, type GarminSnapshot, type GarminStatus } from '../../lib/calories/garmin';
import { DAILY_PROTEIN_GRAMS, macroTargets, nutrientProgress } from '../../lib/calories/nutrition';
import { adjustedBudget, calorieLedger, carryBefore, MAX_DAILY_DEFICIT } from '../../lib/calories/budget';
import { sortFoodsByRecentUse } from '../../lib/calories/food-order';
import { normalizeFoodPhotoDraft, type FoodPhotoDraft, type FoodPhotoRequest } from '../../lib/calories/food-photos';
import FoodPhotoImport from './FoodPhotoImport';
import { FoodBankView, FoodBankEditor, FoodBankConsume } from './FoodBank';
import { consumeFoodBankPortion } from '../../lib/calories/food-bank';
import './calories.css';

type View = 'journal' | 'foods' | 'bank' | 'history';
type Phase = 'loading' | 'setup' | 'login' | 'ready';
type SaveState = 'saved' | 'pending' | 'saving' | 'error';
type ModalState = { type: 'entry'; entry?: Entry; foodId?: string } | { type: 'food'; food?: Food; returnToLog?: boolean } | { type: 'targets' } | { type: 'transfer' } | { type: 'garmin' } | { type: 'bank-prepare' } | { type: 'bank-consume'; portion: BankPortion } | null;
type IconName = 'leaf' | 'plus' | 'left' | 'right' | 'close' | 'edit' | 'trash' | 'settings' | 'download' | 'lock' | 'logout' | 'check' | 'search' | 'calendar' | 'arrow' | 'book' | 'watch' | 'refresh';

function Icon({ name, size = 20, ...props }: { name: IconName; size?: number; className?: string }) {
  const paths: Record<IconName, ReactNode> = {
    leaf: <><path d="M20 4c-1 8-2 15-10 15a6 6 0 0 1-6-6c0-8 8-8 16-9Z"/><path d="m4 20 11-11"/></>,
    plus: <><path d="M12 5v14M5 12h14"/></>,
    left: <path d="m14 6-6 6 6 6"/>,
    right: <path d="m10 6 6 6-6 6"/>,
    close: <path d="m6 6 12 12M6 18 18 6"/>,
    edit: <><path d="m15 5 4 4M4 20l4-1L20 7a2.8 2.8 0 0 0-4-4L4 15v5Z"/></>,
    trash: <><path d="M3 6h18M9 6V3h6v3M6 6l1 15h10l1-15M10 10v7M14 10v7"/></>,
    settings: <><path d="M4 7h16M4 17h16"/><circle cx="9" cy="7" r="3"/><circle cx="15" cy="17" r="3"/></>,
    download: <><path d="M12 3v12m-5-5 5 5 5-5M5 16v5h14v-5"/></>,
    lock: <><rect x="5" y="10" width="14" height="11" rx="3"/><path d="M8 10V7a4 4 0 0 1 8 0v3M12 14v3"/></>,
    logout: <><path d="M10 4H4v16h6M9 12h12m-4-4 4 4-4 4"/></>,
    check: <path d="m5 12 4 4L19 6"/>,
    search: <><circle cx="10" cy="10" r="6"/><path d="m15 15 5 5"/></>,
    calendar: <><rect x="4" y="5" width="16" height="16" rx="3"/><path d="M8 3v4M16 3v4M4 10h16M8 14h2M14 14h2"/></>,
    arrow: <path d="M4 12h16m-6-6 6 6-6 6"/>,
    book: <><path d="M4 4h6a3 3 0 0 1 3 3v14a4 4 0 0 0-4-2H4V4ZM20 4h-4a3 3 0 0 0-3 3v14a4 4 0 0 1 4-2h3V4Z"/></>,
    watch: <><rect x="6" y="6" width="12" height="12" rx="4"/><path d="m9 6 1-4h4l1 4M9 18l1 4h4l1-4M12 9v3l2 1"/></>,
    refresh: <><path d="M20 7v5h-5M4 17v-5h5"/><path d="M6 7a7 7 0 0 1 11-1l3 6M4 12l3 6a7 7 0 0 0 11-1"/></>,
  };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...props}>{paths[name]}</svg>;
}

const number = (value: number, decimals = 0) => new Intl.NumberFormat('en-GB', { maximumFractionDigits: decimals }).format(value);
const signedNumber = (value: number, decimals = 1) => value > 0 ? `+${number(value, decimals)}` : value < 0 ? `−${number(Math.abs(value), decimals)}` : '0';
const readableDate = (date: string, options: Intl.DateTimeFormatOptions) => new Intl.DateTimeFormat('en-GB', options).format(new Date(`${date}T12:00:00`));
const errorMessage = (error: unknown) => error instanceof Error ? error.message : 'Something went wrong. Please try again.';
const newId = () => crypto.randomUUID();
const unitLabel = (unit: string, quantity: number) => unit === 'piece' ? (quantity === 1 ? 'piece' : 'pieces') : unit;
const previewNumber = (value: string) => Number.isFinite(Number(value)) && Number(value) >= 0 ? Number(value) : 0;
const syncTime = (timestamp: string) => new Intl.DateTimeFormat('en-GB', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' }).format(new Date(timestamp));
const burnedLabel = (value: EffectiveBurned) => value.source === 'garmin' ? value.unavailable ? 'Garmin unavailable' : value.restingEstimate ? 'Garmin resting estimate' : value.predicted ? 'Garmin prediction' : value.predictionUnavailable ? 'Garmin partial total' : 'Garmin' : 'Manual override';
function burnedFormula(value: EffectiveBurned) {
  if (value.unavailable) return 'No imported total is available for this date, and no completed Garmin resting reading is available yet';
  if (value.restingEstimate && value.restingDay) return `${number(value.restingDay.resting)} resting kcal from ${readableDate(value.restingDay.date, { day: 'numeric', month: 'short', year: 'numeric' })}, used as an estimate for this day`;
  if (value.predicted && value.day && value.restingDay) return `${number(value.day.active)} active today + ${number(value.restingDay.resting)} resting yesterday (${readableDate(value.restingDay.date, { day: 'numeric', month: 'short' })})`;
  if (value.predictionUnavailable) return 'Yesterday’s resting calories are unavailable. Using Garmin’s partial total so far';
  return value.day ? `${number(value.day.active)} active + ${number(value.day.resting)} resting kcal` : '';
}
function burnedSyncTime(value: EffectiveBurned) {
  if (value.restingEstimate && value.restingDay) return `Resting imported ${syncTime(value.restingDay.synced_at)}`;
  if (!value.day) return '';
  return value.predicted && value.restingDay && value.day.synced_at !== value.restingDay.synced_at ? `Active updated ${syncTime(value.day.synced_at)} · Resting imported ${syncTime(value.restingDay.synced_at)}` : `Updated ${syncTime(value.day.synced_at)}`;
}
const staleImport = (timestamp: string | null) => Boolean(timestamp && Date.now() - Date.parse(timestamp) > 2 * 60 * 60 * 1000);
const garminStatusLabel = (status: GarminStatus) => ({ not_connected: 'Not connected', connected: status.last_sync_at ? staleImport(status.last_sync_at) ? 'Needs a new import' : 'Connected' : 'Waiting for first import', needs_login: 'Sign-in needed', error: 'Import needs attention' })[status.state];
function garminErrorMeaning(code: string | null) {
  if (!code) return '';
  const meanings: Record<string, string> = { missing_data: 'Garmin hasn’t reported complete calorie data yet.', upstream_unavailable: 'Garmin couldn’t be reached. A later import will try again.', needs_login: 'Sign in to Garmin again to resume automatic imports.', rate_limited: 'Garmin is temporarily limiting requests. A later import will try again.' };
  return meanings[code] ?? 'The last import could not finish. Check the connection setup if this continues.';
}

export default function CaloriesApp({ clientOverride }: { clientOverride?: ReturnType<typeof createClient> } = {}) {
  const [client] = useState(() => clientOverride ?? createClient());
  const [phase, setPhase] = useState<Phase>('loading');
  const [data, setData] = useState<TrackerData>(emptyData);
  const dataRef = useRef(data);
  const savedData = useRef(data);
  const [view, setView] = useState<View>('journal');
  const [date, setDate] = useState(todayISO);
  const [currentDay, setCurrentDay] = useState(todayISO);
  const [modal, setModal] = useState<ModalState>(null);
  const modalRef = useRef(modal);
  modalRef.current = modal;
  const [email, setEmail] = useState(client.getEmail());
  const [password, setPassword] = useState('');
  const [remember, setRemember] = useState(false);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState('');
  const [saveState, setSaveState] = useState<SaveState>('saved');
  const [saveError, setSaveError] = useState('');
  const [garmin, setGarmin] = useState<GarminSnapshot>(emptyGarminSnapshot);
  const [garminLoading, setGarminLoading] = useState(false);
  const [garminLoadError, setGarminLoadError] = useState('');
  const [garminDisconnecting, setGarminDisconnecting] = useState(false);
  const disconnectingGarmin = useRef(false);
  const garminFetching = useRef(false);
  const garminRequest = useRef(0);
  const garminSessionScope = useRef(0);
  const deferredGarmin = useRef<GarminSnapshot | null>(null);
  const phaseRef = useRef(phase);
  phaseRef.current = phase;
  const [foodSearch, setFoodSearch] = useState('');
  const [deleteFoodId, setDeleteFoodId] = useState<string | null>(null);
  const [deleteEntryId, setDeleteEntryId] = useState<string | null>(null);
  const queue = useRef<TrackerData | null>(null);
  const saving = useRef(false);
  const blockedSave = useRef(false);
  const refreshing = useRef(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mounted = useRef(true);

  useEffect(() => {
    let dayTimer: ReturnType<typeof setTimeout>;
    const updateCurrentDay = () => {
      const now = new Date();
      setCurrentDay(todayISO());
      clearTimeout(dayTimer);
      const midnight = new Date(now.getFullYear(), now.getMonth(), now.getDate() + 1);
      dayTimer = setTimeout(updateCurrentDay, midnight.getTime() - now.getTime() + 50);
    };
    updateCurrentDay();
    window.addEventListener('focus', updateCurrentDay);
    document.addEventListener('visibilitychange', updateCurrentDay);
    return () => { clearTimeout(dayTimer); window.removeEventListener('focus', updateCurrentDay); document.removeEventListener('visibilitychange', updateCurrentDay); };
  }, []);

  async function refreshGarmin() {
    if (phaseRef.current !== 'ready' || garminFetching.current || disconnectingGarmin.current || (modalRef.current && modalRef.current.type !== 'garmin')) return;
    const request = ++garminRequest.current;
    garminFetching.current = true;
    setGarminLoading(true);
    try {
      const latest = await client.loadGarmin();
      if (!mounted.current || request !== garminRequest.current || phaseRef.current !== 'ready') return;
      if (modalRef.current && modalRef.current.type !== 'garmin') deferredGarmin.current = latest;
      else setGarmin(latest);
      setGarminLoadError('');
    } catch (error) {
      if (mounted.current && request === garminRequest.current && phaseRef.current === 'ready') setGarminLoadError(errorMessage(error));
    } finally {
      if (request === garminRequest.current) { garminFetching.current = false; if (mounted.current) setGarminLoading(false); }
    }
  }

  async function disconnectGarmin() {
    if (!window.confirm('Disconnect Garmin? Automatic imports will stop. Previously imported calorie readings will be kept.')) return;
    const scope = garminSessionScope.current;
    garminRequest.current++; garminFetching.current = false;
    disconnectingGarmin.current = true; setGarminDisconnecting(true); setGarminLoadError('');
    try {
      await client.disconnectGarmin();
      if (scope !== garminSessionScope.current || phaseRef.current !== 'ready') return;
      disconnectingGarmin.current = false;
      setGarmin(current => ({ ...current, status: { ...current.status, state: 'not_connected', last_error_code: null } }));
      if (phaseRef.current === 'ready') await refreshGarmin();
    } catch (error) {
      if (scope === garminSessionScope.current && phaseRef.current === 'ready') setGarminLoadError(errorMessage(error));
    } finally { if (scope === garminSessionScope.current) { disconnectingGarmin.current = false; if (mounted.current) setGarminDisconnecting(false); } }
  }

  async function loadData() {
    const loaded = await client.load();
    dataRef.current = loaded;
    savedData.current = loaded;
    setData(loaded);
    queue.current = null;
    blockedSave.current = false;
    setSaveState('saved');
    setSaveError('');
    setPhase('ready');
    if (phaseRef.current === 'ready') void refreshGarmin();
  }

  async function reloadSavedData() {
    if (saving.current || refreshing.current) { setSaveError('Wait for the current save or refresh to finish before reloading.'); return; }
    if (!window.confirm('Reload the saved journal? Unsaved changes on this device will be discarded.')) return;
    if (timer.current) { clearTimeout(timer.current); timer.current = null; }
    blockedSave.current = true;
    setModal(null);
    setPhase('loading');
    try { await loadData(); }
    catch (error) { setPhase('ready'); setSaveState('error'); setSaveError(errorMessage(error)); }
  }

  useEffect(() => {
    mounted.current = true;
    let active = true;
    async function start() {
      if (!client.configured) { setPhase('setup'); return; }
      try {
        const restored = await client.restoreSession();
        if (!active) return;
        if (restored) { setEmail(client.getEmail()); await loadData(); }
        else setPhase('login');
      } catch (error) {
        if (active) { setNotice(errorMessage(error)); setPhase('login'); }
      }
    }
    void start();
    return () => { active = false; mounted.current = false; if (timer.current) clearTimeout(timer.current); };
  }, []);

  useEffect(() => {
    if (phase === 'ready') void refreshGarmin();
    else { deferredGarmin.current = null; setGarmin(emptyGarminSnapshot()); setGarminLoading(false); setGarminLoadError(''); disconnectingGarmin.current = false; setGarminDisconnecting(false); }
    return () => { garminRequest.current++; garminSessionScope.current++; garminFetching.current = false; };
  }, [phase]);

  useEffect(() => {
    if (!modal && phase === 'ready' && deferredGarmin.current) { setGarmin(deferredGarmin.current); deferredGarmin.current = null; }
  }, [modal, phase]);

  useEffect(() => {
    if (phase !== 'ready') return;
    let active = true;
    let checking = false;
    async function checkSession(refreshData = false) {
      if (!active || checking || document.visibilityState === 'hidden') return;
      checking = true;
      try {
        const valid = await client.restoreSession();
        if (active && valid) void refreshGarmin();
        if (active && !valid) {
          setModal(null);
          setPhase('login');
          setNotice(queue.current ? 'Your session has ended. Sign in again to continue. Unsaved changes are still available to export below.' : 'Your session has ended. Please sign in again.');
        } else if (active && refreshData && !queue.current && !saving.current && !blockedSave.current && !modalRef.current) {
          // Hold autosaves during the read so a new draft cannot overwrite a newer cloud revision.
          refreshing.current = true;
          const baseline = savedData.current;
          try {
            const latest = await client.load();
            if (!active) return;
            if (queue.current || modalRef.current) {
              if (JSON.stringify(latest) !== JSON.stringify(baseline)) {
                blockedSave.current = true; setSaveState('error');
                setSaveError('Another device changed your journal while you were editing. Export any unsaved changes before reloading the saved data.');
              }
            } else {
              dataRef.current = latest; savedData.current = latest; setData(latest);
            }
          } finally {
            refreshing.current = false;
            if (queue.current && !blockedSave.current) void flushSave();
          }
        }
      } catch { /* A temporary network failure does not expose data to another user or discard a draft. */ }
      finally { checking = false; }
    }
    const sessionTimer = setInterval(() => { void checkSession(); }, 60_000);
    const onFocus = () => { void checkSession(true); };
    window.addEventListener('focus', onFocus);
    document.addEventListener('visibilitychange', onFocus);
    return () => { active = false; clearInterval(sessionTimer); window.removeEventListener('focus', onFocus); document.removeEventListener('visibilitychange', onFocus); };
  }, [phase]);

  useEffect(() => {
    function preventUnsavedExit(event: BeforeUnloadEvent) {
      if (queue.current || saving.current) { event.preventDefault(); event.returnValue = ''; }
    }
    window.addEventListener('beforeunload', preventUnsavedExit);
    return () => window.removeEventListener('beforeunload', preventUnsavedExit);
  }, []);

  async function flushSave(): Promise<boolean> {
    if (blockedSave.current || refreshing.current) return false;
    if (saving.current) return false;
    if (timer.current) { clearTimeout(timer.current); timer.current = null; }
    saving.current = true;
    setSaveState('saving');
    try {
      while (queue.current) {
        const pending = queue.current;
        queue.current = null;
        try { await client.save(pending); savedData.current = pending; }
        catch (error) { if (!queue.current) queue.current = pending; throw error; }
      }
      if (mounted.current) { setSaveState('saved'); setSaveError(''); }
      return true;
    } catch (error) {
      blockedSave.current = true;
      if (mounted.current) { setSaveState('error'); setSaveError(errorMessage(error)); }
      return false;
    } finally { saving.current = false; }
  }

  function updateData(transform: (current: TrackerData) => TrackerData) {
    const next = transform(dataRef.current);
    dataRef.current = next;
    setData(next);
    queue.current = next;
    setSaveState(blockedSave.current ? 'error' : 'pending');
    if (timer.current) clearTimeout(timer.current);
    if (!blockedSave.current && !refreshing.current) timer.current = setTimeout(() => { void flushSave(); }, 500);
  }

  async function signIn(event: FormEvent) {
    event.preventDefault();
    setBusy(true); setNotice('');
    try {
      const draft = queue.current;
      const baseline = savedData.current;
      await client.signIn(email.trim(), password, remember);
      setPassword('');
      if (draft) {
        const stored = await client.load();
        dataRef.current = draft; setData(draft); queue.current = draft; setPhase('ready');
        if (JSON.stringify(stored) === JSON.stringify(baseline)) {
          blockedSave.current = false; setSaveError(''); setSaveState('pending'); void flushSave();
        } else {
          blockedSave.current = true; setSaveState('error');
          setSaveError('Another device changed your saved journal. Export your unsaved changes before reloading the saved data.');
        }
      } else await loadData();
    }
    catch (error) { setNotice(errorMessage(error)); }
    finally { setBusy(false); }
  }

  async function signOut() {
    if (saving.current) { setSaveError('Please wait for the current save to finish before signing out.'); return; }
    if (queue.current || saving.current) {
      if (!(await flushSave())) {
        if (!window.confirm('Sign out and discard unsaved changes on this device? Cancel and export a backup first if you want to keep them.')) return;
        queue.current = null;
        if (timer.current) { clearTimeout(timer.current); timer.current = null; }
      }
    }
    setBusy(true);
    const signOutRequest = client.signOut();
    garminRequest.current++; deferredGarmin.current = null; setGarmin(emptyGarminSnapshot()); setGarminLoadError(''); setGarminLoading(false);
    setPhase('login'); setData(emptyData()); dataRef.current = emptyData(); setNotice(''); setSaveError(''); setSaveState('saved'); blockedSave.current = false;
    try { const signOutNotice = await signOutRequest; setNotice(signOutNotice ?? ''); }
    catch (error) { setNotice(errorMessage(error)); }
    finally { setBusy(false); }
  }

  function saveFood(food: Food) {
    const returnToLog = modal?.type === 'food' && modal.returnToLog;
    updateData(current => ({ ...current, foods: current.foods.some(item => item.id === food.id) ? current.foods.map(item => item.id === food.id ? food : item) : [...current.foods, food] }));
    setModal(returnToLog ? { type: 'entry', foodId: food.id } : null);
  }

  function saveEntry(entry: Entry) {
    updateData(current => ({ ...current, days: { ...current.days, [entry.date]: settingsForDay(current, entry.date) }, entries: current.entries.some(item => item.id === entry.id) ? current.entries.map(item => item.id === entry.id ? entry : item) : [...current.entries, entry] }));
    setModal(null);
  }

  function saveBankPortions(portions: BankPortion[]) {
    updateData(current => normalizeData({ ...current, foodBank: [...current.foodBank, ...portions] }));
    setModal(null); setView('bank');
  }

  function eatBankPortion(portionId: string, entryDate: string, meal: string) {
    updateData(current => consumeFoodBankPortion(current, portionId, entryDate, meal));
    setModal(null); openDay(entryDate);
  }

  function openFoodBank() { setModal(null); setView('bank'); }

  function openDay(nextDate: string) { setDate(nextDate); setView('journal'); setDeleteEntryId(null); }

  const dayEntries = data.entries.filter(entry => entry.date === date);
  const totals = dayTotals(dayEntries);
  const targets = settingsForDay(data, date);
  const burned = effectiveBurned(date, targets, garmin.days, currentDay);
  const burnedSyncDay = burned.day ?? burned.restingDay;
  const ledger = useMemo(() => calorieLedger(data, garmin.days, currentDay), [data, garmin.days, currentDay]);
  const carryIn = carryBefore(date, ledger);
  const carryIsProvisional = Boolean(ledger.provisional && date > ledger.provisional.date);
  const allowance = adjustedBudget(burned.burned, targets.deficit, carryIn);
  const budget = allowance.budget;
  const remaining = budget - totals.calories;
  const rawBalance = allowance.rawBudget - totals.calories;
  const selectedBalance = ledger.days[date];
  const hasPendingLoggedDays = Object.values(ledger.days).some(row => row.date < currentDay && !row.available);
  const deficit = burned.burned - totals.calories;
  const calorieProgress = nutrientProgress(totals.calories, budget);
  const caloriePercent = calorieProgress.barPercent;
  const nutritionTargets = macroTargets(budget);
  const meals = [...new Set(dayEntries.map(entry => entry.meal))];
  const sortedFoods = useMemo(() => sortFoodsByRecentUse(data.foods, data.entries, currentDay), [data.foods, data.entries, currentDay]);
  const visibleFoods = sortedFoods.filter(food => food.name.toLowerCase().includes(foodSearch.toLowerCase()));
  const historyDates = [...new Set([...data.entries.map(entry => entry.date), ...Object.keys(data.days)])].sort().reverse();

  return <div className="calorie-app">
    {phase !== 'ready' ? <div className="cal-auth-shell">
      <a className="cal-home-link" href="/">← Back to my website</a>
      <div className="cal-auth-brand"><span className="cal-brand-mark"><Icon name="leaf" size={24}/></span><span>calories<span className="cal-brand-dot">.</span></span></div>
      {phase === 'loading' ? <div className="cal-auth-card cal-loading" role="status"><span className="cal-spinner"/><h1>Opening your journal</h1><p>Checking your saved session.</p></div>
        : phase === 'setup' ? <Setup onConfigure={(config) => { client.configure(config); setPhase('login'); setNotice(''); }} />
        : <div className="cal-auth-card">
          <div className="cal-eyebrow">YOUR PERSONAL FOOD JOURNAL</div>
          <h1>A little space<br/>for your everyday.</h1>
          <p className="cal-auth-intro">Sign in to pick up where you left off.</p>
          <form onSubmit={signIn} className="cal-form">
            <label>Email<input type="email" value={email} onChange={event => setEmail(event.target.value)} autoComplete="username" required autoCapitalize="none" spellCheck={false} placeholder="Your account email" /></label>
            <label>Password<input type="password" value={password} onChange={event => setPassword(event.target.value)} autoComplete="current-password" required placeholder="Your password" /></label>
            <label className="cal-checkbox"><input type="checkbox" checked={remember} onChange={event => setRemember(event.target.checked)}/><span>Remember this device for 30 days<small>Leave unchecked on a shared device.</small></span></label>
            {notice && <p className="cal-form-error" role="alert">{notice}</p>}
            <button className="cal-button cal-button-primary cal-button-full" disabled={busy}>{busy ? 'Signing in…' : 'Open my journal'}<Icon name="arrow" size={18}/></button>
          </form>
          <div className="cal-auth-footnote"><Icon name="lock" size={15}/><span>Your entries sync privately across your devices.</span></div>
          {queue.current && <button className="cal-text-button cal-auth-config" onClick={() => downloadFile(JSON.stringify(queue.current, null, 2), `calories-unsaved-${todayISO()}.json`, 'application/json')}>Export unsaved changes</button>}
          <button className="cal-text-button cal-auth-config" onClick={() => { setPhase('setup'); setNotice(''); }}>Connection settings</button>
        </div>}
      <p className="cal-auth-tagline">A record of your day. At your own pace.</p>
    </div> : <>
      <header className="cal-header">
        <a href="/calories/" className="cal-brand" aria-label="Calories home"><span className="cal-brand-mark"><Icon name="leaf" size={22}/></span><span>calories<span className="cal-brand-dot">.</span></span></a>
        <nav className="cal-nav" aria-label="Journal navigation">
          {([['journal', 'Journal'], ['foods', 'Foods'], ['bank', 'Food bank'], ['history', 'History']] as [View, string][]).map(([key, label]) => <button key={key} className={view === key ? 'is-active' : ''} aria-current={view === key ? 'page' : undefined} onClick={() => { setView(key); setDeleteFoodId(null); setDeleteEntryId(null); }}>{label}</button>)}
        </nav>
        <div className="cal-header-tools">
          <span className={`cal-sync cal-sync-${saveState}`} role="status" aria-live="polite">{saveState === 'saved' ? <Icon name="check" size={13}/> : <span className="cal-status-dot"/>}<span>{saveState === 'saved' ? 'All saved' : saveState === 'error' ? 'Not saved' : 'Saving…'}</span></span>
          <button className="cal-icon-button" onClick={() => setModal({ type: 'transfer' })} title="Backup & import" aria-label="Backup and import"><Icon name="download"/></button>
          <button className="cal-icon-button" onClick={() => void signOut()} title="Sign out" aria-label="Sign out" disabled={busy}><Icon name="logout"/></button>
        </div>
      </header>

      {saveError && <div className="cal-save-alert" role="alert"><div><strong>Your latest changes haven’t synced.</strong><p>{saveError}</p></div><button className="cal-button cal-button-small" onClick={() => setModal({ type: 'transfer' })}>Export unsaved</button><button className="cal-text-button" onClick={() => void reloadSavedData()} disabled={busy || saving.current || refreshing.current}>Reload saved data</button></div>}

      <main className="cal-main">
        {view === 'journal' && <>
          <section className="cal-page-heading">
            <div><div className="cal-eyebrow">{date === currentDay ? 'TODAY’S JOURNAL' : 'DAILY JOURNAL'}</div><h1>{readableDate(date, { weekday: 'long' })}<span className="cal-heading-dot">.</span></h1><p>{readableDate(date, { day: 'numeric', month: 'long', year: 'numeric' })} <span className="cal-date-divider">/</span> One day at a time.</p></div>
            <div className="cal-date-controls"><button className="cal-icon-button" onClick={() => openDay(addDays(date, -1))} aria-label="Previous day"><Icon name="left"/></button><label className="cal-date-picker"><Icon name="calendar" size={17}/><input type="date" aria-label="Choose journal date" value={date} onChange={event => { if (event.target.value) openDay(event.target.value); }}/></label><button className="cal-icon-button" onClick={() => openDay(addDays(date, 1))} aria-label="Next day"><Icon name="right"/></button>{date !== currentDay && <button className="cal-text-button cal-today" onClick={() => openDay(currentDay)}>Today</button>}</div>
          </section>

          <section className="cal-dashboard" aria-label="Daily nutrition summary">
            <div className="cal-energy-card">
              <div className="cal-card-top"><span className="cal-eyebrow">ENERGY TODAY</span><button className="cal-icon-button" onClick={() => setModal({ type: 'targets' })} aria-label="Edit daily targets" title="Edit daily targets"><Icon name="settings" size={19}/></button></div>
              <div className="cal-energy-main"><div><div className="cal-energy-value">{number(totals.calories)}<span>kcal</span></div><p>of {number(budget, 1)} kcal {burned.predicted ? 'predicted ' : ''}budget with carry</p></div><div className="cal-ring" style={{ '--cal-progress': `${caloriePercent}%` } as React.CSSProperties}><div><span>{calorieProgress.percent === null ? '—' : <>{number(caloriePercent)}<small>%</small></>}</span><small>{budget > 0 ? 'of budget' : 'no budget'}</small></div></div></div>
              <div className="cal-energy-bottom"><span>{remaining >= 0 ? 'Remaining' : 'Above budget'}<strong>{number(Math.abs(remaining), 1)} <small>kcal</small></strong></span><span>{burned.predicted ? 'Predicted burned' : burned.restingEstimate ? 'Estimated burned' : 'Burned'}<strong>{burned.unavailable ? '—' : number(burned.burned)} <small>kcal</small></strong></span><span>{burned.unavailable ? 'Deficit / surplus' : burned.predicted ? deficit >= 0 ? 'Predicted deficit' : 'Predicted surplus' : burned.restingEstimate ? deficit >= 0 ? 'Estimated deficit' : 'Estimated surplus' : deficit >= 0 ? 'Deficit' : 'Surplus'}<strong>{burned.unavailable ? '—' : number(Math.abs(deficit))} <small>kcal</small></strong></span></div>
              <div className="cal-target-note"><span>Deficit target: {number(targets.deficit, 1)} kcal</span><button className="cal-text-button" onClick={() => setModal({ type: 'targets' })}>Edit deficit target<Icon name="edit" size={13}/></button></div>
              <div className="cal-burned-source"><span className="cal-source-badge">{burnedLabel(burned)}</span>{burned.source === 'garmin' ? <div className="cal-burned-details"><span>{burnedFormula(burned)}</span>{burnedSyncDay && <small>{burnedSyncTime(burned)}{date === currentDay ? ' · Today is provisional' : ''}{garmin.status.state === 'not_connected' ? ' · Automatic imports stopped' : date === currentDay && staleImport(burnedSyncDay.synced_at) ? ' · Needs a new import' : ''}</small>}</div> : <span>Your saved manual calories burned value.</span>}</div>
            </div>
            <div className="cal-nutrients-card">
              <div className="cal-card-top"><span className="cal-eyebrow">DAILY NUTRIENT TARGETS</span><span className="cal-pill">grams</span></div>
              <div className="cal-macro-progress-list">
                <MacroProgressRow name="Protein" consumed={totals.protein} target={nutritionTargets.protein} tone="protein"/>
                <MacroProgressRow name="Carbs" consumed={totals.carbs} target={nutritionTargets.carbs} tone="carbs"/>
                <MacroProgressRow name="Fat" consumed={totals.fat} target={nutritionTargets.fat} tone="fat"/>
                <MacroProgressRow name="Fibre" consumed={totals.fiber} target={nutritionTargets.fiber} tone="fiber"/>
              </div>
              {nutritionTargets.budgetTooLow && <p className="cal-macro-budget-note">Current calorie budget is too low for 175 g protein and 25% fat. Carb target is 0 g.</p>}
            </div>
          </section>

          <section className="cal-carry-panel" aria-label="Running calorie balance">
            <div className="cal-carry-heading"><h2>Running calorie balance</h2><span>{burned.unavailable ? 'Waiting for Garmin' : selectedBalance?.settled ? 'Carried forward' : carryIsProvisional ? 'Future projection' : date === currentDay ? 'Today is provisional' : dayEntries.length ? 'Not settled yet' : 'Unlogged day'}</span></div>
            <div className="cal-carry-values"><span>Base budget<strong>{burned.unavailable ? '—' : number(allowance.baseBudget, 1)} <small>kcal</small></strong></span><span>Carry from earlier days{carryIsProvisional ? ' (provisional)' : ''}<strong>{signedNumber(carryIn)} <small>kcal</small></strong></span><span>{selectedBalance ? 'Balance after logged food' : 'Projected balance'}<strong>{burned.unavailable ? '—' : signedNumber(rawBalance)} <small>kcal</small></strong></span></div>
            <p>Budgeted deficit is capped at {number(MAX_DAILY_DEFICIT)} kcal/day. All unpaid debt stays in the running balance. Completed logged days settle that balance.{date === currentDay ? ledger.provisional ? ' Today’s balance is provisional and is included when previewing later days.' : dayEntries.length ? ' Today’s carry is waiting for calories burned.' : ' Today has no logged food, so it does not add to carry.' : !dayEntries.length ? ' This day has no logged food, so it does not add to carry.' : date > currentDay ? ' Planned future meals do not carry forward yet.' : ''}</p>
            {carryIsProvisional && <p>Includes today’s provisional balance; updates as food or Garmin data changes.</p>}
            {hasPendingLoggedDays && <p className="cal-carry-pending-note">Some logged days are waiting for Garmin data. Their balances are not included yet.</p>}
            {!burned.unavailable && allowance.deferredDebt > 0 && <p className="cal-carry-debt-note">If you eat this day’s {number(budget, 1)} kcal budget, {number(allowance.deferredDebt, 1)} kcal of debt remains for later days.</p>}
          </section>

          <button className={`cal-garmin-bar ${garmin.status.state === 'needs_login' || garmin.status.state === 'error' || (garmin.status.state === 'connected' && staleImport(garmin.status.last_sync_at)) || garminLoadError ? 'cal-garmin-attention' : ''}`} onClick={() => setModal({ type: 'garmin' })} aria-label="Garmin Connect settings">
            <span className="cal-garmin-symbol"><Icon name="watch" size={20}/></span><span className="cal-garmin-bar-text"><strong>Garmin Connect</strong><span>{garminLoading ? 'Checking imported data…' : garminLoadError ? 'Unable to refresh · saved readings retained' : garminStatusLabel(garmin.status)}{!garminLoading && !garminLoadError && garmin.status.last_sync_at ? ` · Last import ${syncTime(garmin.status.last_sync_at)}` : ''}</span></span><Icon name="right" size={18}/>
          </button>

          <section className="cal-journal-section">
            <div className="cal-section-heading"><div><h2>On the menu</h2><p>{dayEntries.length ? `${dayEntries.length} ${dayEntries.length === 1 ? 'entry' : 'entries'} logged today` : 'Your food, meal by meal.'}</p></div><button className="cal-button cal-button-primary" onClick={() => setModal({ type: 'entry' })}><Icon name="plus" size={18}/>Log food</button></div>
            {dayEntries.length === 0 ? <div className="cal-empty-journal"><span className="cal-empty-art"><Icon name="leaf" size={38}/><span/><span/></span><h3>A fresh page for {date === currentDay ? 'today' : 'this day'}.</h3><p>Choose a saved food or enter a manual estimate for a meal.</p><button className="cal-text-button" onClick={() => setModal({ type: 'entry' })}>Log your first food<Icon name="arrow" size={17}/></button></div>
              : <div className="cal-meals">{meals.map((meal, index) => {
                const entries = dayEntries.filter(entry => entry.meal === meal);
                const mealTotals = dayTotals(entries);
                return <div className="cal-meal" key={meal}><div className="cal-meal-heading"><span className="cal-meal-number">{String(index + 1).padStart(2, '0')}</span><h3>{meal}</h3><span>{number(mealTotals.calories)} kcal</span></div><div className="cal-entry-list">{entries.map(entry => <div className="cal-entry" key={entry.id}><div className="cal-entry-info"><strong>{entry.foodName}</strong><span>{entry.source === 'manual' ? 'Manual estimate' : <>{number(entry.quantity, 2)} {unitLabel(entry.unit, entry.quantity)}<i/> {entry.state}</>}</span></div><div className="cal-entry-macros"><span><b>P</b> {number(entry.protein, 1)}<small> g</small></span><span><b>C</b> {number(entry.carbs, 1)}<small> g</small></span><span><b>F</b> {number(entry.fat, 1)}<small> g</small></span><span><b>Fibre</b> {number(entry.fiber, 1)}<small> g</small></span></div><strong className="cal-entry-energy">{number(entry.calories)}<small> kcal</small></strong><div className="cal-entry-actions">{deleteEntryId === entry.id ? <><button className="cal-text-button cal-danger" onClick={() => { updateData(current => ({ ...current, entries: current.entries.filter(item => item.id !== entry.id) })); setDeleteEntryId(null); }}>Remove</button><button className="cal-icon-button" onClick={() => setDeleteEntryId(null)} aria-label="Cancel removal"><Icon name="close" size={16}/></button></> : <><button className="cal-icon-button" onClick={() => setModal({ type: 'entry', entry })} aria-label={`Edit ${entry.foodName}`}><Icon name="edit" size={16}/></button><button className="cal-icon-button" onClick={() => setDeleteEntryId(entry.id)} aria-label={`Remove ${entry.foodName}`}><Icon name="trash" size={16}/></button></>}</div></div>)}</div></div>;
              })}</div>}
          </section>
        </>}

        {view === 'bank' && <>
          <section className="cal-page-heading"><div><div className="cal-eyebrow">PREPARED FOR LATER</div><h1>Food bank<span className="cal-heading-dot">.</span></h1><p>Your portions, ready to log.</p></div></section>
          <FoodBankView portions={data.foodBank} onPrepare={() => setModal({ type: 'bank-prepare' })} onEat={portion => setModal({ type: 'bank-consume', portion })} onRemove={id => updateData(current => ({ ...current, foodBank: current.foodBank.filter(portion => portion.id !== id) }))}/>
        </>}

        {view === 'foods' && <>
          <section className="cal-page-heading"><div><div className="cal-eyebrow">YOUR OWN LITTLE LIBRARY</div><h1>Everyday foods<span className="cal-heading-dot">.</span></h1><p>Most logged in the last 14 days, first.</p></div><button className="cal-button cal-button-primary" onClick={() => setModal({ type: 'food' })}><Icon name="plus" size={18}/>Add food</button></section>
          <div className="cal-library-bar"><label className="cal-search"><Icon name="search" size={19}/><input type="search" placeholder="Find a food…" aria-label="Search food library" value={foodSearch} onChange={event => setFoodSearch(event.target.value)}/></label><span>{data.foods.length} {data.foods.length === 1 ? 'food' : 'foods'} in your library</span></div>
          {visibleFoods.length ? <div className="cal-food-grid">{visibleFoods.map(food => <article className="cal-food-card" key={food.id}><div className="cal-food-heading"><span className="cal-food-state">{food.state}</span><div>{deleteFoodId === food.id ? <button className="cal-text-button" onClick={() => setDeleteFoodId(null)}>Cancel</button> : <button className="cal-icon-button" onClick={() => setModal({ type: 'food', food })} aria-label={`Edit ${food.name}`}><Icon name="edit" size={17}/></button>}</div></div><h2>{food.name}</h2><p>Per {number(food.baseQuantity, 2)} {unitLabel(food.unit, food.baseQuantity)}</p><div className="cal-food-energy"><strong>{number(food.calories)}</strong><span>kcal</span></div><div className="cal-food-macros"><span>Protein<strong>{number(food.protein, 1)} g</strong></span><span>Carbs<strong>{number(food.carbs, 1)} g</strong></span><span>Fat<strong>{number(food.fat, 1)} g</strong></span><span>Fibre<strong>{number(food.fiber, 1)} g</strong></span></div><div className="cal-food-footer">{deleteFoodId === food.id ? <><p>Existing log entries will be kept.</p><button className="cal-text-button cal-danger" onClick={() => { updateData(current => ({ ...current, foods: current.foods.filter(item => item.id !== food.id) })); setDeleteFoodId(null); }}>Remove food</button></> : <><button className="cal-text-button" onClick={() => setModal({ type: 'entry', foodId: food.id })}>Log this food<Icon name="plus" size={16}/></button><button className="cal-icon-button" onClick={() => setDeleteFoodId(food.id)} aria-label={`Remove ${food.name} from library`}><Icon name="trash" size={16}/></button></>}</div></article>)}</div> : <div className="cal-empty-journal cal-library-empty"><span className="cal-empty-art"><Icon name="book" size={36}/></span><h3>{foodSearch ? 'No foods found.' : 'Your favourites start here.'}</h3><p>{foodSearch ? 'Try a different name, or add a new food.' : 'Use the nutrition on your food label to build a library that feels like you.'}</p><button className="cal-text-button" onClick={() => setModal({ type: 'food' })}>Add a food<Icon name="plus" size={17}/></button></div>}
        </>}

        {view === 'history' && <>
          <section className="cal-page-heading"><div><div className="cal-eyebrow">THE DAYS ADD UP</div><h1>Your history<span className="cal-heading-dot">.</span></h1><p>A simple record, whenever you need it.</p></div><button className="cal-button" onClick={() => setModal({ type: 'transfer' })}><Icon name="download" size={18}/>Export journal</button></section>
          {hasPendingLoggedDays && <p className="cal-history-pending-note">Some logged days are waiting for Garmin data. Their balances are not included yet.</p>}
          {historyDates.length ? <div className="cal-history">
            <div className="cal-history-header"><span>Day</span><span>Energy</span><span className="cal-history-protein">Protein</span><span className="cal-history-deficit">Deficit / surplus</span><span className="cal-history-balance">Running balance</span><span/></div>
            {historyDates.map(day => {
              const entries = data.entries.filter(entry => entry.date === day);
              const dayTotal = dayTotals(entries);
              const goals = settingsForDay(data, day);
              const dayBurned = effectiveBurned(day, goals, garmin.days, currentDay);
              const dayDeficit = dayBurned.burned - dayTotal.calories;
              const balance = ledger.days[day];
              return <button className="cal-history-row" key={day} onClick={() => openDay(day)}>
                <span><strong>{readableDate(day, { weekday: 'short', day: 'numeric', month: 'short' })}</strong><small>{entries.length} {entries.length === 1 ? 'entry' : 'entries'} · {day.slice(0, 4)}<em className="cal-history-source">{burnedLabel(dayBurned)}{dayBurned.source === 'garmin' && !dayBurned.unavailable && day === currentDay ? ' · provisional' : ''}</em>{dayBurned.restingEstimate && dayBurned.restingDay && <em className="cal-history-source">Resting from {readableDate(dayBurned.restingDay.date, { day: 'numeric', month: 'short', year: 'numeric' })} · {burnedSyncTime(dayBurned)}</em>}</small></span>
                <span><strong>{number(dayTotal.calories, 1)}</strong><small> kcal</small></span>
                <span className="cal-history-protein"><strong>{number(dayTotal.protein, 1)}</strong><small> g</small></span>
                <span className="cal-history-deficit"><strong>{entries.length && !dayBurned.unavailable ? number(Math.abs(dayDeficit), 1) : '—'}</strong><small>{dayBurned.unavailable ? 'Burned unavailable' : entries.length ? ` kcal ${dayBurned.restingEstimate ? 'estimated ' : ''}${dayDeficit >= 0 ? 'deficit' : 'surplus'}` : ' Unlogged'}</small></span>
                <span className="cal-history-balance"><strong>{balance?.available ? signedNumber(balance.balance) : '—'}</strong><small>{balance ? !balance.available ? 'Waiting for Garmin' : `kcal · ${balance.settled ? 'carried forward' : day === currentDay ? 'provisional' : ledger.provisional && day > ledger.provisional.date ? 'planned · provisional carry' : 'not settled'}` : 'No logged food'}</small></span>
                <Icon name="right" size={18}/>
              </button>;
            })}
          </div> : <div className="cal-empty-journal"><span className="cal-empty-art"><Icon name="calendar" size={36}/></span><h3>Your story is just beginning.</h3><p>Days you log will appear here, with your daily totals and running calorie balance.</p><button className="cal-text-button" onClick={() => openDay(currentDay)}>Go to today<Icon name="arrow" size={17}/></button></div>}

        </>}
        <footer className="cal-footer"><span>A little more aware, every day.</span><span>Personal journal · <a href="/">fangxk2003</a></span></footer>
      </main>

      {modal && <Modal onClose={() => setModal(null)}>{modal.type === 'entry' ? <EntryEditor key={modal.entry?.id ?? modal.foodId ?? 'new-entry'} entry={modal.entry} initialFoodId={modal.foodId} foods={sortedFoods} date={date} onSave={saveEntry} onAddFood={() => setModal({ type: 'food', returnToLog: true })} onFoodBank={openFoodBank} bankCount={data.foodBank.length} onClose={() => setModal(null)}/>
        : modal.type === 'food' ? <FoodEditor key={modal.food?.id ?? 'new-food'} food={modal.food} readPhotos={(input, signal) => client.readFoodPhotos(input, signal)} onSave={saveFood} onClose={() => setModal(null)}/>
        : modal.type === 'bank-prepare' ? <FoodBankEditor foods={sortedFoods} onSave={saveBankPortions} onClose={() => setModal(null)}/>
        : modal.type === 'bank-consume' ? <FoodBankConsume portion={modal.portion} date={date} onSave={(entryDate, meal) => eatBankPortion(modal.portion.id, entryDate, meal)} onClose={() => setModal(null)}/>
        : modal.type === 'targets' ? <TargetsEditor date={date} currentDay={currentDay} settings={targets} garminDays={garmin.days} carryIn={carryIn} carryIsProvisional={carryIsProvisional} onSave={(settings, makeDefault) => { updateData(current => ({ ...current, days: { ...current.days, [date]: settings }, defaults: makeDefault ? { ...current.defaults, deficit: settings.deficit, protein: DAILY_PROTEIN_GRAMS } : current.defaults })); setModal(null); }} onClose={() => setModal(null)}/>
        : modal.type === 'garmin' ? <GarminConnection date={date} currentDay={currentDay} snapshot={garmin} loading={garminLoading || garminDisconnecting} disconnecting={garminDisconnecting} error={garminLoadError} onRefresh={() => void refreshGarmin()} onDisconnect={() => void disconnectGarmin()} onClose={() => setModal(null)}/>
        : <TransferDialog data={data} onImport={(next) => { updateData(() => next); setModal(null); }} onClose={() => setModal(null)}/>}</Modal>}
    </>}
  </div>;
}

function MacroProgressRow({ name, consumed, target, tone }: { name: string; consumed: number; target: number; tone: 'protein' | 'carbs' | 'fat' | 'fiber' }) {
  const progress = nutrientProgress(consumed, target);
  const percent = progress.percent === null ? '—' : `${number(progress.percent, 1)}%`;
  const remaining = target - consumed;
  const detail = target === 0 ? consumed > 0 ? `${number(consumed, 1)} g over a zero target` : 'Target is 0 g for this budget' : remaining > 0 ? `${number(remaining, 1)} g remaining` : remaining < 0 ? `${number(-remaining, 1)} g over target` : 'Target reached';
  return <div className={`cal-macro-progress-row cal-macro-tone-${tone}`}>
    <div className="cal-macro-progress-heading"><span className="cal-macro-name">{name}</span><span className="cal-macro-consumed"><strong>{number(consumed, 1)}</strong><small> / {number(target, 1)} g</small></span><span className="cal-macro-percentage" aria-label={progress.percent === null ? `${name} percentage unavailable` : `${number(progress.percent, 1)} percent of ${name.toLowerCase()} target consumed`}>{percent}</span></div>
    <div className="cal-progress-track" role="progressbar" aria-label={`${name} target progress`} aria-valuemin={0} aria-valuemax={100} aria-valuenow={progress.barPercent} aria-valuetext={`${number(consumed, 1)} of ${number(target, 1)} grams; ${progress.percent === null ? 'percentage unavailable' : `${number(progress.percent, 1)} percent of target consumed`}`}><span style={{ width: `${progress.barPercent}%` }}/></div>
    <p className={remaining < 0 ? 'cal-macro-over' : ''}>{detail}</p>
  </div>;
}

function Setup({ onConfigure }: { onConfigure: (config: { url: string; key: string; email?: string }) => void }) {
  const [url, setUrl] = useState('');
  const [key, setKey] = useState('');
  const [error, setError] = useState('');
  function submit(event: FormEvent) {
    event.preventDefault(); setError('');
    try { onConfigure({ url: url.trim(), key: key.trim() }); }
    catch (error) { setError(errorMessage(error)); }
  }
  return <div className="cal-auth-card cal-setup-card"><div className="cal-eyebrow">ONE-TIME DEVICE SETUP</div><h1>Connect your<br/>private journal.</h1><p className="cal-auth-intro">Your journal uses your Supabase project to keep your iPhone and Mac in sync.</p><div className="cal-setup-guide"><Icon name="book" size={19}/><div><strong>Set up your project first</strong><p>Follow the short guide to create your database and personal account.</p><a href="/calories-setup/" target="_blank" rel="noreferrer">Open setup guide ↗</a></div></div><form className="cal-form" onSubmit={submit}><label>Supabase project URL<input type="url" value={url} onChange={event => setUrl(event.target.value)} placeholder="https://your-project.supabase.co" autoComplete="off" autoCapitalize="none" spellCheck={false} required/></label><label>Publishable / anon key<textarea value={key} onChange={event => setKey(event.target.value)} placeholder="Paste your public project key" spellCheck={false} required rows={3}/><small className="cal-field-hint">Use only the public publishable or anon key. Never use a service-role or secret key.</small></label>{error && <p className="cal-form-error" role="alert">{error}</p>}<button className="cal-button cal-button-primary cal-button-full">Connect journal<Icon name="arrow" size={18}/></button></form><p className="cal-setup-note">Connection details are saved on this device. Your account password protects access to your data.</p></div>;
}

function Modal({ children, onClose }: { children: ReactNode; onClose: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const element = dialog.current;
    element?.showModal();
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => { document.body.style.overflow = previousOverflow; element?.close(); };
  }, []);
  return <dialog className="cal-dialog" aria-labelledby="cal-dialog-title" ref={dialog} onCancel={(event) => { event.preventDefault(); onClose(); }} onClick={event => { if (event.target === dialog.current) { const rect = dialog.current.getBoundingClientRect(); if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) onClose(); } }}><div className="cal-modal-content">{children}</div></dialog>;
}

function ModalHeader({ title, eyebrow, onClose }: { title: string; eyebrow: string; onClose: () => void }) {
  return <div className="cal-modal-heading"><div><div className="cal-eyebrow">{eyebrow}</div><h2 id="cal-dialog-title">{title}</h2></div><button className="cal-icon-button" type="button" onClick={onClose} aria-label="Close dialog"><Icon name="close"/></button></div>;
}

function NumericField({ label, value, onChange, unit = 'g', min = 0, required = true, describedBy }: { label: string; value: string; onChange: (value: string) => void; unit?: string; min?: number; required?: boolean; describedBy?: string }) {
  return <label>{label}<span className="cal-unit-input"><input type="number" inputMode="decimal" value={value} onChange={event => onChange(event.target.value)} min={min} max={100000000} step="any" required={required} aria-describedby={describedBy}/><span>{unit}</span></span></label>;
}

function EntryEditor({ entry, initialFoodId, foods, date, onSave, onAddFood, onFoodBank, bankCount, onClose }: { entry?: Entry; initialFoodId?: string; foods: Food[]; date: string; onSave: (entry: Entry) => void; onAddFood: () => void; onFoodBank: () => void; bankCount: number; onClose: () => void }) {
  const manualEditing = entry?.source === 'manual' || Boolean(entry && entry.foodId === `manual:${entry.id}`);
  const [mode, setMode] = useState<'library' | 'manual'>(entry ? manualEditing ? 'manual' : 'library' : foods.length ? 'library' : 'manual');
  const initialFood = foods.find(food => food.id === (manualEditing ? initialFoodId : entry?.foodId ?? initialFoodId)) ?? foods[0];
  const [foodId, setFoodId] = useState(manualEditing ? '' : entry ? entry.foodId : initialFood?.id ?? '');
  const [foodFilter, setFoodFilter] = useState('');
  const [quantity, setQuantity] = useState(String(!manualEditing && entry ? entry.quantity : initialFood?.baseQuantity ?? 100));
  const [manualName, setManualName] = useState(manualEditing ? entry?.foodName ?? '' : '');
  const [manualNutrients, setManualNutrients] = useState({ protein: manualEditing ? String(entry?.protein ?? '') : '', carbs: manualEditing ? String(entry?.carbs ?? '') : '', fat: manualEditing ? String(entry?.fat ?? '') : '', fiber: manualEditing ? String(entry?.fiber ?? '') : '' });
  const standardMeals = ['Breakfast', 'Lunch', 'Dinner', 'Snack'];
  const [mealChoice, setMealChoice] = useState(entry ? (standardMeals.includes(entry.meal) ? entry.meal : 'Custom') : 'Breakfast');
  const [customMeal, setCustomMeal] = useState(entry?.meal ?? '');
  const [entryDate, setEntryDate] = useState(entry?.date ?? date);
  const [error, setError] = useState('');
  // Only a library entry contributes its original nutrient snapshot to the food picker.
  const savedFood: Food | undefined = entry && !manualEditing ? { id: entry.foodId, name: entry.foodName, baseQuantity: entry.quantity, unit: entry.unit, state: entry.state, carbs: entry.carbs, protein: entry.protein, fat: entry.fat, fiber: entry.fiber, calories: entry.calories } : undefined;
  const choices = savedFood ? [savedFood, ...foods.filter(food => food.id !== savedFood.id)] : foods;
  const matches = (food: Food, query: string) => food.name.toLowerCase().includes(query.trim().toLowerCase());
  const filteredFoods = choices.filter(food => matches(food, foodFilter));
  const selectedFood = filteredFoods.find(food => food.id === foodId);
  const preview = selectedFood ? macrosForFood(selectedFood, previewNumber(quantity)) : null;
  const estimatedValue = (value: string) => { const parsed = Number(value); return Number.isFinite(parsed) && parsed >= 0 && parsed <= 1e8 ? parsed : 0; };
  const manualPreview = { protein: estimatedValue(manualNutrients.protein), carbs: estimatedValue(manualNutrients.carbs), fat: estimatedValue(manualNutrients.fat), fiber: estimatedValue(manualNutrients.fiber) };
  const manualEnergy = energyFromMacros(manualPreview);
  function filterFoods(value: string) {
    setFoodFilter(value);
    const first = choices.find(food => matches(food, value));
    setFoodId(first?.id ?? '');
    if (first && first.id !== foodId) setQuantity(String(first.baseQuantity));
    setError('');
  }
  function changeMode(next: 'library' | 'manual') {
    if (next === 'library' && !selectedFood) {
      const first = filteredFoods[0];
      setFoodId(first?.id ?? '');
      if (first) setQuantity(String(first.baseQuantity));
    }
    setMode(next); setError('');
  }
  function changeManualNutrient(key: keyof typeof manualNutrients, value: string) { setManualNutrients(current => ({ ...current, [key]: value })); setError(''); }
  function submit(event: FormEvent) {
    event.preventDefault();
    const meal = mealChoice === 'Custom' ? customMeal.trim() : mealChoice;
    if (!meal) { setError('Add a meal name or number.'); return; }
    if (mode === 'manual') {
      if (!manualName.trim()) { setError('Give your estimated meal a name.'); return; }
      if (Object.values(manualNutrients).some(value => !Number.isFinite(Number(value)) || Number(value) < 0 || Number(value) > 1e8) || !Number.isFinite(manualEnergy) || manualEnergy > 1e8) { setError('Enter finite, non-negative nutrient totals. Each value and the calculated energy must be no more than 100,000,000.'); return; }
      try { onSave(manualMealEntry({ id: entry?.id ?? newId(), date: entryDate, meal, name: manualName.trim(), protein: Number(manualNutrients.protein), carbs: Number(manualNutrients.carbs), fat: Number(manualNutrients.fat), fiber: Number(manualNutrients.fiber) })); }
      catch (error) { setError(errorMessage(error)); }
      return;
    }
    if (!selectedFood || !preview || !Number.isFinite(Number(quantity)) || Number(quantity) <= 0 || Number(quantity) > 1e8) { setError('Choose a food and a quantity greater than zero.'); return; }
    // A fresh library object removes a manual source flag when the entry changes modes.
    onSave({ id: entry?.id ?? newId(), date: entryDate, meal, foodId: selectedFood.id, foodName: selectedFood.name, quantity: Number(quantity), unit: selectedFood.unit, state: selectedFood.state, ...preview });
  }
  return <>
    <ModalHeader title={entry ? 'Edit your entry' : 'What’s on the menu?'} eyebrow={entry ? 'DAILY JOURNAL' : 'LOG A FOOD'} onClose={onClose}/>
    <form className="cal-form" onSubmit={submit}>
      <div className={`cal-entry-mode-switch${!entry && bankCount ? ' cal-entry-mode-with-bank' : ''}`} role="group" aria-label="Entry method"><button type="button" className={mode === 'library' ? 'is-active' : ''} aria-pressed={mode === 'library'} onClick={() => changeMode('library')}>Food library</button><button type="button" className={mode === 'manual' ? 'is-active' : ''} aria-pressed={mode === 'manual'} onClick={() => changeMode('manual')}>Manual estimate</button>{!entry && bankCount > 0 && <button type="button" onClick={onFoodBank}>Food bank ({bankCount})</button>}</div>
      {mode === 'library' ? <>
        <div className="cal-food-picker" role="group" aria-labelledby="cal-entry-food-label"><label id="cal-entry-food-label" htmlFor="cal-entry-food-filter">Food</label><div className="cal-food-picker-controls"><input id="cal-entry-food-filter" type="text" aria-label="Filter foods" aria-describedby="cal-entry-food-matches" value={foodFilter} onChange={event => filterFoods(event.target.value)} placeholder="Type part of a food name" autoComplete="off" autoCapitalize="none" spellCheck={false} autoFocus/><select aria-label="Food" aria-describedby="cal-entry-food-matches" value={foodId} onChange={event => { const food = filteredFoods.find(item => item.id === event.target.value); setFoodId(food?.id ?? ''); if (food) setQuantity(String(food.baseQuantity)); setError(''); }} required disabled={filteredFoods.length === 0}>{filteredFoods.length === 0 && <option value="" disabled>No matching foods</option>}{filteredFoods.map(food => <option key={food.id} value={food.id}>{food.name}{food.id === savedFood?.id ? ' (saved entry)' : ''} · {food.state}</option>)}</select></div><small className="cal-field-hint" id="cal-entry-food-matches" role="status">{filteredFoods.length === 0 ? foods.length === 0 && !savedFood ? 'Your library is empty. Add a food below or choose Manual estimate.' : 'No matching foods. Try another name or add a food below.' : foodFilter.trim() ? `${filteredFoods.length} matching ${filteredFoods.length === 1 ? 'food' : 'foods'}` : 'Type to filter the food list.'}</small></div>
        <button className="cal-text-button cal-add-food-inline" type="button" onClick={onAddFood}><Icon name="plus" size={15}/>Add a new food to your library</button>
      </> : <>
        <label>Meal / restaurant name<input value={manualName} onChange={event => { setManualName(event.target.value); setError(''); }} placeholder="e.g. Restaurant lunch" required maxLength={160} autoFocus/></label>
        <p className="cal-field-hint">Enter the estimated nutrient totals for the entire meal or portion you ate. Leave blank for zero.</p>
        <div className="cal-form-grid"><NumericField label="Protein total" value={manualNutrients.protein} onChange={value => changeManualNutrient('protein', value)} required={false}/><NumericField label="Carbohydrate total" value={manualNutrients.carbs} onChange={value => changeManualNutrient('carbs', value)} required={false}/><NumericField label="Fat total" value={manualNutrients.fat} onChange={value => changeManualNutrient('fat', value)} required={false}/><NumericField label="Fibre total" value={manualNutrients.fiber} onChange={value => changeManualNutrient('fiber', value)} required={false}/></div>
      </>}
      <div className="cal-form-grid">{mode === 'library' && <NumericField label="Quantity" value={quantity} onChange={setQuantity} unit={selectedFood?.unit ?? 'g'} min={0.001}/>}<label>Meal<select value={mealChoice} onChange={event => setMealChoice(event.target.value)}>{standardMeals.map(meal => <option key={meal}>{meal}</option>)}<option>Custom</option></select></label>{mode === 'manual' && <label>Date<input type="date" value={entryDate} onChange={event => setEntryDate(event.target.value)} required/></label>}</div>
      {mealChoice === 'Custom' && <label>Meal name or number<input type="text" value={customMeal} onChange={event => setCustomMeal(event.target.value)} placeholder="e.g. Meal 1, late lunch" required maxLength={80}/></label>}
      {mode === 'library' && <label>Date<input type="date" value={entryDate} onChange={event => setEntryDate(event.target.value)} required/></label>}
      {mode === 'manual' ? <div className="cal-budget-preview" aria-live="polite"><span>Estimated energy for this meal</span><strong><output aria-label="Estimated meal energy">{number(manualEnergy, 1)}</output> kcal</strong><p>4 × protein + 4 × carbohydrate + 9 × fat. Fibre is tracked separately.</p></div> : preview && <div className="cal-entry-preview"><div><span>This entry</span><strong>{number(preview.calories)} <small>kcal</small></strong></div><div className="cal-preview-macros">{([['Protein', preview.protein], ['Carbs', preview.carbs], ['Fat', preview.fat], ['Fibre', preview.fiber]] as const).map(([name, amount]) => <span key={name}>{name}<strong>{number(amount, 1)} g</strong></span>)}</div><p>{selectedFood?.state} · {entry && !manualEditing && foodId === entry.foodId ? 'Uses nutrition saved with this entry.' : `Calculated from your food’s values per ${number(selectedFood?.baseQuantity ?? 0, 2)} ${selectedFood?.unit}.`}</p></div>}
      {error && <p className="cal-form-error" role="alert">{error}</p>}
      <div className="cal-modal-actions"><button className="cal-button" type="button" onClick={onClose}>Cancel</button><button className="cal-button cal-button-primary" disabled={mode === 'library' && !selectedFood}>{entry ? 'Save changes' : 'Add to journal'}<Icon name="check" size={17}/></button></div>
    </form>
  </>;
}

function FoodEditor({ food, readPhotos, onSave, onClose }: { food?: Food; readPhotos: (input: FoodPhotoRequest, signal: AbortSignal) => Promise<FoodPhotoDraft>; onSave: (food: Food) => void; onClose: () => void }) {
  const [name, setName] = useState(food?.name ?? '');
  const [baseQuantity, setBaseQuantity] = useState(String(food?.baseQuantity ?? 100));
  const [unit, setUnit] = useState<Food['unit'] | ''>(food?.unit ?? 'g');
  const [state, setState] = useState<Food['state']>(food?.state ?? 'As served');
  const [nutrients, setNutrients] = useState({ carbs: String(food?.carbs ?? ''), protein: String(food?.protein ?? ''), fat: String(food?.fat ?? ''), fiber: String(food?.fiber ?? '') });
  const hasNutrients = Object.values(nutrients).some(value => value.trim() !== '');
  const calories = energyFromMacros({ carbs: Math.max(0, Number(nutrients.carbs) || 0), protein: Math.max(0, Number(nutrients.protein) || 0), fat: Math.max(0, Number(nutrients.fat) || 0), fiber: Math.max(0, Number(nutrients.fiber) || 0) });
  const [error, setError] = useState('');
  const [photoBusy, setPhotoBusy] = useState(false);
  const [photoDraft, setPhotoDraft] = useState<FoodPhotoDraft | null>(null);
  function applyPhotos(raw: FoodPhotoDraft) {
    const draft = normalizeFoodPhotoDraft(raw);
    setName(draft.name ?? ''); setBaseQuantity(draft.baseQuantity === null ? '' : String(draft.baseQuantity));
    setUnit(draft.unit ?? ''); setState(draft.state ?? 'As served');
    setNutrients({ protein: draft.protein === null ? '' : String(draft.protein), carbs: draft.carbs === null ? '' : String(draft.carbs), fat: draft.fat === null ? '' : String(draft.fat), fiber: draft.fiber === null ? '' : String(draft.fiber) });
    setPhotoDraft(draft); setError('');
  }
  function nutrient(key: keyof typeof nutrients, value: string) { setNutrients(current => ({ ...current, [key]: value })); }
  function submit(event: FormEvent) {
    event.preventDefault();
    if (photoBusy) return;
    if (!name.trim() || !unit || !Number.isFinite(Number(baseQuantity)) || Number(baseQuantity) <= 0 || !Number.isFinite(calories) || calories > 1e8 || Object.values(nutrients).some(value => !Number.isFinite(Number(value)) || Number(value) < 0)) { setError('Add a food name, a label quantity greater than zero, a unit, and valid nutrition values.'); return; }
    onSave({ id: food?.id ?? newId(), name: name.trim(), baseQuantity: Number(baseQuantity), unit, state, carbs: Number(nutrients.carbs), protein: Number(nutrients.protein), fat: Number(nutrients.fat), fiber: Number(nutrients.fiber), calories });
  }
  return <>
    <ModalHeader title={food ? 'Edit a favourite' : 'Add an everyday food'} eyebrow="FOOD LIBRARY" onClose={onClose}/>
    <form className="cal-form" onSubmit={submit}>
      {!food && <FoodPhotoImport readPhotos={readPhotos} onDraft={applyPhotos} onBusyChange={setPhotoBusy}/>}
      <fieldset className="cal-form cal-food-editor-fields" disabled={photoBusy}>
        <label>Food name<input value={name} onChange={event => setName(event.target.value)} placeholder="e.g. Greek yoghurt" required maxLength={160} autoFocus/></label>
        <div className="cal-form-grid cal-form-grid-three">
          <NumericField label="Label quantity" value={baseQuantity} onChange={setBaseQuantity} unit="" min={0.001}/>
          <label>Unit<select value={unit} onChange={event => { const nextUnit = event.target.value as Food['unit']; setUnit(nextUnit); if (!food && !photoDraft) setBaseQuantity(nextUnit === 'piece' ? '1' : '100'); }} required><option value="" disabled>Choose unit</option><option value="g">grams (g)</option><option value="ml">millilitres (ml)</option><option value="piece">piece</option></select></label>
          <label>State<select value={state} onChange={event => setState(event.target.value as Food['state'])}><option>As served</option><option>Raw</option><option>Cooked</option></select></label>
        </div>
        <div className="cal-form-section-label"><strong>Nutrition per {baseQuantity || '—'} {unitLabel(unit, Number(baseQuantity))}</strong><p>Enter the label’s values in grams. Leave blank for zero.</p></div>
        <div className="cal-form-grid"><NumericField label="Protein" value={nutrients.protein} onChange={value => nutrient('protein', value)} required={false}/><NumericField label="Carbohydrate" value={nutrients.carbs} onChange={value => nutrient('carbs', value)} required={false}/><NumericField label="Fat" value={nutrients.fat} onChange={value => nutrient('fat', value)} required={false}/><NumericField label="Fibre" value={nutrients.fiber} onChange={value => nutrient('fiber', value)} required={false}/></div>
        <div className="cal-budget-preview" aria-live="polite"><span>Calculated energy</span><strong><output aria-label="Calculated energy">{hasNutrients ? number(calories, 1) : '—'}</output> kcal</strong><p>4 × protein + 4 × carbohydrate + 9 × fat</p></div>
        {food && <p className="cal-field-hint">Changes apply to future entries. Previously logged nutrition stays as recorded.</p>}
      </fieldset>
      {error && <p className="cal-form-error" role="alert">{error}</p>}
      <div className="cal-modal-actions"><button className="cal-button" type="button" onClick={onClose}>Cancel</button><button className="cal-button cal-button-primary" disabled={photoBusy}>{food ? 'Save changes' : 'Save food'}<Icon name="check" size={17}/></button></div>
    </form>
  </>;
}

function TargetsEditor({ date, currentDay, settings, garminDays, carryIn, carryIsProvisional, onSave, onClose }: { date: string; currentDay: string; settings: DaySettings; garminDays: readonly GarminDay[]; carryIn: number; carryIsProvisional: boolean; onSave: (settings: DaySettings, makeDefault: boolean) => void; onClose: () => void }) {
  const [burned, setBurned] = useState(settings.burned === 0 && settings.burnedSource !== 'manual' ? '' : String(settings.burned));
  const [source, setSource] = useState<'manual' | 'garmin'>(settings.burnedSource ?? 'garmin');
  const [deficit, setDeficit] = useState(String(settings.deficit));
  const [makeDefault, setMakeDefault] = useState(false);
  const [error, setError] = useState('');
  const resolved = effectiveBurned(date, { ...settings, burned: previewNumber(burned), burnedSource: source }, garminDays, currentDay);
  const effective = resolved.burned;
  const previewDeficit = previewNumber(deficit);
  const previewAllowance = adjustedBudget(effective, previewDeficit, carryIn);
  const previewBudget = previewAllowance.budget;
  const nutritionTargets = macroTargets(previewBudget);
  function submit(event: FormEvent) {
    event.preventDefault();
    if ([deficit, ...(source === 'manual' ? [burned] : [])].some(value => value === '' || !Number.isFinite(Number(value)) || Number(value) < 0 || Number(value) > 1e8)) { setError('Enter finite, non-negative values no greater than 100,000,000.'); return; }
    if (source === 'manual' && Number(deficit) > Number(burned)) { setError('For a manual estimate, the deficit target cannot be greater than calories burned.'); return; }
    onSave({ burned: previewNumber(burned), deficit: Number(deficit), protein: DAILY_PROTEIN_GRAMS, burnedSource: source }, makeDefault);
  }
  return <>
    <ModalHeader title="Your daily targets" eyebrow={readableDate(date, { day: 'numeric', month: 'long' }).toUpperCase()} onClose={onClose}/>
    <p className="cal-modal-intro">Choose your deficit target and where calories burned comes from.</p>
    <p className="cal-target-date-hint" id="cal-deficit-date-hint">Changes apply to {readableDate(date, { day: 'numeric', month: 'long', year: 'numeric' })}. Leave the default-target option unchecked to edit only this date.</p>
    <form className="cal-form" onSubmit={submit}>
      <label>Calories burned source<select value={source} onChange={event => setSource(event.target.value as 'manual' | 'garmin')}><option value="garmin">Garmin Connect (automatic)</option><option value="manual">Manual estimate / override</option></select></label>
      {source === 'garmin' ? <>
        <label>{resolved.unavailable ? 'Garmin calories burned unavailable' : resolved.restingEstimate ? 'Garmin resting estimate' : resolved.predicted ? 'Predicted calories burned' : resolved.predictionUnavailable ? 'Garmin calories burned so far' : 'Calories burned from Garmin'}<span className="cal-unit-input"><input type="text" value={resolved.unavailable ? '—' : number(resolved.burned, 1)} readOnly aria-describedby="cal-garmin-target-detail"/><span>kcal</span></span></label>
        <p className="cal-field-hint" id="cal-garmin-target-detail">{burnedFormula(resolved)}.{burnedSyncTime(resolved) ? ` ${burnedSyncTime(resolved)}.` : ''}{!resolved.unavailable && date === currentDay ? ' Today’s value is provisional and updates with imported readings.' : ''}</p>
      </> : <>
        <NumericField label="Manual calories burned" value={burned} onChange={setBurned} unit="kcal"/>
        <p className="cal-field-hint">This value overrides Garmin for this date, even when imported readings are available.</p>
      </>}
      <NumericField label="Deficit target for this date" value={deficit} onChange={setDeficit} unit="kcal" describedBy="cal-deficit-date-hint"/>
      <label>Daily protein target<span className="cal-unit-input"><input type="number" value={DAILY_PROTEIN_GRAMS} readOnly aria-describedby="cal-macro-rule"/><span>g</span></span></label>
      <p className="cal-field-hint" id="cal-macro-rule">Protein: 175 g/day. Fat: 25% of daily budget. Carbs: remaining energy after protein and fat. Fibre: 0.014 g per kcal of daily budget.</p>
      <div className="cal-budget-preview"><span>{carryIsProvisional ? 'Provisional food budget with carry' : resolved.predicted ? 'Predicted food budget with carry' : 'Food budget with carry'}</span><strong>{number(previewBudget, 1)} kcal</strong><div className="cal-target-carry-preview"><span>Base budget<strong>{number(previewAllowance.baseBudget, 1)} kcal</strong></span><span>Carry from earlier days{carryIsProvisional ? ' (provisional)' : ''}<strong>{signedNumber(carryIn)} kcal</strong></span></div><p>Base budget is calories burned minus the deficit target, with a minimum of 0. Carry adjusts the food budget, with the budgeted deficit limited to {number(MAX_DAILY_DEFICIT)} kcal/day.</p><div className="cal-target-macro-preview"><span>Protein<strong>{number(nutritionTargets.protein, 1)} g</strong></span><span>Carbs<strong>{number(nutritionTargets.carbs, 1)} g</strong></span><span>Fat<strong>{number(nutritionTargets.fat, 1)} g</strong></span><span>Fibre<strong>{number(nutritionTargets.fiber, 1)} g</strong></span></div></div>
      {carryIsProvisional && <p className="cal-field-hint">Includes today’s provisional balance; updates as food or Garmin data changes.</p>}
      {!resolved.unavailable && previewAllowance.deferredDebt > 0 && <p className="cal-macro-budget-note">If you eat this {number(previewBudget, 1)} kcal budget, {number(previewAllowance.deferredDebt, 1)} kcal of debt remains for later days.</p>}
      {nutritionTargets.budgetTooLow && <p className="cal-macro-budget-note">Current calorie budget is too low for 175 g protein and 25% fat. Carb target is 0 g.</p>}
      <label className="cal-checkbox"><input type="checkbox" checked={makeDefault} onChange={event => setMakeDefault(event.target.checked)}/><span>Use this deficit target as my default<small>Applies to days without their own saved targets. Manual calories burned applies only to this date; new days use Garmin.</small></span></label>
      {error && <p className="cal-form-error" role="alert">{error}</p>}
      <div className="cal-modal-actions"><button className="cal-button" type="button" onClick={onClose}>Cancel</button><button className="cal-button cal-button-primary">Save targets<Icon name="check" size={17}/></button></div>
    </form>
  </>;
}

function GarminConnection({ date, currentDay, snapshot, loading, disconnecting, error, onRefresh, onDisconnect, onClose }: { date: string; currentDay: string; snapshot: GarminSnapshot; loading: boolean; disconnecting: boolean; error: string; onRefresh: () => void; onDisconnect: () => void; onClose: () => void }) {
  const { status } = snapshot;
  const reading = effectiveBurned(date, { burned: 0, deficit: 0, protein: 0, burnedSource: 'garmin' }, snapshot.days, currentDay);
  const day = reading.day;
  const descriptions: Record<GarminStatus['state'], string> = {
    not_connected: snapshot.days.length ? 'Automatic imports are disconnected. Previously imported calorie readings are kept. Follow the guide to reconnect your account.' : 'One-time sign-in is needed. Follow the guide to connect your Garmin account. Hourly imports are already configured.',
    connected: staleImport(status.last_sync_at) ? 'A new import is needed. Your last successful import was more than two hours ago. Previously imported readings are kept.' : 'Your imported active and resting calories are available across your devices.',
    needs_login: 'Garmin needs you to sign in again. Follow the connection step in the setup guide to resume imports.',
    error: 'The latest Garmin import did not complete. Existing readings are kept. Check the setup guide and try refreshing after the next import.',
  };
  return <>
    <ModalHeader title="Garmin Connect" eyebrow="ACTIVITY IMPORTS" onClose={onClose}/>
    <div className={`cal-garmin-status-card cal-garmin-state-${status.state} ${status.state === 'connected' && staleImport(status.last_sync_at) ? 'cal-garmin-stale' : ''}`}><span className="cal-garmin-symbol"><Icon name="watch" size={25}/></span><div><strong>{garminStatusLabel(status)}</strong><p>{descriptions[status.state]}</p></div></div>
    {error && <p className="cal-form-error cal-garmin-read-error" role="alert">Couldn’t refresh imported data. {error} Your food journal is still available.</p>}
    <dl className="cal-garmin-times"><div><dt>Last successful import</dt><dd>{status.last_sync_at ? syncTime(status.last_sync_at) : 'No successful import yet'}</dd></div><div><dt>Last import attempt</dt><dd>{status.last_attempt_at ? syncTime(status.last_attempt_at) : 'No attempt recorded'}</dd></div></dl>
    {status.last_error_code && <p className="cal-garmin-import-note">{garminErrorMeaning(status.last_error_code)}</p>}
    {day && <div className="cal-garmin-day-card">
      <div><strong>{readableDate(date, { day: 'numeric', month: 'long' })}</strong><span>{reading.predicted ? 'Prediction' : date === currentDay ? 'Provisional total so far' : 'Imported reading'}</span></div>
      <div className="cal-garmin-day-values">
        <span>{date === currentDay ? 'Active today' : 'Active'}<strong>{number(day.active)} <small>kcal</small></strong></span>
        <span>{reading.predicted && reading.restingDay ? <>Resting yesterday<em>{readableDate(reading.restingDay.date, { day: 'numeric', month: 'short' })}</em></> : date === currentDay ? 'Resting so far' : 'Resting'}<strong>{number(reading.predicted && reading.restingDay ? reading.restingDay.resting : day.resting)} <small>kcal</small></strong></span>
        <span>{reading.predicted ? 'Predicted burned' : date === currentDay ? 'Total so far' : 'Total burned'}<strong>{number(reading.burned)} <small>kcal</small></strong></span>
      </div>
      <p>{(reading.predicted || reading.predictionUnavailable) && <span className="cal-garmin-formula">{burnedFormula(reading)}.</span>}{burnedSyncTime(reading)}{status.state === 'not_connected' ? '. Automatic imports are stopped; this reading is kept.' : date === currentDay ? '. Today’s value updates with new imported readings.' : '.'}{reading.predicted && <span className="cal-garmin-raw-total">Garmin total so far: {number(day.total)} kcal.</span>}</p>
    </div>}
    {!day && reading.restingEstimate && reading.restingDay && <div className="cal-garmin-day-card">
      <div><strong>{readableDate(date, { day: 'numeric', month: 'long' })}</strong><span>Garmin resting estimate</span></div>
      <div className="cal-garmin-day-values cal-garmin-estimate-values"><span>Reference day<strong>{readableDate(reading.restingDay.date, { day: 'numeric', month: 'short', year: 'numeric' })}</strong></span><span>Estimated burned<strong>{number(reading.burned)} <small>kcal</small></strong></span></div>
      <p>{burnedFormula(reading)}. {burnedSyncTime(reading)}.{status.state === 'not_connected' ? ' Automatic imports are stopped; the reference reading is kept.' : ''}</p>
    </div>}
    {!day && reading.unavailable && <div className="cal-garmin-day-card cal-garmin-unavailable-card">
      <div><strong>{readableDate(date, { day: 'numeric', month: 'long' })}</strong><span>Burned unavailable</span></div>
      <div className="cal-garmin-unavailable-value">— <small>kcal</small></div>
      <p>{burnedFormula(reading)}. The automatic base budget stays at 0 until a reading is available.</p>
    </div>}
    <div className="cal-garmin-help"><a href="/calories/garmin-setup/" target="_blank" rel="noreferrer">{status.state === 'not_connected' ? 'Set up Garmin imports' : status.state === 'needs_login' ? 'Reconnect Garmin' : 'Open Garmin setup guide'} ↗</a><p>Automatic imports run hourly while connected. Refresh checks for the latest data already imported into your journal’s storage.</p></div>
    {status.state !== 'not_connected' && <button className="cal-text-button cal-danger cal-garmin-disconnect" onClick={onDisconnect} disabled={loading}>{disconnecting ? 'Disconnecting…' : 'Disconnect Garmin'}</button>}
    <div className="cal-modal-actions"><button className="cal-button" onClick={onRefresh} disabled={loading}><Icon name="refresh" size={17}/>{loading ? 'Refreshing…' : error ? 'Try refresh again' : 'Refresh imported data'}</button><button className="cal-button cal-button-primary" onClick={onClose}>Done</button></div>
  </>;
}

function downloadFile(contents: string, filename: string, type: string) {
  const url = URL.createObjectURL(new Blob([contents], { type }));
  const anchor = document.createElement('a'); anchor.href = url; anchor.download = filename; anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function TransferDialog({ data, onImport, onClose }: { data: TrackerData; onImport: (data: TrackerData) => void; onClose: () => void }) {
  const [incoming, setIncoming] = useState<TrackerData | null>(null);
  const [filename, setFilename] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const exportDate = todayISO();
  function exportCSV() {
    const columns: (keyof Entry)[] = ['date', 'meal', 'foodName', 'source', 'quantity', 'unit', 'state', 'carbs', 'protein', 'fat', 'fiber', 'calories'];
    const titles = ['Date', 'Meal', 'Food / meal name', 'Entry type', 'Quantity', 'Unit', 'State', 'Carbs (g)', 'Protein (g)', 'Fat (g)', 'Fibre (g)', 'Calories (kcal)'];
    // Prefix formula-like text to keep exported food names safe to open in a spreadsheet.
    const cell = (value: unknown) => { let text = String(value); if (/^[=+@-]/.test(text)) text = `'${text}`; return `"${text.replaceAll('"', '""')}"`; };
    const csv = [titles.map(cell).join(','), ...[...data.entries].sort((a, b) => a.date.localeCompare(b.date)).map(entry => columns.map(column => cell(column === 'source' ? entry.source === 'manual' ? 'Manual estimate' : 'Food library' : entry.source === 'manual' && (column === 'quantity' || column === 'unit' || column === 'state') ? '' : entry[column])).join(','))].join('\r\n');
    downloadFile('\uFEFF' + csv, `calories-entries-${exportDate}.csv`, 'text/csv;charset=utf-8');
  }
  async function selectFile(file: File | undefined) {
    if (!file) return;
    setBusy(true); setError(''); setIncoming(null); setFilename('');
    try { if (file.size > 20_000_000) throw new Error('This backup is too large. Choose a JSON backup smaller than 20 MB.'); const parsed = normalizeData(JSON.parse(await file.text())); setIncoming(parsed); setFilename(file.name); }
    catch (error) { setError(errorMessage(error)); }
    finally { setBusy(false); }
  }
  return <><ModalHeader title="Keep a copy of your days" eyebrow="BACKUP & IMPORT" onClose={onClose}/><p className="cal-modal-intro">Download your journal whenever you like.</p><div className="cal-transfer-option"><div><strong>Complete backup</strong><p>Foods, food bank portions, journal entries, and targets in a JSON file. Use this to restore your journal.</p></div><button className="cal-button cal-button-small" onClick={() => downloadFile(JSON.stringify(data, null, 2), `calories-backup-${exportDate}.json`, 'application/json')}><Icon name="download" size={16}/>JSON</button></div><div className="cal-transfer-option"><div><strong>Spreadsheet export</strong><p>All logged entries, quantities, and nutrition. Opens in Numbers or Google Sheets.</p></div><button className="cal-button cal-button-small" onClick={exportCSV}><Icon name="download" size={16}/>CSV</button></div><div className="cal-import-section"><h3>Restore a backup</h3><p>Choose a JSON file previously exported from this app.</p><label className="cal-file-label">{busy ? 'Reading backup…' : 'Choose JSON backup'}<input type="file" accept=".json,application/json" onChange={event => void selectFile(event.target.files?.[0])} disabled={busy}/></label>{incoming && <div className="cal-import-confirm"><strong>{filename}</strong><p>{incoming.foods.length} foods · {incoming.foodBank.length} bank portions · {incoming.entries.length} entries</p><p>This will replace your current journal, food library, food bank, and targets on every device. Export a backup first if you want to keep them.</p><button className="cal-button cal-button-danger cal-button-small" onClick={() => onImport(incoming)}>Replace journal with this backup</button></div>}{error && <p className="cal-form-error" role="alert">{error}</p>}</div><div className="cal-modal-actions"><button className="cal-button cal-button-primary" onClick={onClose}>Done</button></div></>;
}
