import { useEffect, useRef, useState, type FormEvent, type ReactNode } from 'react';
import { addDays, dayTotals, emptyData, macrosForFood, normalizeData, todayISO, type DaySettings, type Entry, type Food, type TrackerData } from '../../lib/calories/model';
import { createClient } from '../../lib/calories/client';
import './calories.css';

type View = 'journal' | 'foods' | 'history';
type Phase = 'loading' | 'setup' | 'login' | 'ready';
type SaveState = 'saved' | 'pending' | 'saving' | 'error';
type ModalState = { type: 'entry'; entry?: Entry; foodId?: string } | { type: 'food'; food?: Food; returnToLog?: boolean } | { type: 'targets' } | { type: 'transfer' } | null;
type IconName = 'leaf' | 'plus' | 'left' | 'right' | 'close' | 'edit' | 'trash' | 'settings' | 'download' | 'lock' | 'logout' | 'check' | 'search' | 'calendar' | 'arrow' | 'book';

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
  };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...props}>{paths[name]}</svg>;
}

const number = (value: number, decimals = 0) => new Intl.NumberFormat('en-GB', { maximumFractionDigits: decimals }).format(value);
const readableDate = (date: string, options: Intl.DateTimeFormatOptions) => new Intl.DateTimeFormat('en-GB', options).format(new Date(`${date}T12:00:00`));
const errorMessage = (error: unknown) => error instanceof Error ? error.message : 'Something went wrong. Please try again.';
const newId = () => crypto.randomUUID();
const unitLabel = (unit: string, quantity: number) => unit === 'piece' ? (quantity === 1 ? 'piece' : 'pieces') : unit;

export default function CaloriesApp({ clientOverride }: { clientOverride?: ReturnType<typeof createClient> } = {}) {
  const [client] = useState(() => clientOverride ?? createClient());
  const [phase, setPhase] = useState<Phase>('loading');
  const [data, setData] = useState<TrackerData>(emptyData);
  const dataRef = useRef(data);
  const savedData = useRef(data);
  const [view, setView] = useState<View>('journal');
  const [date, setDate] = useState(todayISO);
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
  const [foodSearch, setFoodSearch] = useState('');
  const [deleteFoodId, setDeleteFoodId] = useState<string | null>(null);
  const [deleteEntryId, setDeleteEntryId] = useState<string | null>(null);
  const queue = useRef<TrackerData | null>(null);
  const saving = useRef(false);
  const blockedSave = useRef(false);
  const refreshing = useRef(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mounted = useRef(true);

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
    if (phase !== 'ready') return;
    let active = true;
    let checking = false;
    async function checkSession(refreshData = false) {
      if (!active || checking || document.visibilityState === 'hidden') return;
      checking = true;
      try {
        const valid = await client.restoreSession();
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
    updateData(current => ({ ...current, days: { ...current.days, [entry.date]: current.days[entry.date] ?? current.defaults }, entries: current.entries.some(item => item.id === entry.id) ? current.entries.map(item => item.id === entry.id ? entry : item) : [...current.entries, entry] }));
    setModal(null);
  }

  function openDay(nextDate: string) { setDate(nextDate); setView('journal'); setDeleteEntryId(null); }

  const dayEntries = data.entries.filter(entry => entry.date === date);
  const totals = dayTotals(dayEntries);
  const targets = data.days[date] ?? data.defaults;
  const budget = Math.max(0, targets.burned - targets.deficit);
  const remaining = budget - totals.calories;
  const deficit = targets.burned - totals.calories;
  const caloriePercent = budget > 0 ? Math.min(100, totals.calories / budget * 100) : 0;
  const proteinPercent = targets.protein > 0 ? Math.min(100, totals.protein / targets.protein * 100) : 0;
  const meals = [...new Set(dayEntries.map(entry => entry.meal))];
  const sortedFoods = [...data.foods].sort((a, b) => a.name.localeCompare(b.name));
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
          {([['journal', 'Journal'], ['foods', 'Foods'], ['history', 'History']] as [View, string][]).map(([key, label]) => <button key={key} className={view === key ? 'is-active' : ''} aria-current={view === key ? 'page' : undefined} onClick={() => { setView(key); setDeleteFoodId(null); setDeleteEntryId(null); }}>{label}</button>)}
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
            <div><div className="cal-eyebrow">{date === todayISO() ? 'TODAY’S JOURNAL' : 'DAILY JOURNAL'}</div><h1>{readableDate(date, { weekday: 'long' })}<span className="cal-heading-dot">.</span></h1><p>{readableDate(date, { day: 'numeric', month: 'long', year: 'numeric' })} <span className="cal-date-divider">/</span> One day at a time.</p></div>
            <div className="cal-date-controls"><button className="cal-icon-button" onClick={() => openDay(addDays(date, -1))} aria-label="Previous day"><Icon name="left"/></button><label className="cal-date-picker"><Icon name="calendar" size={17}/><input type="date" aria-label="Choose journal date" value={date} onChange={event => { if (event.target.value) openDay(event.target.value); }}/></label><button className="cal-icon-button" onClick={() => openDay(addDays(date, 1))} aria-label="Next day"><Icon name="right"/></button>{date !== todayISO() && <button className="cal-text-button cal-today" onClick={() => openDay(todayISO())}>Today</button>}</div>
          </section>

          <section className="cal-dashboard" aria-label="Daily nutrition summary">
            <div className="cal-energy-card">
              <div className="cal-card-top"><span className="cal-eyebrow">ENERGY TODAY</span><button className="cal-icon-button" onClick={() => setModal({ type: 'targets' })} aria-label="Edit daily targets" title="Edit daily targets"><Icon name="settings" size={19}/></button></div>
              <div className="cal-energy-main"><div><div className="cal-energy-value">{number(totals.calories)}<span>kcal</span></div><p>of {number(budget)} kcal daily budget</p></div><div className="cal-ring" style={{ '--cal-progress': `${caloriePercent}%` } as React.CSSProperties}><div><span>{number(caloriePercent)}<small>%</small></span><small>of budget</small></div></div></div>
              <div className="cal-energy-bottom"><span>{remaining >= 0 ? 'Remaining' : 'Above budget'}<strong>{number(Math.abs(remaining))} <small>kcal</small></strong></span><span>Burned<strong>{number(targets.burned)} <small>kcal</small></strong></span><span>{deficit >= 0 ? 'Deficit' : 'Surplus'}<strong>{number(Math.abs(deficit))} <small>kcal</small></strong></span></div>
              <p className="cal-target-note">Deficit target: {number(targets.deficit)} kcal</p>
            </div>
            <div className="cal-nutrients-card">
              <div className="cal-card-top"><span className="cal-eyebrow">THE REST OF THE PICTURE</span><span className="cal-pill">grams</span></div>
              <div className="cal-protein"><div><span>Protein</span><strong>{number(totals.protein, 1)}<small> / {number(targets.protein)} g</small></strong></div><div className="cal-progress-track"><span style={{ width: `${proteinPercent}%` }}/></div><p>{targets.protein > 0 && totals.protein >= targets.protein ? 'Daily protein target reached' : `${number(Math.max(0, targets.protein - totals.protein), 1)} g to your protein target`}</p></div>
              <div className="cal-macro-grid">{([['Carbs', totals.carbs, 'carbs'], ['Fat', totals.fat, 'fat'], ['Fibre', totals.fiber, 'fiber']] as const).map(([name, amount, className]) => <div key={name}><span><i className={`cal-macro-dot ${className}`}/>{name}</span><strong>{number(amount, 1)}<small> g</small></strong></div>)}</div>
            </div>
          </section>

          <section className="cal-journal-section">
            <div className="cal-section-heading"><div><h2>On the menu</h2><p>{dayEntries.length ? `${dayEntries.length} ${dayEntries.length === 1 ? 'entry' : 'entries'} logged today` : 'Your food, meal by meal.'}</p></div><button className="cal-button cal-button-primary" onClick={() => setModal(data.foods.length ? { type: 'entry' } : { type: 'food', returnToLog: true })}><Icon name="plus" size={18}/>Log food</button></div>
            {dayEntries.length === 0 ? <div className="cal-empty-journal"><span className="cal-empty-art"><Icon name="leaf" size={38}/><span/><span/></span><h3>A fresh page for {date === todayISO() ? 'today' : 'this day'}.</h3><p>{data.foods.length ? 'Choose a food, add a quantity, and your totals take care of themselves.' : 'Start with a food you eat. Save its nutrition once, then log it in a few taps.'}</p><button className="cal-text-button" onClick={() => setModal(data.foods.length ? { type: 'entry' } : { type: 'food', returnToLog: true })}>{data.foods.length ? 'Log your first food' : 'Add your first food'}<Icon name="arrow" size={17}/></button></div>
              : <div className="cal-meals">{meals.map((meal, index) => {
                const entries = dayEntries.filter(entry => entry.meal === meal);
                const mealTotals = dayTotals(entries);
                return <div className="cal-meal" key={meal}><div className="cal-meal-heading"><span className="cal-meal-number">{String(index + 1).padStart(2, '0')}</span><h3>{meal}</h3><span>{number(mealTotals.calories)} kcal</span></div><div className="cal-entry-list">{entries.map(entry => <div className="cal-entry" key={entry.id}><div className="cal-entry-info"><strong>{entry.foodName}</strong><span>{number(entry.quantity, 2)} {unitLabel(entry.unit, entry.quantity)}<i/> {entry.state}</span></div><div className="cal-entry-macros"><span><b>P</b> {number(entry.protein, 1)}<small> g</small></span><span><b>C</b> {number(entry.carbs, 1)}<small> g</small></span><span><b>F</b> {number(entry.fat, 1)}<small> g</small></span><span><b>Fibre</b> {number(entry.fiber, 1)}<small> g</small></span></div><strong className="cal-entry-energy">{number(entry.calories)}<small> kcal</small></strong><div className="cal-entry-actions">{deleteEntryId === entry.id ? <><button className="cal-text-button cal-danger" onClick={() => { updateData(current => ({ ...current, entries: current.entries.filter(item => item.id !== entry.id) })); setDeleteEntryId(null); }}>Remove</button><button className="cal-icon-button" onClick={() => setDeleteEntryId(null)} aria-label="Cancel removal"><Icon name="close" size={16}/></button></> : <><button className="cal-icon-button" onClick={() => setModal({ type: 'entry', entry })} aria-label={`Edit ${entry.foodName}`}><Icon name="edit" size={16}/></button><button className="cal-icon-button" onClick={() => setDeleteEntryId(entry.id)} aria-label={`Remove ${entry.foodName}`}><Icon name="trash" size={16}/></button></>}</div></div>)}</div></div>;
              })}</div>}
          </section>
        </>}

        {view === 'foods' && <>
          <section className="cal-page-heading"><div><div className="cal-eyebrow">YOUR OWN LITTLE LIBRARY</div><h1>Everyday foods<span className="cal-heading-dot">.</span></h1><p>Save once. Make your next meal easier.</p></div><button className="cal-button cal-button-primary" onClick={() => setModal({ type: 'food' })}><Icon name="plus" size={18}/>Add food</button></section>
          <div className="cal-library-bar"><label className="cal-search"><Icon name="search" size={19}/><input type="search" placeholder="Find a food…" aria-label="Search food library" value={foodSearch} onChange={event => setFoodSearch(event.target.value)}/></label><span>{data.foods.length} {data.foods.length === 1 ? 'food' : 'foods'} in your library</span></div>
          {visibleFoods.length ? <div className="cal-food-grid">{visibleFoods.map(food => <article className="cal-food-card" key={food.id}><div className="cal-food-heading"><span className="cal-food-state">{food.state}</span><div>{deleteFoodId === food.id ? <button className="cal-text-button" onClick={() => setDeleteFoodId(null)}>Cancel</button> : <button className="cal-icon-button" onClick={() => setModal({ type: 'food', food })} aria-label={`Edit ${food.name}`}><Icon name="edit" size={17}/></button>}</div></div><h2>{food.name}</h2><p>Per {number(food.baseQuantity, 2)} {unitLabel(food.unit, food.baseQuantity)}</p><div className="cal-food-energy"><strong>{number(food.calories)}</strong><span>kcal</span></div><div className="cal-food-macros"><span>Protein<strong>{number(food.protein, 1)} g</strong></span><span>Carbs<strong>{number(food.carbs, 1)} g</strong></span><span>Fat<strong>{number(food.fat, 1)} g</strong></span><span>Fibre<strong>{number(food.fiber, 1)} g</strong></span></div><div className="cal-food-footer">{deleteFoodId === food.id ? <><p>Existing log entries will be kept.</p><button className="cal-text-button cal-danger" onClick={() => { updateData(current => ({ ...current, foods: current.foods.filter(item => item.id !== food.id) })); setDeleteFoodId(null); }}>Remove food</button></> : <><button className="cal-text-button" onClick={() => setModal({ type: 'entry', foodId: food.id })}>Log this food<Icon name="plus" size={16}/></button><button className="cal-icon-button" onClick={() => setDeleteFoodId(food.id)} aria-label={`Remove ${food.name} from library`}><Icon name="trash" size={16}/></button></>}</div></article>)}</div> : <div className="cal-empty-journal cal-library-empty"><span className="cal-empty-art"><Icon name="book" size={36}/></span><h3>{foodSearch ? 'No foods found.' : 'Your favourites start here.'}</h3><p>{foodSearch ? 'Try a different name, or add a new food.' : 'Use the nutrition on your food label to build a library that feels like you.'}</p><button className="cal-text-button" onClick={() => setModal({ type: 'food' })}>Add a food<Icon name="plus" size={17}/></button></div>}
        </>}

        {view === 'history' && <>
          <section className="cal-page-heading"><div><div className="cal-eyebrow">THE DAYS ADD UP</div><h1>Your history<span className="cal-heading-dot">.</span></h1><p>A simple record, whenever you need it.</p></div><button className="cal-button" onClick={() => setModal({ type: 'transfer' })}><Icon name="download" size={18}/>Export journal</button></section>
          {historyDates.length ? <div className="cal-history"><div className="cal-history-header"><span>Day</span><span>Energy</span><span>Protein</span><span>Deficit / surplus</span><span/></div>{historyDates.map(day => { const entries = data.entries.filter(entry => entry.date === day); const dayTotal = dayTotals(entries); const goals = data.days[day] ?? data.defaults; const dayDeficit = goals.burned - dayTotal.calories; return <button className="cal-history-row" key={day} onClick={() => openDay(day)}><span><strong>{readableDate(day, { weekday: 'short', day: 'numeric', month: 'short' })}</strong><small>{entries.length} {entries.length === 1 ? 'entry' : 'entries'} · {day.slice(0, 4)}</small></span><span><strong>{number(dayTotal.calories)}</strong><small> kcal</small></span><span><strong>{number(dayTotal.protein, 1)}</strong><small> g</small></span><span><strong>{number(Math.abs(dayDeficit))}</strong><small> kcal {dayDeficit >= 0 ? 'deficit' : 'surplus'}</small></span><Icon name="right" size={18}/></button>; })}</div> : <div className="cal-empty-journal"><span className="cal-empty-art"><Icon name="calendar" size={36}/></span><h3>Your story is just beginning.</h3><p>Days you log will appear here, with a summary of your energy and nutrients.</p><button className="cal-text-button" onClick={() => openDay(todayISO())}>Go to today<Icon name="arrow" size={17}/></button></div>}
        </>}
        <footer className="cal-footer"><span>A little more aware, every day.</span><span>Personal journal · <a href="/">fangxk2003</a></span></footer>
      </main>

      {modal && <Modal onClose={() => setModal(null)}>{modal.type === 'entry' ? <EntryEditor key={modal.entry?.id ?? modal.foodId ?? 'new-entry'} entry={modal.entry} initialFoodId={modal.foodId} foods={sortedFoods} date={date} onSave={saveEntry} onAddFood={() => setModal({ type: 'food', returnToLog: true })} onClose={() => setModal(null)}/>
        : modal.type === 'food' ? <FoodEditor key={modal.food?.id ?? 'new-food'} food={modal.food} onSave={saveFood} onClose={() => setModal(null)}/>
        : modal.type === 'targets' ? <TargetsEditor date={date} settings={targets} onSave={(settings, makeDefault) => { updateData(current => ({ ...current, days: { ...current.days, [date]: settings }, defaults: makeDefault ? settings : current.defaults })); setModal(null); }} onClose={() => setModal(null)}/>
        : <TransferDialog data={data} onImport={(next) => { updateData(() => next); setModal(null); }} onClose={() => setModal(null)}/>}</Modal>}
    </>}
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

function NumericField({ label, value, onChange, unit = 'g', min = 0, required = true }: { label: string; value: string; onChange: (value: string) => void; unit?: string; min?: number; required?: boolean }) {
  return <label>{label}<span className="cal-unit-input"><input type="number" inputMode="decimal" value={value} onChange={event => onChange(event.target.value)} min={min} max={100000000} step="any" required={required}/><span>{unit}</span></span></label>;
}

function EntryEditor({ entry, initialFoodId, foods, date, onSave, onAddFood, onClose }: { entry?: Entry; initialFoodId?: string; foods: Food[]; date: string; onSave: (entry: Entry) => void; onAddFood: () => void; onClose: () => void }) {
  const initialFood = foods.find(food => food.id === (entry?.foodId ?? initialFoodId)) ?? foods[0];
  const [foodId, setFoodId] = useState(entry?.foodId ?? initialFood?.id ?? '');
  const [quantity, setQuantity] = useState(String(entry?.quantity ?? initialFood?.baseQuantity ?? 100));
  const standardMeals = ['Breakfast', 'Lunch', 'Dinner', 'Snack'];
  const [mealChoice, setMealChoice] = useState(entry ? (standardMeals.includes(entry.meal) ? entry.meal : 'Custom') : 'Breakfast');
  const [customMeal, setCustomMeal] = useState(entry?.meal ?? '');
  const [entryDate, setEntryDate] = useState(entry?.date ?? date);
  const [error, setError] = useState('');
  // Saved entries retain the nutrition recorded at the time, even if their library food changes.
  const selectedFood: Food | undefined = entry && foodId === entry.foodId ? { id: entry.foodId, name: entry.foodName, baseQuantity: entry.quantity, unit: entry.unit, state: entry.state, carbs: entry.carbs, protein: entry.protein, fat: entry.fat, fiber: entry.fiber, calories: entry.calories } : foods.find(food => food.id === foodId);
  const preview = selectedFood ? macrosForFood(selectedFood, Math.max(0, Number(quantity) || 0)) : null;
  function submit(event: FormEvent) {
    event.preventDefault();
    const meal = mealChoice === 'Custom' ? customMeal.trim() : mealChoice;
    if (!selectedFood || !preview || !Number.isFinite(Number(quantity)) || Number(quantity) <= 0 || !meal) { setError('Choose a food, a quantity greater than zero, and a meal.'); return; }
    onSave({ id: entry?.id ?? newId(), date: entryDate, meal, foodId: selectedFood.id, foodName: selectedFood.name, quantity: Number(quantity), unit: selectedFood.unit, state: selectedFood.state, ...preview });
  }
  return <><ModalHeader title={entry ? 'Edit your entry' : 'What’s on the menu?'} eyebrow={entry ? 'DAILY JOURNAL' : 'LOG A FOOD'} onClose={onClose}/><form className="cal-form" onSubmit={submit}><label>Food<select value={foodId} onChange={event => { const food = foods.find(item => item.id === event.target.value); setFoodId(event.target.value); if (food) setQuantity(String(food.baseQuantity)); }} required>{entry && !foods.some(food => food.id === entry.foodId) && <option value={entry.foodId}>{entry.foodName} (saved entry)</option>}{foods.map(food => <option key={food.id} value={food.id}>{food.name} · {food.state}</option>)}</select></label><button className="cal-text-button cal-add-food-inline" type="button" onClick={onAddFood}><Icon name="plus" size={15}/>Add a new food to your library</button><div className="cal-form-grid"><NumericField label="Quantity" value={quantity} onChange={setQuantity} unit={selectedFood?.unit ?? 'g'} min={0.001}/><label>Meal<select value={mealChoice} onChange={event => setMealChoice(event.target.value)}>{standardMeals.map(meal => <option key={meal}>{meal}</option>)}<option>Custom</option></select></label></div>{mealChoice === 'Custom' && <label>Meal name or number<input type="text" value={customMeal} onChange={event => setCustomMeal(event.target.value)} placeholder="e.g. Meal 1, late lunch" required maxLength={80}/></label>}<label>Date<input type="date" value={entryDate} onChange={event => setEntryDate(event.target.value)} required/></label>{preview && <div className="cal-entry-preview"><div><span>This entry</span><strong>{number(preview.calories)} <small>kcal</small></strong></div><div className="cal-preview-macros">{([['Protein', preview.protein], ['Carbs', preview.carbs], ['Fat', preview.fat], ['Fibre', preview.fiber]] as const).map(([name, amount]) => <span key={name}>{name}<strong>{number(amount, 1)} g</strong></span>)}</div><p>{selectedFood?.state} · {entry && foodId === entry.foodId ? 'Uses nutrition saved with this entry.' : `Calculated from your food’s values per ${number(selectedFood?.baseQuantity ?? 0, 2)} ${selectedFood?.unit}.`}</p></div>}{error && <p className="cal-form-error" role="alert">{error}</p>}<div className="cal-modal-actions"><button className="cal-button" type="button" onClick={onClose}>Cancel</button><button className="cal-button cal-button-primary" disabled={!selectedFood}>{entry ? 'Save changes' : 'Add to journal'}<Icon name="check" size={17}/></button></div></form></>;
}

function FoodEditor({ food, onSave, onClose }: { food?: Food; onSave: (food: Food) => void; onClose: () => void }) {
  const [name, setName] = useState(food?.name ?? '');
  const [baseQuantity, setBaseQuantity] = useState(String(food?.baseQuantity ?? 100));
  const [unit, setUnit] = useState<Food['unit']>(food?.unit ?? 'g');
  const [state, setState] = useState<Food['state']>(food?.state ?? 'As served');
  const [nutrients, setNutrients] = useState({ carbs: String(food?.carbs ?? 0), protein: String(food?.protein ?? 0), fat: String(food?.fat ?? 0), fiber: String(food?.fiber ?? 0), calories: food ? String(food.calories) : '' });
  const [error, setError] = useState('');
  function nutrient(key: keyof typeof nutrients, value: string) { setNutrients(current => ({ ...current, [key]: value })); }
  function submit(event: FormEvent) {
    event.preventDefault();
    if (!name.trim() || !Number.isFinite(Number(baseQuantity)) || Number(baseQuantity) <= 0 || Object.values(nutrients).some(value => value === '' || !Number.isFinite(Number(value)) || Number(value) < 0)) { setError('Add a food name, a base quantity greater than zero, and valid nutrition values.'); return; }
    onSave({ id: food?.id ?? newId(), name: name.trim(), baseQuantity: Number(baseQuantity), unit, state, carbs: Number(nutrients.carbs), protein: Number(nutrients.protein), fat: Number(nutrients.fat), fiber: Number(nutrients.fiber), calories: Number(nutrients.calories) });
  }
  return <><ModalHeader title={food ? 'Edit a favourite' : 'Add an everyday food'} eyebrow="FOOD LIBRARY" onClose={onClose}/><form className="cal-form" onSubmit={submit}><label>Food name<input value={name} onChange={event => setName(event.target.value)} placeholder="e.g. Greek yoghurt" required maxLength={160} autoFocus/></label><div className="cal-form-grid cal-form-grid-three"><NumericField label="Label quantity" value={baseQuantity} onChange={setBaseQuantity} unit="" min={0.001}/><label>Unit<select value={unit} onChange={event => { const nextUnit = event.target.value as Food['unit']; setUnit(nextUnit); if (!food) setBaseQuantity(nextUnit === 'piece' ? '1' : '100'); }}><option value="g">grams (g)</option><option value="ml">millilitres (ml)</option><option value="piece">piece</option></select></label><label>State<select value={state} onChange={event => setState(event.target.value as Food['state'])}><option>As served</option><option>Raw</option><option>Cooked</option></select></label></div><div className="cal-form-section-label"><strong>Nutrition per {baseQuantity || '—'} {unitLabel(unit, Number(baseQuantity))}</strong><p>Copy the values from the food label or your own reference.</p></div><div className="cal-form-grid"><NumericField label="Energy" value={nutrients.calories} onChange={value => nutrient('calories', value)} unit="kcal"/><NumericField label="Protein" value={nutrients.protein} onChange={value => nutrient('protein', value)}/><NumericField label="Carbohydrate" value={nutrients.carbs} onChange={value => nutrient('carbs', value)}/><NumericField label="Fat" value={nutrients.fat} onChange={value => nutrient('fat', value)}/><NumericField label="Fibre" value={nutrients.fiber} onChange={value => nutrient('fiber', value)}/></div>{food && <p className="cal-field-hint">Changes apply to future entries. Previously logged nutrition stays as recorded.</p>}{error && <p className="cal-form-error" role="alert">{error}</p>}<div className="cal-modal-actions"><button className="cal-button" type="button" onClick={onClose}>Cancel</button><button className="cal-button cal-button-primary">{food ? 'Save changes' : 'Save food'}<Icon name="check" size={17}/></button></div></form></>;
}

function TargetsEditor({ date, settings, onSave, onClose }: { date: string; settings: DaySettings; onSave: (settings: DaySettings, makeDefault: boolean) => void; onClose: () => void }) {
  const [burned, setBurned] = useState(String(settings.burned));
  const [deficit, setDeficit] = useState(String(settings.deficit));
  const [protein, setProtein] = useState(String(settings.protein));
  const [makeDefault, setMakeDefault] = useState(false);
  const [error, setError] = useState('');
  function submit(event: FormEvent) {
    event.preventDefault();
    if ([burned, deficit, protein].some(value => value === '' || !Number.isFinite(Number(value)) || Number(value) < 0) || Number(deficit) > Number(burned)) { setError('Enter non-negative values. The deficit target cannot be greater than calories burned.'); return; }
    onSave({ burned: Number(burned), deficit: Number(deficit), protein: Number(protein) }, makeDefault);
  }
  return <><ModalHeader title="Your daily targets" eyebrow={readableDate(date, { day: 'numeric', month: 'long' }).toUpperCase()} onClose={onClose}/><p className="cal-modal-intro">Set your own numbers for this day.</p><form className="cal-form" onSubmit={submit}><NumericField label="Calories burned" value={burned} onChange={setBurned} unit="kcal"/><NumericField label="Deficit target" value={deficit} onChange={setDeficit} unit="kcal"/><NumericField label="Protein target" value={protein} onChange={setProtein}/><div className="cal-budget-preview"><span>Daily energy budget</span><strong>{number(Math.max(0, (Number(burned) || 0) - (Number(deficit) || 0)))} kcal</strong><p>Calories burned − deficit target</p></div><label className="cal-checkbox"><input type="checkbox" checked={makeDefault} onChange={event => setMakeDefault(event.target.checked)}/><span>Use these as my default targets<small>Applies to days without their own saved targets.</small></span></label>{error && <p className="cal-form-error" role="alert">{error}</p>}<div className="cal-modal-actions"><button className="cal-button" type="button" onClick={onClose}>Cancel</button><button className="cal-button cal-button-primary">Save targets<Icon name="check" size={17}/></button></div></form></>;
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
    const columns: (keyof Entry)[] = ['date', 'meal', 'foodName', 'quantity', 'unit', 'state', 'carbs', 'protein', 'fat', 'fiber', 'calories'];
    const titles = ['Date', 'Meal', 'Food', 'Quantity', 'Unit', 'State', 'Carbs (g)', 'Protein (g)', 'Fat (g)', 'Fibre (g)', 'Calories (kcal)'];
    // Prefix formula-like text to keep exported food names safe to open in a spreadsheet.
    const cell = (value: unknown) => { let text = String(value); if (/^[=+@-]/.test(text)) text = `'${text}`; return `"${text.replaceAll('"', '""')}"`; };
    const csv = [titles.map(cell).join(','), ...[...data.entries].sort((a, b) => a.date.localeCompare(b.date)).map(entry => columns.map(column => cell(entry[column])).join(','))].join('\r\n');
    downloadFile('\uFEFF' + csv, `calories-entries-${exportDate}.csv`, 'text/csv;charset=utf-8');
  }
  async function selectFile(file: File | undefined) {
    if (!file) return;
    setBusy(true); setError(''); setIncoming(null); setFilename('');
    try { if (file.size > 20_000_000) throw new Error('This backup is too large. Choose a JSON backup smaller than 20 MB.'); const parsed = normalizeData(JSON.parse(await file.text())); setIncoming(parsed); setFilename(file.name); }
    catch (error) { setError(errorMessage(error)); }
    finally { setBusy(false); }
  }
  return <><ModalHeader title="Keep a copy of your days" eyebrow="BACKUP & IMPORT" onClose={onClose}/><p className="cal-modal-intro">Download your journal whenever you like.</p><div className="cal-transfer-option"><div><strong>Complete backup</strong><p>Foods, journal entries, and targets in a JSON file. Use this to restore your journal.</p></div><button className="cal-button cal-button-small" onClick={() => downloadFile(JSON.stringify(data, null, 2), `calories-backup-${exportDate}.json`, 'application/json')}><Icon name="download" size={16}/>JSON</button></div><div className="cal-transfer-option"><div><strong>Spreadsheet export</strong><p>All logged entries, quantities, and nutrition. Opens in Numbers or Google Sheets.</p></div><button className="cal-button cal-button-small" onClick={exportCSV}><Icon name="download" size={16}/>CSV</button></div><div className="cal-import-section"><h3>Restore a backup</h3><p>Choose a JSON file previously exported from this app.</p><label className="cal-file-label">{busy ? 'Reading backup…' : 'Choose JSON backup'}<input type="file" accept=".json,application/json" onChange={event => void selectFile(event.target.files?.[0])} disabled={busy}/></label>{incoming && <div className="cal-import-confirm"><strong>{filename}</strong><p>{incoming.foods.length} foods · {incoming.entries.length} entries</p><p>This will replace your current journal, food library, and targets on every device. Export a backup first if you want to keep them.</p><button className="cal-button cal-button-danger cal-button-small" onClick={() => onImport(incoming)}>Replace journal with this backup</button></div>}{error && <p className="cal-form-error" role="alert">{error}</p>}</div><div className="cal-modal-actions"><button className="cal-button cal-button-primary" onClick={onClose}>Done</button></div></>;
}
