import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type Dispatch, type ReactNode, type SetStateAction } from 'react';
import { listFormDrafts, readFormDraft, removeFormDraft, restoreFormDraft, writeFormDraft, type FormDraftDescriptor, type FormDraftSummary } from '../../lib/calories/form-drafts';
import './form-drafts.css';

export { clearFormDrafts } from '../../lib/calories/form-drafts';
export type { FormDraftDescriptor, FormDraftSummary } from '../../lib/calories/form-drafts';
interface DraftContext {
  scope: string | null;
  drafts: FormDraftSummary[];
  refresh: () => void;
  discard: (key: string) => void;
}
const FormDraftContext = createContext<DraftContext>({ scope: null, drafts: [], refresh: () => {}, discard: () => {} });

export function FormDraftProvider({ scope, children }: { scope: string | null; children: ReactNode }) {
  const [drafts, setDrafts] = useState(() => scope ? listFormDrafts(scope) : []);
  const refresh = useCallback(() => setDrafts(scope ? listFormDrafts(scope) : []), [scope]);
  const discard = useCallback((key: string) => { if (scope) removeFormDraft(scope, key); refresh(); }, [scope, refresh]);
  useEffect(() => {
    refresh();
    window.addEventListener('storage', refresh);
    return () => window.removeEventListener('storage', refresh);
  }, [refresh]);
  const context = useMemo(() => ({ scope, drafts, refresh, discard }), [scope, drafts, refresh, discard]);
  return <FormDraftContext.Provider value={context}>{children}</FormDraftContext.Provider>;
}

export function useFormDrafts() {
  const { drafts, discard } = useContext(FormDraftContext);
  return { drafts, discard };
}

export function usePersistentFormDraft<T extends object>(key: string, seed: T, details: FormDraftDescriptor | ((draft: T) => FormDraftDescriptor)) {
  const context = useContext(FormDraftContext);
  const initialSeed = useRef(seed);
  const description = useRef(details);
  description.current = details;
  const initial = useRef<{ value: T; recovered: boolean; needsUpgrade: boolean } | null>(null);
  if (!initial.current) {
    const stored = context.scope ? readFormDraft(context.scope, key) : null;
    const restored = stored ? restoreFormDraft(seed, stored.value) : null;
    initial.current = { value: restored ?? seed, recovered: restored !== null, needsUpgrade: restored !== null && JSON.stringify(restored) !== JSON.stringify(stored?.value) };
  }
  const [draft, setState] = useState(initial.current.value);
  const valueRef = useRef(draft);
  const [recovered, setRecovered] = useState(initial.current.recovered);
  const [storageError, setStorageError] = useState('');
  const mounted = useRef(true);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
  useEffect(() => {
    if (!initial.current?.needsUpgrade || !context.scope) return;
    initial.current.needsUpgrade = false;
    try {
      writeFormDraft(context.scope, key, valueRef.current, typeof description.current === 'function' ? description.current(valueRef.current) : description.current);
      context.refresh();
    } catch { setStorageError('Draft recovery is unavailable on this device. Keep this form open until you save.'); }
  }, [context.scope, key]);
  useEffect(() => {
    if (!storageError) return;
    const warn = (event: BeforeUnloadEvent) => { event.preventDefault(); event.returnValue = ''; };
    window.addEventListener('beforeunload', warn);
    return () => window.removeEventListener('beforeunload', warn);
  }, [storageError]);

  const clearDraft = () => {
    try { context.discard(key); if (mounted.current) { setStorageError(''); setRecovered(false); } }
    catch { if (mounted.current) setStorageError('Draft recovery is unavailable on this device.'); }
  };
  const setDraft: Dispatch<SetStateAction<T>> = update => {
    const value = typeof update === 'function' ? (update as (current: T) => T)(valueRef.current) : update;
    valueRef.current = value;
    setState(value);
    if (!context.scope) return;
    try {
      if (JSON.stringify(value) === JSON.stringify(initialSeed.current)) context.discard(key);
      else writeFormDraft(context.scope, key, value, typeof description.current === 'function' ? description.current(value) : description.current);
      context.refresh();
      setStorageError('');
    } catch { setStorageError('Draft recovery is unavailable on this device. Keep this form open until you save.'); }
  };
  function field<K extends keyof T>(name: K): [T[K], Dispatch<SetStateAction<T[K]>>] {
    return [draft[name], update => setDraft(current => ({ ...current, [name]: typeof update === 'function' ? (update as (value: T[K]) => T[K])(current[name]) : update }))];
  }
  function discardDraft() {
    clearDraft();
    valueRef.current = initialSeed.current;
    setState(initialSeed.current);
  }
  return { draft, setDraft, field, clearDraft, recovered, discardDraft, storageError };
}

export function DraftRestoredNotice({ recovered, discardDraft, storageError, disabled = false }: { recovered: boolean; discardDraft: () => void; storageError?: string; disabled?: boolean }) {
  return <>{recovered && <div className="cal-draft-restored" role="status"><span>Draft restored</span><button className="cal-text-button" type="button" onClick={discardDraft} disabled={disabled}>Discard draft</button></div>}{storageError && <p className="cal-form-error" role="status">{storageError}</p>}</>;
}
