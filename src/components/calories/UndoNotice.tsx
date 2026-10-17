import { useEffect, useRef, useState } from 'react';
import './undo.css';

export interface UndoNoticeProps {
  id: string;
  message: string;
  onUndo: () => Promise<void> | void;
  onDismiss: () => void;
  durationMs?: number;
  busy?: boolean;
  error?: string;
  active?: boolean;
}
interface Deadline {
  generation: number;
  duration: number;
  remaining: number;
  startedAt: number | null;
  timer: ReturnType<typeof setTimeout> | null;
}
const now = () => performance.now();

export function UndoNotice({ id, message, onUndo, onDismiss, durationMs = 20000, busy = false, error = '', active = true }: UndoNoticeProps) {
  const duration = Number.isFinite(durationMs) && durationMs >= 0 ? durationMs : 20000;
  const identity = useRef({ id, generation: 0 });
  if (identity.current.id !== id) identity.current = { id, generation: identity.current.generation + 1 };
  const generation = identity.current.generation;
  const [hovered, setHovered] = useState(false);
  const [focused, setFocused] = useState(false);
  const [hidden, setHidden] = useState(() => typeof document !== 'undefined' && document.hidden);
  const [pending, setPending] = useState<number | null>(null);
  const [failure, setFailure] = useState<{ generation: number; message: string } | null>(null);
  const pendingRef = useRef<number | null>(null);
  const failureRef = useRef<typeof failure>(null);
  const interaction = useRef({ hovered: false, focused: false, hidden });
  const mounted = useRef(true);
  const deadline = useRef<Deadline>({ generation, duration, remaining: duration, startedAt: null, timer: null });
  const latest = useRef({ generation, busy, error, active, onDismiss });
  latest.current = { generation, busy, error, active, onDismiss };
  const working = busy || pending === generation;
  const displayedError = error.trim() || (failure?.generation === generation ? failure.message : '');
  const paused = !active || hovered || focused || hidden || working || Boolean(displayedError);

  function stop(run = deadline.current) {
    if (run.timer !== null) clearTimeout(run.timer);
    run.timer = null;
    if (run.startedAt !== null) run.remaining = Math.max(0, run.remaining - (now() - run.startedAt));
    run.startedAt = null;
  }
  function isPaused() {
    const current = latest.current;
    return !current.active || current.busy || Boolean(current.error.trim()) || pendingRef.current === current.generation || (failureRef.current?.generation === current.generation && Boolean(failureRef.current.message)) || interaction.current.hovered || interaction.current.focused || interaction.current.hidden;
  }

  useEffect(() => {
    mounted.current = true;
    const visibilityChanged = () => {
      interaction.current.hidden = document.hidden;
      if (document.hidden) stop();
      setHidden(document.hidden);
    };
    visibilityChanged();
    document.addEventListener('visibilitychange', visibilityChanged);
    return () => {
      mounted.current = false;
      stop();
      document.removeEventListener('visibilitychange', visibilityChanged);
    };
  }, []);

  useEffect(() => {
    if (active) return;
    interaction.current.hovered = false;
    interaction.current.focused = false;
    setHovered(false);
    setFocused(false);
  }, [active]);

  useEffect(() => {
    let run = deadline.current;
    if (run.generation !== generation || run.duration !== duration) {
      stop(run);
      run = { generation, duration, remaining: duration, startedAt: null, timer: null };
      deadline.current = run;
    }
    if (paused) return;
    run.startedAt = now();
    const timer = setTimeout(() => {
      if (!mounted.current || latest.current.generation !== generation || deadline.current !== run || run.timer !== timer) return;
      stop(run);
      if (!isPaused()) latest.current.onDismiss();
    }, run.remaining);
    run.timer = timer;
    return () => stop(run);
  }, [generation, duration, paused]);

  async function undo() {
    if (working) return;
    stop();
    const startedGeneration = generation;
    pendingRef.current = startedGeneration;
    failureRef.current = null;
    setPending(startedGeneration);
    setFailure(null);
    try {
      await onUndo();
      if (mounted.current && latest.current.generation === startedGeneration) latest.current.onDismiss();
    } catch (caught) {
      if (!mounted.current || latest.current.generation !== startedGeneration) return;
      const detail = caught instanceof Error && caught.message.trim() && caught.message.length <= 500 ? caught.message : 'Couldn’t undo this action. Try again.';
      const nextFailure = { generation: startedGeneration, message: detail };
      failureRef.current = nextFailure;
      setFailure(nextFailure);
    } finally {
      if (pendingRef.current === startedGeneration) pendingRef.current = null;
      if (mounted.current && latest.current.generation === startedGeneration) setPending(null);
    }
  }

  if (!active) return null;
  return <aside className={`cal-undo-notice${displayedError ? ' cal-undo-has-error' : ''}`} aria-label="Undo recent action" aria-busy={working} onPointerEnter={() => { interaction.current.hovered = true; stop(); setHovered(true); }} onPointerLeave={() => { interaction.current.hovered = false; setHovered(false); }} onFocusCapture={() => { interaction.current.focused = true; stop(); setFocused(true); }} onBlurCapture={event => { if (!event.currentTarget.contains(event.relatedTarget as Node | null)) { interaction.current.focused = false; setFocused(false); } }}>
    <div className="cal-undo-content"><p className="cal-undo-message" role="status" aria-live="polite" aria-atomic="true">{message}</p>{displayedError && <p className="cal-undo-error" role="status" aria-live="polite">{displayedError}</p>}</div>
    <div className="cal-undo-actions"><button className="cal-undo-button" type="button" onClick={() => void undo()} disabled={working}>{working ? 'Undoing…' : 'Undo'}</button><button className="cal-undo-dismiss" type="button" onClick={onDismiss} disabled={working} aria-label="Dismiss undo notice"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" aria-hidden="true"><path d="m6 6 12 12M6 18 18 6"/></svg></button></div>
  </aside>;
}

export default UndoNotice;
