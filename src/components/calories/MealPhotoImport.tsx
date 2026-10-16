import { useEffect, useId, useRef, useState } from 'react';
import type { MealPhotoDraft, MealPhotoRequest } from '../../lib/calories/meal-photos';
import { PhotoPreparationError, prepareFoodPhoto, type PreparedFoodPhoto } from '../../lib/calories/prepare-photo';
import './meal-photo.css';

interface MealPhotoImportProps {
  readPhoto: (input: MealPhotoRequest, signal: AbortSignal) => Promise<MealPhotoDraft>;
  description?: string;
  onDraft: (draft: MealPhotoDraft) => void;
  onBusyChange: (busy: boolean) => void;
}
type SelectedPhoto = PreparedFoodPhoto & { filename: string; preview: string };

function MealPhotoIcon() {
  return <svg width="21" height="21" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><rect x="3" y="4" width="18" height="16" rx="3"/><circle cx="8" cy="9" r="1.5"/><path d="m4 17 5-5 4 4 3-3 5 5"/></svg>;
}

export default function MealPhotoImport({ readPhoto, description, onDraft, onBusyChange }: MealPhotoImportProps) {
  const id = useId();
  const [photo, setPhoto] = useState<SelectedPhoto | null>(null);
  const [preparing, setPreparing] = useState(false);
  const [reading, setReading] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [notes, setNotes] = useState<string[]>([]);
  const mounted = useRef(true);
  const operationVersion = useRef(0);
  const preparation = useRef<AbortController | null>(null);
  const request = useRef<AbortController | null>(null);
  const previewUrl = useRef<string | null>(null);
  const busyCallback = useRef(onBusyChange);
  busyCallback.current = onBusyChange;
  const busy = preparing || reading;

  useEffect(() => { busyCallback.current(busy); }, [busy]);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      operationVersion.current++;
      preparation.current?.abort();
      request.current?.abort();
      if (previewUrl.current) URL.revokeObjectURL(previewUrl.current);
      busyCallback.current(false);
    };
  }, []);

  function cancelWork() {
    operationVersion.current++;
    preparation.current?.abort();
    preparation.current = null;
    request.current?.abort();
    request.current = null;
    setPreparing(false);
    setReading(false);
    setError('');
    setNotice('');
    setNotes([]);
  }

  function removePhoto() {
    cancelWork();
    if (previewUrl.current) URL.revokeObjectURL(previewUrl.current);
    previewUrl.current = null;
    setPhoto(null);
  }

  async function selectPhoto(file?: File) {
    if (!file) return;
    removePhoto();
    const version = operationVersion.current;
    const controller = new AbortController();
    preparation.current = controller;
    setPreparing(true);
    try {
      const prepared = await prepareFoodPhoto(file, controller.signal);
      if (!mounted.current || controller.signal.aborted || operationVersion.current !== version) return;
      const preview = URL.createObjectURL(prepared.blob);
      previewUrl.current = preview;
      setPhoto({ ...prepared, filename: file.name, preview });
    } catch (caught) {
      if (!mounted.current || controller.signal.aborted || operationVersion.current !== version) return;
      setError(caught instanceof PhotoPreparationError ? caught.message : 'This photo couldn’t be prepared. Try a JPEG or PNG photo, or enter the meal details below.');
    } finally {
      if (mounted.current && operationVersion.current === version) {
        preparation.current = null;
        setPreparing(false);
      }
    }
  }

  async function estimateFromPhoto() {
    if (!photo || busy) return;
    cancelWork();
    const version = operationVersion.current;
    const controller = new AbortController();
    request.current = controller;
    setReading(true);
    const hint = description?.trim();
    try {
      const draft = await readPhoto({ photo: photo.image, ...(hint ? { description: hint } : {}) }, controller.signal);
      if (!mounted.current || controller.signal.aborted || operationVersion.current !== version) return;
      if (draft.protein === null && draft.carbs === null && draft.fat === null && draft.fiber === null) {
        setError('Couldn’t estimate the meal amounts in this photo. Try a clearer photo, or enter the amounts below.');
        return;
      }
      onDraft(draft);
      setNotice('Photo estimate — review the amounts before logging.');
      setNotes(draft.notes.slice(0, 2).map(note => note.length > 180 ? `${note.slice(0, 177).trimEnd()}…` : note));
    } catch (caught) {
      if (!mounted.current || controller.signal.aborted || operationVersion.current !== version) return;
      setError(caught instanceof Error && caught.message && caught.message.length <= 500 ? caught.message : 'Couldn’t estimate this meal. Check your connection and try again, or enter the amounts below.');
    } finally {
      if (mounted.current && operationVersion.current === version) {
        request.current = null;
        setReading(false);
      }
    }
  }

  return <section className="cal-meal-photo" aria-label="Estimate a meal from a photo">
    <div className="cal-meal-photo-row">
      <label className="cal-meal-photo-choose" htmlFor={`${id}-photo`} title={photo?.filename}>
        {photo ? <img src={photo.preview} alt=""/> : preparing ? <span className="cal-meal-photo-spinner" aria-hidden="true"/> : <MealPhotoIcon/>}
        <span>Meal photo</span>
        <input id={`${id}-photo`} type="file" accept="image/jpeg,image/png,image/webp,image/heic,image/heif,.jpg,.jpeg,.png,.webp,.heic,.heif" aria-label="Meal photo" aria-describedby={error ? `${id}-error` : undefined} onChange={event => { const file = event.currentTarget.files?.[0]; event.currentTarget.value = ''; void selectPhoto(file); }}/>
      </label>
      {(photo || preparing) && <button className="cal-meal-photo-remove" type="button" aria-label="Remove meal photo" onClick={removePhoto}><span aria-hidden="true">×</span></button>}
      <button className="cal-button cal-button-small cal-meal-photo-estimate" type="button" disabled={!photo || busy} onClick={() => void estimateFromPhoto()}>{reading && <span className="cal-meal-photo-spinner" aria-hidden="true"/>}{reading ? 'Estimating…' : 'Estimate from photo'}</button>
      {reading && <button className="cal-text-button cal-meal-photo-cancel" type="button" onClick={() => { cancelWork(); setNotice('Cancelled.'); }}>Cancel</button>}
    </div>
    <p className="cal-meal-photo-status" role="status">{preparing ? 'Preparing photo…' : reading ? 'Estimating the meal…' : notice}</p>
    {error && <p className="cal-meal-photo-error" id={`${id}-error`} role="alert">{error}</p>}
    {notes.length > 0 && <ul className="cal-meal-photo-notes">{notes.map((note, index) => <li key={index}>{note}</li>)}</ul>}
  </section>;
}
