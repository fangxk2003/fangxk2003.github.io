import { useEffect, useId, useRef, useState } from 'react';
import { type FoodPhotoDraft, type FoodPhotoRequest } from '../../lib/calories/food-photos';
import { PhotoPreparationError, prepareFoodPhoto, type PreparedFoodPhoto } from '../../lib/calories/prepare-photo';

interface FoodPhotoImportProps {
  readPhotos: (input: FoodPhotoRequest, signal: AbortSignal) => Promise<FoodPhotoDraft>;
  onDraft: (draft: FoodPhotoDraft) => void;
  onBusyChange: (busy: boolean) => void;
}
type PhotoSlot = 'product' | 'nutrition';
type SelectedPhoto = PreparedFoodPhoto & { filename: string; preview: string };
type SlotState = { photo: SelectedPhoto | null; preparing: boolean; error: string };
const emptySlot = (): SlotState => ({ photo: null, preparing: false, error: '' });
const slotLabels: Record<PhotoSlot, string> = { product: 'Product photo', nutrition: 'Nutrition table photo' };
const photoSlots: PhotoSlot[] = ['product', 'nutrition'];

function PhotoIcon() {
  return <svg width="23" height="23" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><rect x="3" y="4" width="18" height="16" rx="3"/><circle cx="8" cy="9" r="1.5"/><path d="m4 17 5-5 4 4 3-3 5 5"/></svg>;
}

export default function FoodPhotoImport({ readPhotos, onDraft, onBusyChange }: FoodPhotoImportProps) {
  const id = useId();
  const [slots, setSlots] = useState<Record<PhotoSlot, SlotState>>({ product: emptySlot(), nutrition: emptySlot() });
  const [reading, setReading] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const mounted = useRef(true);
  const preparations = useRef<Partial<Record<PhotoSlot, AbortController>>>({});
  const slotVersions = useRef<Record<PhotoSlot, number>>({ product: 0, nutrition: 0 });
  const request = useRef<AbortController | null>(null);
  const requestVersion = useRef(0);
  const previews = useRef<Partial<Record<PhotoSlot, string>>>({});
  const busyCallback = useRef(onBusyChange);
  busyCallback.current = onBusyChange;
  const busy = reading || slots.product.preparing || slots.nutrition.preparing;

  useEffect(() => { busyCallback.current(busy); }, [busy]);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      requestVersion.current++;
      request.current?.abort();
      for (const controller of Object.values(preparations.current)) controller?.abort();
      for (const preview of Object.values(previews.current)) if (preview) URL.revokeObjectURL(preview);
      busyCallback.current(false);
    };
  }, []);

  function cancelReading(message = '') {
    requestVersion.current++;
    request.current?.abort();
    request.current = null;
    setReading(false);
    setError('');
    setNotice(message);
  }

  function clearSlot(slot: PhotoSlot) {
    cancelReading();
    preparations.current[slot]?.abort();
    delete preparations.current[slot];
    slotVersions.current[slot]++;
    const preview = previews.current[slot];
    if (preview) URL.revokeObjectURL(preview);
    delete previews.current[slot];
    setSlots(current => ({ ...current, [slot]: emptySlot() }));
  }

  async function selectPhoto(slot: PhotoSlot, file?: File) {
    if (!file) return;
    clearSlot(slot);
    const version = slotVersions.current[slot];
    const controller = new AbortController();
    preparations.current[slot] = controller;
    setSlots(current => ({ ...current, [slot]: { photo: null, preparing: true, error: '' } }));
    try {
      const prepared = await prepareFoodPhoto(file, controller.signal);
      if (!mounted.current || controller.signal.aborted || slotVersions.current[slot] !== version) return;
      const preview = URL.createObjectURL(prepared.blob);
      previews.current[slot] = preview;
      setSlots(current => ({ ...current, [slot]: { photo: { ...prepared, filename: file.name, preview }, preparing: false, error: '' } }));
    } catch (caught) {
      if (!mounted.current || controller.signal.aborted || slotVersions.current[slot] !== version) return;
      const message = caught instanceof PhotoPreparationError ? caught.message : 'This photo couldn’t be prepared. Try a JPEG or PNG photo, or enter the food details below.';
      setSlots(current => ({ ...current, [slot]: { photo: null, preparing: false, error: message } }));
    } finally {
      if (preparations.current[slot] === controller) delete preparations.current[slot];
    }
  }

  async function fillFromPhotos() {
    const product = slots.product.photo;
    const nutrition = slots.nutrition.photo;
    if (!product || !nutrition || busy) return;
    cancelReading();
    const version = requestVersion.current;
    const controller = new AbortController();
    request.current = controller;
    setReading(true);
    try {
      const draft = await readPhotos({ product: product.image, nutrition: nutrition.image }, controller.signal);
      if (!mounted.current || controller.signal.aborted || requestVersion.current !== version) return;
      onDraft(draft);
      setNotice('Form filled.');
    } catch (caught) {
      if (!mounted.current || controller.signal.aborted || requestVersion.current !== version) return;
      setError(caught instanceof Error && caught.message && caught.message.length <= 500 ? caught.message : 'Couldn’t read these photos. Check your connection and try again, or enter the food details below.');
    } finally {
      if (mounted.current && requestVersion.current === version) {
        request.current = null;
        setReading(false);
      }
    }
  }

  return <section className="cal-photo-import" aria-label="Fill from photos">
    <div className="cal-photo-grid">
      {photoSlots.map(slot => {
        const { photo, preparing, error: slotError } = slots[slot];
        const inputId = `${id}-${slot}`;
        return <div className="cal-photo-card" key={slot}>
          <label className="cal-photo-choose" htmlFor={inputId} title={photo?.filename}>
            {photo ? <img src={photo.preview} alt=""/> : preparing ? <span className="cal-photo-spinner" aria-hidden="true"/> : <PhotoIcon/>}
            <span>{slot === 'product' ? 'Product photo' : 'Nutrition table'}</span>
            <input id={inputId} type="file" accept="image/jpeg,image/png,image/webp,image/heic,image/heif,.jpg,.jpeg,.png,.webp,.heic,.heif" aria-label={slotLabels[slot]} aria-describedby={slotError ? `${inputId}-error` : undefined} onChange={event => { const file = event.currentTarget.files?.[0]; event.currentTarget.value = ''; void selectPhoto(slot, file); }}/>
          </label>
          {(photo || preparing) && <button className="cal-photo-remove" type="button" onClick={() => clearSlot(slot)} aria-label={`Remove ${slotLabels[slot].toLowerCase()}`}><span aria-hidden="true">×</span></button>}
          {slotError && <p className="cal-photo-error" id={`${inputId}-error`} role="alert">{slotError}</p>}
        </div>;
      })}
    </div>
    <div className="cal-photo-actions"><button className="cal-button cal-button-small cal-photo-fill" type="button" onClick={() => void fillFromPhotos()} disabled={!slots.product.photo || !slots.nutrition.photo || busy}>{reading && <span className="cal-photo-spinner" aria-hidden="true"/>}{reading ? 'Reading photos…' : 'Fill from photos'}</button>{reading && <button className="cal-text-button" type="button" onClick={() => cancelReading('Cancelled.')}>Cancel</button>}<span className="cal-photo-status" role="status">{busy ? reading ? '' : 'Preparing…' : notice}</span></div>
    {error && <p className="cal-photo-error" role="alert">{error}</p>}
  </section>;
}
