import { useEffect, useId, useRef, useState } from 'react';
import { PHOTO_NUTRIENTS, PHOTO_NUTRIENT_LABELS, type FoodPhotoDraft, type FoodPhotoRequest } from '../../lib/calories/food-photos';
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
const formatNumber = (value: number) => new Intl.NumberFormat('en-GB', { maximumFractionDigits: 2 }).format(value);

function PhotoIcon() {
  return <svg width="23" height="23" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><rect x="3" y="4" width="18" height="16" rx="3"/><circle cx="8" cy="9" r="1.5"/><path d="m4 17 5-5 4 4 3-3 5 5"/></svg>;
}

export default function FoodPhotoImport({ readPhotos, onDraft, onBusyChange }: FoodPhotoImportProps) {
  const id = useId();
  const [slots, setSlots] = useState<Record<PhotoSlot, SlotState>>({ product: emptySlot(), nutrition: emptySlot() });
  const [reading, setReading] = useState(false);
  const [review, setReview] = useState<FoodPhotoDraft | null>(null);
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
    setReview(null);
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
    setReview(null);
    const version = requestVersion.current;
    const controller = new AbortController();
    request.current = controller;
    setReading(true);
    try {
      const draft = await readPhotos({ product: product.image, nutrition: nutrition.image }, controller.signal);
      if (!mounted.current || controller.signal.aborted || requestVersion.current !== version) return;
      onDraft(draft);
      setReview(draft);
      setNotice('Photo suggestions applied. Review and edit the form below before saving the food.');
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

  const missing = review ? [review.name === null ? 'food name' : null, review.baseQuantity === null ? 'label quantity' : null, review.unit === null ? 'unit' : null, review.state === null ? 'state' : null, ...PHOTO_NUTRIENTS.filter(field => review[field] === null).map(field => PHOTO_NUTRIENT_LABELS[field].toLowerCase())].filter((field): field is string => field !== null) : [];

  return <section className="cal-photo-import" aria-labelledby={`${id}-title`}>
    <div className="cal-photo-heading"><span className="cal-photo-icon"><PhotoIcon/></span><div><h3 id={`${id}-title`}>Fill from two photos</h3><p>Show the product name and a clear nutrition table.</p></div></div>
    <div className="cal-photo-grid">
      {photoSlots.map(slot => {
        const { photo, preparing, error: slotError } = slots[slot];
        const inputId = `${id}-${slot}`;
        return <div className={`cal-photo-card${slotError ? ' cal-photo-card-error' : ''}`} key={slot}>
          <label className="cal-photo-slot-title" htmlFor={inputId}>{slotLabels[slot]}</label>
          <div className={`cal-photo-preview${photo ? ' cal-photo-preview-filled' : ''}`}>
            {photo ? <img src={photo.preview} alt={`Selected ${slotLabels[slot].toLowerCase()}`}/> : <span>{preparing ? <span className="cal-photo-spinner" aria-hidden="true"/> : <PhotoIcon/>}<small>{preparing ? 'Preparing…' : 'No photo selected'}</small></span>}
          </div>
          <div className="cal-photo-file-info">{photo ? <><strong title={photo.filename}>{photo.filename}</strong><span>Ready · prepared on this device</span></> : <span>{slot === 'product' ? 'Show the product name and brand' : 'Keep every nutrient row in focus'}</span>}</div>
          <div className="cal-photo-file-actions">
            <label className="cal-photo-choose" htmlFor={inputId}><span>{photo || preparing ? 'Change photo' : 'Choose photo'}</span><input id={inputId} type="file" accept="image/jpeg,image/png,image/webp,image/heic,image/heif,.jpg,.jpeg,.png,.webp,.heic,.heif" aria-label={slotLabels[slot]} aria-describedby={`${inputId}-hint${slotError ? ` ${inputId}-error` : ''}`} onChange={event => { const file = event.currentTarget.files?.[0]; event.currentTarget.value = ''; void selectPhoto(slot, file); }}/></label>
            {(photo || preparing) && <button className="cal-photo-remove" type="button" onClick={() => clearSlot(slot)} aria-label={`Remove ${slotLabels[slot].toLowerCase()}`}>Remove</button>}
          </div>
          <small className="cal-photo-format-hint" id={`${inputId}-hint`}>JPEG, PNG, WebP or HEIC · up to 15 MB</small>
          {slotError && <p className="cal-photo-error" id={`${inputId}-error`} role="alert">{slotError}</p>}
        </div>;
      })}
    </div>
    <p className="cal-photo-privacy">Photos stay on this device until you choose Fill from photos. Missing nutrients may be estimated using the food’s identity, without a web search.</p>
    <div className="cal-photo-actions"><button className="cal-button cal-button-small cal-photo-fill" type="button" onClick={() => void fillFromPhotos()} disabled={!slots.product.photo || !slots.nutrition.photo || busy}>{reading && <span className="cal-photo-spinner" aria-hidden="true"/>}{reading ? 'Reading photos…' : 'Fill from photos'}</button>{reading && <button className="cal-photo-remove" type="button" onClick={() => cancelReading('Photo reading cancelled. You can try again.')}>Cancel</button>}</div>
    <p className="cal-photo-status" role="status">{busy ? reading ? 'Reading the product and nutrition photos…' : 'Preparing photos on this device…' : notice}</p>
    {error && <p className="cal-photo-error" role="alert">{error}</p>}
    {review && <div className="cal-photo-review">
      <h4>Original photo suggestions</h4>
      <p>These are the original suggestions from the photos. Your edits in the form below may differ.</p>
      <p>{review.name ?? 'Food name not read'}{review.baseQuantity !== null && review.unit !== null ? ` · per ${formatNumber(review.baseQuantity)} ${review.unit === 'piece' && review.baseQuantity !== 1 ? 'pieces' : review.unit}` : ' · label basis needs review'}{review.state ? ` · ${review.state}` : ''}</p>
      <dl className="cal-photo-nutrients">{PHOTO_NUTRIENTS.map(field => <div className={review.estimatedFields.includes(field) ? 'cal-photo-estimated' : ''} key={field}><dt>{PHOTO_NUTRIENT_LABELS[field]}</dt><dd><strong>{review[field] === null ? '—' : `${formatNumber(review[field])} g`}</strong><span>{review[field] === null ? 'Not read' : review.estimatedFields.includes(field) ? 'AI-estimated' : 'From photos'}</span></dd></div>)}</dl>
      {review.estimatedFields.length > 0 && <p className="cal-photo-estimate-note">AI-estimated values are suggestions based on model knowledge. Check them before saving.</p>}
      {missing.length > 0 && <p className="cal-photo-missing"><strong>Not read from photos:</strong> {missing.join(', ')}. Fill in or confirm these fields below; a missing value is not a confirmed zero.</p>}
      {review.notes.length > 0 && <ul className="cal-photo-notes">{review.notes.map((note, index) => <li key={index}>{note}</li>)}</ul>}
      <p className="cal-photo-review-reminder">Review and edit the form below, then choose Save food.</p>
    </div>}
  </section>;
}
