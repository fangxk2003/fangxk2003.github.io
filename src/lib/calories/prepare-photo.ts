import { MAX_PHOTO_BYTES, type FoodPhotoImage } from './food-photos';

const MAX_INPUT_BYTES = 15 * 1024 * 1024;
const MAX_EDGE = 2048;
const supportedTypes = new Set(['image/jpeg', 'image/png', 'image/webp', 'image/heic', 'image/heif', 'image/heic-sequence', 'image/heif-sequence']);

export interface PreparedFoodPhoto {
  image: FoodPhotoImage;
  blob: Blob;
  width: number;
  height: number;
}

export class PhotoPreparationError extends Error {}

function checkAbort(signal: AbortSignal) {
  if (signal.aborted) throw new DOMException('Photo preparation cancelled.', 'AbortError');
}

function loadImage(file: File, signal: AbortSignal): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    checkAbort(signal);
    const image = new Image();
    const url = URL.createObjectURL(file);
    const cleanUp = () => {
      image.onload = null;
      image.onerror = null;
      signal.removeEventListener('abort', abort);
      URL.revokeObjectURL(url);
    };
    const abort = () => {
      cleanUp();
      image.src = '';
      reject(new DOMException('Photo preparation cancelled.', 'AbortError'));
    };
    image.onload = () => { cleanUp(); resolve(image); };
    image.onerror = () => {
      cleanUp();
      reject(new PhotoPreparationError('This browser couldn’t open this photo. Export it as JPEG or PNG, then choose it again.'));
    };
    signal.addEventListener('abort', abort, { once: true });
    image.src = url;
  });
}

function jpegBlob(canvas: HTMLCanvasElement, quality: number, signal: AbortSignal): Promise<Blob> {
  return new Promise((resolve, reject) => {
    checkAbort(signal);
    const abort = () => reject(new DOMException('Photo preparation cancelled.', 'AbortError'));
    signal.addEventListener('abort', abort, { once: true });
    canvas.toBlob(blob => {
      signal.removeEventListener('abort', abort);
      if (signal.aborted) { abort(); return; }
      if (!blob || blob.type !== 'image/jpeg') {
        reject(new PhotoPreparationError('This photo couldn’t be prepared. Try a JPEG or PNG photo.'));
        return;
      }
      resolve(blob);
    }, 'image/jpeg', quality);
  });
}

function base64Data(blob: Blob, signal: AbortSignal): Promise<string> {
  return new Promise((resolve, reject) => {
    checkAbort(signal);
    const reader = new FileReader();
    const cleanUp = () => signal.removeEventListener('abort', abort);
    const abort = () => {
      cleanUp();
      reader.abort();
      reject(new DOMException('Photo preparation cancelled.', 'AbortError'));
    };
    reader.onload = () => {
      cleanUp();
      if (signal.aborted) { abort(); return; }
      const result = typeof reader.result === 'string' ? reader.result : '';
      const data = result.split(',')[1];
      if (!data) { reject(new PhotoPreparationError('This photo couldn’t be prepared. Choose it again.')); return; }
      resolve(data);
    };
    reader.onerror = () => { cleanUp(); reject(new PhotoPreparationError('This photo couldn’t be read. Choose it again.')); };
    signal.addEventListener('abort', abort, { once: true });
    reader.readAsDataURL(blob);
  });
}

/** Decode and re-encode locally so only a resized JPEG, without the original metadata, is sent. */
export async function prepareFoodPhoto(file: File, signal: AbortSignal): Promise<PreparedFoodPhoto> {
  checkAbort(signal);
  if (!file.size) throw new PhotoPreparationError('This photo is empty. Choose another photo.');
  if (file.size > MAX_INPUT_BYTES) throw new PhotoPreparationError('Choose a photo smaller than 15 MB.');
  const type = file.type.toLowerCase();
  if (!supportedTypes.has(type) && !(type === '' && /\.(jpe?g|png|webp|heic|heif)$/i.test(file.name))) {
    throw new PhotoPreparationError('Choose a JPEG, PNG, WebP or HEIC photo. For other formats, export a JPEG or PNG first.');
  }
  const image = await loadImage(file, signal);
  const canvas = document.createElement('canvas');
  try {
    checkAbort(signal);
    if (!image.naturalWidth || !image.naturalHeight) throw new PhotoPreparationError('This photo has no readable image. Choose another photo.');
    const scale = Math.min(1, MAX_EDGE / Math.max(image.naturalWidth, image.naturalHeight));
    let width = Math.max(1, Math.round(image.naturalWidth * scale));
    let height = Math.max(1, Math.round(image.naturalHeight * scale));
    for (let attempt = 0; attempt < 6; attempt++) {
      checkAbort(signal);
      canvas.width = width;
      canvas.height = height;
      const context = canvas.getContext('2d');
      if (!context) throw new PhotoPreparationError('Photos can’t be prepared in this browser. Try another browser, or enter the food details below.');
      context.fillStyle = '#fff';
      context.fillRect(0, 0, width, height);
      context.drawImage(image, 0, 0, width, height);
      for (const quality of [.9, .8, .7, .6, .5]) {
        const blob = await jpegBlob(canvas, quality, signal);
        if (blob.size > MAX_PHOTO_BYTES) continue;
        const data = await base64Data(blob, signal);
        checkAbort(signal);
        return { blob, image: { mimeType: 'image/jpeg', data }, width, height };
      }
      width = Math.max(1, Math.floor(width * .8));
      height = Math.max(1, Math.floor(height * .8));
    }
    throw new PhotoPreparationError('This photo is still too large. Crop it closer to the product or nutrition table, then try again.');
  } finally {
    image.src = '';
    canvas.width = 0;
    canvas.height = 0;
  }
}
