import { MAX_PHOTO_REQUEST_BYTES, normalizeFoodPhotoDraft, normalizeFoodPhotoRequest, type FoodPhotoRequest } from '../../../src/lib/calories/food-photos.ts';
import { normalizeMealPhotoDraft, normalizeMealPhotoRequest, type MealPhotoRequest } from '../../../src/lib/calories/meal-photos.ts';
import { extractFoodPhotos, PhotoProviderError } from './gemini.ts';
import { estimateMealPhoto } from './meal-gemini.ts';

export interface FoodPhotoServerOptions {
  supabaseUrl: string;
  publicKey: string;
  apiKey: string;
  model?: string;
  fetch?: typeof fetch;
  extract?: typeof extractFoodPhotos;
  estimate?: typeof estimateMealPhoto;
}
const ALLOWED_ORIGINS = new Set(['https://fangxk2003.github.io', 'http://127.0.0.1:4321', 'http://localhost:4321']);

async function readBody(request: Request, signal: AbortSignal): Promise<unknown> {
  const length = Number(request.headers.get('content-length'));
  if (Number.isFinite(length) && length > MAX_PHOTO_REQUEST_BYTES) throw new RangeError('Photo upload is too large.');
  if (!request.body) throw new SyntaxError('Choose both photos first.');
  const reader = request.body.getReader();
  const abort = () => { void reader.cancel().catch(() => {}); };
  signal.addEventListener('abort', abort, { once: true });
  let text = '', size = 0;
  const decoder = new TextDecoder();
  try {
    while (true) {
      signal.throwIfAborted();
      const chunk = await reader.read();
      signal.throwIfAborted();
      if (chunk.done) break;
      size += chunk.value.byteLength;
      if (size > MAX_PHOTO_REQUEST_BYTES) { await reader.cancel(); throw new RangeError('Photo upload is too large.'); }
      text += decoder.decode(chunk.value, { stream: true });
    }
    return JSON.parse(text + decoder.decode());
  } finally { signal.removeEventListener('abort', abort); reader.releaseLock(); }
}

/** Only the existing owner/session gate can authorize a provider call. No photos are stored. */
export function createFoodPhotoHandler(options: FoodPhotoServerOptions) {
  const requestUpstream = options.fetch ?? fetch;
  const extract = options.extract ?? extractFoodPhotos;
  const estimate = options.estimate ?? estimateMealPhoto;
  return async (request: Request): Promise<Response> => {
    const origin = request.headers.get('origin');
    const headers = new Headers({ 'Content-Type': 'application/json', 'Cache-Control': 'no-store', Vary: 'Origin' });
    const reply = (status: number, error: string, code: string) => new Response(JSON.stringify({ error, code }), { status, headers });
    if (origin && !ALLOWED_ORIGINS.has(origin)) return reply(403, 'This website cannot use the photo reader.', 'origin_denied');
    if (origin) headers.set('Access-Control-Allow-Origin', origin);
    headers.set('Access-Control-Allow-Headers', 'authorization, apikey, content-type, x-client-info');
    headers.set('Access-Control-Allow-Methods', 'POST, OPTIONS');
    if (request.method === 'OPTIONS') return new Response(null, { status: 204, headers });
    if (request.method !== 'POST') { headers.set('Allow', 'POST, OPTIONS'); return reply(405, 'Use the photo upload form.', 'method_not_allowed'); }
    const authorization = request.headers.get('authorization');
    if (!authorization || !/^Bearer\s+\S+$/i.test(authorization) || authorization.length > 4096) return reply(401, 'Sign in again to read food photos.', 'session_required');
    if (!options.supabaseUrl || !options.publicKey) return reply(503, 'The photo reader is not configured yet.', 'not_configured');
    const signal = AbortSignal.any([request.signal, AbortSignal.timeout(65000)]);
    try {
      const access = await requestUpstream(`${options.supabaseUrl}/rest/v1/rpc/calories_access_allowed`, {
        method: 'POST', headers: { apikey: options.publicKey, Authorization: authorization, 'Content-Type': 'application/json' },
        body: '{}', cache: 'no-store', signal,
      });
      if (access.status === 401) return reply(401, 'Sign in again to read food photos.', 'session_required');
      if (!access.ok) return reply(access.status === 403 ? 403 : 503, 'Your session could not be verified. Sign in again or try later.', 'session_check_failed');
      if (await access.json() !== true) return reply(403, 'Your account or session no longer has access. Sign in again.', 'access_denied');
      if (!options.apiKey) return reply(503, 'Photo reading needs its one-time API setup. You can still enter food manually.', 'not_configured');
      if (!request.headers.get('content-type')?.toLowerCase().startsWith('application/json')) return reply(415, 'Choose photos using the upload form.', 'invalid_content_type');
      let input: { mode: 'meal'; value: MealPhotoRequest } | { mode: 'food'; value: FoodPhotoRequest };
      try {
        const raw = await readBody(request, signal);
        const mode = raw && typeof raw === 'object' && !Array.isArray(raw) ? (raw as Record<string, unknown>).mode : undefined;
        if (mode !== undefined && mode !== 'meal') return reply(400, 'Choose a supported photo reader.', 'invalid_mode');
        input = mode === 'meal'
          ? { mode: 'meal', value: normalizeMealPhotoRequest(raw) }
          : { mode: 'food', value: normalizeFoodPhotoRequest(raw) };
      }
      catch (error) {
        if (signal.aborted) throw error;
        if (error instanceof RangeError) return reply(413, 'The photos are too large. Choose smaller images.', 'photos_too_large');
        return reply(400, 'Choose clear JPEG, PNG or WebP photos, no larger than 2 MB each after preparation.', 'invalid_photos');
      }
      const providerOptions = { apiKey: options.apiKey, model: options.model, signal };
      const draft = input.mode === 'meal'
        ? normalizeMealPhotoDraft(await estimate(input.value, providerOptions))
        : normalizeFoodPhotoDraft(await extract(input.value, providerOptions));
      return new Response(JSON.stringify(draft), { headers });
    } catch (error) {
      if (signal.aborted) return reply(504, 'Photo reading timed out or was cancelled. Try again with clearer photos.', 'reader_timeout');
      if (error instanceof PhotoProviderError) {
        const status = [422, 429, 502, 503, 504].includes(error.status) ? error.status : 502;
        return reply(status, error.message, error.code);
      }
      return reply(502, 'The photos could not be read. Try again or enter the values manually.', 'reader_failed');
    }
  };
}
