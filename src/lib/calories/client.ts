import { emptyData, normalizeData, type TrackerData } from './model.ts';
import { deployedConfiguration } from './config.ts';
import { normalizeGarminSnapshot, type GarminSnapshot } from './garmin.ts';
import { normalizeFoodPhotoDraft, normalizeFoodPhotoRequest, type FoodPhotoDraft, type FoodPhotoRequest } from './food-photos.ts';

export interface Configuration { url: string; key: string; email?: string }
interface Session { accessToken: string; refreshToken: string; userId: string; email: string; expiresAt: number; deadline: number; remember: boolean }
interface StorageLike { getItem(key: string): string | null; setItem(key: string, value: string): void; removeItem(key: string): void }
export interface ClientOptions { url?: string; key?: string; localStorage?: StorageLike; sessionStorage?: StorageLike; fetch?: typeof fetch; now?: () => number }
const CONFIG_KEY = 'calories.connection.v1';
export const REMEMBER_MS = 30 * 24 * 60 * 60 * 1000;

export function validateConfiguration(raw: Configuration): Configuration {
  const parsed = new URL(raw.url.trim());
  const local = ['localhost', '127.0.0.1'].includes(parsed.hostname);
  if ((parsed.protocol !== 'https:' && !(local && parsed.protocol === 'http:')) || parsed.username || parsed.password || parsed.search || parsed.hash || !['', '/'].includes(parsed.pathname)) throw new Error('Enter your Supabase project URL, starting with https://.');
  const key = raw.key.trim();
  if (key.startsWith('sb_secret_')) throw new Error('Use a publishable key, never a secret or service-role key.');
  if (!key.startsWith('sb_publishable_')) {
    try {
      const payload = JSON.parse(atob(key.split('.')[1].replace(/-/g, '+').replace(/_/g, '/')));
      if (payload.role !== 'anon') throw new Error('role');
    } catch { throw new Error('Use your public sb_publishable_ key (or legacy anon key). Secret and service-role keys are not allowed.'); }
  }
  if (key.length < 20 || /\s/.test(key)) throw new Error('Enter a complete publishable key.');
  return { url: parsed.origin, key, email: raw.email?.trim() || '' };
}

export function createClient(options: ClientOptions = {}) {
  const request = options.fetch ?? globalThis.fetch.bind(globalThis);
  const now = options.now ?? Date.now;
  const local = options.localStorage ?? (typeof window !== 'undefined' ? window.localStorage : undefined);
  const tab = options.sessionStorage ?? (typeof window !== 'undefined' ? window.sessionStorage : undefined);
  const buildEnv = (import.meta as ImportMeta & { env?: Record<string, string> }).env;
  let configuration: Configuration | null = null;
  try {
    const defaults = { url: options.url ?? (buildEnv?.PUBLIC_CALORIES_SUPABASE_URL || deployedConfiguration.url), key: options.key ?? (buildEnv?.PUBLIC_CALORIES_SUPABASE_KEY || deployedConfiguration.key) };
    configuration = defaults.url && defaults.key ? validateConfiguration(defaults) : null;
    const stored = local?.getItem(CONFIG_KEY);
    if (stored) configuration = validateConfiguration(JSON.parse(stored));
  } catch { /* Invalid configuration is repaired through Connect storage. */ }
  let session: Session | null = null;
  let revision: number | null | undefined;
  let refreshPromise: Promise<void> | null = null;
  let needsReload = false;
  let saving = false;
  let authGeneration = 0;

  function config(): Configuration {
    if (!configuration) throw new Error('Connect your Supabase project first.');
    return configuration;
  }
  function sessionKey(): string { return `calories.session.v1:${config().url}`; }
  function clearSession() {
    authGeneration += 1;
    if (configuration) { local?.removeItem(sessionKey()); tab?.removeItem(sessionKey()); }
    session = null;
    revision = undefined;
  }
  function storedSession(): Session | null {
    const encoded = tab?.getItem(sessionKey()) || local?.getItem(sessionKey());
    if (!encoded) return null;
    try {
      const value = JSON.parse(encoded) as Session;
      if (!value.accessToken || !value.refreshToken || !value.userId || !Number.isFinite(value.deadline) || !Number.isFinite(value.expiresAt) || typeof value.remember !== 'boolean') return null;
      return value;
    } catch { return null; }
  }
  function persist() {
    if (!session) return;
    const storage = session.remember ? local : tab;
    if (!storage) throw new Error('Browser storage is unavailable. Allow website storage to sign in.');
    storage.setItem(sessionKey(), JSON.stringify(session));
    (session.remember ? tab : local)?.removeItem(sessionKey());
  }
  async function http(path: string, init: RequestInit = {}, token?: string, timeoutMs = 20000): Promise<Response> {
    const headers = new Headers(init.headers);
    headers.set('apikey', config().key);
    if (token) headers.set('Authorization', `Bearer ${token}`);
    if (init.body) headers.set('Content-Type', 'application/json');
    const timeout = AbortSignal.timeout(timeoutMs);
    const signal = init.signal ? AbortSignal.any([init.signal, timeout]) : timeout;
    try { return await request(`${config().url}${path}`, { ...init, headers, cache: 'no-store', signal }); }
    catch {
      if (init.signal?.aborted) throw new DOMException('Photo reading was cancelled.', 'AbortError');
      if (path === '/functions/v1/calories-food-photos') throw new Error('The photo reader could not be reached or took too long. Try again or enter the values manually.');
      throw new Error('Cannot reach your storage. Check your connection and try again. Your changes have not been saved.');
    }
  }
  async function errorFor(response: Response): Promise<Error> {
    const body = await response.json().catch(() => ({}));
    if (response.status === 429) return new Error('Too many sign-in attempts. Please wait a moment and try again.');
    if (body.code === '42P01' || body.code === 'PGRST205') return new Error('Storage setup is incomplete. Run the Calories SQL setup in your Supabase project.');
    if (response.status === 401 || response.status === 403 || body.code === '42501') return new Error('Access was denied. Check your email/password and the owner account in the storage setup.');
    return new Error(body.error_description || body.msg || body.message || 'Your storage request failed. Please try again.');
  }
  async function acceptAuth(response: Response, remember: boolean, deadline: number, stillAllowed: () => boolean = () => true) {
    if (!response.ok) throw await errorFor(response);
    const value = await response.json();
    if (!stillAllowed() || deadline <= now()) {
      const current = storedSession();
      if (current && current.deadline <= now()) clearSession();
      throw new Error('Your session has ended. Sign in again.');
    }
    if (!value.access_token || !value.refresh_token || !value.user?.id) throw new Error('The sign-in response was incomplete.');
    session = { accessToken: value.access_token, refreshToken: value.refresh_token, userId: value.user.id, email: value.user.email || '', expiresAt: (value.expires_at ? value.expires_at * 1000 : now() + value.expires_in * 1000), deadline, remember };
    persist();
  }
  async function refresh(force = false) {
    if (refreshPromise) return refreshPromise;
    const perform = async () => {
      // Re-read rotating credentials after acquiring the cross-tab lock.
      session = storedSession();
      if (!session || session.deadline <= now()) { clearSession(); throw new Error('Your session has ended. Sign in again.'); }
      if (!force && session.expiresAt > now() + 60000) return;
      const previous = session;
      const generation = authGeneration;
      const response = await http('/auth/v1/token?grant_type=refresh_token', { method: 'POST', body: JSON.stringify({ refresh_token: session.refreshToken }) });
      const latest = storedSession();
      if (generation !== authGeneration || !latest || latest.deadline !== previous.deadline || latest.userId !== previous.userId) throw new Error('Your session has ended. Sign in again.');
      if (!response.ok) {
        if (response.status === 400 || response.status === 401 || response.status === 403) clearSession();
        throw await errorFor(response);
      }
      await acceptAuth(response, previous.remember, previous.deadline, () => {
        const stored = storedSession();
        return generation === authGeneration && Boolean(stored && stored.deadline === previous.deadline && stored.userId === previous.userId);
      });
    };
    const locks = typeof navigator !== 'undefined' ? navigator.locks : undefined;
    refreshPromise = (async () => {
      if (locks) await locks.request(`calories-auth:${config().url}`, perform);
      else await perform();
    })();
    try { await refreshPromise; } finally { refreshPromise = null; }
  }
  async function ensureSession(): Promise<Session> {
    session = storedSession();
    if (!session || session.deadline <= now()) { clearSession(); throw new Error('Your session has ended. Sign in again.'); }
    if (session.expiresAt <= now() + 60000) await refresh();
    if (!session) throw new Error('Sign in again.');
    return session;
  }
  async function authenticated(path: string, init: RequestInit = {}, timeoutMs = 20000): Promise<Response> {
    let current = await ensureSession();
    let response = await http(path, init, current.accessToken, timeoutMs);
    if (response.status === 401) {
      await refresh(true);
      current = await ensureSession();
      response = await http(path, init, current.accessToken, timeoutMs);
    }
    return response;
  }

  return {
    get configured() { return Boolean(configuration); },
    getEmail() { return session?.email || configuration?.email || ''; },
    configure(value: Configuration) {
      const next = validateConfiguration(value);
      clearSession();
      configuration = next;
      local?.setItem(CONFIG_KEY, JSON.stringify(next));
      needsReload = false;
    },
    async signIn(email: string, password: string, remember: boolean) {
      clearSession();
      const generation = authGeneration;
      const response = await http('/auth/v1/token?grant_type=password', { method: 'POST', body: JSON.stringify({ email: email.trim(), password }) });
      await acceptAuth(response, remember, now() + REMEMBER_MS, () => generation === authGeneration);
      needsReload = false;
    },
    async restoreSession(): Promise<boolean> {
      if (!configuration) return false;
      session = storedSession();
      if (!session || session.deadline <= now()) { clearSession(); return false; }
      const response = await authenticated('/auth/v1/user');
      if (!response.ok) {
        if (response.status === 401 || response.status === 403) { clearSession(); return false; }
        throw await errorFor(response);
      }
      const user = await response.json();
      if (user.id !== session?.userId) { clearSession(); return false; }
      return true;
    },
    async signOut() {
      const current = storedSession();
      clearSession();
      needsReload = false;
      try {
        if (current) {
          const response = await http('/auth/v1/logout?scope=local', { method: 'POST' }, current.accessToken);
          if (!response.ok) return 'Signed out on this device. The server could not revoke the session; use Supabase to revoke it if needed.';
        }
      } catch { return 'Signed out on this device. The server could not be reached to revoke the session.'; }
      return undefined;
    },
    async load(): Promise<TrackerData> {
      const current = await ensureSession();
      const access = await authenticated('/rest/v1/rpc/calories_access_allowed', { method: 'POST', body: '{}' });
      if (!access.ok) throw await errorFor(access);
      if (await access.json() !== true) throw new Error('This account does not have access, or its 30-day session has expired. Sign in again with the owner account.');
      const response = await authenticated(`/rest/v1/calories_state?user_id=eq.${encodeURIComponent(current.userId)}&select=data,revision`);
      if (!response.ok) throw await errorFor(response);
      const rows = await response.json();
      if (!Array.isArray(rows)) throw new Error('The storage response was invalid.');
      if (rows.length === 0) { revision = null; needsReload = false; return emptyData(); }
      const data = normalizeData(rows[0].data);
      revision = rows[0].revision;
      needsReload = false;
      return data;
    },
    async disconnectGarmin(): Promise<void> {
      const response = await authenticated('/rest/v1/rpc/calories_garmin_disconnect', { method: 'POST', body: '{}' });
      if (!response.ok) throw await errorFor(response);
    },
    async readFoodPhotos(input: FoodPhotoRequest, signal?: AbortSignal): Promise<FoodPhotoDraft> {
      const photos = normalizeFoodPhotoRequest(input);
      signal?.throwIfAborted();
      const response = await authenticated('/functions/v1/calories-food-photos', { method: 'POST', body: JSON.stringify(photos), signal }, 80000);
      signal?.throwIfAborted();
      if (!response.ok) {
        const failure = await response.json().catch(() => ({}));
        const message = typeof failure.error === 'string' && failure.error.length <= 500 ? failure.error : 'The photos could not be read. Try again or enter the food manually.';
        throw new Error(message);
      }
      return normalizeFoodPhotoDraft(await response.json());
    },
    async loadGarmin(): Promise<GarminSnapshot> {
      const current = await ensureSession();
      const access = await authenticated('/rest/v1/rpc/calories_access_allowed', { method: 'POST', body: '{}' });
      if (!access.ok) throw await errorFor(access);
      if (await access.json() !== true) throw new Error('Your session no longer has access. Sign in again.');
      const owner = encodeURIComponent(current.userId);
      const days: unknown[] = [];
      // PostgREST caps a response at 1,000 rows. Page through older journal dates too.
      for (let offset = 0; offset < 100000; offset += 1000) {
        const response = await authenticated(`/rest/v1/calories_garmin_days?owner_id=eq.${owner}&select=date,active,resting,total,synced_at&order=date.desc&limit=1000&offset=${offset}`);
        if (!response.ok) throw await errorFor(response);
        const page = await response.json();
        if (!Array.isArray(page)) throw new Error('The Garmin data response was invalid.');
        days.push(...page);
        if (page.length < 1000) break;
        if (offset === 99000) throw new Error('The Garmin history response is too large.');
      }
      const response = await authenticated(`/rest/v1/calories_garmin_status?owner_id=eq.${owner}&select=state,last_attempt_at,last_sync_at,last_error_code`);
      if (!response.ok) throw await errorFor(response);
      const rows = await response.json();
      if (!Array.isArray(rows) || rows.length !== 1) throw new Error('Garmin storage setup is incomplete, or access has expired.');
      return normalizeGarminSnapshot({ days, status: rows[0] });
    },
    async save(value: TrackerData): Promise<void> {
      const data = normalizeData(value);
      if (revision === undefined || needsReload) throw new Error('Reload your saved data before making more changes. You can export your unsaved changes first.');
      if (saving) throw new Error('A save is already in progress. Wait for it to finish.');
      saving = true;
      try {
        const current = await ensureSession();
        const first = revision === null;
        const path = first ? '/rest/v1/calories_state' : `/rest/v1/calories_state?user_id=eq.${encodeURIComponent(current.userId)}&revision=eq.${revision}`;
        const response = await authenticated(path, { method: first ? 'POST' : 'PATCH', headers: { Prefer: 'return=representation' }, body: JSON.stringify(first ? { user_id: current.userId, data } : { data }) });
        if (!response.ok) {
          // A request may have reached the server even if its response was lost.
          needsReload = true;
          if (response.status === 409) throw new Error('Another device saved changes first. Export any unsaved changes, then reload to see the latest data.');
          throw await errorFor(response);
        }
        const rows = await response.json();
        if (!Array.isArray(rows) || rows.length !== 1 || typeof rows[0].revision !== 'number') {
          needsReload = true;
          throw new Error('Another device saved changes first, or access expired. Export any unsaved changes, then reload your saved data.');
        }
        revision = rows[0].revision;
      } catch (error) { needsReload = true; throw error; }
      finally { saving = false; }
    },
  };
}
