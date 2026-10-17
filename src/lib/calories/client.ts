import { emptyData, normalizeData, type TrackerData } from './model.ts';
import { deployedConfiguration } from './config.ts';
import { normalizeGarminSnapshot, type GarminSnapshot } from './garmin.ts';
import { normalizeFoodPhotoDraft, normalizeFoodPhotoRequest, type FoodPhotoDraft, type FoodPhotoRequest } from './food-photos.ts';
import { normalizeMealPhotoDraft, normalizeMealPhotoRequest, type MealPhotoDraft, type MealPhotoRequest } from './meal-photos.ts';
import { JournalConflictError, RecoverableNetworkError, isRecoverableNetworkError, journalCacheKey, journalEqual, mergeJournals, normalizeJournalCache, withJournalLock, type JournalCache, type JournalSnapshot } from './offline.ts';
export { isRecoverableNetworkError } from './offline.ts';
export type { JournalSnapshot } from './offline.ts';

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
  let journalOffline = false;
  let journalSync: Promise<JournalSnapshot> | null = null;
  let lastJournalVersion: string | undefined;
  const observedJournals = new Map<string, TrackerData>();

  function config(): Configuration {
    if (!configuration) throw new Error('Connect your Supabase project first.');
    return configuration;
  }
  function sessionKey(): string { return `calories.session.v1:${config().url}`; }
  function journalScope(): string | null {
    if (!configuration) return null;
    const current = storedSession();
    return current && current.deadline > now() ? `${config().url}:${current.userId}` : null;
  }
  function cacheFor(current: Session, verified = true): JournalCache | null {
    const encoded = local?.getItem(journalCacheKey(config().url, current.userId));
    if (!encoded) return null;
    const cache = normalizeJournalCache(JSON.parse(encoded), config().url, current.userId);
    return verified && cache.verifiedDeadline !== current.deadline ? null : cache;
  }
  function journalSession(): Session {
    const current = storedSession();
    if (!current || current.deadline <= now()) { clearSession(); throw new Error('Your session has ended. Sign in again.'); }
    return current;
  }
  function assertJournalSession(owner: string, deadline: number, generation: number) {
    const current = storedSession();
    if (authGeneration !== generation || !current || current.deadline !== deadline || current.userId !== owner || current.deadline <= now()) throw new Error('Your session has ended. Sign in again.');
  }
  function putJournal(cache: JournalCache, owner: string, deadline: number, generation: number): JournalCache {
    assertJournalSession(owner, deadline, generation);
    if (!local) throw new Error('Browser storage is unavailable. Your changes cannot be kept safely.');
    const localVersion = globalThis.crypto?.randomUUID?.() ?? `${now()}-${Math.random().toString(36).slice(2)}`;
    const next = normalizeJournalCache({ ...cache, localVersion }, config().url, owner);
    try { local.setItem(journalCacheKey(config().url, owner), JSON.stringify(next)); }
    catch { throw new Error('This device could not keep the journal changes. Free some browser storage and try again before closing this page.'); }
    return next;
  }
  function presentJournal(cache: JournalCache, offline = journalOffline): JournalSnapshot {
    const data = normalizeData(cache.working);
    observedJournals.set(cache.localVersion, data);
    if (observedJournals.size > 32) observedJournals.delete(observedJournals.keys().next().value!);
    lastJournalVersion = cache.localVersion;
    return { data: normalizeData(data), pending: !journalEqual(cache.baseline.data, data), offline, conflict: cache.conflict, localVersion: cache.localVersion };
  }
  function cachedGarmin(): GarminSnapshot | null {
    if (!configuration) return null;
    const current = storedSession();
    if (!current || current.deadline <= now() || !cacheFor(current)) return null;
    const encoded = local?.getItem(`calories.garmin.v1:${config().url}:${current.userId}`);
    if (!encoded) return null;
    try {
      const value = JSON.parse(encoded);
      return value.deadline === current.deadline ? normalizeGarminSnapshot(value.snapshot) : null;
    } catch { return null; }
  }
  function clearSession() {
    authGeneration += 1;
    if (configuration) { local?.removeItem(sessionKey()); tab?.removeItem(sessionKey()); }
    session = null;
    revision = undefined;
    lastJournalVersion = undefined;
    observedJournals.clear();
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
    if (typeof navigator !== 'undefined' && navigator.onLine === false) throw new RecoverableNetworkError();
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
      throw new RecoverableNetworkError('Cannot reach your storage. Check your connection and try again. Any staged journal changes are kept on this device.');
    }
  }
  async function errorFor(response: Response): Promise<Error> {
    const body = await response.json().catch(() => ({}));
    if (response.status === 429) return new Error('Too many sign-in attempts. Please wait a moment and try again.');
    if (body.code === '42P01' || body.code === 'PGRST205') return new Error('Storage setup is incomplete. Run the Calories SQL setup in your Supabase project.');
    if (response.status === 401 || response.status === 403 || body.code === '42501') return new Error('Access was denied. Check your email/password and the owner account in the storage setup.');
    return new Error(body.error_description || body.msg || body.message || 'Your storage request failed. Please try again.');
  }
  async function readStorageJson(response: Response): Promise<any> {
    try { return await response.json(); }
    catch (error) {
      // fetch() resolves at headers; an interrupted body is still a transport failure.
      if (error instanceof TypeError || (error instanceof Error && ['AbortError', 'TimeoutError'].includes(error.name))) throw new RecoverableNetworkError();
      // Invalid JSON is a response error, not authorization to use an offline cache.
      throw new Error('The storage response was invalid. Try again when connected.');
    }
  }
  async function acceptAuth(response: Response, remember: boolean, deadline: number, stillAllowed: () => boolean = () => true) {
    if (!response.ok) throw await errorFor(response);
    const value = await readStorageJson(response);
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
      try { await refresh(true); }
      catch (error) {
        // A received rejection cannot be turned into an offline authorization.
        if (isRecoverableNetworkError(error)) throw new Error('Your session could not be verified. Sign in again when connected.');
        throw error;
      }
      current = await ensureSession();
      response = await http(path, init, current.accessToken, timeoutMs);
    }
    return response;
  }

  async function readCloudJournal(): Promise<{ data: TrackerData; revision: number | null }> {
    const current = await ensureSession();
    const access = await authenticated('/rest/v1/rpc/calories_access_allowed', { method: 'POST', body: '{}' });
    if (!access.ok) {
      if (access.status === 401 || access.status === 403) clearSession();
      if (access.status >= 500 || access.status === 429) throw new RecoverableNetworkError();
      throw await errorFor(access);
    }
    if (await readStorageJson(access) !== true) { clearSession(); throw new Error('This account does not have access, or its 30-day session has expired. Sign in again with the owner account.'); }
    const response = await authenticated(`/rest/v1/calories_state?user_id=eq.${encodeURIComponent(current.userId)}&select=data,revision`);
    if (!response.ok) {
      if (response.status === 401 || response.status === 403) clearSession();
      if (response.status >= 500 || response.status === 429) throw new RecoverableNetworkError();
      throw await errorFor(response);
    }
    const rows = await readStorageJson(response);
    if (!Array.isArray(rows) || rows.length > 1) throw new Error('The storage response was invalid.');
    if (!rows.length) return { data: emptyData(), revision: null };
    if (!Number.isSafeInteger(rows[0].revision) || rows[0].revision < 1) throw new Error('The storage revision was invalid.');
    return { data: normalizeData(rows[0].data), revision: rows[0].revision };
  }
  async function syncJournal(): Promise<JournalSnapshot> {
    if (journalSync) return journalSync;
    const current = journalSession();
    const owner = current.userId, deadline = current.deadline, generation = authGeneration;
    const scope = `${config().url}:${owner}`;
    journalSync = withJournalLock(`calories-journal-sync:${scope}`, async () => {
      try {
        // Drain a bounded number of edits made while a previous request was in flight.
        for (let pass = 0; pass < 4; pass++) {
          const cloud = await readCloudJournal();
          assertJournalSession(owner, deadline, generation);
          journalOffline = false;
          const staged = await withJournalLock(`calories-journal-local:${scope}`, () => {
            const previous = cacheFor(journalSession(), false);
            let working = cloud.data;
            try {
              if (previous) {
                let base = previous.baseline.data;
                // Resolve uncertain responses by recognizing the already committed snapshot.
                for (const attempt of previous.attempted ?? []) {
                  try {
                    if (journalEqual(mergeJournals(base, attempt.data, cloud.data), cloud.data)) base = attempt.data;
                  } catch { /* A genuinely divergent attempted write stays a conflict. */ }
                }
                working = mergeJournals(base, previous.working, cloud.data);
              }
            } catch (error) {
              if (!(error instanceof JournalConflictError) || !previous) throw error;
              return { cache: putJournal({ ...previous, verifiedDeadline: deadline, conflict: error.message }, owner, deadline, generation), write: false };
            }
            const attempted = (previous?.attempted ?? []).filter(attempt => attempt.revision === cloud.revision);
            let cache: JournalCache = { format: 1, project: config().url, owner, verifiedDeadline: deadline, baseline: cloud, working, localVersion: '', conflict: null, ...(attempted.length ? { attempted } : {}) };
            const write = !journalEqual(working, cloud.data);
            if (write) {
              if (attempted.length >= 20) throw new Error('Several saves could not be confirmed. Your local changes are kept; reconnect on a stable connection before syncing again.');
              cache.attempted = [...attempted, { data: working, revision: cloud.revision }];
            } else delete cache.attempted;
            cache = putJournal(cache, owner, deadline, generation);
            return { cache, write };
          });
          revision = cloud.revision;
          needsReload = false;
          if (!staged.write) return presentJournal(staged.cache);
          const first = cloud.revision === null;
          const path = first ? '/rest/v1/calories_state' : `/rest/v1/calories_state?user_id=eq.${encodeURIComponent(owner)}&revision=eq.${cloud.revision}`;
          const response = await authenticated(path, { method: first ? 'POST' : 'PATCH', headers: { Prefer: 'return=representation' }, body: JSON.stringify(first ? { user_id: owner, data: staged.cache.working } : { data: staged.cache.working }) });
          assertJournalSession(owner, deadline, generation);
          if (!response.ok) {
            if (response.status === 401 || response.status === 403) { clearSession(); throw await errorFor(response); }
            if (response.status >= 500 || response.status === 429) throw new RecoverableNetworkError();
            if (response.status === 409) continue;
            throw await errorFor(response);
          }
          const rows = await readStorageJson(response);
          if (!Array.isArray(rows) || rows.length !== 1 || !Number.isSafeInteger(rows[0].revision) || rows[0].revision <= (cloud.revision ?? 0)) continue;
          const savedRevision = rows[0].revision as number;
          const saved = await withJournalLock(`calories-journal-local:${scope}`, () => {
            const latest = cacheFor(journalSession());
            if (!latest) throw new Error('The local journal is no longer available. Sign in again.');
            const next = { ...latest, baseline: { data: staged.cache.working, revision: savedRevision }, conflict: null };
            delete next.attempted;
            return putJournal(next, owner, deadline, generation);
          });
          revision = savedRevision;
          const result = presentJournal(saved);
          if (!result.pending) return result;
        }
        const latest = cacheFor(journalSession());
        if (!latest) throw new Error('Reload your journal before syncing.');
        return presentJournal(latest);
      } catch (error) {
        assertJournalSession(owner, deadline, generation);
        const cached = cacheFor(journalSession());
        if (isRecoverableNetworkError(error) && cached) { journalOffline = true; return presentJournal(cached, true); }
        throw error;
      }
    });
    try { return await journalSync; } finally { journalSync = null; }
  }

  return {
    get configured() { return Boolean(configuration); },
    getEmail() { return session?.email || configuration?.email || ''; },
    getJournalScope: journalScope,
    getJournalSnapshot(): JournalSnapshot | null {
      if (!configuration) return null;
      const current = storedSession();
      if (!current || current.deadline <= now()) return null;
      const cached = cacheFor(current);
      return cached ? presentJournal(cached) : null;
    },
    getCachedGarmin: cachedGarmin,
    openJournal: syncJournal,
    syncJournal,
    async stageJournal(value: TrackerData, expectedLocalVersion?: string): Promise<JournalSnapshot> {
      const data = normalizeData(value);
      const current = journalSession(), generation = authGeneration;
      const scope = `${config().url}:${current.userId}`;
      const version = expectedLocalVersion ?? lastJournalVersion;
      const observed = version ? observedJournals.get(version) : undefined;
      if (!observed) throw new Error('Open your journal before making changes. Your local form is still available.');
      return withJournalLock(`calories-journal-local:${scope}`, () => {
        assertJournalSession(current.userId, current.deadline, generation);
        const cached = cacheFor(current);
        if (!cached) throw new Error('Open your saved journal online once before making offline changes.');
        const working = mergeJournals(observed, data, cached.working);
        const next = putJournal({ ...cached, working, conflict: null }, current.userId, current.deadline, generation);
        return presentJournal(next);
      });
    },
    async discardJournalChanges(): Promise<JournalSnapshot> {
      const current = journalSession(), generation = authGeneration;
      const scope = `${config().url}:${current.userId}`;
      const previousVersion = cacheFor(current, false)?.localVersion;
      return withJournalLock(`calories-journal-sync:${scope}`, async () => {
        const cloud = await readCloudJournal(); // Never discard anything before a successful online read.
        const next = await withJournalLock(`calories-journal-local:${scope}`, () => {
          if (cacheFor(journalSession(), false)?.localVersion !== previousVersion) throw new JournalConflictError('the local journal while saved data was being reloaded');
          return putJournal({
          format: 1, project: config().url, owner: current.userId, verifiedDeadline: current.deadline,
          baseline: cloud, working: cloud.data, localVersion: '', conflict: null,
          }, current.userId, current.deadline, generation);
        });
        journalOffline = false; revision = cloud.revision; needsReload = false;
        return presentJournal(next);
      });
    },
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
      let response: Response;
      try { response = await authenticated('/auth/v1/user'); }
      catch (error) {
        const current = storedSession();
        if (isRecoverableNetworkError(error) && current && current.deadline > now() && cacheFor(current)) { journalOffline = true; return true; }
        throw error;
      }
      if (!response.ok) {
        if (response.status === 401 || response.status === 403) { clearSession(); return false; }
        const current = storedSession();
        if ((response.status >= 500 || response.status === 429) && current && current.deadline > now() && cacheFor(current)) { journalOffline = true; return true; }
        throw await errorFor(response);
      }
      let user;
      try { user = await readStorageJson(response); }
      catch (error) {
        const current = storedSession();
        if (isRecoverableNetworkError(error) && current && current.deadline > now() && cacheFor(current)) { journalOffline = true; return true; }
        throw error;
      }
      if (user.id !== session?.userId || !storedSession() || session.deadline <= now()) { clearSession(); return false; }
      return true;
    },
    async signOut() {
      const current = storedSession();
      clearSession();
      journalOffline = false;
      needsReload = false;
      if (current && configuration) {
        await withJournalLock(`calories-journal-local:${config().url}:${current.userId}`, () => {
          local?.removeItem(journalCacheKey(config().url, current.userId));
          local?.removeItem(`calories.garmin.v1:${config().url}:${current.userId}`);
        });
      }
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
      if (await readStorageJson(access) !== true) throw new Error('This account does not have access, or its 30-day session has expired. Sign in again with the owner account.');
      const response = await authenticated(`/rest/v1/calories_state?user_id=eq.${encodeURIComponent(current.userId)}&select=data,revision`);
      if (!response.ok) throw await errorFor(response);
      const rows = await readStorageJson(response);
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
    async estimateMealPhoto(input: MealPhotoRequest, signal?: AbortSignal): Promise<MealPhotoDraft> {
      const meal = normalizeMealPhotoRequest(input);
      signal?.throwIfAborted();
      const response = await authenticated('/functions/v1/calories-food-photos', { method: 'POST', body: JSON.stringify({ mode: 'meal', ...meal }), signal }, 80000);
      signal?.throwIfAborted();
      if (!response.ok) {
        const failure = await response.json().catch(() => ({}));
        const message = typeof failure.error === 'string' && failure.error.length <= 500 ? failure.error : 'The meal could not be estimated. Try another photo or enter the values manually.';
        throw new Error(message);
      }
      return normalizeMealPhotoDraft(await response.json());
    },
    async loadGarmin(): Promise<GarminSnapshot> {
      try {
      const current = await ensureSession();
      const access = await authenticated('/rest/v1/rpc/calories_access_allowed', { method: 'POST', body: '{}' });
      if (!access.ok) {
        if (access.status >= 500 || access.status === 429) throw new RecoverableNetworkError();
        throw await errorFor(access);
      }
      if (await readStorageJson(access) !== true) { clearSession(); throw new Error('Your session no longer has access. Sign in again.'); }
      const owner = encodeURIComponent(current.userId);
      const days: unknown[] = [];
      // PostgREST caps a response at 1,000 rows. Page through older journal dates too.
      for (let offset = 0; offset < 100000; offset += 1000) {
        const response = await authenticated(`/rest/v1/calories_garmin_days?owner_id=eq.${owner}&select=date,active,resting,total,synced_at&order=date.desc&limit=1000&offset=${offset}`);
        if (!response.ok) {
          if (response.status >= 500 || response.status === 429) throw new RecoverableNetworkError();
          throw await errorFor(response);
        }
        const page = await readStorageJson(response);
        if (!Array.isArray(page)) throw new Error('The Garmin data response was invalid.');
        days.push(...page);
        if (page.length < 1000) break;
        if (offset === 99000) throw new Error('The Garmin history response is too large.');
      }
      const response = await authenticated(`/rest/v1/calories_garmin_status?owner_id=eq.${owner}&select=state,last_attempt_at,last_sync_at,last_error_code`);
      if (!response.ok) {
        if (response.status >= 500 || response.status === 429) throw new RecoverableNetworkError();
        throw await errorFor(response);
      }
      const rows = await readStorageJson(response);
      if (!Array.isArray(rows) || rows.length !== 1) throw new Error('Garmin storage setup is incomplete, or access has expired.');
      const snapshot = normalizeGarminSnapshot({ days, status: rows[0] });
      try {
        const stillCurrent = storedSession();
        if (stillCurrent && stillCurrent.userId === current.userId && stillCurrent.deadline === current.deadline && current.deadline > now()) local?.setItem(`calories.garmin.v1:${config().url}:${current.userId}`, JSON.stringify({ deadline: current.deadline, snapshot }));
      } catch { /* A full auxiliary cache must not discard a successful cloud read. */ }
      return snapshot;
      } catch (error) {
        if (isRecoverableNetworkError(error)) { const cached = cachedGarmin(); if (cached) return cached; }
        throw error;
      }
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
        const rows = await readStorageJson(response);
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
