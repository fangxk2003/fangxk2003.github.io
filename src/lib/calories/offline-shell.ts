type Browser = Pick<Window, 'navigator' | 'location' | 'isSecureContext'>;

/** Production-only, narrowly scoped registration; online use survives cache failures. */
export async function registerCaloriesOfflineShell(production: boolean, browser?: Browser): Promise<ServiceWorkerRegistration | undefined> {
  if (!production) return;
  const environment = browser ?? window;
  if (!environment.isSecureContext || !environment.location.pathname.startsWith('/calories/') || !('serviceWorker' in environment.navigator)) return;
  try {
    return await environment.navigator.serviceWorker.register('/calories/sw.js', { scope: '/calories/', updateViaCache: 'none' });
  } catch {
    // Storage restrictions or a failed precache must not interrupt the online app.
    return;
  }
}
