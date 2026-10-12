// This machine endpoint uses a scoped random credential, not a browser JWT.
// Only the service-role RPC can read the private session; it checks the credential's
// SHA-256 digest before any read/write and binds all operations to one owner.
export interface Runtime { url: string; serviceKey: string; request?: typeof fetch }
const MAX_BYTES = 150000;
const reply = (body: unknown, status = 200) => new Response(JSON.stringify(body), {
  status, headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' },
});

export function createHandler(runtime: Runtime) {
  const request = runtime.request ?? fetch;
  return async (req: Request): Promise<Response> => {
    if (req.method !== 'POST') return reply({ error: 'method_not_allowed' }, 405);
    if (!req.headers.get('Content-Type')?.startsWith('application/json')) return reply({ error: 'invalid_request' }, 400);
    if (Number(req.headers.get('Content-Length')) > MAX_BYTES) return reply({ error: 'request_too_large' }, 413);
    try {
      // Stream with a cap so an absent/misleading Content-Length cannot bypass it.
      const reader = req.body?.getReader();
      if (!reader) return reply({ error: 'invalid_request' }, 400);
      const chunks: Uint8Array[] = [];
      let length = 0;
      while (true) {
        const chunk = await reader.read();
        if (chunk.done) break;
        length += chunk.value.byteLength;
        if (length > MAX_BYTES) { await reader.cancel(); return reply({ error: 'request_too_large' }, 413); }
        chunks.push(chunk.value);
      }
      const bytes = new Uint8Array(length);
      let offset = 0;
      for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
      const body = JSON.parse(new TextDecoder().decode(bytes));
      if (!body || typeof body !== 'object' || Array.isArray(body) || !['acquire', 'checkpoint', 'import', 'release'].includes(body.operation)) return reply({ error: 'invalid_request' }, 400);
      if (typeof body.credential !== 'string' || !/^[a-f0-9]{64}$/.test(body.credential)) return reply({ error: 'unauthorized' }, 401);
      const response = await request(`${runtime.url}/rest/v1/rpc/calories_garmin_worker`, {
        method: 'POST', headers: { 'Content-Type': 'application/json', apikey: runtime.serviceKey, Authorization: `Bearer ${runtime.serviceKey}` },
        body: JSON.stringify({ p_request: body }), signal: AbortSignal.timeout(20000),
      });
      const result = await response.json();
      if (!response.ok) {
        // Neither SQL error details nor the request/opaque session are logged.
        if (result.message === 'worker_auth_failed') return reply({ error: 'unauthorized' }, 401);
        if (result.message === 'stale_lease') return reply({ error: 'stale_lease' }, 409);
        if (result.message === 'invalid_request') return reply({ error: 'invalid_request' }, 400);
        return reply({ error: 'storage_unavailable' }, 503);
      }
      return reply(result);
    } catch (error) {
      return reply({ error: error instanceof SyntaxError ? 'invalid_request' : 'storage_unavailable' }, error instanceof SyntaxError ? 400 : 503);
    }
  };
}

// Importing this file in Node tests leaves the Deno entrypoint inactive.
declare const Deno: { env: { get(name: string): string | undefined }; serve(handler: (req: Request) => Promise<Response>): void };
if (typeof Deno !== 'undefined') {
  const url = Deno.env.get('SUPABASE_URL');
  const serviceKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY');
  if (!url || !serviceKey) throw new Error('Missing server configuration');
  Deno.serve(createHandler({ url, serviceKey }));
}
