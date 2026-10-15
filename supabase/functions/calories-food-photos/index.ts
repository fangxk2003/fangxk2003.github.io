import { createFoodPhotoHandler } from './handler.ts';

declare const Deno: {
  env: { get(name: string): string | undefined };
  serve(handler: (request: Request) => Promise<Response>): void;
};

let publicKey = Deno.env.get('SUPABASE_ANON_KEY') ?? '';
try { publicKey = JSON.parse(Deno.env.get('SUPABASE_PUBLISHABLE_KEYS') ?? '{}').default || publicKey; } catch { /* Compatibility with older projects. */ }

Deno.serve(createFoodPhotoHandler({
  supabaseUrl: Deno.env.get('SUPABASE_URL') ?? '',
  publicKey,
  apiKey: Deno.env.get('GEMINI_API_KEY') ?? '',
  model: Deno.env.get('CALORIES_PHOTO_MODEL') || 'gemini-3.5-flash-lite',
}));
