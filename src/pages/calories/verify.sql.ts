import sql from '../../../backend/calories/verify.sql?raw';
export const prerender = true;
export function GET() {
  return new Response(sql, { headers: { 'Content-Type': 'text/plain; charset=utf-8', 'Content-Disposition': 'attachment; filename="calories-verify.sql"', 'X-Content-Type-Options': 'nosniff' } });
}
