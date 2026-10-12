import sql from '../../../backend/calories/setup.sql?raw';
export const prerender = true;
export function GET() {
  return new Response(sql, { headers: { 'Content-Type': 'text/plain; charset=utf-8', 'Content-Disposition': 'attachment; filename="calories-setup.sql"', 'X-Content-Type-Options': 'nosniff' } });
}
