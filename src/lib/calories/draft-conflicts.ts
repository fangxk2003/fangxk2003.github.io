import { journalEqual } from './offline.ts';

/** Protect a recovered/open edit against overwriting a record changed elsewhere.
 * The proposed state may already be present after a crash between save and draft cleanup. */
export function assertCurrentRecord(current: unknown, originalRecordJson: string, proposed: unknown, label: string): void {
  const subject = typeof label === 'string' && label.trim() && label.trim().length <= 80 ? label.trim() : 'This record';
  const invalid = () => new Error('The original record in this draft cannot be read safely. Your draft is kept; review the saved record before trying again.');
  if (typeof originalRecordJson !== 'string' || !originalRecordJson.trim() || originalRecordJson.length > 65536) throw invalid();
  let original: unknown;
  try { original = JSON.parse(originalRecordJson); } catch { throw invalid(); }
  const validJson = (value: unknown, depth = 0): boolean => {
    if (depth > 32) return false;
    if (value === null || typeof value === 'string' || typeof value === 'boolean') return true;
    if (typeof value === 'number') return Number.isFinite(value);
    if (Array.isArray(value)) return value.every(item => validJson(item, depth + 1));
    return Boolean(value && typeof value === 'object' && Object.getPrototypeOf(value) === Object.prototype && Object.values(value).every(item => validJson(item, depth + 1)));
  };
  if (!validJson(original)) throw invalid();
  try { if (journalEqual(current, original) || journalEqual(current, proposed)) return; }
  catch { /* Unsupported records also fail closed without exposing their contents. */ }
  throw new Error(`${subject} changed since this draft was started. Your draft is kept. Review the current saved record before applying these edits.`);
}
