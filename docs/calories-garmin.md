# Garmin cloud sync

The journal imports Garmin's daily total (active plus resting calories) into separate Supabase rows. Editable food data stays in `calories_state`, so background imports cannot overwrite a device's journal edits. An explicit `burnedSource: 'manual'` wins over imports. For today, predicted burned calories equal today's imported active calories plus the exact previous day's imported resting calories. This prediction drives the budget and deficit and is labeled in the journal, targets, history, and Garmin details. Completed days use Garmin's recorded total. If yesterday is missing, the current partial total is labeled as such; if today's import is missing, the saved estimate is used. Predictions are computed in the interface and never overwrite imported totals. The interface rechecks the current date at midnight and when it regains focus.

## Deployment

Run `backend/calories/garmin.sql` after the existing setup. This creates private session storage, owner/session-gated read-only calorie/status tables, and a machine RPC accessible only by `service_role`. Provision a 256-bit random worker credential; store its SHA-256 hex digest in `private.garmin_worker.credential_hash`, and its raw value only in the private GitHub repository's encrypted `GARMIN_SYNC_CREDENTIAL` secret and a mode-0600 local bootstrap config. Do not commit it.

Deploy `backend/calories/garmin-edge/index.ts` as `calories-garmin-worker`. Set `verify_jwt = false`: its custom machine authentication checks a scoped credential through the service-only RPC before any operation. The Edge runtime's service-role key remains server-side. Request bodies, tokens, raw Garmin errors, and health values are not logged.

Copy `backend/calories/garmin-sync/{worker.py,requirements.lock}` and `workflow.yml` (as `.github/workflows/sync.yml`) into the private `fangxk2003/calories-garmin-sync` repository. The private repository keeps logs private and avoids GitHub's public-repository 60-day inactivity schedule suspension. `GARMIN_SYNC_ENDPOINT` is a repository variable. Hourly jobs run at minute 17, with read-only GitHub permissions and no cached tokens/artifacts. GitHub schedules can be delayed.

## One-time sign-in

On this Mac the prepared helper is `~/Library/Application Support/Calories Garmin/Connect Garmin.command`. It prompts for Garmin email/password and MFA locally, uploads only an encrypted resumable session, imports the last 30 days, and discards plaintext tokens from process memory when it exits. The password is never uploaded to Supabase/GitHub or written to disk. Open the same helper to reconnect.

Local config contains `{endpoint,credential,timezone}`. The worker also accepts `GARMIN_SYNC_ENDPOINT`, `GARMIN_SYNC_CREDENTIAL`, and `GARMIN_TIMEZONE`. Run `python worker.py connect --config /path/to/config.json --days 30` to bootstrap or `python worker.py sync --days 8` for the scheduled job. `--days` supports up to 366 days for explicit backfill.

The session envelope uses AES-256-GCM with fresh nonces and an HKDF-derived encryption key. Purpose/version, project endpoint, and the fixed owner bind the ciphertext. Supabase stores only ciphertext and the credential digest; GitHub supplies the decryption credential. A server lease serializes scheduled runs and local reconnects. Refreshed tokens are checkpointed separately from imported rows, including after upstream failures.

Daily data contains only calendar date, active/resting/total kcal, and server sync time. Missing/non-finite components are skipped, preserving prior valid readings. `needs_login` clears the expired session and pauses Garmin calls until reconnect; network failures retain it. The owner-only disconnect RPC clears the session and invalidates any active worker lease while preserving calorie history.

## Validation

`npm run test:calories` covers source selection, client/session gates, and the machine handler. `python -m unittest test_worker.py` covers the worker. `backend/calories/garmin-verify.sql` verifies live grants, RLS, machine credentials/leases, malformed batches, token checkpointing, and owner-only disconnect in a rolled-back transaction.

This uses the unofficial [`garminconnect` client](https://github.com/cyberjunky/python-garminconnect/tree/0.3.17). Garmin's [official program](https://developer.garmin.com/gc-developer-program/program-faq/) is intended for enterprise/business integrations. [GitHub schedule behavior](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule) and [Supabase function authentication](https://supabase.com/docs/guides/functions/auth-headers) inform the hosting and auth choices.
