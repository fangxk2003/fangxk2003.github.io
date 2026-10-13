# Connect Calories to your Supabase project

The app lives at `https://fangxk2003.github.io/calories/`. GitHub Pages serves the interface; Supabase stores the private food library, daily entries, and targets. Both iPhone and Mac sign in to the same account and read the same saved data.

## 1. Create your one account

1. Create or open a Supabase project in the [Supabase Dashboard](https://supabase.com/dashboard).
2. Under **Authentication → Users**, use **Add user → Create new user** to create your email/password account. Confirm the email when creating it, or complete the email confirmation before signing in. Keep the password in your password manager.
3. Copy that user's **UUID** from its user record. The email address and the UUID serve different purposes: email/password signs you in; the UUID permanently identifies the account allowed to access your tracker.
4. Under **Authentication → Sign In / Providers**, turn off **Allow new users to sign up** and **Allow anonymous sign-ins**. Keep email/password enabled. The app has no registration screen. Supabase documents these controls in its [general Auth configuration](https://supabase.com/docs/guides/auth/general-configuration).

Supabase verifies the password; the app never saves it. Email/password sign-in requires a confirmed user when email confirmation is enabled. See [password authentication](https://supabase.com/docs/guides/auth/passwords).

## 2. Create the private storage

1. Open [`backend/calories/setup.sql`](../backend/calories/setup.sql).
2. The script is already configured for your account UUID. If adapting it for another project, update `owner_uuid` to that project's approved account first.
3. Paste the entire script into your project's **SQL Editor** and run it as the default `postgres` role. It fails before creating anything if the UUID is invalid or the Auth user does not exist.
4. In the project's Data API settings, keep the **`private`** schema out of **Exposed schemas**. The app's table and safe boolean RPC live in `public`.
5. Run [`backend/calories/verify.sql`](../backend/calories/verify.sql) in SQL Editor. It should finish with **“Calories database verification passed; all test changes rolled back.”** It creates temporary test sessions and temporarily changes tracker state in one transaction, then rolls everything back. If an assertion fails, run `ROLLBACK;` before retrying and fix the reported issue.

The setup creates `public.calories_state`, with one JSON document for your food library, entries, daily targets, and default targets. It can be rerun for the same UUID without deleting your records. Changing the configured owner while records exist fails rather than transferring those records.

Database protections apply independently of the interface:

- Anonymous visitors have no table permissions. Other signed-in accounts cannot read or write this tracker.
- The only approved UUID can select its row, insert its first document, and update the `data` column. The browser cannot change ownership, supply revisions, or delete the state row.
- Row Level Security requires both the approved owner and an active Auth session created less than 30 days ago. It rejects revoked sessions and sessions past their explicit `not_after` expiry.
- A database trigger sets the first revision to `1` and increments it on each save. The app updates only the revision it last loaded; another device's newer save causes a conflict instead of an overwrite.
- `public.calories_access_allowed()` is a non-privileged boolean RPC available only to signed-in users. The app checks it before loading, so denied access does not appear as an empty tracker. Its privileged session lookup stays inside the unexposed `private` schema.

These grants and policies follow Supabase's [Row Level Security guidance](https://supabase.com/docs/guides/database/postgres/row-level-security). The session check follows its documented [session validation after sign-out](https://supabase.com/docs/guides/auth/sessions).

## 3. Connect the interface

Open your project's **Connect** dialog, or **Settings → API Keys**, and copy:

- **Project URL**, such as `https://your-project.supabase.co`.
- **Publishable key**, beginning with `sb_publishable_`.

Your provided project URL and publishable key are configured in `src/lib/calories/config.ts`, so both devices connect automatically. Sign in with the account you created. Use **Connection settings** on the sign-in screen only if you want to change the project; such overrides are stored in that browser. Your tracker data stays in Supabase.

The public key identifies the app; your authenticated user token and the database policies control private data access. Only use the publishable key here. Never paste an `sb_secret_` or `service_role` key into the interface, a repository, or a `PUBLIC_` environment variable. The app also accepts a legacy `anon` key for an existing project. See [Supabase API key types](https://supabase.com/docs/guides/getting-started/api-keys).

To override the built-in connection for all devices, add these public build variables:

```dotenv
PUBLIC_CALORIES_SUPABASE_URL=https://your-project.supabase.co
PUBLIC_CALORIES_SUPABASE_KEY=sb_publishable_your_public_key
```

Use a local `.env` file for development; it is ignored by Git. For GitHub Pages, add repository Actions variables with those same names and make them available to the Astro build through the deployment workflow. Rebuild/redeploy after changing them. The values become part of the downloaded JavaScript, as expected for browser connection settings.

## 4. Remember this device

At sign-in, **Remember this device for 30 days** selects persistent browser storage. Its deadline is fixed at sign-in; refreshing the access token does not start a new 30-day period. After that deadline, sign in again. The database independently enforces the same maximum using the Auth session's original creation time.

When the box is unchecked, credentials stay in the current browser tab's session storage. Reloading that tab can keep you signed in; closing it normally ends that stored session. Browser session restoration can restore tab storage, so use **Sign out** when you want to end access immediately.

Signing out clears this browser's stored tokens and attempts to revoke the current Supabase session. If the server is unreachable or rejects revocation, the sign-in screen explains that only local sign-out completed. Your other device remains signed in. Successfully revoked sessions fail the database session check even if their access token has not yet expired. Supabase describes the difference between local and global sign-out in [sign-out scopes](https://supabase.com/docs/guides/auth/signout).

## 5. Verify on iPhone and Mac

1. Sign in on your Mac, add a food and a daily entry, and wait for the saved status.
2. Open the same `/calories/` URL on your iPhone, connect the same project if needed, and sign in to the same account. Verify that the food and entry appear.
3. Add an entry on your iPhone. Reload saved data on your Mac and check the totals.
4. If both devices edit the same older revision, the second save reports a conflict. Export any unsaved changes before reloading the latest saved data.
5. Sign out on one device and confirm it asks for your password again while the other device continues working.
6. On iPhone, Safari's **Share → Add to Home Screen** gives the app a convenient icon. Open that installed view and sign in there if iOS treats its storage separately from Safari.

The screenshot supplies the tracker's fields and starting targets (3,200 kcal burned, 500 kcal deficit, and 175 g protein). It does not include your food database or historical entries. Add the reference values from your spreadsheet through **Foods → Add food**. The import screen restores JSON backups created by this app; it does not directly import Google Sheets. Raw and cooked foods should have separate reference values; changing a food later does not rewrite nutrients already logged in your history.

Use **History → Export journal → JSON** periodically to download a complete backup. The **CSV** option exports logged entries for a spreadsheet. Keep those files somewhere private. The app requires a working network connection to load and save cloud data; a failed save is shown as unsaved.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Sign-in is rejected | Correct email/password, confirmed email, email/password provider enabled, and project not paused. |
| Sign-in works but tracker access is denied | UUID in `setup.sql` matches the signed-in Auth user, the full setup was run, and the session is younger than 30 days. |
| Storage setup is incomplete | `public.calories_state` and `public.calories_access_allowed()` exist in this project; run the full setup script. |
| Save reports another device changed data | Export unsaved edits, reload saved data, then apply the desired changes to the new revision. |
| Device stops remembering you | Website storage may have been cleared, private browsing may be enabled, or the fixed 30-day deadline was reached. Sign in again. |
| Verification reports a permission error | Run the verification as SQL Editor's default `postgres` role, after setup; keep all statements in the same run. |

No live Supabase project is created or changed by editing these repository files. Production sign-in and synchronization can be verified after you connect your own project and run the setup.

## Automatic food energy

The daily macro targets use the journal's food calorie budget (burned calories minus the deficit target). Protein is fixed at 175 g/day (700 kcal); fat receives 25% of the budget, divided by 9 kcal/g; carbs receive the remaining budget after protein and fat, divided by 4 kcal/g. The journal shows consumed grams, target grams, and an uncapped percentage for each. Progress bars stop at 100%; fibre remains a separate total. Carb targets are clamped at zero when the budget cannot accommodate the fixed protein plus 25% fat, with an explanation in the interface. Percentages for zero targets are unavailable rather than divided by zero. Legacy saved protein targets normalize to 175 g, preserving recorded food nutrients.

This journal uses `4 × protein + 4 × carbohydrate + 9 × fat` kcal per reference portion. Fibre is tracked separately and excluded from the app’s energy calculation. New nutrition inputs start empty; any left blank are saved as zero. The food editor updates the calculated energy as you type; there is no energy input. Existing food references are recalculated for future logging, while historical entries retain their recorded nutrients and energy. Rounding is for display only.

## Manual meal estimates

Use **Log food → Manual estimate** for a restaurant meal or another meal without a food-library reference. Enter a description, the meal and date, and estimated protein, carbohydrate, fat and fibre in grams for the entire amount eaten. Nutrition fields start empty; blank values count as zero. Energy calculates automatically using the same 4/4/9 formula. The estimate contributes to daily totals and nutrient percentages, stays editable in the journal, and does not create a food-library item. JSON backups retain the manual estimate marker, and CSV exports identify the entry type.
