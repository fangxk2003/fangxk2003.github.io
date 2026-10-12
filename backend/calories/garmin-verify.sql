-- Run after setup.sql + garmin.sql as postgres, as one SQL Editor request.
-- Uses a disposable verification key and session, never production credentials.
-- All key/session/status/day changes are rolled back. On error run ROLLBACK.
begin;

do $$
declare
  owner_uuid constant uuid := '8c0b4ad9-f74f-4951-bb9d-a4a3d4d1f686'::uuid;
  session_uuid uuid := gen_random_uuid();
  old_session uuid := gen_random_uuid();
begin
  if not (select relrowsecurity from pg_class where oid = 'private.garmin_worker'::regclass)
    or not (select relrowsecurity from pg_class where oid = 'public.calories_garmin_days'::regclass)
    or not (select relrowsecurity from pg_class where oid = 'public.calories_garmin_status'::regclass) then
    raise exception 'Garmin RLS must be enabled.';
  end if;
  if has_function_privilege('anon', 'public.calories_garmin_worker(jsonb)', 'EXECUTE')
    or has_function_privilege('authenticated', 'public.calories_garmin_worker(jsonb)', 'EXECUTE')
    or has_function_privilege('anon', 'private.calories_garmin_worker_impl(jsonb)', 'EXECUTE')
    or has_function_privilege('authenticated', 'private.calories_garmin_worker_impl(jsonb)', 'EXECUTE') then
    raise exception 'Browser roles must not execute Garmin worker functions.';
  end if;
  perform set_config('garmin.test_key', 'garmin-verification-only-not-a-production-credential-0123456789', true);
  perform set_config('garmin.test_session', session_uuid::text, true);
  perform set_config('garmin.test_old_session', old_session::text, true);
  perform set_config('request.jwt.claim', '', true);
  perform set_config('request.jwt.claim.sub', '', true);
  perform set_config('garmin.test_before_count', (select count(*)::text from public.calories_garmin_days), true);
  update private.garmin_worker set credential_hash = null, encrypted_session = null, lease_id = null, lease_until = null
    where owner_id = owner_uuid;
  update public.calories_garmin_status set state = 'not_connected', last_attempt_at = null, last_sync_at = null, last_error_code = null
    where owner_id = owner_uuid;
  insert into auth.sessions (id, user_id, created_at, updated_at, aal)
    values (session_uuid, owner_uuid, now(), now(), 'aal1'),
      (old_session, owner_uuid, now() - interval '30 days', now(), 'aal1');
end;
$$;

set local role service_role;
select set_config('request.jwt.claims', '{"role":"service_role"}', true);
do $$
begin
  begin
    perform public.calories_garmin_worker(jsonb_build_object('operation', 'acquire', 'credential', current_setting('garmin.test_key')));
    raise exception 'Unprovisioned worker credential unexpectedly accepted.';
  exception when invalid_authorization_specification then
    if sqlerrm <> 'worker_auth_failed' then raise; end if;
  end;
end;
$$;
reset role;
update private.garmin_worker set credential_hash = encode(sha256(convert_to(current_setting('garmin.test_key'), 'UTF8')), 'hex');

set local role service_role;
select set_config('request.jwt.claims', '{"role":"service_role"}', true);
do $$
declare
  result jsonb;
  lease_uuid uuid;
  bad_request jsonb;
  bad_rows jsonb;
begin
  begin
    perform public.calories_garmin_worker(jsonb_build_object('operation', 'acquire', 'credential', repeat('wrong', 10)));
    raise exception 'Invalid worker credential unexpectedly accepted.';
  exception when invalid_authorization_specification then
    if sqlerrm <> 'worker_auth_failed' then raise; end if;
  end;
  perform set_config('request.jwt.claims', '{"role":"authenticated"}', true);
  begin
    perform public.calories_garmin_worker(jsonb_build_object('operation', 'acquire', 'credential', current_setting('garmin.test_key')));
    raise exception 'Worker unexpectedly accepted a non-service JWT role.';
  exception when insufficient_privilege then
    if sqlerrm <> 'worker_access_denied' then raise; end if;
  end;
  perform set_config('request.jwt.claims', '{"role":"service_role"}', true);

  result := public.calories_garmin_worker(jsonb_build_object('operation', 'acquire', 'credential', current_setting('garmin.test_key')));
  lease_uuid := (result ->> 'lease_id')::uuid;
  if lease_uuid is null or result -> 'encrypted_session' is distinct from 'null'::jsonb then
    raise exception 'First acquire must return a lease and null encrypted session.';
  end if;
  perform set_config('garmin.test_lease', lease_uuid::text, true);
  if public.calories_garmin_worker(jsonb_build_object('operation', 'acquire', 'credential', current_setting('garmin.test_key')))
    is distinct from '{"busy":true}'::jsonb then
    raise exception 'Concurrent acquire must report busy.';
  end if;

  foreach bad_request in array array[
    jsonb_build_object('encrypted_session', ''),
    jsonb_build_object('encrypted_session', null),
    jsonb_build_object('encrypted_session', repeat('x', 102401)),
    jsonb_build_object('encrypted_session', 'opaque', 'unexpected', 'secret-upstream-message')
  ] loop
    begin
      perform public.calories_garmin_worker(jsonb_build_object('operation', 'checkpoint', 'credential', current_setting('garmin.test_key'), 'lease_id', lease_uuid) || bad_request);
      raise exception 'Invalid checkpoint unexpectedly accepted.';
    exception when invalid_parameter_value then
      if sqlerrm <> 'invalid_request' then raise; end if;
    end;
  end loop;
  if public.calories_garmin_worker(jsonb_build_object('operation', 'checkpoint', 'credential', current_setting('garmin.test_key'), 'lease_id', lease_uuid, 'encrypted_session', 'verification-opaque-envelope'))
    is distinct from '{"ok":true}'::jsonb then
    raise exception 'Valid checkpoint failed.';
  end if;

  foreach bad_rows in array array[
    '[]'::jsonb,
    '[{"date":"2050-01-01","active":100,"resting":1000}]'::jsonb,
    '[{"date":"2050-01-01","active":"100","resting":1000,"total":1100}]'::jsonb,
    '[{"date":"2050-01-01","active":null,"resting":1000,"total":1100}]'::jsonb,
    '[{"date":"2050-01-01","active":-1,"resting":1000,"total":999}]'::jsonb,
    '[{"date":"2050-01-01","active":100001,"resting":0,"total":100001}]'::jsonb,
    '[{"date":"2050-01-01","active":100,"resting":1000,"total":500}]'::jsonb,
    '[{"date":"2050-01-01","active":100,"resting":1000,"total":"NaN"}]'::jsonb,
    '[{"date":"2025-02-29","active":100,"resting":1000,"total":1100}]'::jsonb,
    '[{"date":"2050-1-1","active":100,"resting":1000,"total":1100}]'::jsonb,
    '[{"date":"2050-01-01","active":100,"resting":1000,"total":1100,"password":"never-store"}]'::jsonb,
    '[{"date":"2050-01-01","active":100,"resting":1000,"total":1100},{"date":"2050-01-01","active":200,"resting":1000,"total":1200}]'::jsonb,
    '[{"date":"2050-01-01","active":100,"resting":1000,"total":1100},{"date":"2050-01-02","active":1}]'::jsonb
  ] loop
    begin
      perform public.calories_garmin_worker(jsonb_build_object('operation', 'import', 'credential', current_setting('garmin.test_key'), 'lease_id', lease_uuid, 'rows', bad_rows));
      raise exception 'Incomplete or invalid rows unexpectedly accepted.';
    exception when invalid_parameter_value then
      if sqlerrm <> 'invalid_request' then raise; end if;
    end;
  end loop;
  select jsonb_agg(jsonb_build_object('date', to_char('2050-01-01'::date + day, 'YYYY-MM-DD'), 'active', 100, 'resting', 1000, 'total', 1100))
    into bad_rows from generate_series(0, 400) as days(day);
  begin
    perform public.calories_garmin_worker(jsonb_build_object('operation', 'import', 'credential', current_setting('garmin.test_key'), 'lease_id', lease_uuid, 'rows', bad_rows));
    raise exception 'More than 400 rows unexpectedly accepted.';
  exception when invalid_parameter_value then
    if sqlerrm <> 'invalid_request' then raise; end if;
  end;
  begin
    perform public.calories_garmin_worker(jsonb_build_object('operation', 'release', 'credential', current_setting('garmin.test_key'), 'lease_id', lease_uuid, 'error_code', 'secret-upstream-message'));
    raise exception 'Unapproved upstream error unexpectedly accepted.';
  exception when invalid_parameter_value then
    if sqlerrm <> 'invalid_request' then raise; end if;
  end;
end;
$$;

-- Inspect private effects as administrator, without returning credentials.
reset role;
do $$
begin
  if (select last_sync_at from public.calories_garmin_status) is not null
    or (select state from public.calories_garmin_status) <> 'connected'
    or (select encrypted_session from private.garmin_worker) <> 'verification-opaque-envelope' then
    raise exception 'Checkpoint must connect without claiming a successful import.';
  end if;
  if (select count(*) from public.calories_garmin_days) <> current_setting('garmin.test_before_count')::bigint then
    raise exception 'Invalid imports unexpectedly changed daily rows.';
  end if;
end;
$$;
set local role service_role;
select set_config('request.jwt.claims', '{"role":"service_role"}', true);
do $$
declare
  result jsonb;
begin
  result := public.calories_garmin_worker(jsonb_build_object('operation', 'import', 'credential', current_setting('garmin.test_key'), 'lease_id', current_setting('garmin.test_lease'),
    'rows', '[{"date":"2050-01-01","active":100.25,"resting":1000.5,"total":1100.75},{"date":"2050-01-02","active":200,"resting":1000,"total":1200}]'::jsonb));
  if result is distinct from '{"count":2}'::jsonb then raise exception 'Valid import must return its row count.'; end if;
  result := public.calories_garmin_worker(jsonb_build_object('operation', 'release', 'credential', current_setting('garmin.test_key'), 'lease_id', current_setting('garmin.test_lease'), 'error_code', null));
  if result is distinct from '{"ok":true}'::jsonb then raise exception 'Release failed.'; end if;
  begin
    perform public.calories_garmin_worker(jsonb_build_object('operation', 'checkpoint', 'credential', current_setting('garmin.test_key'), 'lease_id', current_setting('garmin.test_lease'), 'encrypted_session', 'stale'));
    raise exception 'Released lease unexpectedly wrote a checkpoint.';
  exception when object_not_in_prerequisite_state then
    if sqlerrm <> 'stale_lease' then raise; end if;
  end;
  result := public.calories_garmin_worker(jsonb_build_object('operation', 'acquire', 'credential', current_setting('garmin.test_key')));
  if result ->> 'encrypted_session' <> 'verification-opaque-envelope' then raise exception 'Acquire lost its saved envelope.'; end if;
  perform set_config('garmin.test_lease', result ->> 'lease_id', true);
end;
$$;

reset role;
do $$
begin
  if (select last_sync_at from public.calories_garmin_status) is null
    or (select state from public.calories_garmin_status) <> 'connected'
    or not exists (select 1 from public.calories_garmin_days where date = '2050-01-01' and total = 1100.75 and synced_at is not null) then
    raise exception 'Import must persist complete totals and server sync time.';
  end if;
end;
$$;
update private.garmin_worker set lease_until = clock_timestamp() - interval '1 second';
set local role service_role;
select set_config('request.jwt.claims', '{"role":"service_role"}', true);
do $$
declare
  result jsonb;
begin
  begin
    perform public.calories_garmin_worker(jsonb_build_object('operation', 'import', 'credential', current_setting('garmin.test_key'), 'lease_id', current_setting('garmin.test_lease'),
      'rows', '[{"date":"2050-01-01","active":0,"resting":0,"total":0}]'::jsonb));
    raise exception 'Expired lease unexpectedly changed totals.';
  exception when object_not_in_prerequisite_state then
    if sqlerrm <> 'stale_lease' then raise; end if;
  end;
  result := public.calories_garmin_worker(jsonb_build_object('operation', 'acquire', 'credential', current_setting('garmin.test_key')));
  if result ->> 'lease_id' = current_setting('garmin.test_lease') then raise exception 'Expired acquire must replace the lease.'; end if;
  perform set_config('garmin.test_lease', result ->> 'lease_id', true);
  perform public.calories_garmin_worker(jsonb_build_object('operation', 'release', 'credential', current_setting('garmin.test_key'), 'lease_id', current_setting('garmin.test_lease'), 'error_code', 'needs_login'));
  result := public.calories_garmin_worker(jsonb_build_object('operation', 'acquire', 'credential', current_setting('garmin.test_key')));
  if result -> 'encrypted_session' is distinct from 'null'::jsonb then raise exception 'Needs-login must clear stale session ciphertext.'; end if;
  perform public.calories_garmin_worker(jsonb_build_object('operation', 'release', 'credential', current_setting('garmin.test_key'), 'lease_id', result ->> 'lease_id', 'error_code', 'not_connected'));
end;
$$;

reset role;
do $$
begin
  if (select state from public.calories_garmin_status) <> 'needs_login'
    or (select last_error_code from public.calories_garmin_status) <> 'needs_login'
    or (select lease_id from private.garmin_worker) is not null
    or (select encrypted_session from private.garmin_worker) is not null
    or (select total from public.calories_garmin_days where date = '2050-01-01') <> 1100.75 then
    raise exception 'Reconnect requirement must preserve previous daily values and survive no-session skips.';
  end if;
end;
$$;

-- API roles cannot access private worker state or mutate public Garmin tables.
do $$
declare
  api_role text;
  table_name text;
begin
  foreach api_role in array array['anon', 'authenticated', 'service_role'] loop
    if has_table_privilege(api_role, 'private.garmin_worker', 'SELECT')
      or has_any_column_privilege(api_role, 'private.garmin_worker', 'INSERT')
      or has_any_column_privilege(api_role, 'private.garmin_worker', 'UPDATE')
      or has_table_privilege(api_role, 'private.garmin_worker', 'DELETE') then
      raise exception 'API role has direct private-worker privileges.';
    end if;
    foreach table_name in array array['public.calories_garmin_days', 'public.calories_garmin_status'] loop
      if has_any_column_privilege(api_role, table_name, 'INSERT')
        or has_any_column_privilege(api_role, table_name, 'UPDATE')
        or has_table_privilege(api_role, table_name, 'DELETE') then
        raise exception 'API role has direct Garmin table write privileges.';
      end if;
    end loop;
  end loop;
end;
$$;

-- Prepare a real in-flight lease to prove owner disconnect revokes it.
do $$
declare
  lease_uuid uuid := gen_random_uuid();
begin
  update private.garmin_worker set encrypted_session = 'verification-disconnect-envelope', lease_id = lease_uuid,
    lease_until = clock_timestamp() + interval '15 minutes';
  update public.calories_garmin_status set state = 'connected', last_error_code = null;
  perform set_config('garmin.test_disconnect_lease', lease_uuid::text, true);
end;
$$;

set local role authenticated;
select set_config('request.jwt.claims', jsonb_build_object('role', 'authenticated', 'sub', '8c0b4ad9-f74f-4951-bb9d-a4a3d4d1f686', 'session_id', current_setting('garmin.test_session'))::text, true);
do $$
begin
  if (select count(*) from public.calories_garmin_status) <> 1
    or not exists (select 1 from public.calories_garmin_days where date = '2050-01-01' and total = 1100.75) then
    raise exception 'Approved owner session should read Garmin totals and status.';
  end if;
  begin
    perform 1 from private.garmin_worker;
    raise exception 'Browser unexpectedly read encrypted session or credential hash.';
  exception when insufficient_privilege then null;
  end;
  begin
    perform public.calories_garmin_worker('{}'::jsonb);
    raise exception 'Browser unexpectedly called worker RPC.';
  exception when insufficient_privilege then null;
  end;
  begin
    update public.calories_garmin_days set total = 0;
    raise exception 'Browser unexpectedly wrote Garmin totals.';
  exception when insufficient_privilege then null;
  end;
  if public.calories_garmin_disconnect() is distinct from '{"ok":true}'::jsonb then
    raise exception 'Owner disconnect should succeed.';
  end if;
  if (select state from public.calories_garmin_status) <> 'not_connected'
    or (select total from public.calories_garmin_days where date = '2050-01-01') <> 1100.75 then
    raise exception 'Disconnect must preserve daily history.';
  end if;
end;
$$;
reset role;
do $$
begin
  if (select encrypted_session from private.garmin_worker) is not null
    or (select lease_id from private.garmin_worker) is not null
    or (select credential_hash from private.garmin_worker) is null then
    raise exception 'Disconnect must remove the envelope and lease while preserving the worker credential.';
  end if;
end;
$$;
set local role service_role;
select set_config('request.jwt.claims', '{"role":"service_role"}', true);
do $$
begin
  begin
    perform public.calories_garmin_worker(jsonb_build_object('operation', 'import', 'credential', current_setting('garmin.test_key'), 'lease_id', current_setting('garmin.test_disconnect_lease'),
      'rows', '[{"date":"2050-01-01","active":0,"resting":0,"total":0}]'::jsonb));
    raise exception 'Disconnected in-flight lease unexpectedly changed history.';
  exception when object_not_in_prerequisite_state then
    if sqlerrm <> 'stale_lease' then raise; end if;
  end;
  begin
    perform public.calories_garmin_disconnect();
    raise exception 'Service role unexpectedly disconnected the owner directly.';
  exception when insufficient_privilege then null;
  end;
end;
$$;
set local role authenticated;
select set_config('request.jwt.claims', jsonb_build_object('role', 'authenticated', 'sub', gen_random_uuid(), 'session_id', current_setting('garmin.test_session'))::text, true);
do $$
begin
  if exists (select 1 from public.calories_garmin_days) or exists (select 1 from public.calories_garmin_status) then
    raise exception 'Stranger unexpectedly read Garmin data.';
  end if;
  begin
    perform public.calories_garmin_disconnect();
    raise exception 'Stranger unexpectedly disconnected the owner.';
  exception when insufficient_privilege then
    if sqlerrm <> 'garmin_access_denied' then raise; end if;
  end;
end;
$$;
select set_config('request.jwt.claims', jsonb_build_object('role', 'authenticated', 'sub', '8c0b4ad9-f74f-4951-bb9d-a4a3d4d1f686', 'session_id', current_setting('garmin.test_old_session'))::text, true);
do $$
begin
  if exists (select 1 from public.calories_garmin_days) or exists (select 1 from public.calories_garmin_status) then
    raise exception 'Expired owner session unexpectedly read Garmin data.';
  end if;
  begin
    perform public.calories_garmin_disconnect();
    raise exception 'Expired owner session unexpectedly disconnected Garmin.';
  exception when insufficient_privilege then
    if sqlerrm <> 'garmin_access_denied' then raise; end if;
  end;
end;
$$;
select set_config('request.jwt.claims', '{"role":"authenticated","sub":"8c0b4ad9-f74f-4951-bb9d-a4a3d4d1f686"}', true);
do $$
begin
  if exists (select 1 from public.calories_garmin_days) or exists (select 1 from public.calories_garmin_status) then
    raise exception 'Owner without an active session unexpectedly read Garmin data.';
  end if;
  begin
    perform public.calories_garmin_disconnect();
    raise exception 'Owner without an active session unexpectedly disconnected.';
  exception when insufficient_privilege then
    if sqlerrm <> 'garmin_access_denied' then raise; end if;
  end;
end;
$$;

set local role anon;
select set_config('request.jwt.claims', '{}', true);
do $$
begin
  begin
    perform 1 from public.calories_garmin_days;
    raise exception 'Anonymous role unexpectedly read Garmin totals.';
  exception when insufficient_privilege then null;
  end;
  begin
    perform 1 from public.calories_garmin_status;
    raise exception 'Anonymous role unexpectedly read Garmin status.';
  exception when insufficient_privilege then null;
  end;
  begin
    perform public.calories_garmin_worker('{}'::jsonb);
    raise exception 'Anonymous role unexpectedly called Garmin RPC.';
  exception when insufficient_privilege then null;
  end;
  begin
    perform public.calories_garmin_disconnect();
    raise exception 'Anonymous role unexpectedly disconnected Garmin.';
  exception when insufficient_privilege then null;
  end;
end;
$$;

reset role;
rollback;
select 'Garmin verification passed; all test changes rolled back.' as result;
