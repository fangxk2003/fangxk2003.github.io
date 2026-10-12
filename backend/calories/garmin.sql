-- Garmin cloud storage for the existing Calories project.
-- Run after setup.sql as postgres. Keep private out of exposed API schemas.
-- This script provisions no credential. After migration, an administrator must
-- set private.garmin_worker.credential_hash to the lowercase SHA-256 hex digest
-- of a randomly generated worker key. Keep the raw key in server secrets only.
-- The worker RPC is callable by service_role only, never by a browser account.

begin;

do $$
begin
  if not exists (select 1 from auth.users where id = '8c0b4ad9-f74f-4951-bb9d-a4a3d4d1f686'::uuid)
    or to_regprocedure('private.calories_session_active()') is null then
    raise exception 'Run the owner Calories setup before Garmin setup.';
  end if;
end;
$$;

create schema if not exists private;
grant usage on schema private to authenticated, service_role;

create table if not exists private.garmin_worker (
  owner_id uuid primary key references auth.users(id) on delete cascade
    constraint garmin_worker_single_owner check (owner_id = '8c0b4ad9-f74f-4951-bb9d-a4a3d4d1f686'::uuid),
  credential_hash text
    constraint garmin_worker_hash check (credential_hash is null or credential_hash ~ '^[0-9a-f]{64}$'),
  encrypted_session text
    constraint garmin_worker_envelope check (encrypted_session is null or octet_length(encrypted_session) between 1 and 102400),
  lease_id uuid,
  lease_until timestamptz,
  constraint garmin_worker_lease_pair check ((lease_id is null) = (lease_until is null))
);

create table if not exists public.calories_garmin_days (
  owner_id uuid not null references auth.users(id) on delete cascade
    constraint calories_garmin_days_owner check (owner_id = '8c0b4ad9-f74f-4951-bb9d-a4a3d4d1f686'::uuid),
  date date not null constraint calories_garmin_date_finite check (isfinite(date)),
  active numeric not null constraint calories_garmin_active check (active between 0 and 100000),
  resting numeric not null constraint calories_garmin_resting check (resting between 0 and 100000),
  total numeric not null constraint calories_garmin_total check (total between 0 and 100000),
  synced_at timestamptz not null default now(),
  primary key (owner_id, date),
  constraint calories_garmin_total_matches check (abs(total - active - resting) <= 1)
);

create table if not exists public.calories_garmin_status (
  owner_id uuid primary key references auth.users(id) on delete cascade
    constraint calories_garmin_status_owner check (owner_id = '8c0b4ad9-f74f-4951-bb9d-a4a3d4d1f686'::uuid),
  state text not null default 'not_connected'
    constraint calories_garmin_state check (state in ('not_connected', 'connected', 'needs_login', 'error')),
  last_attempt_at timestamptz,
  last_sync_at timestamptz,
  last_error_code text
    constraint calories_garmin_error_code check (last_error_code is null or last_error_code in ('needs_login', 'upstream_unavailable', 'missing_data'))
);

insert into private.garmin_worker (owner_id)
  values ('8c0b4ad9-f74f-4951-bb9d-a4a3d4d1f686'::uuid)
  on conflict (owner_id) do nothing;
insert into public.calories_garmin_status (owner_id)
  values ('8c0b4ad9-f74f-4951-bb9d-a4a3d4d1f686'::uuid)
  on conflict (owner_id) do nothing;

alter table private.garmin_worker enable row level security;
alter table public.calories_garmin_days enable row level security;
alter table public.calories_garmin_status enable row level security;

-- Explicit deny-all documents why the private table has no API read/write path.
-- Its narrowly scoped definer functions own all access.
drop policy if exists garmin_worker_no_direct_access on private.garmin_worker;
create policy garmin_worker_no_direct_access on private.garmin_worker
  for all to public using (false) with check (false);

-- Even the service role goes through the RPC; the definer is the sole writer.
revoke all on private.garmin_worker from public, anon, authenticated, service_role;
revoke all (owner_id, credential_hash, encrypted_session, lease_id, lease_until)
  on private.garmin_worker from public, anon, authenticated, service_role;
revoke all on public.calories_garmin_days from public, anon, authenticated, service_role;
revoke all (owner_id, date, active, resting, total, synced_at)
  on public.calories_garmin_days from public, anon, authenticated, service_role;
revoke all on public.calories_garmin_status from public, anon, authenticated, service_role;
revoke all (owner_id, state, last_attempt_at, last_sync_at, last_error_code)
  on public.calories_garmin_status from public, anon, authenticated, service_role;
grant select on public.calories_garmin_days, public.calories_garmin_status to authenticated;

drop policy if exists calories_garmin_days_gate on public.calories_garmin_days;
create policy calories_garmin_days_gate on public.calories_garmin_days
  as restrictive for all to authenticated
  using (owner_id = '8c0b4ad9-f74f-4951-bb9d-a4a3d4d1f686'::uuid
    and (select auth.uid()) = owner_id and (select private.calories_session_active()))
  with check (false);
drop policy if exists calories_garmin_days_read on public.calories_garmin_days;
create policy calories_garmin_days_read on public.calories_garmin_days
  for select to authenticated using ((select auth.uid()) = owner_id);

drop policy if exists calories_garmin_status_gate on public.calories_garmin_status;
create policy calories_garmin_status_gate on public.calories_garmin_status
  as restrictive for all to authenticated
  using (owner_id = '8c0b4ad9-f74f-4951-bb9d-a4a3d4d1f686'::uuid
    and (select auth.uid()) = owner_id and (select private.calories_session_active()))
  with check (false);
drop policy if exists calories_garmin_status_read on public.calories_garmin_status;
create policy calories_garmin_status_read on public.calories_garmin_status
  for select to authenticated using ((select auth.uid()) = owner_id);

-- Exactly one row lock serializes acquisition and all lease-bound writes.
-- No credential, ciphertext, upstream error, or supplied value appears in errors.
create or replace function private.calories_garmin_worker_impl(p_request jsonb)
returns jsonb
language plpgsql
security definer
set search_path = ''
as $$
declare
  owner_uuid constant uuid := '8c0b4ad9-f74f-4951-bb9d-a4a3d4d1f686'::uuid;
  worker private.garmin_worker%rowtype;
  operation text;
  supplied_lease uuid;
  lease_uuid uuid;
  sync_now timestamptz;
  row_value jsonb;
  row_date date;
  row_active numeric;
  row_resting numeric;
  row_total numeric;
  seen_dates date[] := '{}'::date[];
  error_code text;
  imported_count integer;
begin
  if (auth.jwt() ->> 'role') is distinct from 'service_role' then
    raise exception using errcode = '42501', message = 'worker_access_denied';
  end if;
  if jsonb_typeof(p_request) is distinct from 'object'
    or jsonb_typeof(p_request -> 'credential') is distinct from 'string'
    or length(p_request ->> 'credential') not between 32 and 512 then
    raise exception using errcode = '28000', message = 'worker_auth_failed';
  end if;
  select * into worker from private.garmin_worker where owner_id = owner_uuid for update;
  if not found or worker.credential_hash is null
    or encode(sha256(convert_to(p_request ->> 'credential', 'UTF8')), 'hex') is distinct from worker.credential_hash then
    raise exception using errcode = '28000', message = 'worker_auth_failed';
  end if;
  if jsonb_typeof(p_request -> 'operation') is distinct from 'string'
    or exists (select 1 from jsonb_object_keys(p_request) as keys(key)
      where key <> all(array['operation', 'credential', 'lease_id', 'encrypted_session', 'rows', 'error_code'])) then
    raise exception using errcode = '22023', message = 'invalid_request';
  end if;
  operation := p_request ->> 'operation';
  if operation not in ('acquire', 'checkpoint', 'import', 'release') then
    raise exception using errcode = '22023', message = 'invalid_request';
  end if;
  sync_now := clock_timestamp();

  if operation = 'acquire' then
    if worker.lease_id is not null and worker.lease_until > sync_now then
      return jsonb_build_object('busy', true);
    end if;
    lease_uuid := gen_random_uuid();
    update private.garmin_worker set lease_id = lease_uuid, lease_until = sync_now + interval '15 minutes'
      where owner_id = owner_uuid;
    update public.calories_garmin_status set last_attempt_at = sync_now where owner_id = owner_uuid;
    return jsonb_build_object('lease_id', lease_uuid, 'encrypted_session', worker.encrypted_session);
  end if;

  if jsonb_typeof(p_request -> 'lease_id') is distinct from 'string' then
    raise exception using errcode = '22023', message = 'invalid_request';
  end if;
  begin
    supplied_lease := (p_request ->> 'lease_id')::uuid;
  exception when invalid_text_representation then
    raise exception using errcode = '22023', message = 'invalid_request';
  end;
  if supplied_lease is distinct from worker.lease_id or worker.lease_until is null
    or worker.lease_until <= sync_now then
    raise exception using errcode = '55000', message = 'stale_lease';
  end if;

  if operation = 'checkpoint' then
    if jsonb_typeof(p_request -> 'encrypted_session') is distinct from 'string'
      or octet_length(p_request ->> 'encrypted_session') not between 1 and 102400 then
      raise exception using errcode = '22023', message = 'invalid_request';
    end if;
    update private.garmin_worker set encrypted_session = p_request ->> 'encrypted_session' where owner_id = owner_uuid;
    update public.calories_garmin_status set state = 'connected', last_error_code = null where owner_id = owner_uuid;
    return jsonb_build_object('ok', true);
  end if;

  if operation = 'import' then
    if jsonb_typeof(p_request -> 'rows') is distinct from 'array' then
      raise exception using errcode = '22023', message = 'invalid_request';
    end if;
    if jsonb_array_length(p_request -> 'rows') not between 1 and 400 then
      raise exception using errcode = '22023', message = 'invalid_request';
    end if;
    for row_value in select value from jsonb_array_elements(p_request -> 'rows') loop
      if jsonb_typeof(row_value) is distinct from 'object' then
        raise exception using errcode = '22023', message = 'invalid_request';
      end if;
      if not (row_value ?& array['date', 'active', 'resting', 'total'])
        or (select count(*) from jsonb_object_keys(row_value)) <> 4
        or jsonb_typeof(row_value -> 'date') is distinct from 'string'
        or (row_value ->> 'date') !~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'
        or jsonb_typeof(row_value -> 'active') is distinct from 'number'
        or jsonb_typeof(row_value -> 'resting') is distinct from 'number'
        or jsonb_typeof(row_value -> 'total') is distinct from 'number' then
        raise exception using errcode = '22023', message = 'invalid_request';
      end if;
      begin
        row_date := (row_value ->> 'date')::date;
        row_active := (row_value ->> 'active')::numeric;
        row_resting := (row_value ->> 'resting')::numeric;
        row_total := (row_value ->> 'total')::numeric;
      exception when data_exception then
        raise exception using errcode = '22023', message = 'invalid_request';
      end;
      if to_char(row_date, 'YYYY-MM-DD') <> row_value ->> 'date'
        or row_date = any(seen_dates)
        or row_active not between 0 and 100000
        or row_resting not between 0 and 100000
        or row_total not between 0 and 100000
        or abs(row_total - row_active - row_resting) > 1 then
        raise exception using errcode = '22023', message = 'invalid_request';
      end if;
      seen_dates := array_append(seen_dates, row_date);
    end loop;

    sync_now := clock_timestamp();
    if worker.lease_until <= sync_now then
      raise exception using errcode = '55000', message = 'stale_lease';
    end if;
    insert into public.calories_garmin_days (owner_id, date, active, resting, total, synced_at)
      select owner_uuid, rows.date::date, rows.active, rows.resting, rows.total, sync_now
      from jsonb_to_recordset(p_request -> 'rows') as rows(date text, active numeric, resting numeric, total numeric)
      on conflict (owner_id, date) do update set
        active = excluded.active, resting = excluded.resting, total = excluded.total, synced_at = excluded.synced_at;
    get diagnostics imported_count = row_count;
    update public.calories_garmin_status set state = 'connected', last_sync_at = sync_now, last_error_code = null
      where owner_id = owner_uuid;
    return jsonb_build_object('count', imported_count);
  end if;

  if p_request ? 'error_code' and p_request -> 'error_code' <> 'null'::jsonb
    and jsonb_typeof(p_request -> 'error_code') is distinct from 'string' then
    raise exception using errcode = '22023', message = 'invalid_request';
  end if;
  error_code := p_request ->> 'error_code';
  if error_code is not null and error_code not in ('not_connected', 'needs_login', 'upstream_unavailable', 'missing_data') then
    raise exception using errcode = '22023', message = 'invalid_request';
  end if;
  if error_code = 'needs_login' then
    update private.garmin_worker set encrypted_session = null where owner_id = owner_uuid;
    update public.calories_garmin_status set state = 'needs_login', last_error_code = 'needs_login' where owner_id = owner_uuid;
  elsif error_code in ('upstream_unavailable', 'missing_data') then
    update public.calories_garmin_status set state = 'error', last_error_code = error_code where owner_id = owner_uuid;
  elsif error_code is null then
    update public.calories_garmin_status set
      state = case when worker.encrypted_session is null then 'not_connected' else 'connected' end,
      last_error_code = null where owner_id = owner_uuid;
  end if;
  -- not_connected preserves the existing status, including needs_login, so an
  -- hourly no-session skip does not erase an actionable reconnect request.
  update private.garmin_worker set lease_id = null, lease_until = null where owner_id = owner_uuid;
  return jsonb_build_object('ok', true);
end;
$$;

revoke all on function private.calories_garmin_worker_impl(jsonb) from public, anon, authenticated, service_role;
grant execute on function private.calories_garmin_worker_impl(jsonb) to service_role;

create or replace function public.calories_garmin_worker(p_request jsonb)
returns jsonb
language sql
security invoker
set search_path = ''
as $$
  select private.calories_garmin_worker_impl(p_request);
$$;
revoke all on function public.calories_garmin_worker(jsonb) from public, anon, authenticated, service_role;
grant execute on function public.calories_garmin_worker(jsonb) to service_role;

-- The owner can remove the cloud session without deleting imported history.
-- Clearing the lease also prevents an already-running worker from saving again.
create or replace function private.calories_garmin_disconnect_impl()
returns jsonb
language plpgsql
security definer
set search_path = ''
as $$
declare
  owner_uuid constant uuid := '8c0b4ad9-f74f-4951-bb9d-a4a3d4d1f686'::uuid;
begin
  if auth.uid() is distinct from owner_uuid or not private.calories_session_active() then
    raise exception using errcode = '42501', message = 'garmin_access_denied';
  end if;
  perform 1 from private.garmin_worker where owner_id = owner_uuid for update;
  update private.garmin_worker set encrypted_session = null, lease_id = null, lease_until = null where owner_id = owner_uuid;
  update public.calories_garmin_status set state = 'not_connected', last_error_code = null where owner_id = owner_uuid;
  return jsonb_build_object('ok', true);
end;
$$;
revoke all on function private.calories_garmin_disconnect_impl() from public, anon, authenticated, service_role;
grant execute on function private.calories_garmin_disconnect_impl() to authenticated;

create or replace function public.calories_garmin_disconnect()
returns jsonb
language sql
security invoker
set search_path = ''
as $$
  select private.calories_garmin_disconnect_impl();
$$;
revoke all on function public.calories_garmin_disconnect() from public, anon, authenticated, service_role;
grant execute on function public.calories_garmin_disconnect() to authenticated;

comment on table private.garmin_worker is 'Private worker credential digest and opaque AES-GCM session envelope. No API role can access this table directly.';
comment on table public.calories_garmin_days is 'Owner-readable Garmin total/active/resting daily energy, written only by the lease-bound worker RPC.';
comment on table public.calories_garmin_status is 'Owner-readable generic Garmin connection and sync status; no credentials or upstream messages.';

notify pgrst, 'reload schema';
commit;
