-- Run AFTER setup.sql in the Supabase SQL Editor as postgres.
-- Exercises the actual database roles and policies. It needs no password/key.
-- Temporary Auth sessions and state changes are rolled back at the end.
-- If an assertion fails, run ROLLBACK before running another query.

begin;

do $$
declare
  owner_uuid uuid;
  fresh_session uuid := gen_random_uuid();
  old_session uuid := gen_random_uuid();
  expired_session uuid := gen_random_uuid();
begin
  select (regexp_match(
    pg_get_constraintdef(oid),
    '[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}'
  ))[1]::uuid into owner_uuid
  from pg_constraint
  where conrelid = 'public.calories_state'::regclass
    and conname = 'calories_state_single_owner';
  if owner_uuid is null then
    raise exception 'Run setup.sql with your owner UUID first.';
  end if;
  if not (select relrowsecurity from pg_class where oid = 'public.calories_state'::regclass) then
    raise exception 'Row Level Security must be enabled.';
  end if;
  if has_table_privilege('anon', 'public.calories_state', 'SELECT')
    or has_any_column_privilege('anon', 'public.calories_state', 'INSERT')
    or has_any_column_privilege('anon', 'public.calories_state', 'UPDATE')
    or has_table_privilege('anon', 'public.calories_state', 'DELETE') then
    raise exception 'Anonymous table privileges must be revoked.';
  end if;
  if has_column_privilege('authenticated', 'public.calories_state', 'revision', 'UPDATE')
    or has_column_privilege('authenticated', 'public.calories_state', 'user_id', 'UPDATE')
    or has_table_privilege('authenticated', 'public.calories_state', 'DELETE') then
    raise exception 'Browser role must not write revisions, change ownership, or delete the state row.';
  end if;
  perform set_config('calories.test_owner', owner_uuid::text, true);
  perform set_config('calories.test_stranger', gen_random_uuid()::text, true);
  perform set_config('calories.test_fresh_session', fresh_session::text, true);
  perform set_config('calories.test_old_session', old_session::text, true);
  perform set_config('calories.test_expired_session', expired_session::text, true);
  insert into auth.sessions (id, user_id, created_at, updated_at, aal, not_after) values
    (fresh_session, owner_uuid, now(), now(), 'aal1', null),
    (old_session, owner_uuid, now() - interval '30 days', now(), 'aal1', null),
    (expired_session, owner_uuid, now(), now(), 'aal1', now() - interval '1 second');
end;
$$;

set local role authenticated;
select set_config('request.jwt.claims', jsonb_build_object(
  'sub', current_setting('calories.test_owner'),
  'role', 'authenticated',
  'session_id', current_setting('calories.test_fresh_session')
)::text, true);

do $$
declare
  owner_uuid uuid := current_setting('calories.test_owner')::uuid;
  before_revision bigint;
  after_revision bigint;
  affected integer;
begin
  if not private.calories_session_active() or not public.calories_access_allowed() then
    raise exception 'A fresh owner session should be active.';
  end if;
  select revision into before_revision from public.calories_state where user_id = owner_uuid;
  if before_revision is null then
    insert into public.calories_state (user_id, data)
      values (owner_uuid, '{"verification":"initial"}'::jsonb)
      returning revision into before_revision;
    if before_revision <> 1 then
      raise exception 'Initial save must start at revision 1.';
    end if;
  end if;

  update public.calories_state set data = '{"verification":"fresh"}'::jsonb
    where user_id = owner_uuid and revision = before_revision
    returning revision into after_revision;
  get diagnostics affected = row_count;
  if affected <> 1 or after_revision <> before_revision + 1 then
    raise exception 'Fresh conditional save must update once and advance the revision.';
  end if;
  if (select count(*) from public.calories_state) <> 1 then
    raise exception 'The approved owner should see exactly one state row.';
  end if;

  update public.calories_state set data = '{"verification":"stale"}'::jsonb
    where user_id = owner_uuid and revision = before_revision;
  get diagnostics affected = row_count;
  if affected <> 0 or (select data from public.calories_state where user_id = owner_uuid)
      <> '{"verification":"fresh"}'::jsonb then
    raise exception 'A stale device must not overwrite the newer revision.';
  end if;

  begin
    update public.calories_state set revision = 100 where user_id = owner_uuid;
    raise exception 'Browser role unexpectedly wrote a revision.';
  exception when insufficient_privilege then null;
  end;
  begin
    update public.calories_state set user_id = gen_random_uuid() where user_id = owner_uuid;
    raise exception 'Browser role unexpectedly changed ownership.';
  exception when insufficient_privilege then null;
  end;
  begin
    delete from public.calories_state where user_id = owner_uuid;
    raise exception 'Browser role unexpectedly deleted the state row.';
  exception when insufficient_privilege then null;
  end;
  begin
    update public.calories_state set data = '[]'::jsonb where user_id = owner_uuid;
    raise exception 'State unexpectedly accepted a JSON array.';
  exception when check_violation then null;
  end;
end;
$$;

-- A different signed-in user cannot read, create, or change any state.
select set_config('request.jwt.claims', jsonb_build_object(
  'sub', current_setting('calories.test_stranger'),
  'role', 'authenticated',
  'session_id', current_setting('calories.test_fresh_session')
)::text, true);

do $$
declare
  affected integer;
begin
  if private.calories_session_active() or public.calories_access_allowed()
    or (select count(*) from public.calories_state) <> 0 then
    raise exception 'A stranger must not use the owner session or read owner data.';
  end if;
  update public.calories_state set data = '{}'::jsonb;
  get diagnostics affected = row_count;
  if affected <> 0 then raise exception 'A stranger unexpectedly changed data.'; end if;
  begin
    insert into public.calories_state (user_id, data)
      values (current_setting('calories.test_owner')::uuid, '{}'::jsonb);
    raise exception 'A stranger unexpectedly inserted owner data.';
  exception when insufficient_privilege then null;
  end;
  begin
    insert into public.calories_state (user_id, data)
      values (current_setting('calories.test_stranger')::uuid, '{}'::jsonb);
    raise exception 'A stranger unexpectedly inserted their own state.';
  exception when insufficient_privilege then null;
  end;
end;
$$;

-- Missing/malformed session claims, exactly 30-day-old sessions, and explicit
-- session expiry must all fail closed, even for the approved account.
do $$
declare
  session_value text;
  affected integer;
begin
  foreach session_value in array array[
    '', 'not-a-uuid', gen_random_uuid()::text,
    current_setting('calories.test_old_session'),
    current_setting('calories.test_expired_session')
  ] loop
    perform set_config('request.jwt.claims', jsonb_build_object(
      'sub', current_setting('calories.test_owner'),
      'role', 'authenticated', 'session_id', session_value
    )::text, true);
    if private.calories_session_active() or public.calories_access_allowed()
      or (select count(*) from public.calories_state) <> 0 then
      raise exception 'Inactive session unexpectedly gained access: %', session_value;
    end if;
    update public.calories_state set data = '{}'::jsonb;
    get diagnostics affected = row_count;
    if affected <> 0 then raise exception 'Inactive session unexpectedly wrote data.'; end if;
    begin
      insert into public.calories_state (user_id, data)
        values (current_setting('calories.test_owner')::uuid, '{}'::jsonb);
      raise exception 'Inactive session unexpectedly inserted data.';
    exception when insufficient_privilege then null;
    end;
  end loop;
end;
$$;

-- Removing only the temporary session simulates successful local sign-out.
reset role;
delete from auth.sessions where id = current_setting('calories.test_fresh_session')::uuid;
set local role authenticated;
select set_config('request.jwt.claims', jsonb_build_object(
  'sub', current_setting('calories.test_owner'),
  'role', 'authenticated',
  'session_id', current_setting('calories.test_fresh_session')
)::text, true);
do $$
begin
  if private.calories_session_active() or public.calories_access_allowed()
    or (select count(*) from public.calories_state) <> 0 then
    raise exception 'A signed-out session unexpectedly retained access.';
  end if;
end;
$$;

-- Signed-out requests are denied by grants before they reach the RLS policies.
set local role anon;
select set_config('request.jwt.claims', '{}', true);
do $$
begin
  begin
    perform public.calories_access_allowed();
    raise exception 'Anonymous role unexpectedly called the access RPC.';
  exception when insufficient_privilege then null;
  end;
  begin
    perform 1 from public.calories_state;
    raise exception 'Anonymous role unexpectedly read data.';
  exception when insufficient_privilege then null;
  end;
  begin
    insert into public.calories_state (user_id, data)
      values (current_setting('calories.test_owner')::uuid, '{}'::jsonb);
    raise exception 'Anonymous role unexpectedly inserted data.';
  exception when insufficient_privilege then null;
  end;
  begin
    update public.calories_state set data = '{}'::jsonb;
    raise exception 'Anonymous role unexpectedly updated data.';
  exception when insufficient_privilege then null;
  end;
  begin
    delete from public.calories_state;
    raise exception 'Anonymous role unexpectedly deleted data.';
  exception when insufficient_privilege then null;
  end;
end;
$$;

reset role;
rollback;
select 'Calories database verification passed; all test changes rolled back.' as result;
