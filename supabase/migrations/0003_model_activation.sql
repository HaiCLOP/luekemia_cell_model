-- Atomic model promotion, artifact identity by hash, and an explicit "unverified" mark on public feedback.

alter table model_versions add column if not exists onnx_sha256 text;
alter table model_versions add column if not exists activated_at timestamptz;
create unique index if not exists model_versions_onnx_sha256 on model_versions (onnx_sha256) where onnx_sha256 is not null;

-- Register the served artifact (keyed by its ONNX hash) and make it the only active version, in one transaction.
-- The previous active row is kept (inactive) for rollback: call this again with that row's metadata.
-- A different artifact under an existing name is refused, so a version name always means one file.
create or replace function public.activate_model_version(p jsonb)
returns uuid
language plpgsql
security definer
set search_path = public
as $$
declare
  v_id uuid;
  v_sha text := p->>'onnx_sha256';
begin
  if v_sha is null or length(v_sha) <> 64 then
    raise exception 'onnx_sha256 is required';
  end if;

  select id into v_id from model_versions where onnx_sha256 = v_sha;
  if v_id is null then
    -- rows created before hashes were recorded: adopt them only if the name matches and no hash is set
    select id into v_id from model_versions where name = p->>'name' and onnx_sha256 is null;
  end if;
  if v_id is null and exists (select 1 from model_versions where name = p->>'name') then
    raise exception 'model name % already belongs to a different artifact', p->>'name';
  end if;

  if v_id is null then
    insert into model_versions (name, architecture, git_sha, checkpoint_uri, temperature, t_low, t_high,
                                band_target, metrics, onnx_sha256, is_active)
    values (p->>'name', p->>'architecture', p->>'git_sha', p->>'onnx_file', (p->>'temperature')::float8,
            (p->>'t_low')::float8, (p->>'t_high')::float8, p->>'band_target', p->'metrics', v_sha, false)
    returning id into v_id;
  else
    update model_versions
       set onnx_sha256 = v_sha, git_sha = p->>'git_sha', temperature = (p->>'temperature')::float8,
           t_low = (p->>'t_low')::float8, t_high = (p->>'t_high')::float8, band_target = p->>'band_target',
           metrics = p->'metrics'
     where id = v_id;
  end if;

  update model_versions set is_active = false where is_active and id <> v_id;
  update model_versions set is_active = true, activated_at = case when is_active then coalesce(activated_at, now()) else now() end where id = v_id;
  return v_id;
end;
$$;

revoke execute on function public.activate_model_version(jsonb) from public, anon, authenticated;

-- Public feedback is a user's opinion, not a reference label. Nothing enters training without expert review.
alter table feedback add column if not exists source text not null default 'public_unverified'
  check (source in ('public_unverified', 'expert_reviewed'));
