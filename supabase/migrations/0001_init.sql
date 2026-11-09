-- Leukemia cell classifier: initial schema.
-- Research demo. Not for clinical use. No patient data is stored; predictions log only an image SHA-256.

create extension if not exists "pgcrypto";

create table model_versions (
  id uuid primary key default gen_random_uuid(),
  name text not null unique,
  architecture text not null,
  git_sha text,
  checkpoint_uri text,
  temperature double precision not null default 1.0,
  t_low double precision,
  t_high double precision,
  band_target text,
  metrics jsonb,
  is_active boolean not null default false,
  created_at timestamptz not null default now()
);

-- at most one active model
create unique index model_versions_one_active on model_versions (is_active) where is_active;

create table training_runs (
  id uuid primary key default gen_random_uuid(),
  model_version_id uuid references model_versions(id),
  seed int not null,
  fold int,
  config jsonb not null,
  status text not null default 'queued',
  notes text,
  started_at timestamptz,
  finished_at timestamptz
);

create table run_metrics (
  id bigserial primary key,
  run_id uuid not null references training_runs(id) on delete cascade,
  split text not null check (split in ('train','val','test','external')),
  metric text not null,
  value double precision not null,
  ci_low double precision,
  ci_high double precision
);
create index on run_metrics (run_id);

create table dataset_manifest (
  id bigserial primary key,
  image_path text not null unique,
  patient_id text not null,
  label smallint not null check (label in (0,1)),
  split text not null check (split in ('train','val','test')),
  fold int
);
create index on dataset_manifest (patient_id);

create table predictions (
  id uuid primary key default gen_random_uuid(),
  model_version_id uuid not null references model_versions(id),
  image_sha256 text not null,
  prob_raw double precision not null,
  prob_calibrated double precision not null,
  decision text not null check (decision in ('leukemic','normal','refer_to_expert')),
  latency_ms int,
  session_id text,
  created_at timestamptz not null default now()
);
create index on predictions (created_at desc);

create table feedback (
  id uuid primary key default gen_random_uuid(),
  prediction_id uuid not null references predictions(id) on delete cascade,
  user_label text not null check (user_label in ('leukemic','normal','unsure')),
  comment text check (char_length(comment) <= 500),
  created_at timestamptz not null default now()
);

-- Row Level Security: everything is service-role only except reading the active model card.
alter table model_versions enable row level security;
alter table training_runs enable row level security;
alter table run_metrics enable row level security;
alter table dataset_manifest enable row level security;
alter table predictions enable row level security;
alter table feedback enable row level security;

create policy "read active model" on model_versions
  for select using (is_active = true);

-- Opt-in image uploads: private bucket, no public policies (service role only).
-- Objects older than 24 h are removed by api/cleanup.py (run on a schedule).
insert into storage.buckets (id, name, public)
values ('uploads', 'uploads', false)
on conflict (id) do nothing;
