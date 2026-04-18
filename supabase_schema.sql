create table if not exists public.analytics (
  id bigint generated always as identity primary key,
  created_at timestamptz not null default now(),
  sequence text not null,
  actual_label text not null check (actual_label in ('Human', 'Random')),
  p_human double precision not null,
  p_random double precision not null,
  model_prediction text not null check (model_prediction in ('Human', 'Random')),
  user_guess text check (user_guess is null or user_guess in ('Human', 'Random')),
  session_id text,
  batch_id text,
  batch_position integer check (batch_position is null or batch_position > 0),
  model_version text,
  sequence_length integer check (sequence_length is null or sequence_length > 0),
  source_mode text,
  explanation_tags jsonb
);

alter table public.analytics
alter column user_guess drop not null;

alter table public.analytics
add column if not exists session_id text;

alter table public.analytics
add column if not exists batch_id text;

alter table public.analytics
add column if not exists batch_position integer;

alter table public.analytics
add column if not exists model_version text;

alter table public.analytics
add column if not exists sequence_length integer;

alter table public.analytics
add column if not exists source_mode text;

alter table public.analytics
add column if not exists explanation_tags jsonb;

alter table public.analytics enable row level security;

drop policy if exists "Allow public analytics inserts" on public.analytics;
create policy "Allow public analytics inserts"
on public.analytics
for insert
to anon
with check (
  actual_label in ('Human', 'Random')
  and model_prediction in ('Human', 'Random')
  and (user_guess is null or user_guess in ('Human', 'Random'))
  and (batch_position is null or batch_position > 0)
  and (sequence_length is null or sequence_length > 0)
);

drop policy if exists "Allow public analytics reads" on public.analytics;

drop view if exists public.analytics_public_summary;
create view public.analytics_public_summary as
select
  count(*)::integer as total_rows,
  count(*) filter (where actual_label = 'Human')::integer as human_rows,
  count(*) filter (where actual_label = 'Random')::integer as random_rows,
  avg((model_prediction = actual_label)::integer::double precision) as model_accuracy,
  (
    count(*) filter (where model_prediction = 'Human' and actual_label = 'Human')::double precision
    / nullif(count(*) filter (where model_prediction = 'Human'), 0)
  ) as human_precision,
  (
    count(*) filter (where model_prediction = 'Human' and actual_label = 'Human')::double precision
    / nullif(count(*) filter (where actual_label = 'Human'), 0)
  ) as human_recall,
  (
    count(*) filter (where model_prediction = 'Random' and actual_label = 'Random')::double precision
    / nullif(count(*) filter (where model_prediction = 'Random'), 0)
  ) as random_precision,
  (
    count(*) filter (where model_prediction = 'Random' and actual_label = 'Random')::double precision
    / nullif(count(*) filter (where actual_label = 'Random'), 0)
  ) as random_recall,
  avg(p_human) filter (where actual_label = 'Human') as avg_p_human_for_human,
  avg(p_human) filter (where actual_label = 'Random') as avg_p_human_for_random,
  count(*) filter (where user_guess in ('Human', 'Random'))::integer as guessed_rows,
  avg((user_guess = actual_label)::integer::double precision)
    filter (where user_guess in ('Human', 'Random')) as user_accuracy
from public.analytics;

grant select on public.analytics_public_summary to anon;
