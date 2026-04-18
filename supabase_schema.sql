create table if not exists public.analytics (
  id bigint generated always as identity primary key,
  created_at timestamptz not null default now(),
  sequence text not null,
  actual_label text not null check (actual_label in ('Human', 'Random')),
  p_human double precision not null,
  p_random double precision not null,
  model_prediction text not null check (model_prediction in ('Human', 'Random')),
  user_guess text not null check (user_guess in ('Human', 'Random'))
);

alter table public.analytics enable row level security;

drop policy if exists "Allow public analytics inserts" on public.analytics;
create policy "Allow public analytics inserts"
on public.analytics
for insert
to anon
with check (
  actual_label in ('Human', 'Random')
  and model_prediction in ('Human', 'Random')
  and user_guess in ('Human', 'Random')
);

drop policy if exists "Allow public analytics reads" on public.analytics;
create policy "Allow public analytics reads"
on public.analytics
for select
to anon
using (true);
