-- Row Level Security policies for Supabase with Clerk JWTs.
-- Assumes Clerk user id is in auth.jwt()->>'sub'

-- Enable RLS
alter table public.profiles enable row level security;
alter table public.sessions enable row level security;
alter table public.session_metrics enable row level security;
alter table public.subscriptions enable row level security;

-- PROFILES
create policy "Profiles: users can view their profile"
on public.profiles
for select
using (user_id = auth.jwt()->>'sub');

create policy "Profiles: users can insert their profile"
on public.profiles
for insert
with check (user_id = auth.jwt()->>'sub');

create policy "Profiles: users can update their profile"
on public.profiles
for update
using (user_id = auth.jwt()->>'sub')
with check (user_id = auth.jwt()->>'sub');

-- SESSIONS
create policy "Sessions: users can view their sessions"
on public.sessions
for select
using (user_id = auth.jwt()->>'sub');

create policy "Sessions: users can insert their sessions"
on public.sessions
for insert
with check (user_id = auth.jwt()->>'sub');

-- Typically no direct update/delete from clients; handle via backend if needed.

-- SESSION METRICS
create policy "SessionMetrics: users can view metrics of their sessions"
on public.session_metrics
for select
using (
  exists (
    select 1 from public.sessions s
    where s.id = session_metrics.session_id
      and s.user_id = auth.jwt()->>'sub'
  )
);

create policy "SessionMetrics: users can insert metrics for their sessions"
on public.session_metrics
for insert
with check (
  exists (
    select 1 from public.sessions s
    where s.id = session_metrics.session_id
      and s.user_id = auth.jwt()->>'sub'
  )
);

-- SUBSCRIPTIONS
create policy "Subscriptions: users can view their subscriptions"
on public.subscriptions
for select
using (user_id = auth.jwt()->>'sub');

-- Writes are typically from secure server-side Stripe webhooks; do not allow client inserts/updates.

