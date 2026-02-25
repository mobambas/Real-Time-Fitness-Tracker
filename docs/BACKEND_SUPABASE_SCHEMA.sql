-- Core Supabase schema for Real-Time Fitness Tracker

-- Enable UUID extension if not already enabled
create extension if not exists "uuid-ossp";

-- Profiles table, 1:1 with auth.users (Clerk user id in sub claim)
create table if not exists public.profiles (
  id uuid primary key default uuid_generate_v4(),
  user_id text unique not null,
  email text,
  first_name text,
  last_name text,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create index if not exists idx_profiles_user_id on public.profiles (user_id);

-- Workout sessions (high-level per workout)
create table if not exists public.sessions (
  id uuid primary key default uuid_generate_v4(),
  user_id text not null,
  exercise_type text not null,
  started_at timestamptz not null,
  ended_at timestamptz,
  duration_seconds integer,
  correct_reps integer default 0,
  incorrect_reps integer default 0,
  device text,
  platform text, -- android / ios / web
  created_at timestamptz default now()
);

create index if not exists idx_sessions_user_id on public.sessions (user_id);
create index if not exists idx_sessions_exercise_type on public.sessions (exercise_type);

-- Optional detailed metrics per session (JSON for flexibility)
create table if not exists public.session_metrics (
  id uuid primary key default uuid_generate_v4(),
  session_id uuid not null references public.sessions (id) on delete cascade,
  metric_type text not null, -- e.g. 'time_series', 'summary'
  payload jsonb not null,
  created_at timestamptz default now()
);

create index if not exists idx_session_metrics_session_id on public.session_metrics (session_id);

-- Stripe subscriptions (if you store subscription state in Supabase)
create table if not exists public.subscriptions (
  id uuid primary key default uuid_generate_v4(),
  user_id text not null,
  stripe_customer_id text not null,
  stripe_subscription_id text not null,
  plan_id text not null,
  status text not null, -- active / trialing / canceled / past_due
  current_period_start timestamptz,
  current_period_end timestamptz,
  cancel_at_period_end boolean default false,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create index if not exists idx_subscriptions_user_id on public.subscriptions (user_id);
create index if not exists idx_subscriptions_stripe_customer_id on public.subscriptions (stripe_customer_id);

