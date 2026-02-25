## End-to-End Deployment Guide

This guide explains how to turn this repo into a working product with:

- A Flutter (or native) mobile app that performs real-time pose tracking on-device.
- A Supabase + Vercel backend with Clerk auth and Stripe billing.
- Analytics (PostHog), error tracking (Sentry), emails (Resend), and optional Redis/Pinecone.

Use this as a checklist when you’re ready to move beyond the Python prototype.

### 1. Supabase Setup

- Create a new Supabase project (free tier).
- In the SQL editor:
  - Run `docs/BACKEND_SUPABASE_SCHEMA.sql`.
  - Run `docs/SUPABASE_RLS_POLICIES.sql`.
- In **Authentication → Providers**, add **Clerk** as an external provider (see Clerk docs for exact values).
- Copy:
  - `SUPABASE_URL`
  - `SUPABASE_ANON_KEY`
  - `SUPABASE_SERVICE_ROLE_KEY`
- Paste these into:
  - Vercel project env vars.
  - `.env` for local backend work (or `.env.local` in a separate backend repo).

### 2. Clerk Authentication

- Create a project in Clerk (free tier) and configure:
  - Sign-in methods you want (email/password, Google, Apple, etc.).
  - JWT templates with `sub` claim equal to Clerk user id.
- In Clerk Dashboard:
  - Note `CLERK_PUBLISHABLE_KEY` and `CLERK_SECRET_KEY`.
  - (Optional) Add metadata fields for subscription tier, plan, etc.
- In your mobile app:
  - Use Clerk’s **Flutter / React Native** SDK.
  - After sign-in, retrieve the user’s JWT and attach it as `Authorization: Bearer <token>` on calls to the Vercel API.

### 3. Stripe Billing

- In Stripe Dashboard:
  - Create Products and Prices (e.g. `basic_monthly`, `premium_monthly`).
  - Configure webhooks to point to your deployed `backend/api/stripe-webhook` function on Vercel.
  - Add signing secret to `STRIPE_WEBHOOK_SECRET`.
- Map Clerk users to Stripe customers:
  - When creating Checkout sessions, pass `clerk_user_id` in Stripe `customer` or `subscription` metadata.
  - The webhook handler in `backend/api/stripe-webhook.ts` reads that metadata and upserts `public.subscriptions`.
- In mobile:
  - Use Stripe’s Flutter plugin or a WebView to open Stripe Checkout / Billing portal.
  - After payment, the mobile app can poll the `/api/profile` or a `/api/subscription` endpoint to update local UI.

### 4. Vercel Backend

- Either:
  - Move the `backend/` folder into its own repo, or
  - Point a separate Vercel project at this repo and set the root to `backend/`.
- On Vercel:
  - Connect the GitHub repo.
  - Set build command to `npm install` (or `pnpm install`) in `backend/` if needed; otherwise Vercel will auto-detect.
  - Configure env vars:
    - `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`
    - `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET`
    - `CLERK_SECRET_KEY` (if you choose to verify JWTs server-side here)
    - `POSTHOG_API_KEY`, `POSTHOG_HOST` (optional)
    - `SENTRY_DSN` (optional)
    - `RESEND_API_KEY` (optional)
    - `UPSTASH_REDIS_REST_URL`, `UPSTASH_REDIS_REST_TOKEN` (optional)
    - `PINECONE_API_KEY`, `PINECONE_INDEX_NAME` (optional)
- Endpoints:
  - `POST /api/session` → records a workout session + metrics into Supabase.
  - `GET /api/profile?user_id=...` → returns profile row for a Clerk user id.
  - `POST /api/stripe-webhook` → handles subscription lifecycle events.

### 5. Mobile App (Flutter) Scaffold

- Create a Flutter project (outside this repo):
  ```bash
  flutter create fitmaster_mobile
  cd fitmaster_mobile
  ```
- Add packages:
  - Camera & vision:
    - `camera`
    - A MediaPipe or TFLite plugin for pose (e.g. mediapipe via platform channel or tflite_flutter).
  - Networking / auth:
    - `http` or `dio`
    - Clerk SDK for Flutter (or use REST/JWT dance).
  - State management of your choice (e.g. `riverpod` or `bloc`).
- Implement feature flow:
  - Auth screens (via Clerk).
  - Home screen listing supported exercises.
  - Workout screen:
    - Camera preview.
    - Pose model inference pipeline.
    - Ported exercise logic as shown in `docs/MOBILE_ARCHITECTURE.md` (angle, stage machine, rep counting).
    - Overlay UI: rep counters, correct/incorrect counts, stars/form rating, alerts.
  - On session end:
    - Build a JSON payload (`exercise_type`, `started_at`, `ended_at`, `correct_reps`, `incorrect_reps`, `metrics`).
    - Send it to `POST /api/session` with the user’s Clerk JWT.

### 6. Analytics, Error Tracking, and Emails

- **PostHog**:
  - Sign up for PostHog (cloud or self-hosted).
  - In mobile, initialize PostHog SDK with `POSTHOG_API_KEY`.
  - Track key events:
    - `user_sign_up`, `session_started`, `session_completed`, `purchase_completed`.
- **Sentry**:
  - In mobile, integrate Sentry Flutter SDK; initialize with `SENTRY_DSN`.
  - In Vercel backend, wrap handlers with Sentry middleware (optional but recommended).
- **Resend**:
  - Configure `RESEND_API_KEY`.
  - From backend, send:
    - Welcome email on new profile created.
    - Optional weekly summary email after sessions aggregation.

### 7. Optional: Redis (Upstash) and Pinecone

- **Upstash Redis**:
  - Use for small, ephemeral data:
    - Caching last session summary.
    - Rate limiting (e.g. max N sessions per day on free plan).
  - Use REST API from `backend/api/*` handlers; store URLs and tokens in env vars.
- **Pinecone**:
  - Use for semantic search / AI:
    - Store embeddings of help articles or workout tips.
    - From mobile, send a question; backend embeds it and queries Pinecone, then returns top matches.

### 8. Google Play Store Preparation

- Ensure your app:
  - Targets latest SDK (Android 15 / API 35 or newer when required).
  - Declares camera permission:
    ```xml
    <uses-permission android:name="android.permission.CAMERA" />
    <uses-feature android:name="android.hardware.camera.any" />
    ```
  - Has a clear Privacy Policy hosted at a public URL describing:
    - Use of camera and pose data.
    - What is stored in Supabase (sessions, metrics).
    - How billing and analytics work.
- Prepare assets:
  - 512×512 icon (no rounded corners, PNG).
  - 1024×500 feature graphic.
  - At least 2–4 in-app screenshots (portrait recommended for fitness).
- Use internal / closed testing tracks first to validate performance and stability.

### 9. Recommended Development Workflow in Cursor

- Keep Python scripts as a **reference implementation** for angles and thresholds.
- When porting logic:
  - Open the relevant Python file (e.g. `push-up.py`) for thresholds and heuristics.
  - Implement a small, pure Dart function with clear inputs/outputs (no UI).
  - Write unit tests in Dart for known landmark positions.
- Use feature branches:
  - `feature/mobile-pushup-logic`
  - `feature/backend-stripe-integration`
  - `feature/analytics`
- Let Cursor help:
  - Ask for single-purpose functions (e.g. “convert this Python angle logic to Dart”).
  - Use the docs in this repo (`docs/MOBILE_ARCHITECTURE.md`, this guide) as shared reference.

Following these steps, the current Python-based prototype becomes:

- A desktop **reference** for pose logic and UX experiments.
- A **mobile-first** app with on-device pose estimation and rep counting.
- A **scalable backend** for persistence, subscriptions, analytics, and monitoring.

