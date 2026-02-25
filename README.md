# Real-Time Fitness Tracker (FitMaster AI)

Python-based real-time form tracking and rep counting for common exercises using OpenCV and MediaPipe Pose. This repo also includes a high-level backend and mobile architecture so you can turn the core CV logic into a cross-platform mobile app with Supabase, Clerk, Stripe, and other services.

## Python Desktop Prototype

- **Exercises**: squats, push-ups, lunges, bicep curls, pull-ups, lat pulldown, sit-ups.
- **Tech**: `opencv-python`, `mediapipe`, `numpy`, `simpleaudio`, `google-genai` (for the chatbot).

### Setup

1. Create and activate a virtualenv (recommended):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Create a `.env` file:
   ```bash
   copy .env.example .env
   ```
   Fill in `GOOGLE_API_KEY` if you want to use the fitness chatbot (`chatbot.py`).

### Running an Exercise Tracker

Each exercise script opens its own window and uses your default camera:

```bash
python squats.py
python push-up.py
python lunges.py
python bicep_curls.py
python pull_ups.py
python lat_pulldown.py
python sit-up.py
```

Close the window or press the on-screen/keyboard exit control (usually `q` or `e`) to end a session. Sessions are recorded as `.mp4` files alongside the scripts.

### Fitness Chatbot

```bash
python chatbot.py
```

This uses Gemini via `google-genai` and your `GOOGLE_API_KEY` from `.env`.

## Turning This Repo into a Mobile App

This repo includes documentation and scaffolding to build a full mobile product around the CV core:

- **`docs/MOBILE_ARCHITECTURE.md`**: Flutter/native mobile architecture, camera + on-device pose estimation, and how to port the angle/rep logic.
- **`docs/BACKEND_SUPABASE_SCHEMA.sql`**: Core Supabase tables (`profiles`, `sessions`, etc.).
- **`docs/SUPABASE_RLS_POLICIES.sql`**: Example row-level security policies integrated with Clerk JWTs.
- **`backend/`**: Example Vercel-compatible API functions for sessions and user profile, plus Stripe webhook handling.
- **`docs/DEPLOYMENT_GUIDE.md`**: End-to-end steps: Supabase, Clerk, Stripe, Vercel, PostHog, Sentry, Resend, and Google Play preparation.

Follow the step-by-step guide in `docs/DEPLOYMENT_GUIDE.md` after you’ve reviewed the Python prototype. That guide walks you from this repo to:

- A Flutter (or native) mobile app with on-device pose estimation and rep counting.
- A Supabase + Vercel backend with Clerk auth and Stripe billing.
- Production-ready analytics, error tracking, and email infrastructure.