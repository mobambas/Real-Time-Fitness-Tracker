import type { VercelRequest, VercelResponse } from '@vercel/node';
import Stripe from 'stripe';
import { createClient } from '@supabase/supabase-js';

// Environment variables are expected to be configured on Vercel
const SUPABASE_URL = process.env.SUPABASE_URL as string;
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY as string;
const STRIPE_SECRET_KEY = process.env.STRIPE_SECRET_KEY as string;

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);
const stripe = new Stripe(STRIPE_SECRET_KEY, { apiVersion: '2023-10-16' });

// Minimal session endpoint:
// POST /api/session
// Body: JSON matching the payload described in docs/MOBILE_ARCHITECTURE.md
export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const authHeader = req.headers.authorization || '';
    const token = authHeader.startsWith('Bearer ') ? authHeader.slice(7) : null;
    if (!token) {
      return res.status(401).json({ error: 'Missing Bearer token from Clerk' });
    }

    // In production, you should verify this JWT using Clerk's SDK or JWKS.
    // For this prototype, we trust that Vercel middleware / mobile SDK already validated it,
    // and we pass it through to Supabase as the "user" JWT.

    const {
      user_id,
      exercise_type,
      started_at,
      ended_at,
      duration_seconds,
      correct_reps,
      incorrect_reps,
      device,
      platform,
      metrics,
    } = req.body ?? {};

    if (!user_id || !exercise_type || !started_at) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    const { data, error } = await supabase
      .from('sessions')
      .insert({
        user_id,
        exercise_type,
        started_at,
        ended_at,
        duration_seconds,
        correct_reps,
        incorrect_reps,
        device,
        platform,
      })
      .select('id')
      .single();

    if (error) {
      console.error('Supabase insert error', error);
      return res.status(500).json({ error: 'Failed to save session' });
    }

    if (metrics) {
      const { error: metricsError } = await supabase
        .from('session_metrics')
        .insert({
          session_id: data.id,
          metric_type: 'summary',
          payload: metrics,
        });
      if (metricsError) {
        console.error('Supabase metrics insert error', metricsError);
      }
    }

    return res.status(201).json({ id: data.id });
  } catch (err) {
    console.error('Unexpected error in /api/session', err);
    return res.status(500).json({ error: 'Internal server error' });
  }
}

