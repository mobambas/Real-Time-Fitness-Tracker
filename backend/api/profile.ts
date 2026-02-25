import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

const SUPABASE_URL = process.env.SUPABASE_URL as string;
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY as string;

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

// GET /api/profile?user_id=...
export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const userId = (req.query.user_id as string) || null;
  if (!userId) {
    return res.status(400).json({ error: 'Missing user_id query param' });
  }

  try {
    const { data, error } = await supabase
      .from('profiles')
      .select('*')
      .eq('user_id', userId)
      .single();

    if (error) {
      if (error.code === 'PGRST116') {
        // not found
        return res.status(404).json({ error: 'Profile not found' });
      }
      console.error('Supabase profile select error', error);
      return res.status(500).json({ error: 'Failed to fetch profile' });
    }

    return res.status(200).json({ profile: data });
  } catch (err) {
    console.error('Unexpected error in /api/profile', err);
    return res.status(500).json({ error: 'Internal server error' });
  }
}

