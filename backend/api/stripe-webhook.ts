import type { VercelRequest, VercelResponse } from '@vercel/node';
import Stripe from 'stripe';
import { createClient } from '@supabase/supabase-js';

export const config = {
  api: {
    bodyParser: false,
  },
};

const STRIPE_SECRET_KEY = process.env.STRIPE_SECRET_KEY as string;
const STRIPE_WEBHOOK_SECRET = process.env.STRIPE_WEBHOOK_SECRET as string;
const SUPABASE_URL = process.env.SUPABASE_URL as string;
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY as string;

const stripe = new Stripe(STRIPE_SECRET_KEY, { apiVersion: '2023-10-16' });
const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const sig = req.headers['stripe-signature'] as string;
  const buf = await getRawBody(req);

  let event: Stripe.Event;
  try {
    event = stripe.webhooks.constructEvent(buf, sig, STRIPE_WEBHOOK_SECRET);
  } catch (err: any) {
    console.error('Stripe webhook signature verification failed', err.message);
    return res.status(400).send(`Webhook Error: ${err.message}`);
  }

  try {
    switch (event.type) {
      case 'customer.subscription.created':
      case 'customer.subscription.updated':
      case 'customer.subscription.deleted': {
        const subscription = event.data.object as Stripe.Subscription;
        const customerId = subscription.customer as string;
        const status = subscription.status;

        // Clerk user id is often stored as metadata on the Stripe customer or subscription
        const userId =
          (subscription.metadata && subscription.metadata.clerk_user_id) ||
          (subscription.items.data[0]?.price?.metadata?.clerk_user_id as string | undefined);

        if (!userId) {
          console.warn('No clerk_user_id metadata on subscription; skipping Supabase sync');
          break;
        }

        const planId = subscription.items.data[0]?.price?.id ?? 'unknown';

        const { error: upsertError } = await supabase.from('subscriptions').upsert(
          {
            user_id: userId,
            stripe_customer_id: customerId,
            stripe_subscription_id: subscription.id,
            plan_id: planId,
            status,
            current_period_start: new Date(subscription.current_period_start * 1000).toISOString(),
            current_period_end: new Date(subscription.current_period_end * 1000).toISOString(),
            cancel_at_period_end: subscription.cancel_at_period_end,
          },
          { onConflict: 'stripe_subscription_id' }
        );

        if (upsertError) {
          console.error('Supabase upsert error in subscriptions', upsertError);
          // Returning a 500 here tells Stripe to retry the webhook, keeping
          // Stripe and Supabase in sync rather than silently dropping failures.
          return res.status(500).json({ error: 'Failed to sync subscription' });
        }
        break;
      }
      default:
        // ignore other events
        break;
    }

    res.json({ received: true });
  } catch (err) {
    console.error('Error handling Stripe webhook', err);
    res.status(500).json({ error: 'Internal server error' });
  }
}

function getRawBody(req: VercelRequest): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on('data', (chunk) => chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk)));
    req.on('end', () => resolve(Buffer.concat(chunks)));
    req.on('error', reject);
  });
}

