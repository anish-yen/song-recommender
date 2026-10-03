import { NextResponse } from 'next/server';
import { getRecommendations } from '@/lib/recommendation-client';

export async function POST(request) {
  try {
    const body = await request.json();
    const query = String(body.query || '').trim(); const seed = body.seed && typeof body.seed === 'object' ? body.seed : null;
    if (!query && !seed?.title) return NextResponse.json({ error: 'Describe what you want or select a seed song.' }, { status: 400 });
    return NextResponse.json(await getRecommendations({ query: query.slice(0, 400), seed, filters: body.filters || {} }));
  } catch (error) { return NextResponse.json({ error: error.message || 'Recommendations are unavailable.' }, { status: 503 }); }
}
