import { NextResponse } from 'next/server';
import { getRecommendations } from '@/lib/recommendation-client';

// Results requested per search. Kept at 5 while the catalog is small (40 tracks): a top 10
// would be a quarter of the catalog, so unrelated queries share most of their lists.
// Raise to 10 once the full FMA-small index (8,000 tracks) is serving.
const RESULT_COUNT = 5;

export async function POST(request) {
  try {
    const body = await request.json();
    const query = String(body.query || '').trim(); const seed = body.seed && typeof body.seed === 'object' ? body.seed : null;
    if (!query && !seed?.title) return NextResponse.json({ error: 'Describe what you want or select a seed song.' }, { status: 400 });
    return NextResponse.json(await getRecommendations({ query: query.slice(0, 400), seed, filters: body.filters || {}, k: RESULT_COUNT }));
  } catch (error) { return NextResponse.json({ error: error.message || 'Recommendations are unavailable.' }, { status: error.status && error.status < 500 ? error.status : 503 }); }
}
