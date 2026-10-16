import { NextResponse } from 'next/server';
import { findTracks } from '@/lib/recommendation-client';

export async function GET(request) {
  const q = (new URL(request.url).searchParams.get('q') || '').trim().slice(0, 200);
  if (!q) return NextResponse.json({ tracks: [] });
  try { return NextResponse.json(await findTracks(q)); } catch (error) { return NextResponse.json({ tracks: [], error: error.message }, { status: error.status || 503 }); }
}
