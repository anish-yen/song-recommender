import { NextResponse } from 'next/server';
import { getFilterOptions } from '@/lib/recommendation-client';

export async function GET() {
  try { return NextResponse.json(await getFilterOptions()); } catch (error) { return NextResponse.json({ available: false, error: error.message }, { status: 503 }); }
}
