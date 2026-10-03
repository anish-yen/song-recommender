import { NextResponse } from 'next/server';
import { currentUserId } from '@/lib/auth';
import { prisma } from '@/lib/prisma';
import { trackIdentity } from '@/lib/tracks';

async function persistTrack(input) {
  const data = trackIdentity(input);
  if (!data.title || !data.artist) throw new Error('Track details are incomplete.');
  const existing = data.providerId && data.providerUrl ? await prisma.track.findUnique({ where: { providerId_providerUrl: { providerId: data.providerId, providerUrl: data.providerUrl } } }) : await prisma.track.findFirst({ where: { title: data.title, artist: data.artist, providerUrl: data.providerUrl } });
  return existing || prisma.track.create({ data });
}

export async function POST(request) {
  try {
    const userId = await currentUserId(); if (!userId) return NextResponse.json({ error: 'Sign in to save feedback.' }, { status: 401 });
    const body = await request.json(); if (!['SAVE', 'REJECT', 'OPEN', 'SKIP'].includes(body.action)) return NextResponse.json({ error: 'Unsupported feedback action.' }, { status: 400 });
    const track = await persistTrack(body.track || {});
    await prisma.feedbackEvent.create({ data: { userId, trackId: track.id, queryId: body.queryId || null, action: body.action } });
    return NextResponse.json({ ok: true, trackId: track.id });
  } catch (error) { return NextResponse.json({ error: error.message || 'Unable to record feedback.' }, { status: 400 }); }
}
