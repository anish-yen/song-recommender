import { NextResponse } from 'next/server';
import { currentUserId } from '@/lib/auth';
import { prisma } from '@/lib/prisma';

export async function GET() {
  const userId = await currentUserId(); if (!userId) return NextResponse.json({ error: 'Sign in to view playlists.' }, { status: 401 });
  const playlists = await prisma.playlist.findMany({ where: { ownerId: userId }, include: { _count: { select: { tracks: true } } }, orderBy: { updatedAt: 'desc' } });
  return NextResponse.json({ playlists });
}

export async function POST(request) {
  try {
    const userId = await currentUserId(); if (!userId) return NextResponse.json({ error: 'Sign in to create playlists.' }, { status: 401 });
    const title = String((await request.json()).title || '').trim();
    if (title.length < 1 || title.length > 100) return NextResponse.json({ error: 'Playlist titles must be 1–100 characters.' }, { status: 400 });
    return NextResponse.json({ playlist: await prisma.playlist.create({ data: { ownerId: userId, title } }) }, { status: 201 });
  } catch { return NextResponse.json({ error: 'Unable to create playlist.' }, { status: 500 }); }
}
