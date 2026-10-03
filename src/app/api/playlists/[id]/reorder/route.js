import { NextResponse } from 'next/server';
import { currentUserId } from '@/lib/auth';
import { prisma } from '@/lib/prisma';
import { assertOwner, orderedPositions } from '@/lib/playlist-access';

export async function PATCH(request, { params }) {
  try { const { id } = await params; const userId = await currentUserId(); if (!userId) return NextResponse.json({ error: 'Sign in required.' }, { status: 401 }); const playlist = await prisma.playlist.findUnique({ where: { id }, include: { tracks: true } }); assertOwner(playlist, userId); const positions = orderedPositions((await request.json()).trackIds, playlist.tracks.map((track) => track.trackId)); await prisma.$transaction(async (tx) => { await tx.playlistTrack.updateMany({ where: { playlistId: id }, data: { position: { increment: 10000 } } }); await Promise.all(positions.map(({ trackId, position }) => tx.playlistTrack.update({ where: { playlistId_trackId: { playlistId: id, trackId } }, data: { position } }))); }); return NextResponse.json({ ok: true }); } catch (error) { return NextResponse.json({ error: error.message === 'Playlist not found.' ? error.message : 'Unable to reorder playlist.' }, { status: error.status || 400 }); }
}
