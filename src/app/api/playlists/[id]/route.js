import { NextResponse } from 'next/server';
import { currentUserId } from '@/lib/auth';
import { prisma } from '@/lib/prisma';
import { assertOwner } from '@/lib/playlist-access';

async function owned(id) { const userId = await currentUserId(); if (!userId) return [null, NextResponse.json({ error: 'Sign in required.' }, { status: 401 })]; const playlist = await prisma.playlist.findUnique({ where: { id }, include: { tracks: { include: { track: true }, orderBy: { position: 'asc' } } } }); try { return [assertOwner(playlist, userId)]; } catch { return [null, NextResponse.json({ error: 'Playlist not found.' }, { status: 404 })]; } }
export async function GET(_request, { params }) { const { id } = await params; const [playlist, error] = await owned(id); return error || NextResponse.json({ playlist }); }
export async function DELETE(_request, { params }) { const { id } = await params; const [playlist, error] = await owned(id); if (error) return error; await prisma.playlist.delete({ where: { id: playlist.id } }); return NextResponse.json({ ok: true }); }
