import bcrypt from 'bcryptjs';
import { NextResponse } from 'next/server';
import { prisma } from '@/lib/prisma';
import { signSession, sessionCookie } from '@/lib/auth';
import { validateCredentials } from '@/lib/validation';

export async function POST(request) {
  try {
    const body = await request.json();
    const valid = validateCredentials(body, true);
    if (valid.error) return NextResponse.json({ error: valid.error }, { status: 400 });
    if (body.password !== body.confirmPassword) return NextResponse.json({ error: 'Passwords do not match.' }, { status: 400 });
    const existing = await prisma.user.findUnique({ where: { email: valid.data.email } });
    if (existing) return NextResponse.json({ error: 'An account with that email already exists.' }, { status: 409 });
    const user = await prisma.user.create({ data: { name: valid.data.name, email: valid.data.email, passwordHash: await bcrypt.hash(valid.data.password, 12) } });
    const response = NextResponse.json({ user: { id: user.id, name: user.name, email: user.email } }, { status: 201 });
    const cookie = sessionCookie(signSession(user)); response.cookies.set(cookie.name, cookie.value, cookie.options); return response;
  } catch { return NextResponse.json({ error: 'Unable to create your account right now.' }, { status: 500 }); }
}
