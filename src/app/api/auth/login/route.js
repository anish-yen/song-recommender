import bcrypt from 'bcryptjs';
import { NextResponse } from 'next/server';
import { prisma } from '@/lib/prisma';
import { signSession, sessionCookie } from '@/lib/auth';
import { validateCredentials } from '@/lib/validation';

export async function POST(request) {
  try {
    const valid = validateCredentials(await request.json());
    if (valid.error) return NextResponse.json({ error: valid.error }, { status: 400 });
    const user = await prisma.user.findUnique({ where: { email: valid.data.email } });
    if (!user || !(await bcrypt.compare(valid.data.password, user.passwordHash))) return NextResponse.json({ error: 'Email or password is incorrect.' }, { status: 401 });
    const response = NextResponse.json({ user: { id: user.id, name: user.name, email: user.email } });
    const cookie = sessionCookie(signSession(user)); response.cookies.set(cookie.name, cookie.value, cookie.options); return response;
  } catch { return NextResponse.json({ error: 'Unable to sign in right now.' }, { status: 500 }); }
}
