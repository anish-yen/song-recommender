import { NextResponse } from 'next/server';
import { clearSessionCookie } from '@/lib/auth';
export async function POST() { const response = NextResponse.json({ ok: true }); const cookie = clearSessionCookie(); response.cookies.set(cookie.name, cookie.value, cookie.options); return response; }
