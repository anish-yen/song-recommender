import jwt from 'jsonwebtoken';
import { cookies } from 'next/headers';

const COOKIE_NAME = 'song_session';
const maxAge = 60 * 60 * 24 * 7;

function secret() {
  if (!process.env.JWT_SECRET || process.env.JWT_SECRET.length < 32) {
    throw new Error('Session configuration is unavailable.');
  }
  return process.env.JWT_SECRET;
}

export function signSession(user) {
  return jwt.sign({ sub: user.id, name: user.name }, secret(), { expiresIn: maxAge });
}

export function verifySession(token) {
  try { return jwt.verify(token, secret()); } catch { return null; }
}

export async function currentUserId() {
  const token = (await cookies()).get(COOKIE_NAME)?.value;
  return token ? verifySession(token)?.sub || null : null;
}

export function sessionCookie(token) {
  return { name: COOKIE_NAME, value: token, options: {
    httpOnly: true, sameSite: 'lax', secure: process.env.NODE_ENV === 'production', path: '/', maxAge,
  }};
}

export function clearSessionCookie() {
  return { name: COOKIE_NAME, value: '', options: { httpOnly: true, sameSite: 'lax', secure: process.env.NODE_ENV === 'production', path: '/', maxAge: 0 } };
}
