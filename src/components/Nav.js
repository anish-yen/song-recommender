'use client';
import Link from 'next/link';
import { useEffect, useState } from 'react';

export default function Nav() {
  const [user, setUser] = useState(null);
  useEffect(() => { fetch('/api/auth/session').then((r) => r.json()).then(({ user }) => setUser(user)).catch(() => {}); }, []);
  async function logout() { await fetch('/api/auth/logout', { method: 'POST' }); window.location.href = '/'; }
  return <nav className="nav"><Link className="wordmark" href="/">afterglow<span>•</span></Link><div className="nav-links"><Link href="/discover">Discover</Link>{user ? <><Link href="/playlists">Library</Link><button className="text-button" onClick={logout}>Log out</button></> : <><Link href="/login">Log in</Link><Link className="button small" href="/signup">Join free</Link></>}</div></nav>;
}
