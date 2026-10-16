export function trackIdentity(input) {
  return {
    title: String(input.title || '').slice(0, 160),
    artist: String(input.artist || '').slice(0, 160),
    album: input.album ? String(input.album).slice(0, 160) : null,
    coverArtUrl: safeUrl(input.coverArtUrl), previewUrl: safeUrl(input.previewUrl), providerUrl: safeUrl(input.providerUrl),
    providerId: input.providerId ? String(input.providerId).slice(0, 180) : null,
    durationSeconds: Number.isInteger(input.durationSeconds) ? input.durationSeconds : null,
    source: input.source ? String(input.source).slice(0, 120) : null,
    license: input.license ? String(input.license).slice(0, 160) : null,
  };
}

function safeUrl(value) {
  if (!value) return null;
  try { const url = new URL(value); return ['http:', 'https:'].includes(url.protocol) ? url.toString() : null; } catch { return null; }
}
