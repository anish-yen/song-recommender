const mockResults = [
  { providerId: 'mock-01', title: 'Daylight Arcade', artist: 'Mira Vale', album: 'Night Notes', coverArtUrl: null, providerUrl: 'https://example.com/track/daylight-arcade', source: 'development mock', why: 'Mock result for checking the discovery flow before the recommendation service is connected.' },
  { providerId: 'mock-02', title: 'Quiet Current', artist: 'Elo Pine', album: 'Soft Signal', coverArtUrl: null, providerUrl: 'https://example.com/track/quiet-current', source: 'development mock', why: 'Mock result for testing save, reject, and playlist actions.' },
  { providerId: 'mock-03', title: 'After the Metro', artist: 'North Bloom', album: 'Late Transit', coverArtUrl: null, providerUrl: 'https://example.com/track/after-the-metro', source: 'development mock', why: 'Mock result, clearly marked until a recommendation backend returns live matches.' },
];

const baseUrl = () => (process.env.RECOMMENDATION_API_URL || '').replace(/\/$/, '');

// The engine reports problems as { detail: "..." }; surface that message rather than a generic one.
async function engineJson(response, fallback) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(typeof data.detail === 'string' ? data.detail : fallback);
    error.status = response.status;
    throw error;
  }
  return data;
}

export async function getRecommendations(payload) {
  if (!baseUrl()) return { results: mockResults, mode: 'mock', queryId: `mock-${Date.now()}` };
  const response = await fetch(`${baseUrl()}/recommendations`, {
    method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(payload), cache: 'no-store',
  });
  const data = await engineJson(response, 'The recommendation service is currently unavailable.');
  return { results: Array.isArray(data.results) ? data.results : [], mode: data.mode || 'live', queryId: data.queryId || null, hidden: data.hidden || null };
}

export async function getFilterOptions() {
  if (!baseUrl()) return { available: false };
  return engineJson(await fetch(`${baseUrl()}/filters`, { cache: 'no-store' }), 'Filter options are unavailable.');
}

export async function findTracks(q) {
  if (!baseUrl()) return { tracks: [] };
  const response = await fetch(`${baseUrl()}/tracks?${new URLSearchParams({ q, limit: '8' })}`, { cache: 'no-store' });
  return engineJson(response, 'Track search is unavailable.');
}
