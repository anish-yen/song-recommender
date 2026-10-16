const mockResults = [
  { providerId: 'mock-01', title: 'Daylight Arcade', artist: 'Mira Vale', album: 'Night Notes', coverArtUrl: null, providerUrl: 'https://example.com/track/daylight-arcade', source: 'development mock', why: 'Mock result for checking the discovery flow before the recommendation service is connected.' },
  { providerId: 'mock-02', title: 'Quiet Current', artist: 'Elo Pine', album: 'Soft Signal', coverArtUrl: null, providerUrl: 'https://example.com/track/quiet-current', source: 'development mock', why: 'Mock result for testing save, reject, and playlist actions.' },
  { providerId: 'mock-03', title: 'After the Metro', artist: 'North Bloom', album: 'Late Transit', coverArtUrl: null, providerUrl: 'https://example.com/track/after-the-metro', source: 'development mock', why: 'Mock result, clearly marked until a recommendation backend returns live matches.' },
];

export async function getRecommendations(payload) {
  if (!process.env.RECOMMENDATION_API_URL) return { results: mockResults, mode: 'mock', queryId: `mock-${Date.now()}` };
  const response = await fetch(`${process.env.RECOMMENDATION_API_URL.replace(/\/$/, '')}/recommendations`, {
    method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(payload), cache: 'no-store',
  });
  if (!response.ok) throw new Error('The recommendation service is currently unavailable.');
  const data = await response.json();
  return { results: Array.isArray(data.results) ? data.results : [], mode: 'live', queryId: data.queryId || null };
}
