export function assertOwner(playlist, userId) {
  if (!playlist || playlist.ownerId !== userId) {
    const error = new Error('Playlist not found.');
    error.status = 404;
    throw error;
  }
  return playlist;
}

export function orderedPositions(trackIds, knownTrackIds) {
  if (!Array.isArray(trackIds) || trackIds.length !== knownTrackIds.length) throw new Error('Playlist order is incomplete.');
  if (new Set(trackIds).size !== trackIds.length || trackIds.some((id) => !knownTrackIds.includes(id))) throw new Error('Playlist order is invalid.');
  return trackIds.map((trackId, position) => ({ trackId, position }));
}
