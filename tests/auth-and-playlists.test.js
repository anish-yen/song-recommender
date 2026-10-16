import test from 'node:test';
import assert from 'node:assert/strict';

test('credential validation normalizes email and rejects weak passwords', async () => {
  const { validateCredentials } = await import('../src/lib/validation.js');
  assert.equal(validateCredentials({ email: '  ME@EXAMPLE.COM ', password: 'strong-pass' }).data.email, 'me@example.com');
  assert.match(validateCredentials({ email: 'me@example.com', password: 'short' }).error, /8 characters/);
});

test('playlist ownership prevents an IDOR-style access attempt', async () => {
  const { assertOwner } = await import('../src/lib/playlist-access.js');
  assert.equal(assertOwner({ id: 'list-a', ownerId: 'user-a' }, 'user-a').id, 'list-a');
  assert.throws(() => assertOwner({ id: 'list-a', ownerId: 'user-a' }, 'user-b'), /not found/);
});

test('playlist reordering accepts only a complete unique set of owned tracks', async () => {
  const { orderedPositions } = await import('../src/lib/playlist-access.js');
  assert.deepEqual(orderedPositions(['b', 'a'], ['a', 'b']), [{ trackId: 'b', position: 0 }, { trackId: 'a', position: 1 }]);
  assert.throws(() => orderedPositions(['a', 'a'], ['a', 'b']), /invalid/);
  assert.throws(() => orderedPositions(['a'], ['a', 'b']), /incomplete/);
});
