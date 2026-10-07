import test from 'node:test';
import assert from 'node:assert/strict';
import { encodePeer, decodePeer } from '../../src/gradlab/web_player/desktop-workspace.js';

test('desktop workspace transports exact frame bytes with cursor identity', async () => {
  const bytes = new Uint8Array(25000).map((_, index) => index % 256);
  const message = { type: 'inspection-frame', session_epoch: 2, sequence: 17,
    source: 'main', target: 'stats', blob: new Blob([bytes], { type: 'image/png' }) };
  const received = decodePeer(await encodePeer(message));
  assert.equal(received.sequence, 17);
  assert.equal(received.session_epoch, 2);
  assert.equal(received.target, 'stats');
  assert.equal(received.blob.type, 'image/png');
  assert.deepEqual(new Uint8Array(await received.blob.arrayBuffer()), bytes);
});

test('desktop workspace preserves layout and range messages', async () => {
  const message = { type: 'chart-range', range: [0, 17], source: 'stats', epoch: 2 };
  assert.deepEqual(decodePeer(await encodePeer(message)), message);
});
