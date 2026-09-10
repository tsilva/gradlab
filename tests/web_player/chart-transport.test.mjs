import test from 'node:test';
import assert from 'node:assert/strict';
import { ChartVersions, frameScheduler } from '../../src/gradlab/web_player/chart-transport.js';

test('chart revisions reconstruct rows, changed constants and removed points', () => {
  const versions = new ChartVersions();
  const first = { format:'chart-columns-v1', revision:'a', base:null, removed:[],
    fields:[['step'],['signals','ball']], constants:[[['episode'],1]], rows:[[1,[1,4]],[2,[2,6]]] };
  assert.deepEqual(versions.accept(first), [
    {step:1, episode:1, signals:{ball:4}}, {step:2, episode:1, signals:{ball:6}},
  ]);
  assert.deepEqual(versions.accept({...first, revision:'b', base:'a', removed:[1],
    constants:[[['episode'],1],[['return_estimate_step'],3]], rows:[[3,[3,9]]]}), [
    {step:2, episode:1, return_estimate_step:3, signals:{ball:6}},
    {step:3, episode:1, return_estimate_step:3, signals:{ball:9}},
  ]);
  assert.throws(() => versions.accept({...first, base:'missing'}), /revision/);
  versions.reset();
  assert.equal(versions.revision, null);
});

test('multiple history invalidations render once with current state', () => {
  let value=0; const callbacks=[]; const rendered=[];
  const schedule=frameScheduler(() => rendered.push(value), cb=>callbacks.push(cb));
  schedule(); value=1; schedule(); value=2; schedule();
  assert.equal(callbacks.length, 1);
  callbacks.shift()();
  assert.deepEqual(rendered,[2]);
  schedule(); callbacks.shift()();
  assert.deepEqual(rendered,[2,2]);
});
