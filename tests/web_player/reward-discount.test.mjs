import test from 'node:test';
import assert from 'node:assert/strict';
import { rewardContribution, contributionLabel } from '../../src/gradlab/web_player/panels/reward-discount.js';
test('discount uses distance from selected state and preserves reward signs', () => {
  assert.equal(rewardContribution({step: 5, reward_shaped: 1}, 5, .99).contribution, 1);
  assert.equal(rewardContribution({step: 105, reward_shaped: -1}, 5, .99).contribution, -(.99 ** 100));
  assert.equal(rewardContribution({step: 105, reward_shaped: 1}, 104, .99).contribution, .99);
  assert.equal(rewardContribution({step: 4, reward_shaped: 1}, 5, .99).past, true);
});
test('missing data is unavailable and zero and unit discounts are valid', () => {
  for (const gamma of [null, undefined, NaN, -1, 2, '0.99']) assert.equal(rewardContribution({step: 5, reward_shaped: 1}, 5, gamma), null);
  assert.equal(rewardContribution({step: 5, reward_shaped: null}, 5, .99), null);
  assert.equal(rewardContribution({step: 5, reward_shaped: 1}, 5, 0).contribution, 1);
  assert.equal(rewardContribution({step: 6, reward_shaped: 1}, 5, 0).contribution, 0);
  assert.equal(rewardContribution({step: 1000, reward_shaped: 1}, 5, 1).contribution, 1);
  assert.match(contributionLabel({step: 105, reward_shaped: 1}, 5, .99), /100 steps later · weight 0.366/);
});
