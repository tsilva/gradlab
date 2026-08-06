import assert from "node:assert/strict";
import test from "node:test";

import {
  magnitudeShareLabel,
  rewardBreakdownPresentation,
  signedContributionLabel,
} from "../../src/gradlab/web_player/panels/reward-breakdown.js";

function snapshot({ scale = 1, clip = null, protocol = 8, status = "available" } = {}) {
  return {
    protocol,
    session: {
      reward_accounting: status === "available"
        ? {
          status,
          reason: null,
          reward_scale: scale,
          clip_bounds: clip,
        }
        : {
          status,
          reason: "recording does not contain raw reward",
          reward_scale: null,
          clip_bounds: null,
        },
    },
  };
}

function point({
  sequence = 1,
  step = sequence,
  raw,
  final,
  total = final,
  components = {},
  error = null,
} = {}) {
  return {
    sequence,
    episode: 1,
    step,
    reward_raw: raw,
    reward_shaped: final,
    reward_accounting_error: error,
    return: total,
    components,
  };
}

function rowById(presentation, id) {
  return presentation.rows.find((row) => row.id === id);
}

test("mixed bonuses, penalties, unattributed reward, and clipping reconcile exactly", () => {
  const history = [point({
    raw: 5,
    final: 1,
    components: {
      progress_reward: 6,
      death_penalty: -2,
    },
  })];
  const result = rewardBreakdownPresentation({
    snapshot: snapshot({ scale: 0.5, clip: [-1, 1] }),
    history,
    view: { selectedSequence: 1 },
    scope: "step",
  });

  assert.equal(result.status, "available");
  assert.equal(result.raw, 5);
  assert.equal(result.preclip, 2.5);
  assert.equal(result.final, 1);
  assert.equal(result.positive, 3.5);
  assert.equal(result.negative, -2.5);
  assert.equal(rowById(result, "progress_reward").impact, 3);
  assert.equal(rowById(result, "progress_reward").signedContribution, 300);
  assert.equal(rowById(result, "death_penalty").signedContribution, -100);
  assert.equal(rowById(result, "unattributed").impact, 0.5);
  assert.equal(rowById(result, "clip_adjustment").impact, -1.5);
  assert.equal(
    result.rows.reduce((sum, row) => sum + row.signedContribution, 0),
    100,
  );
  assert.equal(
    result.rows.reduce((sum, row) => sum + row.magnitudeShare, 0),
    100,
  );
});

test("negative final rewards retain negative signed contribution totals", () => {
  const history = [point({
    raw: -2,
    final: -2,
    components: { death_penalty: -3, progress_reward: 1 },
  })];
  const result = rewardBreakdownPresentation({
    snapshot: snapshot(),
    history,
    view: { selectedSequence: 1 },
  });

  assert.equal(result.status, "available");
  assert.equal(
    result.rows.reduce((sum, row) => sum + row.signedContribution, 0),
    -100,
  );
  assert.equal(rowById(result, "death_penalty").signedContribution, -150);
  assert.equal(rowById(result, "progress_reward").signedContribution, 50);
});

test("zero-net steps omit signed percentages but preserve gross activity", () => {
  const history = [point({
    raw: 0,
    final: 0,
    components: { progress_reward: 2, death_penalty: -2 },
  })];
  const result = rewardBreakdownPresentation({
    snapshot: snapshot(),
    history,
    view: { selectedSequence: 1 },
  });

  assert.equal(result.signedAvailable, false);
  assert.equal(rowById(result, "progress_reward").signedContribution, null);
  assert.equal(rowById(result, "progress_reward").magnitudeShare, 50);
  assert.equal(rowById(result, "death_penalty").magnitudeShare, 50);
  assert.equal(signedContributionLabel(null), "N/A");
  assert.equal(magnitudeShareLabel(null), "N/A");
});

test("reward scale accepts zero and rejects values above one", () => {
  const muted = rewardBreakdownPresentation({
    snapshot: snapshot({ scale: 0 }),
    history: [point({ raw: 4, final: 0, components: { progress_reward: 4 } })],
    view: { selectedSequence: 1 },
  });
  assert.equal(muted.status, "available");
  assert.equal(muted.preclip, 0);

  const invalid = rewardBreakdownPresentation({
    snapshot: snapshot({ scale: 2 }),
    history: [point({ raw: 1, final: 2 })],
    view: { selectedSequence: 1 },
  });
  assert.equal(invalid.status, "protocol-error");
  assert.match(invalid.message, /scale is invalid/);
});

test("episode activity sums absolute per-transition impacts through cancellation", () => {
  const history = [
    point({ sequence: 1, raw: 1, final: 1, total: 1, components: { progress_reward: 1 } }),
    point({ sequence: 2, raw: -1, final: -1, total: 0, components: { progress_reward: -1 } }),
  ];
  const result = rewardBreakdownPresentation({
    snapshot: snapshot(),
    history,
    view: { selectedSequence: 2 },
    scope: "episode",
  });

  assert.equal(result.status, "available");
  assert.equal(result.final, 0);
  assert.equal(result.positive, 1);
  assert.equal(result.negative, -1);
  assert.equal(rowById(result, "progress_reward").impact, 0);
  assert.equal(rowById(result, "progress_reward").magnitude, 2);
  assert.equal(rowById(result, "progress_reward").magnitudeShare, 100);
});

test("episode clipping is accounted per transition rather than on the aggregate", () => {
  const history = [
    point({ sequence: 1, raw: 4, final: 1, total: 1 }),
    point({ sequence: 2, raw: -4, final: -1, total: 0 }),
  ];
  const result = rewardBreakdownPresentation({
    snapshot: snapshot({ scale: 0.5, clip: [-1, 1] }),
    history,
    view: { selectedSequence: 2 },
    scope: "episode",
  });

  assert.equal(result.status, "available");
  assert.equal(result.raw, 0);
  assert.equal(result.preclip, 0);
  assert.equal(result.final, 0);
  assert.equal(rowById(result, "clip_adjustment").impact, 0);
  assert.equal(rowById(result, "clip_adjustment").magnitude, 2);
});

test("episode accounting fails closed when bounded history has lost its beginning", () => {
  const history = [point({ sequence: 9, step: 4, raw: 1, final: 1, total: 4 })];
  const result = rewardBreakdownPresentation({
    snapshot: snapshot(),
    history,
    view: { selectedSequence: 9 },
    scope: "episode",
  });

  assert.equal(result.status, "partial-history");
  assert.match(result.message, /step 1/);
});

test("selected-step accounting remains available without complete episode history", () => {
  const history = [point({ sequence: 9, step: 4, raw: 1, final: 1, total: 4 })];
  const result = rewardBreakdownPresentation({
    snapshot: snapshot(),
    history,
    view: { selectedSequence: 9 },
    scope: "step",
  });

  assert.equal(result.status, "available");
  assert.equal(result.step, 4);
});

test("unavailable, old, missing, malformed, and inconsistent accounting stay distinct", () => {
  assert.equal(
    rewardBreakdownPresentation({ snapshot: snapshot({ status: "unavailable" }) }).status,
    "unavailable",
  );
  assert.equal(
    rewardBreakdownPresentation({ snapshot: snapshot({ protocol: 7 }) }).status,
    "protocol-error",
  );
  assert.equal(
    rewardBreakdownPresentation({ snapshot: snapshot(), history: [] }).status,
    "not-yet-observed",
  );
  const malformed = point({ raw: 1, final: 1, error: "raw_reward is malformed" });
  assert.equal(
    rewardBreakdownPresentation({
      snapshot: snapshot(),
      history: [malformed],
      view: { selectedSequence: 1 },
    }).status,
    "protocol-error",
  );
  const inconsistent = point({ raw: 2, final: 1 });
  assert.match(
    rewardBreakdownPresentation({
      snapshot: snapshot(),
      history: [inconsistent],
      view: { selectedSequence: 1 },
    }).message,
    /does not match/,
  );
});

test("episode shaped sum must match the authoritative selected return", () => {
  const history = [point({ raw: 1, final: 1, total: 2 })];
  const result = rewardBreakdownPresentation({
    snapshot: snapshot(),
    history,
    view: { selectedSequence: 1 },
    scope: "episode",
  });

  assert.equal(result.status, "protocol-error");
  assert.match(result.message, /authoritative episode return/);
});
