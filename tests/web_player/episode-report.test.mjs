import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { episodeReport } from "../../src/gradlab/web_player/episode-report.js";

const html = readFileSync(new URL("../../src/gradlab/web_player/index.html", import.meta.url), "utf8");

test("the player omits the redundant episode outcome panel", () => {
  assert.doesNotMatch(html, /id="episode-report"/);
  assert.doesNotMatch(html, /data-episode-outcome/);
  assert.doesNotMatch(html, /<dt>Outcome<\/dt>/);
  assert.match(html, /id="playback-evidence-status"[\s\S]*role="status"[\s\S]*hidden/);
});

test("episode report leads with outcome, boundary, and evidence semantics", () => {
  const report = episodeReport({
    mode: "playback",
    app: { route: { checkpoint_id: "checkpoint-10002432-b285ff3b" } },
    session: {
      step: 847,
      seed: 42,
      total_reward: 18.125,
      playback_contract: { mode: "evaluation" },
    },
    transition: {
      step: 847,
      seed: 42,
      boundary: true,
      outcome: "success",
      boundary_reasons: ["target_reached"],
      reward: { return: 18.125 },
    },
  });

  assert.equal(report.outcome, "Success");
  assert.equal(report.outcomeTone, "success");
  assert.equal(report.boundary, "Target Reached");
  assert.equal(report.episodeReturn, "18.125");
  assert.equal(report.steps, "847");
  assert.equal(report.semantics, "Published evaluation contract");
  assert.equal(report.source, "Checkpoint at step 10,002,432");
  assert.match(report.disclaimer, /not acceptance or promotion evidence/);
});

test("terminal evidence stays attached to the completed transition after the provider resets", () => {
  const report = episodeReport({
    session: { step: 0, seed: 999, total_reward: 0 },
    transition: {
      step: 528,
      seed: 123,
      boundary: true,
      outcome: "success",
      reward: { return: 1917560605 },
    },
  });

  assert.equal(report.steps, "528");
  assert.equal(report.seed, "123");
  assert.equal(report.episodeReturn, "1,917,560,605");
});

test("counterfactual playback is visibly ineligible as evidence", () => {
  const report = episodeReport({
    session: { playback_contract: { mode: "counterfactual" } },
    transition: { boundary: false },
  });

  assert.equal(report.outcome, "In progress");
  assert.equal(report.boundary, "Not reached");
  assert.match(report.semantics, /not evidence/);
});

test("terminal reports do not borrow reset-state return or a continuing outcome", () => {
  const report = episodeReport({
    session: { total_reward: 0 },
    transition: { boundary: true, outcome: "continuing", terminated: true, reward: { return: null } },
  });

  assert.equal(report.outcome, "Terminated");
  assert.equal(report.episodeReturn, "—");
});
