import assert from "node:assert/strict";
import test from "node:test";

import {
  gameFramePhase,
  gameFrameTerminationDetail,
  gameFrameTerminationTone,
} from "../../src/gradlab/web_player/panels/game.js";

test("game frame phase distinguishes initial, after-action, and terminal frames", () => {
  assert.equal(gameFramePhase({ transition: null }), "Initial observation");
  assert.equal(
    gameFramePhase({ transition: { after: { frame_role: "after_action_observation" } } }),
    "After-action observation",
  );
  assert.equal(
    gameFramePhase({ transition: { after: { frame_role: "terminal_observation" } } }),
    "Terminal observation",
  );
  assert.equal(
    gameFramePhase({
      transition: { after: { frame_role: "next_episode_initial_observation" } },
    }),
    "Next episode initial observation",
  );
});

test("terminal observations show the most specific termination fact", () => {
  assert.equal(gameFrameTerminationDetail({
    transition: {
      after: { frame_role: "terminal_observation" },
      events: ["player_died"],
      boundary_reasons: ["provider_terminated"],
    },
  }), "Player died");
  assert.equal(gameFrameTerminationDetail({
    transition: {
      after: { frame_role: "terminal_observation" },
      events: [],
      info: { termination_reason: "time_limit" },
      boundary_reasons: ["provider_truncated"],
    },
  }), "Time limit");
  assert.equal(gameFrameTerminationDetail({
    transition: {
      after: { frame_role: "terminal_observation" },
      events: [],
      boundary_reasons: ["task_terminated"],
    },
  }), "Task terminated");
  assert.equal(gameFrameTerminationDetail({
    transition: {
      after: { frame_role: "after_action_observation" },
      events: ["player_died"],
    },
  }), "");
});

test("terminal badge tone follows the canonical success or failure outcome", () => {
  const terminal = (outcome) => ({
    transition: {
      after: { frame_role: "terminal_observation" },
      outcome,
    },
  });

  assert.equal(gameFrameTerminationTone(terminal("success")), "success");
  assert.equal(gameFrameTerminationTone(terminal("failure")), "failure");
  assert.equal(gameFrameTerminationTone(terminal("timeout")), "timeout");
  assert.equal(gameFrameTerminationTone(terminal("neutral")), "");
  assert.equal(gameFrameTerminationTone({
    transition: {
      after: { frame_role: "after_action_observation" },
      outcome: "failure",
    },
  }), "");
});
