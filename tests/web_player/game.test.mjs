import assert from "node:assert/strict";
import test from "node:test";

import { gameFramePhase } from "../../src/rlab/web_player/panels/game.js";

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
