import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  playbackSourceTitle,
  transportPresentation,
  workspaceIsEditable,
} from "../../src/gradlab/web_player/player-presentation.js";
import { setSvgUseHref } from "../../src/gradlab/web_player/panels/shared.js";

const settings = readFileSync(
  new URL("../../src/gradlab/web_player/playback-settings.js", import.meta.url),
  "utf8",
);
const controls = readFileSync(
  new URL("../../src/gradlab/web_player/panels/controls.js", import.meta.url),
  "utf8",
);
const app = readFileSync(
  new URL("../../src/gradlab/web_player/app.js", import.meta.url),
  "utf8",
);
const page = readFileSync(
  new URL("../../src/gradlab/web_player/index.html", import.meta.url),
  "utf8",
);
const styles = readFileSync(
  new URL("../../src/gradlab/web_player/styles.css", import.meta.url),
  "utf8",
);
const icons = readFileSync(
  new URL("../../src/gradlab/web_player/tabler-icons.svg", import.meta.url),
  "utf8",
);

test("one contextual transport covers play, pause, replay, and next episode", () => {
  assert.deepEqual(
    transportPresentation({ hasControl: true }),
    {
      action: "play",
      label: "Play",
      icon: "player-play",
      disabled: false,
      reason: "Play the current episode",
    },
  );
  assert.equal(
    transportPresentation({ running: true, hasControl: true }).action,
    "pause",
  );
  assert.equal(
    transportPresentation({ canReplay: true, hasControl: true }).action,
    "replay",
  );
  assert.deepEqual(
    transportPresentation({
      hasControl: true,
      session: { awaiting_next_episode: true, can_start_next_episode: true },
    }).action,
    "next_episode",
  );
  const exhausted = transportPresentation({
    hasControl: true,
    session: { awaiting_next_episode: true, can_start_next_episode: false },
  });
  assert.equal(exhausted.disabled, true);
  assert.match(exhausted.reason, /limit/);
});

test("task views are fixed while Customize is explicitly editable", () => {
  assert.equal(workspaceIsEditable("watch"), false);
  assert.equal(workspaceIsEditable("explain"), false);
  assert.equal(workspaceIsEditable("debug"), false);
  assert.equal(workspaceIsEditable("custom"), true);
  assert.equal(
    playbackSourceTitle({
      environment_id: "SuperMarioBros-Nes-v0",
      checkpoint_id: "checkpoint-10002432-b285ff3b",
    }),
    "Super Mario Bros · 10,002,432 steps",
  );
});

test("the scrubber stays left of a fixed-width playback action rail", () => {
  assert.equal((page.match(/id="timeline-playback-toggle"/g) || []).length, 1);
  const timelineStart = page.indexOf('<div class="timeline-track">');
  const timelineEnd = page.indexOf("</section>", timelineStart);
  const scrubberPosition = page.indexOf('id="timeline-scrubber"', timelineStart);
  const actionsPosition = page.indexOf('class="timeline-actions"', timelineStart);
  assert.ok(scrubberPosition > timelineStart && scrubberPosition < actionsPosition);
  for (const id of ["timeline-playback-toggle", "timeline-reset", "playback-settings-toggle"]) {
    const position = page.indexOf(`id="${id}"`, timelineStart);
    assert.ok(position > actionsPosition && position < timelineEnd, id);
  }
  assert.match(styles, /\.timeline-track \{[^}]*grid-template-columns: minmax\(0, 1fr\) 20rem/);
  assert.match(styles, /\.timeline-actions \{[^}]*width: 20rem/);
  assert.match(page, /id="playback-settings-toggle"[^>]*>.*#ti-settings/);
  assert.match(icons, /id="ti-settings"/);
  assert.equal(settings.includes("data-next-episode"), false);
  assert.equal(settings.includes("data-reset-episode"), false);
});

test("playback tuning is one reusable on-demand settings form", () => {
  assert.match(page, /id="playback-settings-menu"[^>]*hidden/);
  assert.match(controls, /mountPlaybackSettings/);
  for (const className of [
    "playback-fps",
    "next-episode-seed",
    "playback-sampling",
    "playback-contract",
  ]) {
    assert.match(settings, new RegExp(`class="playback-field ${className}"`));
  }
  assert.match(settings, /data-termination-settings/);
  assert.match(styles, /\.playback-settings-menu \{/);
});

test("episode termination selections flow through Reset and Next episode", () => {
  assert.match(
    settings,
    /enabled_termination_conditions: enabledTerminationConditions\(\)/,
  );
  assert.match(
    app,
    /command\("next_episode", \{[\s\S]*enabled_termination_conditions: options\.enabled_termination_conditions/,
  );
  assert.match(
    app,
    /command\("reset_episode", \{[\s\S]*seed: options\.seed,[\s\S]*enabled_termination_conditions: options\.enabled_termination_conditions/,
  );
  assert.match(settings, /Selections apply with Reset or Next episode\./);
});

test("seed control keeps the loaded checkpoint default across automatic resets", () => {
  assert.match(
    settings,
    /const defaultSeed = text\(session\.default_seed, session\.seed\);/,
  );
  assert.match(settings, /seed\.dataset\.defaultSeed !== defaultSeed/);
  assert.match(settings, /seed\.value = defaultSeed;/);
  assert.match(settings, /text\(snapshot\.transition\?\.seed, text\(session\.seed, defaultSeed\)\)/);
});

test("playback setting values share the compact field layout", () => {
  assert.match(
    styles,
    /\.playback-field select \{[^}]*font-size: var\(--font-size-xs\);/,
  );
  assert.match(
    styles,
    /\.termination-settings \{[^}]*border: 0;[^}]*\}/,
  );
});

test("unchanged playback icons do not invalidate the SVG glyph", () => {
  const icon = {
    href: "/assets/tabler-icons.svg#ti-player-pause",
    writes: 0,
    getAttribute(name) {
      assert.equal(name, "href");
      return this.href;
    },
    setAttribute(name, value) {
      assert.equal(name, "href");
      this.href = value;
      this.writes += 1;
    },
  };

  assert.equal(
    setSvgUseHref(icon, "/assets/tabler-icons.svg#ti-player-pause"),
    false,
  );
  assert.equal(icon.writes, 0);
  assert.equal(
    setSvgUseHref(icon, "/assets/tabler-icons.svg#ti-player-play"),
    true,
  );
  assert.equal(icon.writes, 1);
});
