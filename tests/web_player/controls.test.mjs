import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { setSvgUseHref } from "../../src/gradlab/web_player/panels/shared.js";

const source = readFileSync(
  new URL("../../src/gradlab/web_player/panels/controls.js", import.meta.url),
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

test("common episode actions stay visible while playback tuning is disclosed on demand", () => {
  const controlsStart = source.indexOf('<div class="control-components">');
  const advancedStart = source.indexOf('data-advanced-playback', controlsStart);

  for (const marker of ['data-reset-episode', 'data-next-episode']) {
    const position = source.indexOf(marker);
    assert.ok(position > controlsStart && position < advancedStart, marker);
  }
  for (const marker of [
    'id="playback-fps"',
    'id="next-episode-seed"',
    'id="playback-sampling"',
    'id="playback-contract-mode"',
    'data-termination-settings',
  ]) {
    const position = source.indexOf(marker);
    assert.ok(position > advancedStart, marker);
  }
  assert.match(source, /<details class="advanced-playback-settings"/);
  assert.match(source, /<summary>Advanced playback<\/summary>/);
  assert.equal(source.includes("control-section"), false);
  assert.equal(source.includes("control-label"), false);
});

test("playback setting labels use the shared stacked field layout", () => {
  for (const className of [
    "playback-fps",
    "next-episode-seed",
    "playback-sampling",
    "playback-contract",
  ]) {
    assert.match(
      source,
      new RegExp(`class="playback-field ${className}"`),
      className,
    );
  }
});

test("playback setting values use the same compact text size", () => {
  assert.match(
    styles,
    /\.playback-field select \{[^}]*font-size: \.64rem;/,
  );
});

test("episode termination selections apply through the existing episode actions", () => {
  assert.equal(source.includes("Apply and reset episode"), false);
  assert.equal(source.includes('data-command="apply-termination"'), false);
  assert.match(
    source,
    /"reset-episode":[\s\S]*enabled_termination_conditions: enabledTerminationConditions\(\)/,
  );
  assert.match(
    source,
    /"next-episode":[\s\S]*enabled_termination_conditions: enabledTerminationConditions\(\)/,
  );
  assert.match(
    source,
    /Selections apply with Reset episode or Play next episode\./,
  );
});

test("seed control keeps the loaded checkpoint default across automatic resets", () => {
  assert.match(
    source,
    /const defaultSeed = text\(session\.default_seed, session\.seed\);/,
  );
  assert.match(source, /seed\.dataset\.defaultSeed !== defaultSeed/);
  assert.match(source, /seed\.value = defaultSeed;/);
  assert.match(source, /text\(snapshot\.transition\?\.seed, text\(session\.seed, defaultSeed\)\)/);
});

test("advanced playback is visually separated from common episode actions", () => {
  assert.equal(styles.includes(".playback-settings"), false);
  assert.equal(styles.includes(".next-episode-settings {"), false);
  assert.match(styles, /\.advanced-playback-settings \{[^}]*border-top:/);
  assert.match(
    styles,
    /\.termination-settings \{[^}]*border: 0;[^}]*\}/,
  );
  assert.doesNotMatch(
    styles,
    /\.termination-settings \{[^}]*border-top:/,
  );
});

test("playback transport lives beside the timeline scrubber", () => {
  assert.equal(
    (page.match(/id="timeline-playback-toggle"/g) || []).length,
    1,
  );
  const timelineStart = page.indexOf('<div class="timeline-track">');
  const timelineEnd = page.indexOf("</section>", timelineStart);
  const play = page.indexOf('id="timeline-playback-toggle"', timelineStart);
  const scrubber = page.indexOf('id="timeline-scrubber"', timelineStart);
  assert.ok(play > timelineStart && play < scrubber);
  assert.ok(scrubber < timelineEnd);
  assert.equal(source.includes("data-playback-toggle"), false);
  for (const command of ["step", "step-ten", "continue-event"]) {
    assert.equal(source.includes(`data-command="${command}"`), false);
  }
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
