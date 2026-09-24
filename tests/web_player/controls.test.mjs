import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  playbackSourceTitle,
  statusMessageShouldToast,
  timelineProgress,
  transportPresentation,
  workspaceIsEditable,
} from "../../src/gradlab/web_player/player-presentation.js";
import { setSvgUseHref } from "../../src/gradlab/web_player/panels/shared.js";
import {
  applyStopConditionSuggestion,
  stopConditionHighlightSegments,
  stopConditionPopoverPlacement,
  stopConditionSuggestions,
} from "../../src/gradlab/web_player/playback-settings.js";

const settings = readFileSync(
  new URL("../../src/gradlab/web_player/playback-settings.js", import.meta.url),
  "utf8",
);
const app = readFileSync(
  new URL("../../src/gradlab/web_player/app.js", import.meta.url),
  "utf8",
);
const page = readFileSync(
  new URL("../../frontend/components/Shell.svelte", import.meta.url),
  "utf8",
).replace(/\s+/g, " ").replaceAll(" >", ">");
const playbackSettingsComponent = readFileSync(
  new URL("../../frontend/components/PlaybackSettings.svelte", import.meta.url),
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

test("one contextual transport uses Play for every live paused state", () => {
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
      session: {
        awaiting_next_episode: true,
        can_start_next_episode: true,
        stop_condition: { valid: true },
      },
    }),
    {
      action: "play",
      label: "Play",
      icon: "player-play",
      disabled: false,
    reason: "Start the next episode",
    },
  );
  const exhausted = transportPresentation({
    hasControl: true,
    session: { awaiting_next_episode: true, can_start_next_episode: false },
  });
  assert.equal(exhausted.disabled, true);
  assert.match(exhausted.reason, /limit/);
});

test("episode completion uses the player controls instead of a toast", () => {
  assert.equal(statusMessageShouldToast({ status_message: "Paused" }), true);
  assert.equal(statusMessageShouldToast({
    status_message: "playing next episode",
    session: { awaiting_next_episode: false },
  }), false);
  assert.equal(statusMessageShouldToast({
    status_message: "episode complete · choose Play next episode",
    session: { awaiting_next_episode: true },
  }), false);
  assert.equal(statusMessageShouldToast({
    status_message: "stop condition matched · episode.terminated = 2",
    session: { awaiting_next_episode: true },
  }), true);
  assert.equal(statusMessageShouldToast({
    status_message: "Checkpoint expired before the next episode",
    session: { awaiting_next_episode: true },
  }), true);
  assert.match(app, /if \(statusMessageShouldToast\(snapshot\)\) \{/);
});

test("the single all-panels workspace is editable", () => {
  assert.equal(workspaceIsEditable("watch"), false);
  assert.equal(workspaceIsEditable("explain"), false);
  assert.equal(workspaceIsEditable("debug"), false);
  assert.equal(workspaceIsEditable("custom"), false);
  assert.equal(workspaceIsEditable("all"), true);
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
  assert.match(styles, /\.timeline-track \{[^}]*grid-template-columns: minmax\(0, 1fr\) 8rem/);
  assert.match(styles, /\.timeline-actions \{[^}]*width: 8rem/);
  assert.match(page, /id="playback-settings-toggle"[^>]*>.*#ti-settings/);
  assert.match(icons, /id="ti-settings"/);
  assert.equal(settings.includes("data-next-episode"), false);
  assert.equal(settings.includes("data-reset-episode"), false);
});

test("the timeline track is fully filled at its final retained step", () => {
  assert.equal(timelineProgress(0, 0), 0);
  assert.equal(timelineProgress(0, 1), 100);
  assert.equal(timelineProgress(2, 5), 50);
  assert.equal(timelineProgress(4, 5), 100);
  assert.match(
    styles,
    /#timeline-scrubber::-(?:webkit-slider-runnable-track|moz-range-track)[\s\S]*var\(--timeline-progress\)/,
  );
});

test("timeline event markers render below the progress bar", () => {
  const scrubberPosition = page.indexOf('id="timeline-scrubber"');
  const markersPosition = page.indexOf('id="timeline-markers"');
  assert.ok(scrubberPosition >= 0 && markersPosition > scrubberPosition);
  assert.match(
    styles,
    /\.timeline-scrubber-track \{[^}]*display: grid;[^}]*grid-template-rows: 1rem \.6rem;/,
  );
  assert.match(
    styles,
    /\.timeline-markers \{[^}]*position: relative;[^}]*margin-inline: \.45rem;/,
  );
});

test("timeline keyboard focus uses a discreet visible outline", () => {
  assert.match(
    styles,
    /#timeline-scrubber:focus-visible \{[^}]*outline: 1px solid color-mix\([^;]+45%, transparent\);[^}]*outline-offset: 1px;/,
  );
});

test("the scrubber delegates arrow keys to native one-step range behavior", () => {
  assert.match(
    page,
    /id="timeline-scrubber"[^>]*type="range"[^>]*step="1"/,
  );
  const bindTimeline = app.slice(
    app.indexOf("function bindTimeline()"),
    app.indexOf("function initWorkspace()"),
  );
  assert.doesNotMatch(bindTimeline, /ArrowLeft|ArrowRight/);
  assert.match(bindTimeline, /event\.code !== "Space"/);
});


test("the player has no view selector", () => {
  assert.doesNotMatch(page, /workspace-preset|Workspace view|<option[^>]*>(?:Watch|Explain|Debug|Customize)<\/option>/);
  assert.doesNotMatch(app, /workspace-preset|applyWorkspacePreset/);
  assert.doesNotMatch(styles, /workspace-preset|data-workspace-view="(?:watch|explain|debug)"/);
});

test("the game stage reserves space for an always-visible transport", () => {
  assert.match(app, /stage\.append\(timeline\)/);
  assert.match(app, /timeline\.classList\.add\("game-timeline-docked"\)/);
  assert.doesNotMatch(app, /TIMELINE_HIDE_DELAY_MS|scheduleTimelineOverlayHide/);
  assert.match(styles, /\.game-stage \{[^}]*grid-template-rows: minmax\(0, 1fr\) auto/);
  assert.doesNotMatch(styles, /game-timeline-overlay/);
});

test("timeline controls use accessible icons and distinct action colors", () => {
  for (const [id, label] of [
    ["timeline-reset", "Reset episode"],
    ["playback-settings-toggle", "Playback settings"],
  ]) {
    assert.match(page, new RegExp(`id="${id}"[^>]*class="[^"]*icon-only[^"]*"[^>]*aria-label="${label}"`));
  }
  assert.equal(page.includes("timeline-playback-label"), false);
  assert.doesNotMatch(styles, /data-action="next_episode"/);
  assert.match(styles, /#timeline-playback-toggle\[data-action="play"\][^{]*\{[^}]*var\(--color-interaction\)/);
  assert.match(styles, /#timeline-playback-toggle\[data-action="pause"\][^{]*\{[^}]*var\(--color-series-amber\)/);
  assert.match(styles, /#timeline-reset:not\(:disabled\)[^{]*\{[^}]*var\(--color-series-coral\)/);
  assert.match(styles, /#playback-settings-toggle:not\(:disabled\)[^{]*\{[^}]*var\(--color-series-aqua\)/);
});

test("header checkpoint and overflow controls share one height", () => {
  assert.match(styles, /--header-control-height: 2\.25rem/);
  assert.match(
    styles,
    /\.header-status > \.checkpoint-navigation,[\s\S]*\.header-status > #more-toggle \{ height: var\(--header-control-height\); \}/,
  );
  assert.match(styles, /\.checkpoint-navigation-button \{[^}]*height: 100%/);
  assert.match(styles, /\.checkpoint-navigation-position \{[^}]*height: 100%/);
  assert.match(styles, /\.checkpoint-navigation-position \{[^}]*min-width: 4\.25rem/);
  assert.match(styles, /#more-toggle\.icon-only \{ width: var\(--header-control-height\); \}/);
  assert.match(
    page,
    /id="more-toggle" class="quiet icon-only"[^>]*aria-label="More playback actions"[^>]*>[\s\S]*?#ti-dots-vertical[\s\S]*?<\/button>/,
  );
  assert.doesNotMatch(page, /id="more-toggle"[^>]*>[\s\S]*?<span>More<\/span>/);
});

test("checkpoint navigation uses accessible directional icons", () => {
  assert.match(
    page,
    /data-checkpoint-position aria-label="Checkpoint position" aria-live="polite">— \/ —<\/span>/,
  );
  assert.match(
    page,
    /class="quiet checkpoint-navigation-button icon-only"[^>]*aria-label="Previous checkpoint"[^>]*data-checkpoint-previous[^>]*>[\s\S]*?#ti-arrow-left/,
  );
  assert.match(
    page,
    /class="quiet checkpoint-navigation-button icon-only"[^>]*aria-label="Next checkpoint"[^>]*data-checkpoint-next[^>]*>[\s\S]*?#ti-arrow-right/,
  );
  assert.equal(page.includes(">Prev</button>"), false);
  assert.equal(page.includes(">Next</button>"), false);
  assert.match(icons, /id="ti-arrow-right"/);
  assert.match(styles, /\.checkpoint-navigation-button\.icon-only \{ width: calc\(var\(--header-control-height\) - 2px\); \}/);
});

test("the overflow menu omits the redundant change-checkpoint action", () => {
  assert.doesNotMatch(page, /change-source|Change checkpoint/);
  assert.doesNotMatch(app, /change-source/);
  assert.doesNotMatch(styles, /change-source/);
  assert.match(page, /id="source-back"[^>]*aria-label="Back to source selection"/);
});


test("playback tuning is one reusable on-demand settings form", () => {
  assert.match(page, /id="playback-settings-menu"[^>]*hidden/);
  for (const className of [
    "playback-fps",
    "next-episode-seed",
    "playback-sampling",
    "playback-contract",
  ]) {
  }
  assert.match(styles, /\.playback-settings-menu \{/);
  assert.doesNotMatch(playbackSettingsComponent, /playback-glance|data-playback-frame-skip/);
  assert.doesNotMatch(
    playbackSettingsComponent,
    /1 uses the original distribution|Training-compatible critic comparison|Use event counts/,
  );
});

test("playback settings apply on change", () => {
  assert.doesNotMatch(app, /command\("next_episode"/);
  assert.match(app, /command\("reset_episode", \{[\s\S]*seed: options\.seed/);
  assert.doesNotMatch(app, /enabled_termination_conditions/);
});

test("stop-condition autocomplete understands expression context", () => {
  const symbols = [
    { name: "episode.terminated", kind: "episode", value: 1 },
    { name: "x_pos", kind: "signal", value: 2940 },
  ];
  assert.deepEqual(
    stopConditionSuggestions("episode.t", 9, symbols).items.map((item) => item.value),
    ["episode.terminated"],
  );
  assert.deepEqual(
    stopConditionSuggestions("x_pos ", 6, symbols).items.map((item) => item.value),
    ["==", "!=", "<", "<=", ">", ">="],
  );
  assert.deepEqual(
    stopConditionSuggestions("x_pos >= 3000 ", 14, symbols).items.map((item) => item.value),
    ["and", "or"],
  );
  for (const source of [
    "episode.terminated >=",
    "episode.terminated >= ",
    "episode.terminated >= -",
    "episode.terminated >= 1e",
    "x_pos >= nope",
    "x_pos >= 1 or episode.truncated >=",
  ]) {
    assert.deepEqual(
      stopConditionSuggestions(source, source.length, symbols).items,
      [],
      source,
    );
  }
  assert.deepEqual(
    applyStopConditionSuggestion("episode.t", 9, 0, 9, "episode.terminated"),
    { source: "episode.terminated", cursor: 18 },
  );
  const middle = stopConditionSuggestions("episode.terminated == 2", 9, symbols);
  assert.deepEqual(
    applyStopConditionSuggestion(
      "episode.terminated == 2",
      9,
      middle.from,
      middle.to,
      "episode.terminated",
    ),
    { source: "episode.terminated == 2", cursor: 18 },
  );
});

test("stop-condition highlighting preserves source text and marks exact parse errors", () => {
  const source = "episode.terminated >= 2 or x_pos >=";
  const error = { offset: source.length, length: 0 };
  const segments = stopConditionHighlightSegments(source, error, [
    { name: "episode.terminated", kind: "episode" },
    { name: "x_pos", kind: "signal" },
  ]);
  assert.equal(segments.map((segment) => segment.text).join(""), source);
  assert.ok(segments.some((segment) => segment.kind === "symbol-episode"));
  assert.ok(segments.some((segment) => segment.kind === "symbol-signal"));
  assert.ok(segments.some((segment) => segment.kind === "keyword"));
  assert.equal(segments.some((segment) => segment.kind === "invalid"), false);
  assert.ok(segments.some((segment) => segment.text === ">="));
  assert.deepEqual(segments.at(-1), {
    text: "",
    kind: "error-marker",
    error: true,
    from: source.length,
    to: source.length,
  });

  const ranged = stopConditionHighlightSegments("life_loss >> 1", {
    offset: 11,
    length: 1,
  });
  assert.deepEqual(
    ranged.filter((segment) => segment.error).map((segment) => segment.text),
    [">"],
  );
});

test("stop-condition parse errors retain a visible message and strong red editor outline", () => {
  assert.match(playbackSettingsComponent, /\{#if stopError\}[\s\S]*data-stop-condition-status/);
  assert.match(
    styles,
    /\.stop-condition-input-shell\.invalid \{[^}]*outline: 3px solid var\(--color-error-text\);/,
  );
});

test("stop-condition suggestions stay anchored above the editor outside modal overflow", () => {
  assert.deepEqual(
    stopConditionPopoverPlacement(
      { left: 100, top: 100, bottom: 180, width: 300 },
      { width: 800, height: 600 },
    ),
    { left: 100, top: 94, width: 300, maxHeight: 86, placement: "above" },
  );
  assert.deepEqual(
    stopConditionPopoverPlacement(
      { left: 620, top: 500, bottom: 580, width: 300 },
      { width: 800, height: 600 },
    ),
    { left: 492, top: 494, width: 300, maxHeight: 240, placement: "above" },
  );
  assert.match(styles, /\.stop-condition-suggestions \{[^}]*transform: translateY\(-100%\);/);
  assert.match(playbackSettingsComponent, /use:portalToBody/);
  assert.match(playbackSettingsComponent, /data-playback-settings-popover/);
  assert.match(app, /closest\("\[data-playback-settings-popover\]"\)/);
});


test("playback setting values share the compact field layout", () => {
  assert.match(
    styles,
    /\.playback-field select \{[^}]*font-size: var\(--font-size-xs\);/,
  );
  assert.match(
    styles,
    /\.stop-condition-editor \{[^}]*position: relative;[^}]*\}/,
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
