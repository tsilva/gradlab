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
  frameSkipPresentation,
  orderedTerminationConditions,
  terminationOutcomeClass,
} from "../../src/gradlab/web_player/playback-settings.js";

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
    }),
    {
      action: "next_episode",
      label: "Next episode",
      icon: "player-skip-forward",
      disabled: false,
      reason: "Start the prepared next episode",
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
  assert.match(app, /--timeline-progress/);
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

test("the scrubber retains snapshots before waiting for their frames", () => {
  const snapshotMessage = app.slice(
    app.indexOf('if (message.type === "snapshot")'),
    app.indexOf('if (message.type === "command_result")'),
  );
  const retainPosition = snapshotMessage.indexOf("state.snapshots.set");
  const frameWaitPosition = snapshotMessage.indexOf("requiredFramesAvailable");
  assert.ok(retainPosition >= 0 && retainPosition < frameWaitPosition);
});

test("the player has no view selector", () => {
  assert.doesNotMatch(page, /workspace-preset|Workspace view|<option[^>]*>(?:Watch|Explain|Debug|Customize)<\/option>/);
  assert.doesNotMatch(app, /workspace-preset|applyWorkspacePreset/);
  assert.doesNotMatch(styles, /workspace-preset|data-workspace-view="(?:watch|explain|debug)"/);
});

test("the all-panels workspace overlays an idle-hiding transport on the game stage", () => {
  assert.match(page, /id="timeline-home" hidden/);
  assert.equal((page.match(/id="timeline"/g) || []).length, 1);
  assert.match(app, /const TIMELINE_HIDE_DELAY_MS = 1600/);
  assert.match(app, /stage\.append\(timeline\)/);
  assert.match(app, /timeline\.classList\.add\("game-timeline-overlay", "visible"\)/);
  assert.match(app, /\["pointerenter", "pointermove", "pointerdown", "focusin"\]/);
  assert.match(app, /stage\.addEventListener\("pointerleave", scheduleTimelineOverlayHide/);
  assert.match(app, /timeline\.addEventListener\("focusout", scheduleTimelineOverlayHide/);
  assert.match(app, /timeline\?\.contains\(document\.activeElement\)/);
  assert.match(app, /settingsOpen/);
  assert.doesNotMatch(app, /state\.layout\?\.preset !== "debug"/);
  assert.match(
    styles,
    /\.game-stage > \.timeline\.game-timeline-overlay \{[^}]*position: absolute;[^}]*inset: auto 0 0;[^}]*opacity: 0;[^}]*pointer-events: none;/,
  );
  assert.match(
    styles,
    /\.timeline\.game-timeline-overlay\.visible,[\s\S]*\.timeline\.game-timeline-overlay:focus-within \{[^}]*opacity: 1;[^}]*pointer-events: auto;/,
  );
});

test("timeline controls use accessible icons and distinct action colors", () => {
  for (const [id, label] of [
    ["timeline-playback-toggle", "Play"],
    ["timeline-reset", "Reset episode"],
    ["playback-settings-toggle", "Playback settings"],
  ]) {
    assert.match(page, new RegExp(`id="${id}"[^>]*class="[^"]*icon-only[^"]*"[^>]*aria-label="${label}"`));
  }
  assert.equal(page.includes("timeline-playback-label"), false);
  assert.match(icons, /id="ti-player-skip-forward"/);
  assert.match(styles, /#timeline-playback-toggle\[data-action="next_episode"\][^{]*\{[^}]*var\(--color-series-teal\)/);
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

test("checkpoint changes block the whole player until the new frame is renderable", () => {
  assert.match(
    page,
    /id="checkpoint-loading-mask" class="checkpoint-loading-mask" role="status" aria-live="polite" hidden/,
  );
  assert.match(page, /<strong>Loading checkpoint…<\/strong>/);
  assert.match(
    styles,
    /\.checkpoint-loading-mask \{[^}]*position: fixed;[^}]*inset: 0;[^}]*z-index: 200;/,
  );
  assert.match(app, /function beginCheckpointLoad\(\{ commandId, checkpointId \}\)/);
  assert.match(
    app,
    /showFramesForSequence\(Number\(snapshot\.sequence\)\)\.then\(\(\) => \{\s*if \(snapshotCompletesCheckpointLoad\(snapshot\)\) finishCheckpointLoad\(\);/,
  );
  assert.match(app, /message\.id === state\.checkpointLoad\?\.commandId && !message\.ok/);
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

test("playback settings distinguish training and active frame skip", () => {
  assert.deepEqual(
    frameSkipPresentation({ frame_skip: { training: 4, playback: 2 } }),
    {
      training: 4,
      playback: 2,
      differs: true,
      label: "Frame skip · training 4 · playback 2",
    },
  );
  assert.equal(frameSkipPresentation({ frame_skip: { training: 4 } }), null);
  assert.match(settings, /data-playback-frame-skip hidden/);
  assert.match(settings, /frameSkip\.classList\.toggle\("contract-mismatch"/);
  assert.match(
    styles,
    /\[data-playback-frame-skip\]\.contract-mismatch \{[^}]*var\(--color-series-amber\)/,
  );
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

test("episode termination settings prioritize and color semantic outcomes", () => {
  const failure = { id: "event:life_loss", outcome: "failure" };
  const success = { id: "event:level_change", outcome: "success" };
  const firstTimeout = { id: "event:stalled", outcome: "timeout" };
  const secondTimeout = { id: "limit:max_episode_steps", outcome: "timeout" };

  assert.deepEqual(
    orderedTerminationConditions([failure, firstTimeout, success, secondTimeout]),
    [success, failure, firstTimeout, secondTimeout],
  );
  assert.equal(terminationOutcomeClass("success"), "outcome-success");
  assert.equal(terminationOutcomeClass("failure"), "outcome-failure");
  assert.equal(terminationOutcomeClass("timeout"), "outcome-timeout");
  assert.equal(terminationOutcomeClass("neutral"), "");
  assert.match(settings, /outcome\.className = `termination-outcome/);
  assert.match(
    styles,
    /\.game-frame-detail\.outcome-success,[\s\S]*\.termination-outcome\.outcome-success \{[^}]*var\(--color-evaluation-text\)/,
  );
  assert.match(
    styles,
    /\.game-frame-detail\.outcome-failure,[\s\S]*\.termination-outcome\.outcome-failure \{[^}]*var\(--color-error-text\)/,
  );
  assert.match(
    styles,
    /\.game-frame-detail\.outcome-timeout,[\s\S]*\.termination-outcome\.outcome-timeout \{[^}]*var\(--color-series-amber\)/,
  );
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
