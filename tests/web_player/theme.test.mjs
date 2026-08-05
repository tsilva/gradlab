import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const ROOT = new URL("../../src/gradlab/web_player/", import.meta.url);

function luminance(hex) {
  const channels = hex.match(/[0-9a-f]{2}/gi).map((value) => Number.parseInt(value, 16) / 255);
  const linear = channels.map((value) => (
    value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4
  ));
  return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2];
}

function contrast(foreground, background) {
  const [lighter, darker] = [luminance(foreground), luminance(background)]
    .sort((left, right) => right - left);
  return (lighter + 0.05) / (darker + 0.05);
}

test("scientific editorial theme tokens are the single CSS color source", async () => {
  const styles = await readFile(new URL("styles.css", ROOT), "utf8");
  const expected = new Map([
    ["color-canvas", "#FCF9F8"],
    ["color-surface-primary", "#FFFFFF"],
    ["color-surface-secondary", "#F6F3F2"],
    ["color-surface-tertiary", "#F0EDED"],
    ["color-surface-quaternary", "#EAE7E7"],
    ["color-text", "#1C1B1B"],
    ["color-text-muted", "#4B4356"],
    ["color-border", "#CDC2DA"],
    ["color-outline-strong", "#7C7388"],
    ["color-interaction", "#6100C5"],
    ["color-interaction-active", "#7F00FF"],
    ["color-evaluation-surface", "#8DF5E4"],
    ["color-evaluation-text", "#006B5F"],
    ["color-training-surface", "#FFDCC6"],
    ["color-training-text", "#743800"],
    ["color-error-surface", "#FFDAD6"],
    ["color-error-text", "#BA1A1A"],
    ["color-inverse-surface", "#313030"],
    ["color-inverse-text", "#F3F0EF"],
  ]);
  for (const [name, value] of expected) {
    assert.match(styles, new RegExp(`--${name}: ${value};`, "i"));
  }

  const withoutRootTokens = styles.replace(/:root \{[\s\S]*?\n\}/, "");
  assert.doesNotMatch(withoutRootTokens, /#[0-9a-f]{3,8}|rgba?\(/i);
  assert.equal((styles.match(/var\(--color-media-black\)/g) || []).length, 1);
  assert.match(
    styles,
    /button:disabled \{[^}]*color: var\(--color-text-muted\);[^}]*background: var\(--color-surface-quaternary\);[^}]*opacity: 1;/,
  );
});

test("canvas colors resolve the same CSS theme properties", async () => {
  const styles = await readFile(new URL("styles.css", ROOT), "utf8");
  const shared = await readFile(new URL("panels/shared.js", ROOT), "utf8");
  const canvasProperties = [
    "--color-chart-surface",
    "--color-chart-grid",
    "--color-chart-axis",
    "--color-chart-bar",
    "--color-chart-highlight",
    "--color-series-violet",
    "--color-series-teal",
    "--color-series-amber",
    "--color-series-coral",
    "--color-series-lavender",
    "--color-series-aqua",
    "--color-series-deep-violet",
    "--color-series-burnt-orange",
  ];
  for (const property of canvasProperties) {
    assert.match(styles, new RegExp(`${property}:`));
    assert.match(shared, new RegExp(`"${property}"`));
  }
  assert.match(shared, /getComputedStyle\(document\.documentElement\)/);
  assert.doesNotMatch(shared, /#[0-9a-f]{3,8}|rgba?\(/i);
});

test("telemetry descriptors retain a fixed eight-color sequence", async () => {
  const telemetry = await readFile(new URL("panels/telemetry.js", ROOT), "utf8");
  const sequence = [
    "seriesViolet",
    "seriesTeal",
    "seriesAmber",
    "seriesCoral",
    "seriesLavender",
    "seriesAqua",
    "seriesDeepViolet",
    "seriesBurntOrange",
  ];
  let cursor = -1;
  for (const name of sequence) {
    const next = telemetry.indexOf(`color: "${name}"`, cursor + 1);
    assert.ok(next > cursor, `${name} must appear in palette order`);
    cursor = next;
  }
  assert.doesNotMatch(telemetry, /#[0-9a-f]{3,8}|rgba?\(/i);
});

test("theme text, evidence, syntax, disabled, and focus colors meet contrast targets", () => {
  const normalTextPairs = [
    ["#1C1B1B", "#FCF9F8"],
    ["#4B4356", "#FCF9F8"],
    ["#4B4356", "#EAE7E7"],
    ["#006B5F", "#8DF5E4"],
    ["#743800", "#FFDCC6"],
    ["#BA1A1A", "#FFDAD6"],
    ["#F3F0EF", "#313030"],
    ["#6100C5", "#FFFFFF"],
    ["#00796B", "#FFFFFF"],
    ["#B14E00", "#FFFFFF"],
    ["#BA1A1A", "#FFFFFF"],
    ["#8A5BB7", "#FFFFFF"],
    ["#007C91", "#FFFFFF"],
    ["#3F006E", "#FFFFFF"],
    ["#934200", "#FFFFFF"],
  ];
  for (const [foreground, background] of normalTextPairs) {
    assert.ok(
      contrast(foreground, background) >= 4.5,
      `${foreground} on ${background} must meet WCAG AA`,
    );
  }
  assert.ok(contrast("#7F00FF", "#FCF9F8") >= 3, "focus indicator must meet 3:1");
  assert.ok(contrast("#7C7388", "#FCF9F8") >= 3, "strong outline must meet 3:1");
});
