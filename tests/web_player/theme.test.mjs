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

test("scientific editorial dark-theme tokens are the single CSS color source", async () => {
  const styles = await readFile(new URL("styles.css", ROOT), "utf8");
  assert.match(styles, /color-scheme: dark;/);
  const expected = new Map([
    ["color-canvas", "#121015"],
    ["color-surface-primary", "#1B181F"],
    ["color-surface-secondary", "#24202A"],
    ["color-surface-tertiary", "#2D2833"],
    ["color-surface-quaternary", "#37313E"],
    ["color-text", "#F7F2F8"],
    ["color-text-muted", "#C9BFCE"],
    ["color-border", "#51475B"],
    ["color-outline-strong", "#83748F"],
    ["color-interaction", "#B978FF"],
    ["color-interaction-active", "#D2A8FF"],
    ["color-evaluation-surface", "#275B53"],
    ["color-evaluation-text", "#8DF5E4"],
    ["color-training-surface", "#5A3829"],
    ["color-training-text", "#FFDCC6"],
    ["color-error-surface", "#5A302F"],
    ["color-error-text", "#FFDAD6"],
    ["color-inverse-surface", "#09090A"],
    ["color-inverse-text", "#F3F0EF"],
  ]);
  for (const [name, value] of expected) {
    assert.match(styles, new RegExp(`--${name}: ${value};`, "i"));
  }
  assert.doesNotMatch(
    styles,
    /#FCF9F8|#FFFFFF|#F6F3F2|#F0EDED|#EAE7E7|#1C1B1B|#4B4356|#CDC2DA|#7C7388|#6100C5|#7F00FF/i,
  );

  const withoutRootTokens = styles.replace(/:root \{[\s\S]*?\n\}/, "");
  assert.doesNotMatch(withoutRootTokens, /#[0-9a-f]{3,8}|rgba?\(/i);
  assert.equal((styles.match(/var\(--color-media-black\)/g) || []).length, 1);
  assert.match(
    styles,
    /button:disabled \{[^}]*color: var\(--color-text-muted\);[^}]*background: var\(--color-surface-quaternary\);[^}]*opacity: 1;/,
  );
});

test("typography uses stable family roles and a five-step scale", async () => {
  const styles = await readFile(new URL("styles.css", ROOT), "utf8");
  for (const token of ["brand", "ui", "mono"]) {
    assert.match(styles, new RegExp(`--font-family-${token}:`));
  }
  for (const [token, value] of [
    ["xs", ".75rem"],
    ["sm", ".8125rem"],
    ["md", ".875rem"],
    ["lg", "1rem"],
    ["xl", "1.5rem"],
  ]) {
    assert.match(styles, new RegExp(`--font-size-${token}: ${value.replace(".", "\\.")}`));
  }
  assert.match(styles, /:root \{[\s\S]*?font-size: 100%;/);
  assert.equal((styles.match(/font-family: var\(--font-family-brand\)/g) || []).length, 1);
  assert.match(styles, /\.app-wordmark \{ font-family: var\(--font-family-brand\); \}/);
  assert.match(styles, /\.eyebrow, h1, h2, h3 \{ font-family: var\(--font-family-ui\); \}/);
  const withoutRootTokens = styles.replace(/:root \{[\s\S]*?\n\}/, "");
  assert.doesNotMatch(withoutRootTokens, /font-size:\s*(?:clamp|[.\d])/);
  assert.doesNotMatch(styles, /font-weight:\s*(?:650|750|800);/);
  assert.doesNotMatch(styles, /\bfont:\s/);
  assert.match(styles, /ChivoVariable\.woff2/);
  assert.match(styles, /InterVariable\.woff2/);
  assert.match(styles, /JetBrainsMonoVariable\.woff2/);
  assert.doesNotMatch(styles, /InterVariable-Italic\.woff2/);
});

test("scrollbars use compact cross-browser theme styling", async () => {
  const styles = await readFile(new URL("styles.css", ROOT), "utf8");
  assert.match(
    styles,
    /\* \{[^}]*scrollbar-color: var\(--color-outline-strong\) transparent;[^}]*scrollbar-width: thin;/,
  );
  assert.match(styles, /\*::-webkit-scrollbar \{[^}]*width: \.65rem;[^}]*height: \.65rem;/);
  assert.match(
    styles,
    /\*::-webkit-scrollbar-thumb \{[^}]*border-radius: 999px;[^}]*background: var\(--color-outline-strong\);/,
  );
  assert.match(
    styles,
    /\*::-webkit-scrollbar-thumb:hover \{ background-color: var\(--color-interaction\); \}/,
  );
  assert.match(styles, /\*::-webkit-scrollbar-corner \{ background: transparent; \}/);
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
    ["#F7F2F8", "#121015"],
    ["#C9BFCE", "#121015"],
    ["#C9BFCE", "#37313E"],
    ["#8DF5E4", "#275B53"],
    ["#FFDCC6", "#5A3829"],
    ["#FFDAD6", "#5A302F"],
    ["#F3F0EF", "#09090A"],
    ["#B978FF", "#1B181F"],
    ["#5FD6C7", "#1B181F"],
    ["#F3A35B", "#1B181F"],
    ["#FF8A86", "#1B181F"],
    ["#D2A8FF", "#1B181F"],
    ["#68CFE3", "#1B181F"],
    ["#9A6CFF", "#1B181F"],
    ["#E8793E", "#1B181F"],
  ];
  for (const [foreground, background] of normalTextPairs) {
    assert.ok(
      contrast(foreground, background) >= 4.5,
      `${foreground} on ${background} must meet WCAG AA`,
    );
  }
  assert.ok(contrast("#D2A8FF", "#121015") >= 3, "focus indicator must meet 3:1");
  assert.ok(contrast("#83748F", "#121015") >= 3, "strong outline must meet 3:1");
});
