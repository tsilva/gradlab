import { createHash } from "node:crypto";
import { readdirSync, readFileSync } from "node:fs";
import { relative, resolve } from "node:path";
export function playerBuildInputs() {
  const paths = [];
  function walk(directory) {
    for (const entry of readdirSync(directory, { withFileTypes: true })) {
      if (entry.name === "dist") continue;
      const path = resolve(directory, entry.name);
      if (entry.isDirectory()) walk(path);
      else if (entry.isFile()) paths.push(path);
    }
  }
  walk(resolve("frontend"));
  walk(resolve("src/gradlab/web_player"));
  paths.push(
    ...[
      "package.json",
      "pnpm-lock.yaml",
      "pnpm-workspace.yaml",
      ".npmrc",
      "vite.config.ts",
      "svelte.config.js",
      "tsconfig.json",
      "scripts/player-build-inputs.mjs",
    ].map((path) => resolve(path)),
  );
  return Object.fromEntries(
    paths
      .sort()
      .map((path) => [
        relative(process.cwd(), path),
        createHash("sha256").update(readFileSync(path)).digest("hex"),
      ]),
  );
}

export function playerBuildOutputs(directory) {
  return Object.fromEntries(
    readdirSync(directory, { recursive: true, withFileTypes: true })
      .filter((entry) => entry.isFile() && entry.name !== "build-manifest.json")
      .map((entry) => resolve(entry.parentPath, entry.name))
      .sort()
      .map((path) => [
        relative(directory, path),
        createHash("sha256").update(readFileSync(path)).digest("hex"),
      ]),
  );
}
