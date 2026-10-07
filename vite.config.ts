import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import { resolve } from "node:path";
import { cpSync, writeFileSync } from "node:fs";
import {
  playerBuildInputs,
  playerBuildOutputs,
} from "./scripts/player-build-inputs.mjs";

const source = resolve("src/gradlab/web_player");
const output = resolve("src/gradlab/web_player/dist");
export default defineConfig(({ mode }) =>
  mode === "fixtures"
    ? {
        plugins: [svelte()],
        build: {
          outDir: "logs/web-fixtures",
          rollupOptions: {
            input: {
              "chart-panels": resolve(
                "tests/web_player/fixtures/chart-panels.mjs",
              ),
            },
            output: { entryFileNames: "[name].js" },
          },
        },
      }
    : {
        plugins: [
          svelte(),
          {
            name: "player-static-assets",
            closeBundle() {
              for (const file of [
                "index.html",
                "styles.css",
                "fonts",
                "vendor",
                "favicon.svg",
                "tabler-icons.svg",
                "tabler-chevron-down.svg",
                "oauth_complete.html",
                "oauth_complete.js",
              ]) {
                cpSync(resolve(source, file), resolve(output, file), {
                  recursive: true,
                });
              }
              writeFileSync(
                resolve(output, "build-manifest.json"),
                JSON.stringify(
                  {
                    inputs: playerBuildInputs(),
                    outputs: playerBuildOutputs(output),
                  },
                  null,
                  2,
                ) + "\n",
              );
            },
          },
        ],
        build: {
          outDir: output,
          emptyOutDir: true,
          minify: true,
          rollupOptions: {
            input: {
              app: resolve("frontend/main.ts"),
              "sources/browser": resolve(source, "sources/browser.js"),
            },
            output: {
              entryFileNames: "[name].js",
              chunkFileNames: "chunks/[name]-[hash].js",
              assetFileNames: "[name][extname]",
            },
          },
        },
      },
);
