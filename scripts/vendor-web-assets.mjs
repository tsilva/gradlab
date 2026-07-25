import { copyFile, mkdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const source = resolve(root, "node_modules/gridstack/dist");
const destination = resolve(root, "src/rlab/web_player/vendor/gridstack");

await mkdir(destination, { recursive: true });
await Promise.all([
  copyFile(resolve(source, "gridstack-all.js"), resolve(destination, "gridstack-all.js")),
  copyFile(resolve(source, "gridstack.min.css"), resolve(destination, "gridstack.min.css")),
]);
