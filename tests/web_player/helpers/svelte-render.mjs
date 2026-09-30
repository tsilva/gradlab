import { mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve, basename } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { compile } from "svelte/compiler";
import { render } from "svelte/server";

// Compile the actual components for server rendering. Generated modules stay in
// the ignored package cache and resolve the same installed Svelte runtime.
const cache = resolve("node_modules/.cache/gradlab-svelte-tests");
await mkdir(cache, { recursive: true });
const directory = await mkdtemp(`${cache}/render-`);
const modules = new Map();
async function load(path) {
  if (modules.has(path)) return modules.get(path);
  const target = resolve(directory, `${modules.size}-${basename(path)}.mjs`);
  modules.set(path, target);
  let { js } = compile(await readFile(path, "utf8"), { filename: path, generate: "server" });
  let code = js.code;
  for (const match of code.matchAll(/from ['"]([^'"]+)['"]/g)) {
    const specifier = match[1];
    if (!specifier.startsWith(".")) continue;
    const original = resolve(dirname(path), specifier);
    const imported = specifier.endsWith(".svelte") ? await load(original) : original;
    code = code.replace(match[0], `from ${JSON.stringify(pathToFileURL(imported).href)}`);
  }
  await writeFile(target, code);
  return target;
}
export async function renderComponent(path, props) {
  const { default: component } = await import(pathToFileURL(await load(fileURLToPath(path))).href);
  return render(component, { props }).body;
}
