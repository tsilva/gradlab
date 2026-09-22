import { flushSync, mount } from "svelte";
import Shell from "./components/Shell.svelte";
mount(Shell, { target: document.body });
flushSync();
await import("../src/gradlab/web_player/app.js");
