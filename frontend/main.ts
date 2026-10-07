import { flushSync, mount } from "svelte";
import Shell from "./components/Shell.svelte";
import { installButtonTooltips } from "./button-tooltips.js";
mount(Shell, { target: document.body });
flushSync();
installButtonTooltips();
await import("../src/gradlab/web_player/app.js");
