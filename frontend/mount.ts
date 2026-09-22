import { flushSync, mount, unmount } from "svelte";
import type { Component } from "svelte";
import PlaybackSettings from "./components/PlaybackSettings.svelte";

// Synchronized presentation owns preparation and commitment. Flush the component
// metadata in that boundary so it cannot trail its matching canvas into a frame.
export function mountPanel(component: Component<any, any>, options: any) {
  const target = document.createElement("div");
  target.style.display = "contents";
  const instance = mount(component, { target, props: options });
  flushSync();
  const result: Record<string, any> = { element: target };
  for (const [method, callback] of Object.entries(instance)) {
    if (typeof callback !== "function") continue;
    result[method] = (...args: unknown[]) => {
      const value = flushSync(() => callback(...args));
      return value instanceof Promise
        ? value.then((result) => {
            flushSync();
            return result;
          })
        : value;
    };
  }
  result.destroy = () => {
    void unmount(instance);
  };
  return result;
}
export const mountPlaybackSettings = (options: any) =>
  mountPanel(PlaybackSettings, options);
