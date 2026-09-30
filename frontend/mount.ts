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

export async function mountSourceBrowser(
  controller: any,
  root: HTMLElement,
  breadcrumbsRoot: HTMLElement,
  navigationRoot: HTMLElement,
) {
  const [{ default: Browser }, { default: Navigation }] = await Promise.all([
    import("./components/SourceBrowser.svelte"),
    import("./components/SourceNavigation.svelte"),
  ]);
  const browser = mountPanel(Browser, { controller });
  const breadcrumbs = mountPanel(Navigation, {
    controller,
    kind: "breadcrumbs",
  });
  const navigation = mountPanel(Navigation, {
    controller,
    kind: "checkpoints",
  });
  root.replaceChildren(browser.element);
  breadcrumbsRoot.replaceChildren(breadcrumbs.element);
  navigationRoot.replaceChildren(navigation.element);
  return {
    render: browser.render,
    breadcrumbs(items: any[]) {
      breadcrumbs.render(items);
      breadcrumbsRoot.hidden = !items.length;
    },
    navigation(value: any) {
      if (value) navigation.render(value);
      navigationRoot.hidden = !value;
      navigationRoot.classList.toggle("warning", Boolean(value?.warning));
    },
  };
}

export async function mountContractViewer(
  controller: any,
  dialog: HTMLElement,
) {
  const { default: Viewer } =
    await import("./components/ContractViewer.svelte");
  const viewer = mountPanel(Viewer, { controller });
  dialog.replaceChildren(viewer.element);
  return viewer;
}
