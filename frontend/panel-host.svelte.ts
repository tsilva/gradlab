import { flushSync, mount, unmount } from "svelte";
import { PanelDelivery } from "./panel-delivery.js";
import Workspace from "./components/Workspace.svelte";

/** Svelte owns panel membership and disposal; GridStack owns only geometry.
 * Frame preparation remains an explicit operation of synchronized presentation.
 */
export class PanelHost extends PanelDelivery {
  entries = $state.raw<any[]>([]);
  component: Record<string, any>;
  constructor(options: any) {
    super(options, flushSync);
    this.component = mount(Workspace, {
      target: options.container,
      props: { host: this },
    });
  }
  async sync(workspace: any, windowId: string) {
    flushSync(() => {
      this.entries = Object.entries<any>(workspace.panels)
        .filter(
          ([, panel]) =>
            panel.placement.visible && panel.placement.window === windowId,
        )
        .map(([id, panel]) => {
          const definition = this.options.definitionFor(workspace, id);
          return {
            id,
            definition,
            placement: panel.placement,
            key: JSON.stringify([
              id,
              definition.type,
              definition.title,
              definition.config,
            ]),
          };
        });
    });
    for (const entry of this.entries) {
      const instance = this.instances.get(entry.id);
      if (instance) instance.definition = entry.definition;
      if (instance)
        this.options.onLayout?.(
          instance.element,
          entry.id,
          entry.placement,
          instance.gridItem,
          entry.definition,
        );
    }
  }
  register(id: string, instance: any) {
    this.instances.set(id, {
      ...instance,
      ...instance.component,
      definition: this.entries.find((entry) => entry.id === id).definition,
    });
    const entry = this.entries.find((entry) => entry.id === id);
    this.options.onMount?.(
      instance.element,
      id,
      entry.definition,
      instance.gridItem,
    );
    // Mount can occur during a Svelte flush. Updating component state is enough;
    // the owner flush completes metadata before returning to presentation.
    if (entry.definition.enabled && !this.options.isSuspended?.(id)) {
      instance.component.render?.(this.view.snapshot, this.view);
      instance.component.renderHistory?.(
        this.view.history,
        this.view.snapshot,
        this.view,
      );
    }
  }
  unregister(id: string, gridItem: HTMLElement) {
    const instance = this.instances.get(id);
    if (!instance || instance.gridItem !== gridItem) return;
    this.options.onUnmount?.(instance.element, id, instance.gridItem);
    this.instances.delete(id);
  }
  destroy() {
    void unmount(this.component);
  }
}
