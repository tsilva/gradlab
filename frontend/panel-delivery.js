/** Explicit delivery boundary for snapshots and synchronized bitmap frames.
 * Panel membership belongs to Svelte; this controller only gates expensive work.
 */
export class PanelDelivery {
  instances = new Map();
  view = { snapshot: null, history: [], inspection: false };
  /** @param {any} options @param {any} flush */
  constructor(options = {}, flush = (callback) => callback?.()) {
    this.options = options;
    this.flush = flush;
  }
  invoke(id, method, ...args) {
    if (
      (!this.instances.get(id)?.definition.enabled ||
        this.options.isSuspended?.(id)) &&
      [
        "render",
        "renderHistory",
        "renderFrame",
        "prepareFrame",
        "resize",
      ].includes(method)
    )
      return;
    const callback = this.instances.get(id)?.[method];
    if (typeof callback !== "function") return;
    try {
      return callback(...args);
    } catch (error) {
      this.options.onError?.(id, error);
    }
  }
  enabled() {
    return [...this.instances]
      .filter(
        ([id, panel]) =>
          panel.definition.enabled && !this.options.isSuspended?.(id),
      )
      .map(([id, panel]) => ({ id, definition: panel.definition }));
  }
  renderSnapshot(snapshot, view = {}) {
    this.view = { ...this.view, ...view, snapshot };
    this.flush(() => {
      for (const { id } of this.enabled())
        this.invoke(id, "render", snapshot, this.view);
    });
  }
  renderHistory(history, snapshot = this.view.snapshot, view = {}) {
    this.view = { ...this.view, ...view, history, snapshot };
    this.flush(() => {
      for (const { id } of this.enabled())
        this.invoke(id, "renderHistory", history, snapshot, this.view);
    });
  }
  async frame(method, kind, blob, metadata = {}) {
    const results = await Promise.all(
      this.enabled()
        .filter((entry) => entry.definition.frameKinds.includes(kind))
        .map(async ({ id }) => {
          const instance = this.instances.get(id);
          const isCurrent = () =>
            metadata.isCurrent?.() !== false &&
            this.instances.get(id) === instance &&
            instance.definition.enabled &&
            !this.options.isSuspended?.(id);
          try {
            return await this.invoke(id, method, kind, blob, {
              ...metadata,
              isCurrent,
            });
          } catch (error) {
            if (isCurrent()) this.options.onError?.(id, error);
            return false;
          }
        }),
    );
    this.flush();
    return results.some(Boolean);
  }
  prepareFrame(kind, blob, metadata = {}) {
    return this.frame("prepareFrame", kind, blob, metadata);
  }
  renderFrame(kind, blob, metadata = {}) {
    return this.frame("renderFrame", kind, blob, metadata);
  }
  resetFrames() {
    this.flush(() => {
      for (const id of this.instances.keys()) this.invoke(id, "resetFrames");
    });
  }
  resize() {
    for (const { id } of this.enabled()) this.invoke(id, "resize");
  }
}
