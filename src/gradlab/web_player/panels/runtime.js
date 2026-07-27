export class PanelRuntime {
  constructor({
    definitionFor,
    container,
    services,
    onMount,
    onLayout,
    onUnmount,
    onError,
  }) {
    this.definitionFor = definitionFor;
    this.container = container;
    this.services = services;
    this.onMount = onMount;
    this.onLayout = onLayout;
    this.onUnmount = onUnmount;
    this.onError = onError;
    this.instances = new Map();
    this.loading = new Map();
    this.desired = new Map();
    this.generation = 0;
    this.view = { snapshot: null, history: [], inspection: false };
  }

  async sync(workspace, windowId) {
    const generation = ++this.generation;
    this.desired = new Map(
      Object.entries(workspace.panels)
        .filter(([, panel]) => (
          panel.placement.visible && panel.placement.window === windowId
        ))
        .map(([id]) => [id, this.definitionFor(workspace, id)])
        .filter(([, definition]) => definition),
    );

    [...this.instances.entries()].forEach(([id, instance]) => {
      const desired = this.desired.get(id);
      if (!desired || instance.fingerprint !== this.fingerprint(desired)) {
        this.unmount(id);
      }
    });

    await Promise.all([...this.desired.keys()].map((id) => this.ensureMounted(id)));
    if (generation !== this.generation) return;
    await Promise.all(
      [...this.desired.keys()]
        .filter((id) => !this.instances.has(id))
        .map((id) => this.ensureMounted(id)),
    );
    if (generation !== this.generation) return;
    this.instances.forEach((instance, id) => {
      const panel = workspace.panels[id];
      if (!panel || !this.desired.has(id)) return;
      this.onLayout?.(
        instance.element,
        id,
        panel.placement,
        instance.gridItem,
        instance.definition,
      );
    });
  }

  fingerprint(definition) {
    return JSON.stringify([
      definition.type,
      definition.title,
      definition.config,
    ]);
  }

  async ensureMounted(id) {
    if (this.instances.has(id)) return this.instances.get(id);
    if (this.loading.has(id)) return this.loading.get(id);
    const definition = this.desired.get(id);
    if (!definition) return null;
    const loading = this.loadPanel(id, definition);
    this.loading.set(id, loading);
    try {
      return await loading;
    } finally {
      this.loading.delete(id);
    }
  }

  async loadPanel(id, definition) {
    try {
      const module = await import(definition.module);
      const current = this.desired.get(id);
      if (!current || current.module !== definition.module) return null;
      definition = current;
      const instance = await module.mount({
        definition,
        services: this.services,
      });
      if (!instance?.element) throw new Error(`Panel ${id} did not return an element.`);
      const desired = this.desired.get(id);
      if (
        !desired
        || this.fingerprint(desired) !== this.fingerprint(definition)
      ) {
        instance.destroy?.();
        return null;
      }
      instance.element.dataset.panel = id;
      instance.element.classList.add("grid-stack-item-content");
      const gridItem = document.createElement("div");
      gridItem.className = "grid-stack-item";
      gridItem.dataset.panel = id;
      gridItem.append(instance.element);
      this.container.append(gridItem);
      const observer = new ResizeObserver(() => this.safeCall(id, "resize"));
      observer.observe(instance.element);
      Object.assign(instance, {
        definition,
        fingerprint: this.fingerprint(definition),
        observer,
        gridItem,
      });
      this.instances.set(id, instance);
      this.onMount?.(instance.element, id, definition, gridItem);
      this.safeCall(id, "render", this.view.snapshot, this.view);
      this.safeCall(
        id,
        "renderHistory",
        this.view.history,
        this.view.snapshot,
        this.view,
      );
      return instance;
    } catch (error) {
      this.onError?.(id, error);
      return null;
    }
  }

  unmount(id) {
    const instance = this.instances.get(id);
    if (!instance) return;
    instance.observer?.disconnect();
    this.onUnmount?.(instance.element, id, instance.gridItem);
    this.safeCall(id, "destroy");
    instance.gridItem.remove();
    this.instances.delete(id);
  }

  safeCall(id, method, ...args) {
    const callback = this.instances.get(id)?.[method];
    if (typeof callback !== "function") return undefined;
    try {
      return callback(...args);
    } catch (error) {
      this.onError?.(id, error);
      return undefined;
    }
  }

  renderSnapshot(snapshot, view = {}) {
    this.view = { ...this.view, ...view, snapshot };
    this.instances.forEach((_, id) => this.safeCall(id, "render", snapshot, this.view));
  }

  renderHistory(history, snapshot = this.view.snapshot, view = {}) {
    this.view = { ...this.view, ...view, history, snapshot };
    this.instances.forEach((_, id) => this.safeCall(
      id,
      "renderHistory",
      history,
      snapshot,
      this.view,
    ));
  }

  async renderFrame(kind, blob) {
    const tasks = [...this.instances.entries()]
      .filter(([, instance]) => instance.definition.frameKinds.includes(kind))
      .map(([id]) => Promise.resolve(this.safeCall(id, "renderFrame", kind, blob))
        .catch((error) => {
          this.onError?.(id, error);
          return false;
        }));
    const results = await Promise.all(tasks);
    return results.some(Boolean);
  }

  invoke(id, method, ...args) {
    return this.safeCall(id, method, ...args);
  }

  resize() {
    this.instances.forEach((_, id) => this.safeCall(id, "resize"));
  }
}
