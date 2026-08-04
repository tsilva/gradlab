import {
  compatibleMetricKeys,
  descriptorCatalog,
  metricOptions,
} from "./telemetry.js";

const BLOCK_LABELS = Object.freeze({
  stats: "Stats",
  line: "Line chart",
  histogram: "Histogram",
  distribution: "Distribution",
  "namespace-explorer": "Metric explorer",
  "reward-breakdown": "Reward breakdown",
});

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function option(value, label, selected = false) {
  const item = document.createElement("option");
  item.value = value;
  item.textContent = label;
  item.selected = selected;
  return item;
}

export function defaultBlockForKind(kind) {
  if (kind === "stats" || kind === "line") {
    return { kind, metrics: ["reward/shaped"] };
  }
  if (kind === "namespace-explorer") {
    return { kind, namespace: "signal", metric: "" };
  }
  if (kind === "reward-breakdown") return { kind, scope: "step" };
  return {
    kind,
    metric: kind === "distribution" ? "policy/distribution" : "action/executed",
  };
}

export function editorFieldsForBlock(block) {
  return {
    metric: block?.kind !== "reward-breakdown",
    namespace: block?.kind === "namespace-explorer",
    scope: block?.kind === "reward-breakdown",
  };
}

export class PanelManager {
  constructor({
    getWorkspace,
    getContext,
    getWindowId,
    onReveal,
    onCreate,
    onUpdate,
    onDuplicate,
    onRemove,
    showToast,
  }) {
    this.getWorkspace = getWorkspace;
    this.getContext = getContext;
    this.getWindowId = getWindowId;
    this.onReveal = onReveal;
    this.onCreate = onCreate;
    this.onUpdate = onUpdate;
    this.onDuplicate = onDuplicate;
    this.onRemove = onRemove;
    this.showToast = showToast;
    this.dialog = document.querySelector("#panel-editor");
    this.form = document.querySelector("#panel-editor-form");
    this.title = document.querySelector("#panel-editor-title");
    this.blocks = document.querySelector("#panel-editor-blocks");
    this.heading = document.querySelector("#panel-editor-heading");
    this.editingId = null;
    this.draftBlocks = [];
    this.bind();
  }

  bind() {
    document.querySelector("#panel-add")?.addEventListener("click", () => {
      this.openEditor();
    });
    document.querySelector("#panel-editor-add-block")?.addEventListener("click", () => {
      this.draftBlocks.push({ kind: "line", metrics: ["reward/shaped"] });
      this.renderEditorBlocks();
    });
    document.querySelector("#panel-editor-cancel")?.addEventListener("click", () => {
      this.dialog.close();
    });
    this.form?.addEventListener("submit", (event) => {
      event.preventDefault();
      this.save();
    });
  }

  contextCatalog() {
    const { snapshot, history } = this.getContext();
    return descriptorCatalog(snapshot, history);
  }

  renderShelf() {
    const workspace = this.getWorkspace();
    const windowId = this.getWindowId();
    const target = document.querySelector("#panel-shelf-items");
    const entries = Object.entries(workspace.panels)
      .filter(([, panel]) => (
        !panel.placement.visible || panel.placement.window !== windowId
      ))
      .sort((left, right) => left[1].title.localeCompare(right[1].title));
    const buttons = entries.map(([id, panel]) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "shelf-item";
      const label = panel.placement.visible
        ? `Move ${panel.title} to this window`
        : `Show ${panel.title}`;
      button.setAttribute("aria-label", label);
      button.title = label;
      const title = document.createElement("span");
      title.textContent = panel.title;
      button.append(title);
      if (panel.placement.visible) {
        const status = document.createElement("small");
        status.textContent = "Other window";
        button.append(status);
      }
      button.addEventListener("click", () => this.onReveal(id));
      return button;
    });
    if (!buttons.length) {
      const empty = document.createElement("span");
      empty.className = "empty-state";
      empty.textContent = "Every panel is visible in this window.";
      target.replaceChildren(empty);
    } else target.replaceChildren(...buttons);
  }

  openEditor(id = null) {
    const panel = id ? this.getWorkspace().panels[id] : null;
    if (panel && panel.type !== "telemetry") {
      this.showToast("Only telemetry panels have editable visualizations.", true);
      return;
    }
    this.editingId = id;
    this.heading.textContent = panel ? `Edit ${panel.title}` : "Add telemetry panel";
    this.title.value = panel?.title || "Telemetry";
    this.draftBlocks = clone(panel?.config?.blocks || [
      { kind: "line", metrics: ["reward/shaped"] },
    ]);
    this.renderEditorBlocks();
    this.dialog.showModal();
    this.title.focus();
    this.title.select();
  }

  blockOptions(kind, selectedValues) {
    const catalog = this.contextCatalog();
    const { snapshot } = this.getContext();
    if (kind === "namespace-explorer") {
      return [...catalog.values()]
        .filter((descriptor) => descriptor.namespace)
        .map((descriptor) => option(
          descriptor.key,
          descriptor.label,
          selectedValues.includes(descriptor.key),
        ));
    }
    return metricOptions(catalog, kind, snapshot, selectedValues).map((descriptor) => option(
      descriptor.key,
      descriptor.label,
      selectedValues.includes(descriptor.key),
    ));
  }

  renderEditorBlocks() {
    this.blocks.replaceChildren(...this.draftBlocks.map((block, index) => {
      const row = document.createElement("fieldset");
      row.className = "panel-editor-block";
      const legend = document.createElement("legend");
      legend.textContent = `Visualization ${index + 1}`;
      const kindLabel = document.createElement("label");
      kindLabel.textContent = "Type";
      const kind = document.createElement("select");
      kind.append(...Object.entries(BLOCK_LABELS).map(([value, label]) => (
        option(value, label, value === block.kind)
      )));
      kind.addEventListener("change", () => {
        this.draftBlocks[index] = defaultBlockForKind(kind.value);
        this.renderEditorBlocks();
      });
      kindLabel.append(kind);

      const titleLabel = document.createElement("label");
      titleLabel.textContent = "Block title (optional)";
      const title = document.createElement("input");
      title.type = "text";
      title.maxLength = 80;
      title.value = block.title || "";
      title.addEventListener("input", () => {
        if (title.value) this.draftBlocks[index].title = title.value;
        else delete this.draftBlocks[index].title;
      });
      titleLabel.append(title);

      const metricLabel = document.createElement("label");
      metricLabel.textContent = block.kind === "stats" || block.kind === "line"
        ? "Metrics"
        : "Metric";
      const metrics = document.createElement("select");
      const multiple = block.kind === "stats" || block.kind === "line";
      metrics.multiple = multiple;
      metrics.size = multiple ? 6 : 1;
      const selectedValues = multiple ? block.metrics || [] : [block.metric || ""];
      const options = this.blockOptions(block.kind, selectedValues);
      if (!multiple) {
        metrics.append(option("", "Choose a metric", !selectedValues[0]));
      }
      metrics.append(...options);
      metrics.addEventListener("change", () => {
        if (multiple) {
          this.draftBlocks[index].metrics = [...metrics.selectedOptions]
            .map((item) => item.value);
        } else this.draftBlocks[index].metric = metrics.value;
      });
      metricLabel.append(metrics);

      const namespaceLabel = document.createElement("label");
      namespaceLabel.textContent = "Namespace";
      const namespace = document.createElement("select");
      namespace.append(
        option("signal", "Environment signals", block.namespace !== "reward-component"),
        option(
          "reward-component",
          "Reward components",
          block.namespace === "reward-component",
        ),
      );
      namespace.addEventListener("change", () => {
        this.draftBlocks[index].namespace = namespace.value;
        this.draftBlocks[index].metric = "";
        this.renderEditorBlocks();
      });
      namespaceLabel.append(namespace);

      const scopeLabel = document.createElement("label");
      scopeLabel.textContent = "Scope";
      const scope = document.createElement("select");
      scope.append(
        option("step", "Selected step", block.scope !== "episode"),
        option("episode", "Episode to cursor", block.scope === "episode"),
      );
      scope.addEventListener("change", () => {
        this.draftBlocks[index].scope = scope.value;
      });
      scopeLabel.append(scope);

      if (block.kind === "namespace-explorer") {
        const namespaceOptions = [...metrics.options];
        namespaceOptions.forEach((item) => {
          const descriptor = this.contextCatalog().get(item.value);
          item.hidden = Boolean(
            descriptor?.namespace && descriptor.namespace !== block.namespace,
          );
        });
      }

      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "quiet danger";
      remove.textContent = "Remove visualization";
      remove.addEventListener("click", () => {
        this.draftBlocks.splice(index, 1);
        this.renderEditorBlocks();
      });

      const fields = editorFieldsForBlock(block);
      row.append(legend, kindLabel, titleLabel);
      if (fields.namespace) row.append(namespaceLabel);
      if (fields.scope) row.append(scopeLabel);
      if (fields.metric) row.append(metricLabel);
      row.append(remove);
      return row;
    }));
  }

  save() {
    const title = this.title.value.trim();
    if (!title) {
      this.showToast("Give the panel a title.", true);
      this.title.focus();
      return;
    }
    if (!this.draftBlocks.length) {
      this.showToast("Add at least one visualization.", true);
      return;
    }
    const catalog = this.contextCatalog();
    const invalid = this.draftBlocks.find((block) => {
      if (block.kind === "stats") return !block.metrics?.length;
      if (block.kind === "line") {
        return !block.metrics?.length
          || !compatibleMetricKeys(block.metrics, catalog);
      }
      if (block.kind === "namespace-explorer") return !block.namespace;
      if (block.kind === "reward-breakdown") {
        return !["step", "episode"].includes(block.scope);
      }
      return !block.metric;
    });
    if (invalid?.kind === "line") {
      this.showToast("A line chart can only combine scalar metrics with the same unit.", true);
      return;
    }
    if (invalid) {
      this.showToast("Choose metrics for every visualization.", true);
      return;
    }
    const value = { title, config: { blocks: clone(this.draftBlocks) } };
    if (this.editingId) this.onUpdate(this.editingId, value);
    else this.onCreate(value);
    this.dialog.close();
  }

  duplicate(id) {
    this.onDuplicate(id);
  }

  remove(id) {
    this.onRemove(id);
  }
}
