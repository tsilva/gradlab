function makeTab(label, value, group, onSelect) {
  const tab = document.createElement("button");
  tab.type = "button";
  tab.className = "contract-tab";
  tab.setAttribute("role", "tab");
  tab.dataset.value = value;
  tab.textContent = label;
  tab.addEventListener("click", () => onSelect(value));
  group.append(tab);
  return tab;
}

async function jsonRequest(url, token, signal) {
  const response = await fetch(url, {
    headers: { Authorization: `Bearer ${token}` },
    cache: "no-store",
    signal,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || `Contract request failed (${response.status})`);
  }
  return payload;
}

export class ContractViewer {
  constructor(dialog, { token, showToast }) {
    this.dialog = dialog;
    this.token = token;
    this.showToast = showToast;
    this.heading = dialog.querySelector("#contract-viewer-heading");
    this.status = dialog.querySelector("#contract-viewer-status");
    this.message = dialog.querySelector("#contract-viewer-message");
    this.loading = dialog.querySelector("#contract-viewer-loading");
    this.content = dialog.querySelector("#contract-viewer-content");
    this.code = this.content.querySelector("code");
    this.search = dialog.querySelector("#contract-search-input");
    this.searchCount = dialog.querySelector("#contract-search-count");
    this.copy = dialog.querySelector("#contract-copy");
    this.documentTabs = dialog.querySelector("#contract-document-tabs");
    this.viewTabs = dialog.querySelector("#contract-view-tabs");
    this.recipePickerLabel = dialog.querySelector("#contract-recipe-picker-label");
    this.recipePicker = dialog.querySelector("#contract-recipe-picker");
    this.payload = null;
    this.documentKind = "goal";
    this.view = "resolved";
    this.requestSerial = 0;
    this.controller = null;
    this.returnFocus = null;
    this.recipeEndpoint = null;
    this.recipeDocuments = new Map();
    this.documentTabButtons = new Map();
    this.viewTabButtons = new Map();

    ["goal", "recipe"].forEach((kind) => {
      this.documentTabButtons.set(
        kind,
        makeTab(
          kind === "goal" ? "Goal" : "Recipe",
          kind,
          this.documentTabs,
          (value) => this.selectDocument(value),
        ),
      );
    });
    [
      ["changes", "Changes"],
      ["base", "Base"],
      ["resolved", "Resolved"],
    ].forEach(([value, label]) => {
      this.viewTabButtons.set(
        value,
        makeTab(label, value, this.viewTabs, (selected) => {
          this.view = selected;
          this.render();
        }),
      );
    });
    dialog.querySelector("#contract-viewer-close").addEventListener(
      "click",
      () => this.close(),
    );
    this.copy.addEventListener("click", () => this.copyCurrent());
    this.search.addEventListener("input", () => this.renderContent());
    this.recipePicker.addEventListener("change", () => {
      void this.selectRecipe(this.recipePicker.value).catch((error) => {
        this.showToast(String(error?.message || error), true);
      });
    });
    dialog.addEventListener("cancel", (event) => {
      event.preventDefault();
      this.close();
    });
    dialog.addEventListener("close", () => {
      this.controller?.abort();
      this.controller = null;
      this.returnFocus?.focus?.({ preventScroll: true });
      this.returnFocus = null;
    });
  }

  async open(
    endpoint,
    {
      preferredDocument = "goal",
      recipesEndpoint = null,
      recipeEndpoint = null,
    } = {},
  ) {
    this.returnFocus = document.activeElement;
    this.documentKind = preferredDocument;
    this.view = "resolved";
    this.payload = null;
    this.recipeEndpoint = recipeEndpoint;
    this.recipeDocuments.clear();
    this.search.value = "";
    this.recipePicker.replaceChildren();
    this.recipePickerLabel.hidden = true;
    this.message.hidden = true;
    this.message.textContent = "";
    this.code.textContent = "";
    this.content.hidden = true;
    this.setLoading(true);
    if (!this.dialog.open) this.dialog.showModal();
    this.controller?.abort();
    const controller = new AbortController();
    this.controller = controller;
    const serial = ++this.requestSerial;
    try {
      const [payload, recipes] = await Promise.all([
        jsonRequest(endpoint, this.token, controller.signal),
        recipesEndpoint
          ? jsonRequest(recipesEndpoint, this.token, controller.signal)
          : Promise.resolve(null),
      ]);
      if (serial !== this.requestSerial) return;
      this.payload = payload;
      const recipeItems = Array.isArray(recipes?.items) ? recipes.items : [];
      recipeItems.forEach((item) => {
        const option = document.createElement("option");
        option.value = String(item.recipe_id || "");
        option.textContent = String(item.title || item.recipe_id || "Recipe");
        this.recipePicker.append(option);
      });
      this.recipePickerLabel.hidden = recipeItems.length === 0;
      if (recipeItems.length && !payload?.documents?.recipe) {
        await this.selectRecipe(String(recipeItems[0].recipe_id || ""), { serial });
      }
      if (serial !== this.requestSerial) return;
      this.chooseInitialView();
      this.render();
    } catch (error) {
      if (controller.signal.aborted || serial !== this.requestSerial) return;
      this.payload = {
        source: {},
        documents: {},
      };
      this.message.hidden = false;
      this.message.textContent = String(error?.message || error);
      this.status.textContent = "Could not load contract";
      this.code.textContent = "";
      this.showToast(String(error?.message || error), true);
    } finally {
      if (serial === this.requestSerial) this.setLoading(false);
    }
  }

  async selectRecipe(recipeId, { serial = this.requestSerial } = {}) {
    if (!recipeId || !this.recipeEndpoint) return;
    this.recipePicker.value = recipeId;
    if (this.recipeDocuments.has(recipeId)) {
      this.payload.documents.recipe = this.recipeDocuments.get(recipeId);
      this.render();
      return;
    }
    this.setLoading(true);
    try {
      const endpoint = this.recipeEndpoint(recipeId);
      const payload = await jsonRequest(endpoint, this.token, this.controller?.signal);
      if (serial !== this.requestSerial) return;
      const recipe = payload?.documents?.recipe;
      if (!recipe) throw new Error("Recipe preview response has no recipe document");
      this.recipeDocuments.set(recipeId, recipe);
      this.payload.documents.recipe = recipe;
      this.render();
    } finally {
      if (serial === this.requestSerial) this.setLoading(false);
    }
  }

  chooseInitialView() {
    const document = this.currentDocument();
    if (!document && this.payload?.documents?.recipe) this.documentKind = "recipe";
    const selected = this.currentDocument();
    this.view = selected?.is_variant && selected?.views?.base
      ? "changes"
      : "resolved";
  }

  selectDocument(kind) {
    if (!this.payload?.documents?.[kind]) return;
    this.documentKind = kind;
    const document = this.currentDocument();
    if (this.view === "base" && !document?.views?.base) this.view = "resolved";
    if (this.view === "changes" && !document?.views?.changes?.unified_diff) {
      this.view = "resolved";
    }
    this.render();
  }

  currentDocument() {
    return this.payload?.documents?.[this.documentKind] || null;
  }

  currentText() {
    const document = this.currentDocument();
    if (!document) return "";
    if (this.view === "resolved") return document.views?.resolved || "";
    if (this.view === "base") return document.views?.base || "";
    return document.views?.changes?.unified_diff || "";
  }

  render() {
    const current = this.currentDocument();
    this.heading.textContent = current?.title || "Goal and recipe YAML";
    this.status.textContent = current
      ? [
          current.availability === "static-preview" ? "Static preview" : current.availability,
          current.variant_id ? `variant ${current.variant_id}` : "",
        ].filter(Boolean).join(" · ")
      : "Unavailable";
    this.documentTabButtons.forEach((tab, kind) => {
      const available = Boolean(this.payload?.documents?.[kind]);
      tab.disabled = !available;
      tab.setAttribute("aria-selected", String(kind === this.documentKind));
      tab.tabIndex = kind === this.documentKind ? 0 : -1;
    });
    const hasBase = Boolean(current?.views?.base);
    const hasChanges = Boolean(current?.views?.changes?.unified_diff);
    this.viewTabButtons.forEach((tab, view) => {
      const available = view === "resolved"
        ? Boolean(current?.views?.resolved)
        : view === "base"
          ? hasBase
          : hasChanges;
      tab.disabled = !available;
      tab.setAttribute("aria-selected", String(view === this.view));
      tab.tabIndex = view === this.view ? 0 : -1;
    });
    this.message.hidden = !current?.message;
    this.message.textContent = current?.message || "";
    this.content.hidden = !this.currentText();
    this.copy.disabled = !this.currentText();
    this.renderContent();
  }

  renderContent() {
    const value = this.currentText();
    const query = this.search.value;
    this.code.replaceChildren();
    if (!query || !value) {
      this.code.textContent = value;
      this.searchCount.textContent = "";
      return;
    }
    const normalized = query.toLocaleLowerCase();
    const source = value.toLocaleLowerCase();
    let cursor = 0;
    let count = 0;
    while (cursor < value.length) {
      const index = source.indexOf(normalized, cursor);
      if (index < 0) {
        this.code.append(document.createTextNode(value.slice(cursor)));
        break;
      }
      this.code.append(document.createTextNode(value.slice(cursor, index)));
      const mark = document.createElement("mark");
      mark.textContent = value.slice(index, index + query.length);
      this.code.append(mark);
      count += 1;
      cursor = index + query.length;
    }
    this.searchCount.textContent = count
      ? `${count.toLocaleString()} match${count === 1 ? "" : "es"}`
      : "No matches";
    this.code.querySelector("mark")?.scrollIntoView({ block: "center" });
  }

  setLoading(loading) {
    this.loading.hidden = !loading;
    this.content.setAttribute("aria-busy", String(loading));
  }

  async copyCurrent() {
    const value = this.currentText();
    if (!value) return;
    try {
      await navigator.clipboard.writeText(value);
      this.showToast("Contract copied.");
    } catch {
      this.showToast("Could not copy the contract.", true);
    }
  }

  close() {
    this.requestSerial += 1;
    this.controller?.abort();
    if (this.dialog.open) this.dialog.close();
  }
}
