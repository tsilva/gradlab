import {
  contractSearchRanges,
  contractSyntaxTokens,
} from "./syntax.js";
import {
  buildSideBySideRows,
  sideBySideSearchCounts,
} from "./diff.js";

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
    this.singleContent = dialog.querySelector("#contract-single-content");
    this.code = this.singleContent.querySelector("code");
    this.diffContent = dialog.querySelector("#contract-diff-content");
    this.diffError = dialog.querySelector("#contract-diff-error");
    this.diffBaseName = dialog.querySelector("#contract-diff-base-name");
    this.diffResolvedName = dialog.querySelector("#contract-diff-resolved-name");
    this.diffBaseScroll = dialog.querySelector("#contract-diff-base-scroll");
    this.diffResolvedScroll = dialog.querySelector("#contract-diff-resolved-scroll");
    this.diffBaseLines = dialog.querySelector("#contract-diff-base-lines");
    this.diffResolvedLines = dialog.querySelector("#contract-diff-resolved-lines");
    this.searchDisclosure = dialog.querySelector("#contract-search-disclosure");
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
    this.diffScrollSyncPending = false;

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
    [
      [this.diffBaseScroll, this.diffResolvedScroll],
      [this.diffResolvedScroll, this.diffBaseScroll],
    ].forEach(([source, target]) => {
      source.addEventListener("scroll", () => {
        if (this.diffScrollSyncPending) return;
        this.diffScrollSyncPending = true;
        target.scrollTop = source.scrollTop;
        requestAnimationFrame(() => {
          this.diffScrollSyncPending = false;
        });
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
    this.searchDisclosure.open = false;
    this.search.value = "";
    this.recipePicker.replaceChildren();
    this.recipePickerLabel.hidden = true;
    this.message.hidden = true;
    this.message.textContent = "";
    this.code.textContent = "";
    this.diffBaseLines.replaceChildren();
    this.diffResolvedLines.replaceChildren();
    this.diffContent.hidden = true;
    this.diffError.hidden = true;
    this.diffError.textContent = "";
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
    this.view = selected?.is_variant && this.hasChanges(selected)
      ? "changes"
      : "resolved";
  }

  selectDocument(kind) {
    if (!this.payload?.documents?.[kind]) return;
    this.documentKind = kind;
    const document = this.currentDocument();
    if (this.view === "base" && !document?.views?.base) this.view = "resolved";
    if (this.view === "changes" && !this.hasChanges(document)) {
      this.view = "resolved";
    }
    this.render();
  }

  currentDocument() {
    return this.payload?.documents?.[this.documentKind] || null;
  }

  hasChanges(document = this.currentDocument()) {
    return Boolean(
      document?.views?.base
      && document?.views?.resolved
      && document?.views?.changes?.unified_diff,
    );
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
    const hasChanges = this.hasChanges(current);
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
    this.content.hidden = this.view === "changes"
      ? !hasChanges
      : !this.currentText();
    this.copy.disabled = !this.currentText();
    this.renderContent();
  }

  renderContent() {
    this.diffError.hidden = true;
    this.diffError.textContent = "";
    if (this.view === "changes") {
      this.renderDiffContent();
      return;
    }
    this.renderSingleContent();
  }

  renderSingleContent() {
    const value = this.currentText();
    const query = this.search.value;
    const matches = contractSearchRanges(value, query);
    const tokens = contractSyntaxTokens(value, this.view);
    this.singleContent.hidden = false;
    this.diffContent.hidden = true;
    this.code.replaceChildren();
    this.appendDecoratedText(this.code, value, tokens, matches);
    this.searchCount.textContent = query
      ? (
        matches.length
          ? `${matches.length.toLocaleString()} match${matches.length === 1 ? "" : "es"}`
          : "No matches"
      )
      : "";
    if (query) this.code.querySelector("mark")?.scrollIntoView({ block: "center" });
  }

  renderDiffContent() {
    const current = this.currentDocument();
    this.singleContent.hidden = true;
    this.diffContent.hidden = false;
    const kind = current?.kind === "recipe" ? "recipe" : "goal";
    const baseName = `${kind}-base.yaml`;
    const resolvedName = `${kind}-resolved.yaml`;
    this.diffBaseName.textContent = baseName;
    this.diffResolvedName.textContent = resolvedName;
    this.diffBaseScroll.setAttribute("aria-label", `Base YAML, ${baseName}`);
    this.diffResolvedScroll.setAttribute("aria-label", `Resolved YAML, ${resolvedName}`);

    let rows;
    try {
      rows = buildSideBySideRows({
        baseText: current?.views?.base || "",
        resolvedText: current?.views?.resolved || "",
        unifiedDiff: current?.views?.changes?.unified_diff || "",
      });
    } catch {
      this.diffContent.hidden = true;
      this.diffError.hidden = false;
      this.diffError.textContent = (
        "Could not render this comparison because the diff does not match "
        + "the supplied Base and Resolved YAML."
      );
      this.searchCount.textContent = "";
      return;
    }

    const query = this.search.value;
    const counts = sideBySideSearchCounts(rows, query);
    const baseFragment = document.createDocumentFragment();
    const resolvedFragment = document.createDocumentFragment();
    rows.forEach((row) => {
      baseFragment.append(this.diffRow(row.base, query));
      resolvedFragment.append(this.diffRow(row.resolved, query));
    });
    this.diffBaseLines.replaceChildren(baseFragment);
    this.diffResolvedLines.replaceChildren(resolvedFragment);
    const totalMatches = counts.base + counts.resolved;
    this.searchCount.textContent = query
      ? (
        totalMatches
          ? `Base ${counts.base.toLocaleString()} · Resolved ${counts.resolved.toLocaleString()}`
          : "No matches"
      )
      : "";
    if (query) {
      const first = this.diffBaseLines.querySelector("mark")
        || this.diffResolvedLines.querySelector("mark");
      first?.scrollIntoView({ block: "center", inline: "center" });
    }
  }

  diffRow(item, query) {
    const row = document.createElement("div");
    row.className = `contract-diff-row contract-diff-${item?.change || "spacer"}`;
    const number = document.createElement("span");
    number.className = "contract-diff-line-number";
    number.setAttribute("aria-hidden", "true");
    number.textContent = item ? String(item.number) : "";
    const code = document.createElement("code");
    code.className = "contract-diff-line-code";
    if (item) {
      const matches = contractSearchRanges(item.text, query);
      const tokens = contractSyntaxTokens(item.text, "base");
      this.appendDecoratedText(code, item.text, tokens, matches, item.emphasis);
    }
    row.append(number, code);
    return row;
  }

  appendDecoratedText(parent, value, tokens, matches = [], emphasis = []) {
    let offset = 0;
    tokens.forEach((token) => {
      const tokenEnd = offset + token.text.length;
      const boundaries = new Set([offset, tokenEnd]);
      [...matches, ...emphasis].forEach((range) => {
        if (range.start > offset && range.start < tokenEnd) boundaries.add(range.start);
        if (range.end > offset && range.end < tokenEnd) boundaries.add(range.end);
      });
      const ordered = [...boundaries].sort((left, right) => left - right);
      for (let index = 0; index < ordered.length - 1; index += 1) {
        const start = ordered[index];
        const end = ordered[index + 1];
        if (end <= start) continue;
        let node = document.createTextNode(value.slice(start, end));
        if (token.className) {
          const syntax = document.createElement("span");
          syntax.className = token.className;
          syntax.append(node);
          node = syntax;
        }
        if (emphasis.some((range) => range.start < end && range.end > start)) {
          const inline = document.createElement("span");
          inline.className = "contract-diff-inline";
          inline.append(node);
          node = inline;
        }
        if (matches.some((range) => range.start < end && range.end > start)) {
          const mark = document.createElement("mark");
          mark.append(node);
          node = mark;
        }
        parent.append(node);
      }
      offset = tokenEnd;
    });
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
