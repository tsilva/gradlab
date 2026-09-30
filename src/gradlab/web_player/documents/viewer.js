import {
  contractSearchRanges,
  contractSyntaxTokens,
} from "./syntax.js";
import {
  buildSideBySideRows,
  sideBySideSearchCounts,
} from "./diff.js";

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
    this.payload = null;
    this.documentKind = "goal";
    this.view = "resolved";
    this.requestSerial = 0;
    this.controller = null;
    this.returnFocus = null;
    this.recipeEndpoint = null;
    this.recipeDocuments = new Map();
    this.query = "";
    this.searchOpen = false;
    this.recipeItems = [];
    this.selectedRecipeId = "";
    this.loading = false;
    this.error = "";
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
    this.searchOpen = false;
    this.query = "";
    this.recipeItems = [];
    this.selectedRecipeId = "";
    this.error = "";
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
      this.recipeItems = recipeItems;
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
      this.error = String(error?.message || error);
      this.render();
      this.showToast(String(error?.message || error), true);
    } finally {
      if (serial === this.requestSerial) this.setLoading(false);
    }
  }

  async selectRecipe(recipeId, { serial = this.requestSerial } = {}) {
    if (!recipeId || !this.recipeEndpoint) return;
    this.selectedRecipeId = recipeId;
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
    const text = this.currentText();
    let rows = [];
    let diffError = "";
    if (this.view === "changes") {
      try {
        rows = buildSideBySideRows({
          baseText: current?.views?.base || "",
          resolvedText: current?.views?.resolved || "",
          unifiedDiff: text,
        });
      } catch {
        diffError = "Could not render this comparison because the diff does not match the supplied Base and Resolved YAML.";
      }
    }
    const counts = this.view === "changes" ? sideBySideSearchCounts(rows, this.query) : null;
    const matches = counts ? counts.base + counts.resolved : contractSearchRanges(text, this.query).length;
    this.presentation?.render({
      current, documents: this.payload?.documents || {}, documentKind: this.documentKind,
      view: this.view, hasChanges: this.hasChanges(), text, rows, diffError,
      query: this.query, searchOpen: this.searchOpen,
      count: !this.query || diffError ? "" : !matches ? "No matches" : counts
        ? `Base ${counts.base.toLocaleString()} · Resolved ${counts.resolved.toLocaleString()}`
        : `${matches.toLocaleString()} match${matches === 1 ? "" : "es"}`,
      loading: this.loading, error: this.error, recipeItems: this.recipeItems,
      selectedRecipeId: this.selectedRecipeId,
    });
  }

  setLoading(loading) {
    this.loading = loading;
    this.render();
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

export function decoratedText(value, view, query, emphasis = []) {
  const matches = contractSearchRanges(value, query);
  let offset = 0;
  return contractSyntaxTokens(value, view).flatMap((token) => {
    const end = offset + token.text.length;
    const boundaries = new Set([offset, end]);
    [...matches, ...emphasis].forEach((range) => {
      if (range.start > offset && range.start < end) boundaries.add(range.start);
      if (range.end > offset && range.end < end) boundaries.add(range.end);
    });
    const ordered = [...boundaries].sort((a, b) => a - b);
    offset = end;
    return ordered.slice(0, -1).map((start, index) => ({
      text: value.slice(start, ordered[index + 1]), className: token.className,
      emphasis: emphasis.some((r) => r.start < ordered[index + 1] && r.end > start),
      match: matches.some((r) => r.start < ordered[index + 1] && r.end > start),
    }));
  });
}
