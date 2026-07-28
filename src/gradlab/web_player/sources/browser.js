const ICONS = "/assets/tabler-icons.svg";

function icon(name) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.classList.add("icon");
  svg.setAttribute("aria-hidden", "true");
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", `${ICONS}#ti-${name}`);
  svg.append(use);
  return svg;
}

function button(label, { iconName = null, quiet = false, primary = false } = {}) {
  const element = document.createElement("button");
  element.type = "button";
  element.className = [
    iconName ? "button-with-icon" : "",
    quiet ? "quiet" : "",
    primary ? "primary" : "",
  ].filter(Boolean).join(" ");
  if (iconName) element.append(icon(iconName));
  const text = document.createElement("span");
  text.textContent = label;
  element.append(text);
  return element;
}

export function formatDate(value, nowValue = Date.now()) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  const now = new Date(nowValue);
  if (
    !Number.isNaN(now.getTime())
    && date.getFullYear() === now.getFullYear()
    && date.getMonth() === now.getMonth()
    && date.getDate() === now.getDate()
  ) {
    const elapsedMilliseconds = now.getTime() - date.getTime();
    const absoluteMilliseconds = Math.abs(elapsedMilliseconds);
    const [amount, unit] = absoluteMilliseconds < 60_000
      ? [Math.floor(absoluteMilliseconds / 1_000), "second"]
      : absoluteMilliseconds < 3_600_000
        ? [Math.floor(absoluteMilliseconds / 60_000), "minute"]
        : [Math.floor(absoluteMilliseconds / 3_600_000), "hour"];
    const label = `${unit}${amount === 1 ? "" : "s"}`;
    return elapsedMilliseconds >= 0 ? `${amount} ${label} ago` : `in ${amount} ${label}`;
  }
  return date.toLocaleString();
}

function formatBytes(value) {
  const bytes = Number(value);
  if (!Number.isFinite(bytes) || bytes <= 0) return "—";
  const units = ["B", "KiB", "MiB", "GiB"];
  const unit = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
  return `${(bytes / (1024 ** unit)).toFixed(unit ? 1 : 0)} ${units[unit]}`;
}

function runStatePresentation(value) {
  const state = String(value || "").trim().toLowerCase();
  if (state === "finished") {
    return { iconName: "check", tone: "finished", label: "Finished" };
  }
  if (state === "running") {
    return { iconName: "activity-heartbeat", tone: "running", label: "Running" };
  }
  if (state === "failed" || state === "crashed") {
    return {
      iconName: "x",
      tone: "failed",
      label: state === "crashed" ? "Crashed" : "Failed",
    };
  }
  if (["pending", "queued", "starting"].includes(state)) {
    return {
      iconName: "player-pause",
      tone: "pending",
      label: state ? humanizeMetricPart(state) : "Pending",
    };
  }
  if (["killed", "cancelled", "canceled", "preempted"].includes(state)) {
    return { iconName: "x", tone: "stopped", label: humanizeMetricPart(state) };
  }
  return {
    iconName: "activity-heartbeat",
    tone: "unknown",
    label: state ? humanizeMetricPart(state) : "Unknown",
  };
}

function humanizeMetricPart(value) {
  return String(value || "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

export function metricLabel(metric) {
  const name = String(metric || "");
  const known = {
    "leader/checkpoint/step": "Checkpoint step",
    "train/global_step": "Global step",
    "train/episode/return/shaped/from/target/mean": "Target return",
    "train/outcome/success/window_100/rate/min": "Min success (100)",
    "eval/full/outcome/success/rate/min": "Min success",
    "eval/full/outcome/success/rate/mean": "Mean success",
    "eval/full/episode/return/mean": "Mean return",
    "eval/full/episode/return/best": "Best return",
  };
  if (known[name]) return known[name];
  const reason = name.match(/^eval\/full\/outcome\/reason\/([^/]+)\/rate$/);
  if (reason) return `${humanizeMetricPart(reason[1])} failure rate`;
  const progress = name.match(/^eval\/full\/progress\/([^/]+)\/(mean|max)$/);
  if (progress) {
    return `${humanizeMetricPart(progress[1])} ${progress[2]}`;
  }
  return name
    .replace(/^(eval\/full|leader|train)\//, "")
    .split("/")
    .map(humanizeMetricPart)
    .join(" · ");
}

export function formatMetricValue(metric, value) {
  if (value === null || value === undefined || value === "") return "—";
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "—";
  if (String(metric).includes("/rate/") || String(metric).endsWith("/rate")) {
    return `${(numeric * 100).toLocaleString(undefined, { maximumFractionDigits: 2 })}%`;
  }
  return numeric.toLocaleString(undefined, { maximumFractionDigits: 3 });
}

function compactRecipeOverride(value) {
  const text = String(value || "").trim();
  const separator = text.indexOf("=");
  if (separator < 0) return text;
  let key = text.slice(0, separator).trim();
  const overrideValue = text.slice(separator + 1).trim();
  for (const prefix of [
    "train.backend.config.",
    "train.environment.env_config.",
    "train.environment.task.",
    "train.",
  ]) {
    if (key.startsWith(prefix)) {
      key = key.slice(prefix.length);
      break;
    }
  }
  return `${key}=${overrideValue}`;
}

export function recipeVariantPresentation(item) {
  const overrides = Array.isArray(item?.recipe_overrides)
    ? item.recipe_overrides.map((value) => String(value || "").trim()).filter(Boolean)
    : [];
  const variantId = String(item?.recipe_variant_id || "").trim();
  if (!overrides.length) {
    if (variantId === "base") {
      return {
        summary: "base",
        detail: "Checked-in recipe with no launch-time configuration overrides.",
      };
    }
    return {
      summary: "variation unknown",
      detail: "This run predates explicit recipe-variation metadata.",
    };
  }
  const visible = overrides.slice(0, 2).map(compactRecipeOverride);
  if (overrides.length > visible.length) visible.push(`+${overrides.length - visible.length}`);
  if (variantId) visible.push(variantId);
  return {
    summary: visible.join(" · "),
    detail: [
      variantId ? `Recipe variant ${variantId}` : "Launch-time recipe overrides",
      ...overrides,
    ].join("\n"),
  };
}

export function sortRunItems(items, sort) {
  const metric = String(sort?.metric || "");
  const direction = sort?.direction === "ascending" ? "ascending" : "descending";
  if (!metric) return [...items];
  return items
    .map((item, index) => ({ item, index }))
    .sort((left, right) => {
      const leftRaw = left.item?.metrics?.[metric];
      const rightRaw = right.item?.metrics?.[metric];
      const leftValue = Number(leftRaw);
      const rightValue = Number(rightRaw);
      const leftMissing = leftRaw === null
        || leftRaw === undefined
        || !Number.isFinite(leftValue);
      const rightMissing = rightRaw === null
        || rightRaw === undefined
        || !Number.isFinite(rightValue);
      if (leftMissing !== rightMissing) return leftMissing ? 1 : -1;
      if (leftMissing) return left.index - right.index;
      const difference = direction === "ascending"
        ? leftValue - rightValue
        : rightValue - leftValue;
      return difference || left.index - right.index;
    })
    .map(({ item }) => item);
}

function hasCompleteRunMetrics(item, columns) {
  return columns.length > 0 && columns.every((column) => {
    const raw = item?.metrics?.[column.metric];
    return raw !== null && raw !== undefined && Number.isFinite(Number(raw));
  });
}

export function activeRunMetricColumns(items, primaryColumns, fallbackColumns = []) {
  const primary = Array.isArray(primaryColumns) ? primaryColumns : [];
  const fallback = Array.isArray(fallbackColumns) ? fallbackColumns : [];
  if (primary.some(Boolean) && items.some((item) => hasCompleteRunMetrics(item, primary))) {
    return primary;
  }
  return fallback.length ? fallback : primary;
}

export function rankRunItems(items, columns) {
  const criteria = Array.isArray(columns) ? columns : [];
  if (!criteria.length) return [...items];
  return items
    .map((item, index) => ({ item, index }))
    .sort((left, right) => {
      const leftComplete = hasCompleteRunMetrics(left.item, criteria);
      const rightComplete = hasCompleteRunMetrics(right.item, criteria);
      if (leftComplete !== rightComplete) return leftComplete ? -1 : 1;
      if (!leftComplete) return left.index - right.index;
      for (const criterion of criteria) {
        const leftValue = Number(left.item.metrics[criterion.metric]);
        const rightValue = Number(right.item.metrics[criterion.metric]);
        const difference = criterion.direction === "min"
          ? leftValue - rightValue
          : rightValue - leftValue;
        if (difference) return difference;
      }
      return left.index - right.index;
    })
    .map(({ item }) => item);
}

export function bestRunEfficiency(items, primaryColumns, fallbackColumns = []) {
  const columns = activeRunMetricColumns(items, primaryColumns, fallbackColumns);
  const ranked = rankRunItems(items, columns);
  if (!ranked.length || !hasCompleteRunMetrics(ranked[0], columns)) return null;
  const usesPrimary = columns === primaryColumns;
  const evaluated = usesPrimary && columns.some(
    (column) => /^(eval\/|leader\/)/.test(String(column.metric || "")),
  );
  return {
    item: ranked[0],
    columns,
    evidence: evaluated ? "evaluation" : "training",
  };
}

export function checkpointPlaybackSeed(item) {
  const value = item?.playback_seed;
  if (value === null || value === undefined || value === "") return null;
  const seed = Number(value);
  return Number.isSafeInteger(seed) && seed >= 0 ? seed : null;
}

function checkpointEvaluationCell(item) {
  const evaluation = item?.evaluation;
  if (!evaluation || typeof evaluation !== "object") return ["—"];
  const details = [];
  const completed = Number(evaluation.episodes_completed);
  const planned = Number(evaluation.episodes_planned);
  if (Number.isFinite(completed) && Number.isFinite(planned)) {
    details.push(`${completed.toLocaleString()} / ${planned.toLocaleString()} episodes`);
  }
  (Array.isArray(evaluation.criteria) ? evaluation.criteria : []).forEach((criterion) => {
    const label = metricLabel(criterion.metric);
    const threshold = formatMetricValue(criterion.metric, criterion.threshold);
    if (
      criterion.value !== null
      && criterion.value !== undefined
      && criterion.value !== ""
      && Number.isFinite(Number(criterion.value))
    ) {
      const value = formatMetricValue(criterion.metric, criterion.value);
      details.push(`${label}: ${value} ${criterion.operator} ${threshold}`);
    } else {
      details.push(`${label} ${criterion.operator} ${threshold}`);
    }
  });
  const failureCount = Number(evaluation.failure_count);
  if (Number.isFinite(failureCount) && failureCount > 0) {
    details.push(`${failureCount.toLocaleString()} failed`);
  }
  return [
    evaluation.pass ? "Passed" : "Failed",
    details.join(" · "),
    `evaluation-cell ${evaluation.pass ? "accepted" : "rejected"}`,
  ];
}

function decodePathPart(value) {
  try {
    return decodeURIComponent(value);
  } catch {
    return "";
  }
}

export function sourceRouteFromPath(pathname = location.pathname) {
  const parts = String(pathname || "").split("/").filter(Boolean);
  if (!parts.length) {
    return {
      level: "environments",
      entity: "",
      project: "",
      goal_id: "",
      goal_variant_id: "",
      run_id: "",
      checkpoint_id: "",
    };
  }
  if (!["environments", "projects"].includes(parts[0]) || !parts[1]) return null;
  const legacy = parts[0] === "projects";
  const project = decodePathPart(parts[1]);
  if (!project) return null;
  if (parts.length === 2) {
    return {
      level: "goals",
      entity: "",
      project,
      goal_id: "",
      goal_variant_id: "",
      run_id: "",
      checkpoint_id: "",
    };
  }
  if (parts[2] !== "goals" || !parts[3]) return null;
  const goal_id = decodePathPart(parts[3]);
  if (!goal_id) return null;
  if (parts.length === 4) {
    return {
      level: "goal_variants",
      entity: "",
      project,
      goal_id,
      goal_variant_id: "",
      run_id: "",
      checkpoint_id: "",
    };
  }
  if (legacy) return null;
  if (parts[4] !== "variants" || !parts[5]) return null;
  const goal_variant_id = decodePathPart(parts[5]);
  if (!goal_variant_id) return null;
  if (parts.length === 6) {
    return {
      level: "runs",
      entity: "",
      project,
      goal_id,
      goal_variant_id,
      run_id: "",
      checkpoint_id: "",
    };
  }
  if (parts[6] !== "runs" || !parts[7]) return null;
  const run_id = decodePathPart(parts[7]);
  if (!run_id) return null;
  if (parts.length === 8) {
    return {
      level: "runs",
      entity: "",
      project,
      goal_id,
      goal_variant_id,
      run_id,
      checkpoint_id: "",
    };
  }
  if (parts.length !== 10 || parts[8] !== "checkpoints" || !parts[9]) return null;
  const checkpoint_id = decodePathPart(parts[9]);
  if (!checkpoint_id) return null;
  return {
    level: "runs",
    entity: "",
    project,
    goal_id,
    goal_variant_id,
    run_id,
    checkpoint_id,
  };
}

export function sourceRoutePath(route) {
  const project = String(route?.project || "").trim();
  const goalId = String(route?.goal_id || "").trim();
  const goalVariantId = String(route?.goal_variant_id || "").trim();
  const runId = String(route?.run_id || "").trim();
  const checkpointId = String(route?.checkpoint_id || "").trim();
  if (!project) return "/";
  let path = `/environments/${encodeURIComponent(project)}`;
  if (!goalId) return path;
  path += `/goals/${encodeURIComponent(goalId)}`;
  if (!goalVariantId) return path;
  path += `/variants/${encodeURIComponent(goalVariantId)}`;
  if (!runId) return path;
  path += `/runs/${encodeURIComponent(runId)}`;
  if (!checkpointId) return path;
  return `${path}/checkpoints/${encodeURIComponent(checkpointId)}`;
}

function routeSignature(route) {
  return JSON.stringify({
    level: route?.level || "environments",
    entity: route?.entity || "",
    project: route?.project || "",
    goal_id: route?.goal_id || "",
    goal_variant_id: route?.goal_variant_id || "",
    run_id: route?.run_id || "",
    checkpoint_id: route?.checkpoint_id || "",
  });
}

export class SourceBrowser {
  constructor(
    root,
    breadcrumbsRoot,
    {
      token,
      command,
      getState,
      showToast,
      catalogRequestTimeoutMs = 30_000,
    },
  ) {
    this.root = root;
    this.breadcrumbsRoot = breadcrumbsRoot;
    this.token = token;
    this.command = command;
    this.getState = getState;
    this.showToast = showToast;
    this.route = {
      level: "environments",
      entity: "",
      project: "",
      goal_id: "",
      goal_variant_id: "",
      run_id: "",
      checkpoint_id: "",
    };
    this.query = "";
    this.items = [];
    this.metricColumns = [];
    this.fallbackMetricColumns = [];
    this.sort = { metric: "", direction: "" };
    this.nextCursor = null;
    this.loading = false;
    this.error = "";
    this.app = { phase: "selecting" };
    this.lastAppRoute = "";
    this.loadedKey = "";
    this.requestSerial = 0;
    this.requestController = null;
    this.loadingKey = "";
    this.catalogRequestTimeoutMs = catalogRequestTimeoutMs;
    this.searchTimer = null;
    this.pollTimer = null;
    this.autoSelectedRoute = "";
    this.initialProjectCatalog = null;
    this.historyEnabled = (
      location.pathname === "/"
      || location.pathname.startsWith("/environments/")
      || location.pathname.startsWith("/projects/")
    );
    this.pendingLocationRoute = this.historyEnabled
      ? sourceRouteFromPath(location.pathname)
      : null;
    this.onPopState = () => {
      const route = sourceRouteFromPath(location.pathname);
      if (!route) return;
      this.applyRoute(route);
      this.command("browse_sources", { route: { ...this.route } });
    };
    if (this.historyEnabled) window.addEventListener("popstate", this.onPopState);
  }

  render(snapshot) {
    this.app = snapshot?.app || { phase: "active" };
    if (this.app.catalog && typeof this.app.catalog === "object") {
      this.initialProjectCatalog = this.app.catalog;
    }
    const appRoute = this.app.route || {};
    if (
      this.pendingLocationRoute
      && location.pathname !== "/"
      && this.app.phase === "selecting"
      && !this.app.source
    ) {
      const pending = {
        ...this.pendingLocationRoute,
        entity: (
          this.pendingLocationRoute.entity
          || appRoute.entity
          || this.initialProjectCatalog?.entity
          || ""
        ),
      };
      this.pendingLocationRoute = null;
      this.lastAppRoute = routeSignature(appRoute);
      this.applyRoute(pending);
      this.command("browse_sources", { route: { ...pending } });
    } else {
      this.pendingLocationRoute = null;
    }
    const signature = routeSignature(appRoute);
    if (signature !== this.lastAppRoute) {
      this.lastAppRoute = signature;
      this.route = {
        level: appRoute.level || "environments",
        entity: appRoute.entity || "",
        project: appRoute.project || "",
        goal_id: appRoute.goal_id || "",
        goal_variant_id: appRoute.goal_variant_id || "",
        run_id: appRoute.run_id || "",
        checkpoint_id: appRoute.checkpoint_id || "",
      };
      this.query = "";
      this.items = [];
      this.metricColumns = [];
      this.fallbackMetricColumns = [];
      this.sort = { metric: "", direction: "" };
      this.nextCursor = null;
      this.loadedKey = "";
      this.error = "";
      this.autoSelectedRoute = "";
      this.syncUrl("replace");
    }
    this.hydrateInitialProjects();
    this.renderView();
    if (this.app.phase === "selecting") this.ensureLoaded();
    this.updatePolling();
  }

  stop() {
    clearTimeout(this.searchTimer);
    clearInterval(this.pollTimer);
    this.pollTimer = null;
    this.requestController?.abort();
    this.requestController = null;
    this.requestSerial += 1;
    this.loading = false;
    this.loadingKey = "";
    this.breadcrumbsRoot.replaceChildren();
    this.breadcrumbsRoot.hidden = true;
  }

  hasControl() {
    return Boolean(this.getState()?.hasControl);
  }

  routeKey() {
    return `${routeSignature(this.route)}:${this.query.trim().toLocaleLowerCase()}`;
  }

  hydrateInitialProjects() {
    const catalog = this.initialProjectCatalog;
    if (
      !catalog
      || this.route.level !== "environments"
      || this.query.trim()
      || this.loadedKey
    ) {
      return false;
    }
    if (catalog.entity && !this.route.entity) this.route.entity = catalog.entity;
    this.items = Array.isArray(catalog.items) ? [...catalog.items] : [];
    this.metricColumns = Array.isArray(catalog.metric_columns)
      ? [...catalog.metric_columns]
      : [];
    this.fallbackMetricColumns = Array.isArray(catalog.fallback_metric_columns)
      ? [...catalog.fallback_metric_columns]
      : [];
    this.nextCursor = catalog.next_cursor || null;
    this.loadedKey = this.routeKey();
    this.error = "";
    return true;
  }

  endpoint(cursor = null) {
    const query = new URLSearchParams();
    if (this.query.trim()) query.set("q", this.query.trim());
    if (cursor) query.set("cursor", cursor);
    if (this.route.level === "goals") {
      return `/api/catalog/environments/${encodeURIComponent(this.route.entity)}/${encodeURIComponent(this.route.project)}/goals?${query}`;
    }
    if (this.route.level === "goal_variants") {
      return `/api/catalog/environments/${encodeURIComponent(this.route.entity)}/${encodeURIComponent(this.route.project)}/goals/${encodeURIComponent(this.route.goal_id)}/variants?${query}`;
    }
    if (this.route.level === "runs" && this.route.run_id) {
      if (this.route.entity) query.set("entity", this.route.entity);
      if (this.route.project) query.set("project", this.route.project);
      if (this.route.goal_variant_id) {
        query.set("goal_variant_id", this.route.goal_variant_id);
      }
      return `/api/catalog/runs/${encodeURIComponent(this.route.run_id)}/checkpoints?${query}`;
    }
    if (this.route.level === "runs") {
      return `/api/catalog/environments/${encodeURIComponent(this.route.entity)}/${encodeURIComponent(this.route.project)}/goals/${encodeURIComponent(this.route.goal_id)}/variants/${encodeURIComponent(this.route.goal_variant_id)}/runs?${query}`;
    }
    if (this.route.entity) query.set("entity", this.route.entity);
    return `/api/catalog/environments?${query}`;
  }

  async ensureLoaded() {
    const key = this.routeKey();
    if ((this.loading && this.loadingKey === key) || this.loadedKey === key) return;
    await this.load();
  }

  async load({ append = false, quiet = false } = {}) {
    const key = this.routeKey();
    if (this.loading && this.loadingKey === key) return;
    this.requestController?.abort();
    const controller = new AbortController();
    this.requestController = controller;
    const cursor = append ? this.nextCursor : null;
    const serial = ++this.requestSerial;
    this.loading = true;
    this.loadingKey = key;
    let timedOut = false;
    const timeout = setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, this.catalogRequestTimeoutMs);
    if (!quiet) this.error = "";
    this.renderView();
    try {
      const response = await fetch(this.endpoint(cursor), {
        headers: { Authorization: `Bearer ${this.token}` },
        cache: "no-store",
        signal: controller.signal,
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.error || `Catalog request failed (${response.status})`);
      if (serial !== this.requestSerial || key !== this.routeKey()) return;
      if (this.route.level === "environments" && payload.entity && !this.route.entity) {
        this.route.entity = payload.entity;
      }
      const received = Array.isArray(payload.items) ? payload.items : [];
      this.items = append ? [...this.items, ...received] : received;
      if (!append) {
        this.metricColumns = Array.isArray(payload.metric_columns)
          ? payload.metric_columns
          : [];
        this.fallbackMetricColumns = Array.isArray(payload.fallback_metric_columns)
          ? payload.fallback_metric_columns
          : [];
        if (
          this.sort.metric
          && ![...this.metricColumns, ...this.fallbackMetricColumns]
            .some((column) => column.metric === this.sort.metric)
        ) {
          this.sort = { metric: "", direction: "" };
        }
      }
      this.nextCursor = payload.next_cursor || null;
      this.loadedKey = this.routeKey();
      this.error = "";
      if (this.route.checkpoint_id && !append) {
        const selected = received.find(
          (item) => item.checkpoint_id === this.route.checkpoint_id,
        );
        if (!selected) {
          throw new Error(`Checkpoint ${this.route.checkpoint_id} was not found`);
        }
        if (this.autoSelectedRoute !== key) {
          this.autoSelectedRoute = key;
          this.selectCheckpoint(selected, { historyMode: "replace" });
        }
      }
    } catch (error) {
      if (serial !== this.requestSerial || key !== this.routeKey()) return;
      this.error = timedOut
        ? "Catalog request timed out. Try Refresh."
        : String(error?.message || error);
      if (quiet) this.showToast(this.error, true);
    } finally {
      clearTimeout(timeout);
      if (serial === this.requestSerial) {
        this.requestController = null;
        this.loading = false;
        this.loadingKey = "";
        this.renderView();
        if (
          key !== this.routeKey()
          && this.loadedKey !== this.routeKey()
          && this.app.phase === "selecting"
        ) {
          this.ensureLoaded();
        }
      }
    }
  }

  updatePolling() {
    const shouldPoll = (
      this.app.phase === "selecting"
      && this.route.level === "runs"
      && Boolean(this.route.run_id)
    );
    if (shouldPoll && this.pollTimer === null) {
      this.pollTimer = window.setInterval(() => {
        this.loadedKey = "";
        this.load({ quiet: true });
      }, 5000);
    } else if (!shouldPoll && this.pollTimer !== null) {
      clearInterval(this.pollTimer);
      this.pollTimer = null;
    }
  }

  applyRoute(route) {
    this.route = { ...this.route, ...route };
    this.query = "";
    this.items = [];
    this.metricColumns = [];
    this.fallbackMetricColumns = [];
    this.sort = { metric: "", direction: "" };
    this.nextCursor = null;
    this.loadedKey = "";
    this.error = "";
    this.autoSelectedRoute = "";
    this.hydrateInitialProjects();
    this.renderView();
    this.ensureLoaded();
    this.updatePolling();
  }

  syncUrl(mode = "push") {
    if (!this.historyEnabled) return;
    const path = sourceRoutePath(this.route);
    const target = `${path}${location.search}${location.hash}`;
    const current = `${location.pathname}${location.search}${location.hash}`;
    if (target === current) return;
    if (mode === "replace") history.replaceState(null, "", target);
    else history.pushState(null, "", target);
  }

  navigate(route, { historyMode = "push" } = {}) {
    this.applyRoute(route);
    this.syncUrl(historyMode);
    this.command("browse_sources", { route: { ...this.route } });
  }

  browseCurrentSource() {
    const route = this.app.route || this.route;
    const next = route.run_id
      ? { ...route, level: "runs", checkpoint_id: "" }
      : {
          level: "environments",
          entity: route.entity || "",
          project: "",
          goal_id: "",
          goal_variant_id: "",
          run_id: "",
          checkpoint_id: "",
        };
    this.navigate(next);
  }

  goHome() {
    this.navigate({
      level: "environments",
      entity: this.route.entity || this.app.route?.entity || "",
      project: "",
      goal_id: "",
      goal_variant_id: "",
      run_id: "",
      checkpoint_id: "",
    });
  }

  selectCheckpoint(item, { historyMode = "push" } = {}) {
    const route = {
      ...this.route,
      level: "runs",
      checkpoint_id: item.checkpoint_id,
    };
    this.route = route;
    this.syncUrl(historyMode);
    this.command("select_source", {
      source: {
        kind: "manifest",
        value: item.manifest_url,
        entity: route.entity,
        project: route.project,
        run_id: item.run_id,
        checkpoint_id: item.checkpoint_id,
        seed: checkpointPlaybackSeed(item),
      },
      route: { ...route },
    });
  }

  back() {
    if (this.route.level === "runs" && this.route.run_id) {
      this.navigate({ level: "runs", run_id: "", checkpoint_id: "" });
    } else if (this.route.level === "runs") {
      this.navigate({
        level: "goal_variants",
        goal_variant_id: "",
        run_id: "",
        checkpoint_id: "",
      });
    } else if (this.route.level === "goal_variants") {
      this.navigate({
        level: "goals",
        goal_id: "",
        goal_variant_id: "",
        run_id: "",
        checkpoint_id: "",
      });
    } else if (this.route.level === "goals") {
      this.navigate({
        level: "environments",
        project: "",
        goal_id: "",
        goal_variant_id: "",
        run_id: "",
        checkpoint_id: "",
      });
    }
  }

  setSearch(value) {
    this.query = value;
    clearTimeout(this.searchTimer);
    this.searchTimer = window.setTimeout(() => {
      this.items = [];
      this.nextCursor = null;
      this.loadedKey = "";
      this.load();
    }, 220);
  }

  renderView() {
    const activeElement = document.activeElement;
    const restoreSearchFocus = (
      activeElement instanceof HTMLInputElement
      && activeElement.type === "search"
      && this.root.contains(activeElement)
    );
    const selectionStart = restoreSearchFocus ? activeElement.selectionStart : null;
    const selectionEnd = restoreSearchFocus ? activeElement.selectionEnd : null;
    const shell = document.createElement("section");
    shell.className = "source-shell";
    const head = document.createElement("div");
    head.className = "source-head";
    const titleBlock = document.createElement("div");
    titleBlock.className = "source-title";
    const eyebrow = document.createElement("span");
    eyebrow.className = "eyebrow";
    eyebrow.textContent = "PLAYBACK SOURCE";
    const heading = document.createElement("h2");
    heading.textContent = this.heading();
    titleBlock.append(eyebrow, heading);
    head.append(titleBlock);

    if (this.app.phase === "selecting") {
      const refresh = button("Refresh", { iconName: "refresh", quiet: true });
      refresh.title = "Refresh this list";
      refresh.disabled = this.loading;
      refresh.addEventListener("click", () => {
        this.loadedKey = "";
        this.load();
      });
      head.append(refresh);
    }
    shell.append(head);

    if (!this.hasControl()) {
      const observer = document.createElement("p");
      observer.className = "source-notice";
      observer.textContent = "Observer window — choose Control here to change the shared run.";
      shell.append(observer);
    }

    const showsCatalog = ![
      "approval_required",
      "error",
      "resolving",
      "verifying",
      "loading",
    ].includes(this.app.phase);
    if (showsCatalog) {
      this.renderBreadcrumbs(this.breadcrumbsRoot);
      this.breadcrumbsRoot.hidden = false;
    } else {
      this.breadcrumbsRoot.replaceChildren();
      this.breadcrumbsRoot.hidden = true;
    }

    if (this.app.phase === "approval_required") {
      shell.append(this.renderApproval());
    } else if (this.app.phase === "error") {
      shell.append(this.renderFailure());
    } else if (["resolving", "verifying", "loading"].includes(this.app.phase)) {
      shell.append(this.renderProgress());
    } else {
      shell.append(this.renderSearch(), this.renderResults());
    }
    this.root.replaceChildren(shell);
    if (restoreSearchFocus) {
      const search = this.root.querySelector('input[type="search"]');
      search?.focus({ preventScroll: true });
      if (search && selectionStart !== null && selectionEnd !== null) {
        search.setSelectionRange(selectionStart, selectionEnd);
      }
    }
  }

  heading() {
    if (this.app.phase === "approval_required") return "Approve executable model";
    if (this.app.phase === "error") return "Could not open checkpoint";
    if (["resolving", "verifying", "loading"].includes(this.app.phase)) return "Opening checkpoint";
    if (this.route.level === "runs" && this.route.run_id) {
      return "Runs · choose a checkpoint";
    }
    if (this.route.level === "runs") return "Choose a run";
    if (this.route.level === "goal_variants") return "Choose a goal variant";
    if (this.route.level === "goals") return "Choose a goal";
    return "Choose an environment";
  }

  renderBreadcrumbs(nav) {
    nav.replaceChildren();
    const environments = button("Environments", { quiet: true });
    environments.disabled = this.route.level === "environments";
    environments.addEventListener("click", () => this.navigate({
      level: "environments",
      project: "",
      goal_id: "",
      goal_variant_id: "",
      run_id: "",
      checkpoint_id: "",
    }));
    nav.append(environments);
    if (this.route.project) {
      const project = button(this.route.project, { quiet: true });
      project.disabled = this.route.level === "goals";
      project.addEventListener("click", () => this.navigate({
        level: "goals",
        goal_id: "",
        goal_variant_id: "",
        run_id: "",
        checkpoint_id: "",
      }));
      nav.append(project);
    }
    if (this.route.goal_id) {
      const goal = button(this.route.goal_id, { quiet: true });
      goal.disabled = this.route.level === "goal_variants";
      goal.addEventListener("click", () => this.navigate({
        level: "goal_variants",
        goal_variant_id: "",
        run_id: "",
        checkpoint_id: "",
      }));
      nav.append(goal);
    }
    if (this.route.goal_variant_id) {
      const variant = button(this.route.goal_variant_id, { quiet: true });
      variant.disabled = this.route.level === "runs" && !this.route.run_id;
      variant.addEventListener("click", () => this.navigate({
        level: "runs",
        run_id: "",
        checkpoint_id: "",
      }));
      nav.append(variant);
    }
    if (this.route.run_id) {
      const run = button(this.route.run_id, { quiet: true });
      run.disabled = true;
      nav.append(run);
    }
    return nav;
  }

  renderSearch() {
    const wrap = document.createElement("label");
    wrap.className = "source-search";
    wrap.append(icon("search"));
    const input = document.createElement("input");
    input.type = "search";
    input.value = this.query;
    input.autocomplete = "off";
    input.placeholder = this.route.level === "environments"
      ? "Search environments"
      : this.route.level === "goals"
        ? "Search goals"
        : this.route.level === "goal_variants"
          ? "Search goal variant, diff, status, or contract hash"
          : this.route.level === "runs" && !this.route.run_id
          ? "Search run, description, recipe, variant, override, or seed"
          : "Search checkpoint, step, hash, purpose, or evaluation";
    input.setAttribute("aria-label", input.placeholder);
    input.addEventListener("input", (event) => this.setSearch(event.target.value));
    wrap.append(input);
    return wrap;
  }

  renderResults() {
    const body = document.createElement("div");
    body.className = "source-results";
    if (this.error) {
      const error = document.createElement("div");
      error.className = "source-inline-error";
      const message = document.createElement("p");
      message.textContent = this.error;
      const retry = button("Retry", { iconName: "refresh" });
      retry.addEventListener("click", () => {
        this.loadedKey = "";
        this.load();
      });
      error.append(message, retry);
      body.append(error);
      return body;
    }
    if (this.loading && !this.items.length) {
      body.append(this.loadingState("Loading catalog…"));
      return body;
    }
    if (!this.items.length) {
      const empty = document.createElement("div");
      empty.className = "source-empty";
      const heading = document.createElement("strong");
      heading.textContent = this.route.level === "runs" && this.route.run_id
        ? "No public checkpoints yet"
        : "No matching results";
      const detail = document.createElement("p");
      detail.textContent = this.route.level === "runs" && this.route.run_id
        ? "This run has not published a playable checkpoint to public model storage."
        : "Try a broader search.";
      empty.append(heading, detail);
      body.append(empty);
      return body;
    }
    body.append(
      this.route.level === "environments"
        ? this.renderProjects()
        : this.route.level === "goals"
          ? this.renderGoals()
          : this.route.level === "goal_variants"
            ? this.renderGoalVariants()
          : this.route.level === "runs"
            ? this.renderRunResults()
            : this.renderTable(),
    );
    if (this.nextCursor) {
      const more = button(this.loading ? "Loading…" : "Load more");
      more.classList.add("source-load-more");
      more.disabled = this.loading;
      more.addEventListener("click", () => this.load({ append: true }));
      body.append(more);
    }
    return body;
  }

  renderProjects() {
    const list = document.createElement("div");
    list.className = "project-list";
    this.items.forEach((project) => {
      const row = document.createElement("button");
      row.type = "button";
      row.className = "project-row";
      row.disabled = !this.hasControl();
      const name = document.createElement("strong");
      name.textContent = project.name;
      const meta = document.createElement("span");
      const goalLabel = Number(project.goal_count) === 1 ? "goal" : "goals";
      meta.textContent = `${project.entity} · ${Number(project.goal_count).toLocaleString()} ${goalLabel}`;
      row.append(name, meta);
      row.addEventListener("click", () => this.navigate({
        level: "goals",
        entity: project.entity || this.route.entity,
        project: project.name,
        goal_id: "",
        goal_variant_id: "",
        run_id: "",
        checkpoint_id: "",
      }));
      list.append(row);
    });
    return list;
  }

  renderGoals() {
    const list = document.createElement("div");
    list.className = "goal-list";
    this.items.forEach((goal) => {
      const row = document.createElement("button");
      row.type = "button";
      row.className = "goal-row";
      row.disabled = !this.hasControl();
      const identity = document.createElement("div");
      const name = document.createElement("strong");
      name.textContent = goal.goal_id;
      identity.append(name);
      const meta = document.createElement("span");
      const recipeLabel = Number(goal.recipe_count) === 1 ? "recipe" : "recipes";
      meta.textContent = `${goal.title || goal.goal_slug} · ${Number(goal.recipe_count).toLocaleString()} ${recipeLabel}`;
      identity.append(meta);
      if (goal.goal_slug && goal.goal_slug !== goal.goal_id) {
        const slug = document.createElement("small");
        slug.textContent = goal.goal_slug;
        identity.append(slug);
      }
      row.append(identity);
      row.addEventListener("click", () => this.navigate({
        level: "goal_variants",
        goal_id: goal.goal_id,
        goal_variant_id: "",
        run_id: "",
        checkpoint_id: "",
      }));
      list.append(row);
    });
    return list;
  }

  renderGoalVariants() {
    const list = document.createElement("div");
    list.className = "goal-list goal-variant-list";
    this.items.forEach((variant) => {
      const row = document.createElement("button");
      row.type = "button";
      row.className = "goal-row goal-variant-row";
      row.disabled = !this.hasControl();
      const identity = document.createElement("div");
      const name = document.createElement("strong");
      name.textContent = variant.label || variant.variant_id;
      identity.append(name);
      const meta = document.createElement("span");
      meta.textContent = [
        variant.status || "historical",
        variant.source_relation || "",
        String(variant.effective_goal_contract_sha256 || "").slice(0, 12),
      ].filter(Boolean).join(" · ");
      identity.append(meta);
      const diff = Array.isArray(variant.diff) ? variant.diff : [];
      if (diff.length) {
        const detail = document.createElement("small");
        detail.textContent = diff.slice(0, 3).map((entry) => {
          const before = entry.before === null || entry.before === undefined
            ? "unset"
            : String(entry.before);
          const after = entry.after === null || entry.after === undefined
            ? "unset"
            : String(entry.after);
          return `${entry.path}: ${before} → ${after}`;
        }).join(" · ");
        identity.append(detail);
      }
      row.append(identity);
      row.addEventListener("click", () => this.navigate({
        level: "runs",
        goal_variant_id: variant.variant_id,
        run_id: "",
        checkpoint_id: "",
      }));
      list.append(row);
    });
    return list;
  }

  activeRunMetricColumns() {
    return activeRunMetricColumns(
      this.items,
      this.metricColumns,
      this.fallbackMetricColumns,
    );
  }

  runEfficiency() {
    return bestRunEfficiency(
      this.items,
      this.metricColumns,
      this.fallbackMetricColumns,
    );
  }

  renderRunResults() {
    if (this.route.run_id) return this.renderTable();
    const efficiency = this.runEfficiency();
    return this.renderTable(efficiency);
  }

  renderTable(efficiency = null) {
    const scroll = document.createElement("div");
    scroll.className = "source-table-scroll";
    if (efficiency?.evidence === "training") {
      scroll.classList.add("training-leader");
    }
    const table = document.createElement("table");
    table.className = "source-table";
    const head = document.createElement("thead");
    const headerRow = document.createElement("tr");
    const showingRuns = this.route.level === "runs" && !this.route.run_id;
    const runMetricColumns = showingRuns ? this.activeRunMetricColumns() : [];
    const columns = showingRuns
      ? [
          { label: "Run" },
          { label: "Recipe / variant" },
          { label: "Seed" },
          ...runMetricColumns.map((column) => ({
            ...column,
            label: metricLabel(column.metric),
          })),
          { label: "Updated" },
        ]
      : [
          { label: "Checkpoint" },
          { label: "Purpose" },
          { label: "Step" },
          { label: "Evaluation" },
          { label: "Size" },
          { label: "Created" },
        ];
    columns.forEach((column) => {
      const cell = document.createElement("th");
      cell.scope = "col";
      if (column.metric) {
        const active = this.sort.metric === column.metric;
        const defaultDirection = column.direction === "min" ? "ascending" : "descending";
        const nextDirection = active && this.sort.direction === "ascending"
          ? "descending"
          : active && this.sort.direction === "descending"
            ? "ascending"
            : defaultDirection;
        cell.setAttribute("aria-sort", active ? this.sort.direction : "none");
        const sortButton = document.createElement("button");
        sortButton.type = "button";
        sortButton.className = "source-sort";
        sortButton.title = `${column.label} · ${
          column.direction === "min" ? "lower is better" : "higher is better"
        }`;
        sortButton.setAttribute(
          "aria-label",
          `Sort by ${column.label}, ${nextDirection}`,
        );
        const label = document.createElement("span");
        label.textContent = column.label;
        const indicator = document.createElement("span");
        indicator.className = "source-sort-indicator";
        indicator.setAttribute("aria-hidden", "true");
        indicator.textContent = active
          ? this.sort.direction === "ascending" ? "↑" : "↓"
          : "↕";
        sortButton.append(label, indicator);
        sortButton.addEventListener("click", () => {
          this.sort = { metric: column.metric, direction: nextDirection };
          this.renderView();
        });
        cell.append(sortButton);
      } else {
        cell.textContent = column.label;
      }
      headerRow.append(cell);
    });
    head.append(headerRow);
    const body = document.createElement("tbody");
    const items = showingRuns
      ? this.sort.metric
        ? sortRunItems(this.items, this.sort)
        : rankRunItems(this.items, runMetricColumns)
      : this.items;
    items.forEach((item) => {
      const row = document.createElement("tr");
      const isEfficiencyLeader = (
        showingRuns
        && efficiency?.item?.run_id === item.run_id
      );
      if (isEfficiencyLeader) row.classList.add("efficiency-leader");
      row.tabIndex = this.hasControl() ? 0 : -1;
      row.setAttribute("role", "button");
      row.setAttribute("aria-disabled", String(!this.hasControl()));
      const values = showingRuns
        ? [
            [item.description || item.name || item.run_id, item.run_id, "run-cell"],
            [
              item.recipe || "—",
              "",
              "recipe-cell",
            ],
            [item.seed ?? "—"],
            ...runMetricColumns.map((column) => [
              formatMetricValue(column.metric, item.metrics?.[column.metric]),
            ]),
            [formatDate(item.updated_at || item.created_at)],
          ]
        : [
            [item.promoted ? `${item.checkpoint_id} · promoted` : item.checkpoint_id, item.sha256],
            [item.purpose],
            [Number(item.step).toLocaleString()],
            checkpointEvaluationCell(item),
            [formatBytes(item.size_bytes)],
            [formatDate(item.created_at)],
          ];
      values.forEach(([primary, secondary = "", className = ""]) => {
        const cell = document.createElement("td");
        if (className) cell.className = className;
        const main = document.createElement("span");
        if (className.includes("evaluation-cell")) main.className = "evaluation-verdict";
        main.textContent = String(primary);
        if (className.includes("run-cell")) {
          const presentation = runStatePresentation(item.state);
          const identity = document.createElement("div");
          identity.className = "run-identity";
          const state = document.createElement("span");
          state.className = `run-state ${presentation.tone}`;
          state.title = `Run state: ${presentation.label}`;
          state.setAttribute("aria-label", `Run state: ${presentation.label}`);
          state.append(icon(presentation.iconName));
          const text = document.createElement("div");
          text.className = "run-identity-text";
          main.className = "run-name";
          text.append(main);
          if (secondary) {
            const small = document.createElement("small");
            small.textContent = String(secondary);
            text.append(small);
          }
          identity.append(state, text);
          cell.append(identity);
        } else {
          cell.append(main);
        }
        if (className.includes("recipe-cell") && isEfficiencyLeader) {
          const badge = document.createElement("span");
          badge.className = "source-leader-badge";
          badge.textContent = efficiency.evidence === "evaluation"
            ? "Most efficient"
            : "Training lead";
          cell.append(badge);
        }
        if (className.includes("recipe-cell")) {
          const variant = recipeVariantPresentation(item);
          const variation = document.createElement("small");
          variation.className = "recipe-variant";
          variation.textContent = variant.summary;
          variation.title = variant.detail;
          cell.append(variation);
          if (item.recipe_sha256) {
            const revision = document.createElement("small");
            revision.className = "recipe-revision";
            revision.textContent = `rev ${String(item.recipe_sha256).slice(0, 12)}`;
            revision.title = `Recipe SHA-256: ${item.recipe_sha256}`;
            cell.append(revision);
          }
        }
        if (secondary && !className.includes("run-cell")) {
          const small = document.createElement("small");
          small.textContent = String(secondary);
          cell.append(small);
        }
        row.append(cell);
      });
      const activate = () => {
        if (!this.hasControl()) return;
        if (showingRuns) {
          this.navigate({
            level: "runs",
            run_id: item.run_id,
            checkpoint_id: "",
          });
        } else {
          this.selectCheckpoint(item);
        }
      };
      row.addEventListener("click", activate);
      row.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          activate();
        }
      });
      body.append(row);
    });
    table.append(head, body);
    scroll.append(table);
    return scroll;
  }

  loadingState(message) {
    const state = document.createElement("div");
    state.className = "source-loading";
    const spinner = document.createElement("span");
    spinner.className = "spinner";
    spinner.setAttribute("aria-hidden", "true");
    const text = document.createElement("p");
    text.textContent = message;
    state.append(spinner, text);
    return state;
  }

  renderProgress() {
    const wrap = document.createElement("div");
    wrap.className = "source-centered";
    wrap.append(this.loadingState(this.app.message || "Preparing playback…"));
    if (this.app.has_active_runner) {
      const cancel = button("Back to current run", { iconName: "arrow-left", quiet: true });
      cancel.disabled = !this.hasControl();
      cancel.addEventListener("click", () => this.command("cancel_source"));
      wrap.append(cancel);
    }
    return wrap;
  }

  renderFailure() {
    const wrap = document.createElement("div");
    wrap.className = "source-centered source-failure";
    const message = document.createElement("p");
    message.textContent = this.app.error || "The checkpoint could not be opened.";
    const actions = document.createElement("div");
    actions.className = "source-actions";
    const retry = button("Retry", { iconName: "refresh", primary: true });
    retry.disabled = !this.hasControl();
    retry.addEventListener("click", () => this.command("retry_source"));
    const choose = button("Choose another", { iconName: "folder-search", quiet: true });
    choose.disabled = !this.hasControl();
    choose.addEventListener("click", () => this.browseCurrentSource());
    actions.append(retry, choose);
    if (this.app.has_active_runner) {
      const cancel = button("Back to current run", { iconName: "arrow-left", quiet: true });
      cancel.disabled = !this.hasControl();
      cancel.addEventListener("click", () => this.command("cancel_source"));
      actions.append(cancel);
    }
    wrap.append(message, actions);
    return wrap;
  }

  renderApproval() {
    const approval = this.app.approval || {};
    const wrap = document.createElement("div");
    wrap.className = "approval-card";
    const warning = document.createElement("p");
    warning.className = "approval-warning";
    warning.textContent = approval.warning || "This model contains executable Python content.";
    const source = document.createElement("dl");
    source.className = "approval-summary";
    [["Source", approval.source], ["Manifest", approval.manifest_hash]].forEach(([label, value]) => {
      const term = document.createElement("dt");
      term.textContent = label;
      const detail = document.createElement("dd");
      detail.textContent = value || "—";
      source.append(term, detail);
    });
    const files = document.createElement("div");
    files.className = "approval-files";
    (approval.files || []).forEach((file) => {
      const row = document.createElement("div");
      const path = document.createElement("span");
      path.textContent = file.path;
      const digest = document.createElement("code");
      digest.textContent = file.sha256;
      row.append(path, digest);
      files.append(row);
    });
    const actions = document.createElement("div");
    actions.className = "source-actions";
    const approve = button("Approve exact closure", { iconName: "shield-check", primary: true });
    approve.disabled = !this.hasControl();
    approve.addEventListener("click", () => this.command("approve_source", {
      manifest_hash: approval.manifest_hash,
    }));
    const cancel = button(
      this.app.has_active_runner ? "Back to current run" : "Cancel",
      { iconName: "x", quiet: true },
    );
    cancel.disabled = !this.hasControl();
    cancel.addEventListener("click", () => this.command("cancel_source"));
    actions.append(approve, cancel);
    wrap.append(warning, source, files, actions);
    return wrap;
  }
}
