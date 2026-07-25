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

function evaluationMetricLabel(metric) {
  const name = String(metric || "");
  const known = {
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
  return name.replace(/^eval\/full\//, "").replaceAll("/", " · ");
}

function formatEvaluationValue(metric, value) {
  if (value === null || value === undefined || value === "") return "—";
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "—";
  if (String(metric).includes("/rate/") || String(metric).endsWith("/rate")) {
    return `${(numeric * 100).toLocaleString(undefined, { maximumFractionDigits: 2 })}%`;
  }
  return numeric.toLocaleString(undefined, { maximumFractionDigits: 3 });
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
    const label = evaluationMetricLabel(criterion.metric);
    const threshold = formatEvaluationValue(criterion.metric, criterion.threshold);
    if (
      criterion.value !== null
      && criterion.value !== undefined
      && criterion.value !== ""
      && Number.isFinite(Number(criterion.value))
    ) {
      const value = formatEvaluationValue(criterion.metric, criterion.value);
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
      level: "projects",
      entity: "",
      project: "",
      goal_id: "",
      run_id: "",
      checkpoint_id: "",
    };
  }
  if (parts[0] !== "projects" || !parts[1]) return null;
  const project = decodePathPart(parts[1]);
  if (!project) return null;
  if (parts.length === 2) {
    return {
      level: "goals",
      entity: "",
      project,
      goal_id: "",
      run_id: "",
      checkpoint_id: "",
    };
  }
  if (parts[2] !== "goals" || !parts[3]) return null;
  const goal_id = decodePathPart(parts[3]);
  if (!goal_id) return null;
  if (parts.length === 4) {
    return {
      level: "runs",
      entity: "",
      project,
      goal_id,
      run_id: "",
      checkpoint_id: "",
    };
  }
  if (parts[4] !== "runs" || !parts[5]) return null;
  const run_id = decodePathPart(parts[5]);
  if (!run_id) return null;
  if (parts.length === 6) {
    return {
      level: "checkpoints",
      entity: "",
      project,
      goal_id,
      run_id,
      checkpoint_id: "",
    };
  }
  if (parts.length !== 8 || parts[6] !== "checkpoints" || !parts[7]) return null;
  const checkpoint_id = decodePathPart(parts[7]);
  if (!checkpoint_id) return null;
  return {
    level: "checkpoints",
    entity: "",
    project,
    goal_id,
    run_id,
    checkpoint_id,
  };
}

export function sourceRoutePath(route) {
  const project = String(route?.project || "").trim();
  const goalId = String(route?.goal_id || "").trim();
  const runId = String(route?.run_id || "").trim();
  const checkpointId = String(route?.checkpoint_id || "").trim();
  if (!project) return "/";
  let path = `/projects/${encodeURIComponent(project)}`;
  if (!goalId) return path;
  path += `/goals/${encodeURIComponent(goalId)}`;
  if (!runId) return path;
  path += `/runs/${encodeURIComponent(runId)}`;
  if (!checkpointId) return path;
  return `${path}/checkpoints/${encodeURIComponent(checkpointId)}`;
}

function routeSignature(route) {
  return JSON.stringify({
    level: route?.level || "projects",
    entity: route?.entity || "",
    project: route?.project || "",
    goal_id: route?.goal_id || "",
    run_id: route?.run_id || "",
    checkpoint_id: route?.checkpoint_id || "",
  });
}

export class SourceBrowser {
  constructor(root, { token, command, getState, showToast }) {
    this.root = root;
    this.token = token;
    this.command = command;
    this.getState = getState;
    this.showToast = showToast;
    this.route = {
      level: "projects",
      entity: "",
      project: "",
      goal_id: "",
      run_id: "",
      checkpoint_id: "",
    };
    this.query = "";
    this.items = [];
    this.nextCursor = null;
    this.loading = false;
    this.error = "";
    this.app = { phase: "selecting" };
    this.lastAppRoute = "";
    this.loadedKey = "";
    this.requestSerial = 0;
    this.searchTimer = null;
    this.pollTimer = null;
    this.autoSelectedRoute = "";
    this.historyEnabled = (
      location.pathname === "/"
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
    const appRoute = this.app.route || {};
    if (this.pendingLocationRoute) {
      const pending = {
        ...this.pendingLocationRoute,
        entity: this.pendingLocationRoute.entity || appRoute.entity || "",
      };
      this.pendingLocationRoute = null;
      this.lastAppRoute = routeSignature(appRoute);
      this.applyRoute(pending);
      this.command("browse_sources", { route: { ...pending } });
    }
    const signature = routeSignature(appRoute);
    if (signature !== this.lastAppRoute) {
      this.lastAppRoute = signature;
      this.route = {
        level: appRoute.level || "projects",
        entity: appRoute.entity || "",
        project: appRoute.project || "",
        goal_id: appRoute.goal_id || "",
        run_id: appRoute.run_id || "",
        checkpoint_id: appRoute.checkpoint_id || "",
      };
      this.query = "";
      this.items = [];
      this.nextCursor = null;
      this.loadedKey = "";
      this.error = "";
      this.autoSelectedRoute = "";
      this.syncUrl("replace");
    }
    this.renderView();
    if (this.app.phase === "selecting") this.ensureLoaded();
    this.updatePolling();
  }

  stop() {
    clearTimeout(this.searchTimer);
    clearInterval(this.pollTimer);
    this.pollTimer = null;
    this.requestSerial += 1;
    this.loading = false;
  }

  hasControl() {
    return Boolean(this.getState()?.hasControl);
  }

  routeKey() {
    return `${routeSignature(this.route)}:${this.query.trim().toLocaleLowerCase()}`;
  }

  endpoint(cursor = null) {
    const query = new URLSearchParams();
    if (this.query.trim()) query.set("q", this.query.trim());
    if (cursor) query.set("cursor", cursor);
    if (this.route.level === "goals") {
      return `/api/catalog/projects/${encodeURIComponent(this.route.entity)}/${encodeURIComponent(this.route.project)}/goals?${query}`;
    }
    if (this.route.level === "runs") {
      return `/api/catalog/projects/${encodeURIComponent(this.route.entity)}/${encodeURIComponent(this.route.project)}/goals/${encodeURIComponent(this.route.goal_id)}/runs?${query}`;
    }
    if (this.route.level === "checkpoints") {
      if (this.route.entity) query.set("entity", this.route.entity);
      if (this.route.project) query.set("project", this.route.project);
      return `/api/catalog/runs/${encodeURIComponent(this.route.run_id)}/checkpoints?${query}`;
    }
    if (this.route.entity) query.set("entity", this.route.entity);
    return `/api/catalog/projects?${query}`;
  }

  async ensureLoaded() {
    const key = this.routeKey();
    if (this.loading || this.loadedKey === key) return;
    await this.load();
  }

  async load({ append = false, quiet = false } = {}) {
    if (this.loading) return;
    const key = this.routeKey();
    const cursor = append ? this.nextCursor : null;
    const serial = ++this.requestSerial;
    this.loading = true;
    if (!quiet) this.error = "";
    this.renderView();
    try {
      const response = await fetch(this.endpoint(cursor), {
        headers: { Authorization: `Bearer ${this.token}` },
        cache: "no-store",
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.error || `Catalog request failed (${response.status})`);
      if (serial !== this.requestSerial) return;
      if (this.route.level === "projects" && payload.entity && !this.route.entity) {
        this.route.entity = payload.entity;
      }
      const received = Array.isArray(payload.items) ? payload.items : [];
      this.items = append ? [...this.items, ...received] : received;
      this.nextCursor = payload.next_cursor || null;
      this.loadedKey = key;
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
      if (serial !== this.requestSerial) return;
      this.error = String(error?.message || error);
      if (quiet) this.showToast(this.error, true);
    } finally {
      if (serial === this.requestSerial) {
        this.loading = false;
        this.renderView();
      }
    }
  }

  updatePolling() {
    const shouldPoll = this.app.phase === "selecting" && this.route.level === "checkpoints";
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
    this.nextCursor = null;
    this.loadedKey = "";
    this.error = "";
    this.autoSelectedRoute = "";
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
      ? { ...route, level: "checkpoints", checkpoint_id: "" }
      : {
          level: "projects",
          entity: route.entity || "",
          project: "",
          goal_id: "",
          run_id: "",
          checkpoint_id: "",
        };
    this.navigate(next);
  }

  selectCheckpoint(item, { historyMode = "push" } = {}) {
    const route = {
      ...this.route,
      level: "checkpoints",
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
      },
      route: { ...route },
    });
  }

  back() {
    if (this.route.level === "checkpoints") {
      this.navigate({ level: "runs", run_id: "", checkpoint_id: "" });
    } else if (this.route.level === "runs") {
      this.navigate({
        level: "goals",
        goal_id: "",
        run_id: "",
        checkpoint_id: "",
      });
    } else if (this.route.level === "goals") {
      this.navigate({
        level: "projects",
        project: "",
        goal_id: "",
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

    if (this.app.phase === "approval_required") {
      shell.append(this.renderApproval());
    } else if (this.app.phase === "error") {
      shell.append(this.renderFailure());
    } else if (["resolving", "verifying", "loading"].includes(this.app.phase)) {
      shell.append(this.renderProgress());
    } else {
      shell.append(this.renderBreadcrumbs(), this.renderSearch(), this.renderResults());
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
    if (this.route.level === "checkpoints") return "Choose a checkpoint";
    if (this.route.level === "runs") return "Choose a run";
    if (this.route.level === "goals") return "Choose a goal";
    return "Choose a project";
  }

  renderBreadcrumbs() {
    const nav = document.createElement("nav");
    nav.className = "source-breadcrumbs";
    nav.setAttribute("aria-label", "Playback source");
    const projects = button("Projects", { quiet: true });
    projects.disabled = this.route.level === "projects";
    projects.addEventListener("click", () => this.navigate({
      level: "projects",
      project: "",
      goal_id: "",
      run_id: "",
      checkpoint_id: "",
    }));
    nav.append(projects);
    if (this.route.project) {
      const project = button(this.route.project, { quiet: true });
      project.disabled = this.route.level === "goals";
      project.addEventListener("click", () => this.navigate({
        level: "goals",
        goal_id: "",
        run_id: "",
        checkpoint_id: "",
      }));
      nav.append(project);
    }
    if (this.route.goal_id) {
      const goal = button(this.route.goal_id, { quiet: true });
      goal.disabled = this.route.level === "runs";
      goal.addEventListener("click", () => this.navigate({
        level: "runs",
        run_id: "",
        checkpoint_id: "",
      }));
      nav.append(goal);
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
    input.placeholder = this.route.level === "projects"
      ? "Search projects"
      : this.route.level === "goals"
        ? "Search goals"
        : this.route.level === "runs"
          ? "Search run ID, name, recipe, or seed"
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
      heading.textContent = this.route.level === "checkpoints"
        ? "No public checkpoints yet"
        : "No matching results";
      const detail = document.createElement("p");
      detail.textContent = this.route.level === "checkpoints"
        ? "This run has not published a playable checkpoint to public model storage."
        : "Try a broader search.";
      empty.append(heading, detail);
      body.append(empty);
      return body;
    }
    body.append(
      this.route.level === "projects"
        ? this.renderProjects()
        : this.route.level === "goals"
          ? this.renderGoals()
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
      if (goal.goal_slug && goal.goal_slug !== goal.goal_id) {
        const slug = document.createElement("small");
        slug.textContent = goal.goal_slug;
        identity.append(slug);
      }
      const meta = document.createElement("span");
      const recipeLabel = Number(goal.recipe_count) === 1 ? "recipe" : "recipes";
      meta.textContent = `${goal.title || goal.goal_slug} · ${Number(goal.recipe_count).toLocaleString()} ${recipeLabel}`;
      row.append(identity, meta);
      row.addEventListener("click", () => this.navigate({
        level: "runs",
        goal_id: goal.goal_id,
        run_id: "",
        checkpoint_id: "",
      }));
      list.append(row);
    });
    return list;
  }

  renderTable() {
    const scroll = document.createElement("div");
    scroll.className = "source-table-scroll";
    const table = document.createElement("table");
    table.className = "source-table";
    const head = document.createElement("thead");
    const headerRow = document.createElement("tr");
    const columns = this.route.level === "runs"
      ? ["Run", "Recipe", "Seed", "Updated"]
      : ["Checkpoint", "Purpose", "Step", "Evaluation", "Size", "Created"];
    columns.forEach((label) => {
      const cell = document.createElement("th");
      cell.scope = "col";
      cell.textContent = label;
      headerRow.append(cell);
    });
    head.append(headerRow);
    const body = document.createElement("tbody");
    this.items.forEach((item) => {
      const row = document.createElement("tr");
      row.tabIndex = this.hasControl() ? 0 : -1;
      row.setAttribute("role", "button");
      row.setAttribute("aria-disabled", String(!this.hasControl()));
      const values = this.route.level === "runs"
        ? [
            [item.name || item.run_id, item.run_id, "run-cell"],
            [item.recipe || "—"],
            [item.seed ?? "—"],
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
        if (secondary && !className.includes("run-cell")) {
          const small = document.createElement("small");
          small.textContent = String(secondary);
          cell.append(small);
        }
        row.append(cell);
      });
      const activate = () => {
        if (!this.hasControl()) return;
        if (this.route.level === "runs") {
          this.navigate({
            level: "checkpoints",
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
