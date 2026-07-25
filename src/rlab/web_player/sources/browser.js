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

function formatDate(value) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "—" : date.toLocaleString();
}

function formatBytes(value) {
  const bytes = Number(value);
  if (!Number.isFinite(bytes) || bytes <= 0) return "—";
  const units = ["B", "KiB", "MiB", "GiB"];
  const unit = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
  return `${(bytes / (1024 ** unit)).toFixed(unit ? 1 : 0)} ${units[unit]}`;
}

function routeSignature(route) {
  return JSON.stringify({
    level: route?.level || "projects",
    entity: route?.entity || "",
    project: route?.project || "",
    run_id: route?.run_id || "",
  });
}

export class SourceBrowser {
  constructor(root, { token, command, getState, showToast }) {
    this.root = root;
    this.token = token;
    this.command = command;
    this.getState = getState;
    this.showToast = showToast;
    this.route = { level: "projects", entity: "", project: "", run_id: "" };
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
  }

  render(snapshot) {
    this.app = snapshot?.app || { phase: "active" };
    const appRoute = this.app.route || {};
    const signature = routeSignature(appRoute);
    if (signature !== this.lastAppRoute) {
      this.lastAppRoute = signature;
      this.route = {
        level: appRoute.level || "projects",
        entity: appRoute.entity || "",
        project: appRoute.project || "",
        run_id: appRoute.run_id || "",
      };
      this.query = "";
      this.items = [];
      this.nextCursor = null;
      this.loadedKey = "";
      this.error = "";
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
    if (this.route.level === "runs") {
      return `/api/catalog/projects/${encodeURIComponent(this.route.entity)}/${encodeURIComponent(this.route.project)}/runs?${query}`;
    }
    if (this.route.level === "checkpoints") {
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

  navigate(route) {
    this.route = { ...this.route, ...route };
    this.query = "";
    this.items = [];
    this.nextCursor = null;
    this.loadedKey = "";
    this.error = "";
    this.renderView();
    this.ensureLoaded();
    this.updatePolling();
  }

  back() {
    if (this.route.level === "checkpoints") {
      this.navigate({ level: "runs", run_id: "" });
    } else if (this.route.level === "runs") {
      this.navigate({ level: "projects", project: "", run_id: "" });
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
    return "Choose a W&B project";
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
      run_id: "",
    }));
    nav.append(projects);
    if (this.route.project) {
      const project = button(this.route.project, { quiet: true });
      project.disabled = this.route.level === "runs";
      project.addEventListener("click", () => this.navigate({ level: "runs", run_id: "" }));
      nav.append(project);
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
      : this.route.level === "runs"
        ? "Search run ID, name, recipe, goal, or seed"
        : "Search checkpoint, step, hash, or purpose";
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
      meta.textContent = `${project.entity} · created ${formatDate(project.created_at)}`;
      row.append(name, meta);
      row.addEventListener("click", () => this.navigate({
        level: "runs",
        entity: project.entity || this.route.entity,
        project: project.name,
        run_id: "",
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
      ? ["Run", "State", "Recipe / goal", "Seed", "Updated"]
      : ["Checkpoint", "Purpose", "Step", "Size", "Created"];
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
            [item.name || item.run_id, item.run_id],
            [item.state || "—"],
            [item.recipe || item.goal || "—", item.recipe && item.goal ? item.goal : ""],
            [item.seed ?? "—"],
            [formatDate(item.updated_at || item.created_at)],
          ]
        : [
            [item.promoted ? `${item.checkpoint_id} · promoted` : item.checkpoint_id, item.sha256],
            [item.purpose],
            [Number(item.step).toLocaleString()],
            [formatBytes(item.size_bytes)],
            [formatDate(item.created_at)],
          ];
      values.forEach(([primary, secondary = ""]) => {
        const cell = document.createElement("td");
        const main = document.createElement("span");
        main.textContent = String(primary);
        cell.append(main);
        if (secondary) {
          const small = document.createElement("small");
          small.textContent = String(secondary);
          cell.append(small);
        }
        row.append(cell);
      });
      const activate = () => {
        if (!this.hasControl()) return;
        if (this.route.level === "runs") {
          this.navigate({ level: "checkpoints", run_id: item.run_id });
        } else {
          this.command("select_source", {
            source: {
              kind: "manifest",
              value: item.manifest_url,
              entity: this.route.entity,
              project: this.route.project,
              run_id: item.run_id,
              checkpoint_id: item.checkpoint_id,
            },
            route: { ...this.route },
          });
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
    choose.addEventListener("click", () => this.command("browse_sources"));
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
