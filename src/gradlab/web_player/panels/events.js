import { createPanel } from "./shared.js";
import { eventColor, eventColorFill, eventLabels } from "../event-colors.js";

export function eventAtCursor(point, view) {
  if (Number.isInteger(view.selectedStep)) {
    return point.step === view.selectedStep
      && (view.selectedEpisode == null || point.episode === view.selectedEpisode);
  }
  return view.selectedSequence != null && point.sequence === view.selectedSequence;
}

export function mount({ definition, services }) {
  const element = createPanel({
    id: definition.id,
    label: definition.label,
    body: '<ol data-list class="event-list"><li class="empty-state">No events observed.</li></ol>',
  });
  const list = element.querySelector("[data-list]");

  const status = document.createElement("div");
  status.setAttribute("role", "status");
  list.after(status);
  let identity = null;
  let nextLast = null;
  let expanded = false;
  let pending = false;
  let revision = 0;
  let updated = 0;
  let disposed = false;
  let latestView = {};
  let currentPoints = [];
  let selectedCursor = null;
  let cursorPoint = null;

  async function load(append = false) {
    if (pending || !identity || disposed) return;
    pending = true;
    const request = revision;
    status.textContent = "Loading events…";
    const episodeId = identity.split(":").slice(1).join(":");
    try {
      const result = await services.loadEvents(episodeId, append ? nextLast : null);
      if (request !== revision || disposed) return;
      if (append || !expanded) nextLast = result.next_last;
      if (append) {
        const steps = new Set(currentPoints.map((point) => point.step));
        currentPoints.push(...result.points.filter((point) => !steps.has(point.step)));
        expanded = true;
      } else if (expanded) {
        const points = new Map(currentPoints.map((point) => [point.step, point]));
        for (const point of result.points) points.set(point.step, point);
        currentPoints = [...points.values()].sort((a, b) => b.step - a.step);
      } else {
        currentPoints = result.points;
      }
      const scrollTop = element.scrollTop;
      draw(currentPoints, latestView, true);
      element.scrollTop = scrollTop;
      status.textContent = nextLast === null ? "" : "Scroll down for older events";
      updated = Date.now();
    } catch (error) {
      if (request === revision && !disposed) status.textContent = error.message;
    } finally {
      pending = false;
      if (request !== revision && !disposed) void load();
    }
  }
  function loadAtBottom() {
    if (nextLast !== null && element.clientHeight > 0
      && element.scrollHeight - element.scrollTop - element.clientHeight <= 1) {
      void load(true);
    }
  }
  element.addEventListener("scroll", loadAtBottom, { passive: true });

  function draw(visible, view, recorded) {
    if (cursorPoint && !visible.some((point) => eventAtCursor(point, view))) {
      visible = [...visible, cursorPoint].sort((a, b) => b.step - a.step);
    }
    const selected = visible.find((point) => eventAtCursor(point, view));
    if (!visible.length) {
      const empty = document.createElement("li");
      empty.className = "empty-state";
      empty.textContent = "No events observed.";
      list.replaceChildren(empty);
      return;
    }
    list.replaceChildren(...visible.map((point) => {
      const labels = eventLabels(point);
      const item = document.createElement("li");
      const isSelected = point === selected;
      item.className = [
        "event-item",
        point.boundary ? "boundary" : "",
        isSelected ? "selected" : "",
      ].filter(Boolean).join(" ");
      item.style.setProperty("--event-colors", eventColorFill(labels));
      const jump = document.createElement("button");
      jump.type = "button";
      jump.className = "event-jump";
      const label = document.createElement("div");
      label.className = "event-labels";
      label.append(...labels.map((eventLabel) => {
        const part = document.createElement("span");
        part.className = "event-label";
        part.style.setProperty("--event-color", eventColor(eventLabel));
        part.textContent = eventLabel;
        return part;
      }));
      const meta = document.createElement("div");
      meta.className = "event-meta";
      meta.textContent = `ep ${point.episode} · step ${point.step}`;
      jump.setAttribute(
        "aria-label",
        `Inspect ${labels.join(" · ")} at episode ${point.episode}, step ${point.step}`,
      );
      if (isSelected) jump.setAttribute("aria-current", "step");
      jump.addEventListener("click", () => recorded
        ? services.inspectStep(point.step)
        : services.inspectSequence(point.sequence));
      jump.append(label, meta);
      item.append(jump);
      return item;
    }));
  }
  return {
    element,
    renderHistory(history, snapshot = null, view = {}) {
      view = { ...view, selectedStep: snapshot?.transition?.step,
        selectedEpisode: snapshot?.transition?.episode ?? snapshot?.session?.episode };
      latestView = view;
      cursorPoint = history.find((point) => eventAtCursor(point, view)
        && (point.boundary || point.events?.length)) || null;
      const episodeId = services.getState?.().liveSnapshot?.trajectory?.episode_id;
      const key = episodeId ? `${view.sessionEpoch}:${episodeId}` : null;
      status.hidden = !key;
      if (key !== identity) {
        identity = key;
        revision += 1;
        expanded = false;
        element.scrollTop = 0;
        status.textContent = "";
        nextLast = null;
        updated = 0;
        currentPoints = [];
        list.replaceChildren();
      }
      if (key) {
        const cursor = JSON.stringify([view.selectedStep, view.selectedEpisode, view.selectedSequence]);
        if (selectedCursor !== cursor) {
          selectedCursor = cursor;
          draw(currentPoints, view, true);
        }
        if (element.scrollTop === 0 && Date.now() - updated >= 1000) void load();
      } else {
        draw(history.filter((point) => point.boundary || point.events?.length).reverse(), view, false);
      }
    },
    destroy() {
      disposed = true;
      revision += 1;
      element.removeEventListener("scroll", loadAtBottom);
    },
  };
}
