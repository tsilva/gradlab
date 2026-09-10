import { createPanel } from "./shared.js";
import { eventColor, eventColorFill, eventLabels } from "../event-colors.js";

export function mount({ definition, services }) {
  const element = createPanel({
    id: definition.id,
    label: definition.label,
    body: '<ol data-list class="event-list"><li class="empty-state">No events observed.</li></ol>',
  });
  const list = element.querySelector("[data-list]");

  const navigation = document.createElement("div");
  const newer = document.createElement("button");
  const older = document.createElement("button");
  const status = document.createElement("span");
  newer.textContent = "Newer events";
  older.textContent = "Older events";
  navigation.append(newer, status, older);
  list.after(navigation);
  let identity = null;
  let pages = [null];
  let cursor = 0;
  let nextLast = null;
  let pending = false;
  let revision = 0;
  let updated = 0;
  let disposed = false;
  let latestView = {};
  let currentPoints = [];
  let selectedSequence = null;

  async function load() {
    if (pending || !identity || disposed) return;
    pending = true;
    const request = revision;
    const episodeId = identity.split(":").slice(1).join(":");
    try {
      const result = await services.loadEvents(episodeId, pages[cursor]);
      if (request !== revision || disposed) return;
      nextLast = result.next_last;
      currentPoints = result.points;
      draw(currentPoints, latestView, true);
      status.textContent = ` Page ${cursor + 1} `;
      updated = Date.now();
    } catch (error) {
      if (request === revision && !disposed) status.textContent = error.message;
    } finally {
      pending = false;
      newer.disabled = cursor === 0;
      older.disabled = nextLast === null;
      if (request !== revision && !disposed) void load();
    }
  }
  newer.addEventListener("click", () => {
    if (pending || cursor === 0) return;
    cursor -= 1;
    void load();
  });
  older.addEventListener("click", () => {
    if (pending || nextLast === null) return;
    pages = pages.slice(0, cursor + 1);
    pages.push(nextLast);
    cursor += 1;
    void load();
  });

  function draw(visible, view, recorded) {
    const selected = view.inspection
      ? visible.find((point) => Number(point.sequence) === Number(view.selectedSequence))
      : null;
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
      const isSelected = selected
        && Number(point.sequence) === Number(selected.sequence);
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
      latestView = view;
      const episodeId = services.getState?.().liveSnapshot?.trajectory?.episode_id;
      const key = episodeId ? `${view.sessionEpoch}:${episodeId}` : null;
      navigation.hidden = !key;
      if (key !== identity) {
        identity = key;
        revision += 1;
        pages = [null];
        cursor = 0;
        nextLast = null;
        updated = 0;
        currentPoints = [];
        list.replaceChildren();
      }
      if (key) {
        if (selectedSequence !== view.selectedSequence) {
          selectedSequence = view.selectedSequence;
          draw(currentPoints, view, true);
        }
        if (cursor === 0 && Date.now() - updated >= 1000) void load();
      } else {
        draw(history.filter((point) => point.boundary || point.events?.length).reverse(), view, false);
      }
    },
    destroy() { disposed = true; revision += 1; },
  };
}
