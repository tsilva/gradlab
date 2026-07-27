import { createPanel } from "./shared.js";

export function mount({ definition }) {
  const element = createPanel({
    id: definition.id,
    label: definition.label,
    body: '<ol data-list class="event-list"><li class="empty-state">No events observed.</li></ol>',
  });
  const list = element.querySelector("[data-list]");

  return {
    element,
    renderHistory(history, _snapshot = null, view = {}) {
      const events = history
        .filter((point) => point.boundary || point.events?.length);
      const selected = view.inspection
        ? events.find(
          (point) => Number(point.sequence) === Number(view.selectedSequence),
        )
        : null;
      const visible = events.slice(-100);
      if (
        selected
        && !visible.some((point) => Number(point.sequence) === Number(selected.sequence))
      ) {
        visible.shift();
        visible.unshift(selected);
      }
      visible.reverse();
      if (!visible.length) {
        const empty = document.createElement("li");
        empty.className = "empty-state";
        empty.textContent = "No events observed.";
        list.replaceChildren(empty);
        return;
      }
      list.replaceChildren(...visible.map((point) => {
        const item = document.createElement("li");
        const isSelected = selected
          && Number(point.sequence) === Number(selected.sequence);
        item.className = [
          "event-item",
          point.boundary ? "boundary" : "",
          isSelected ? "selected" : "",
        ].filter(Boolean).join(" ");
        const label = document.createElement("div");
        label.textContent = point.events?.length ? point.events.join(" · ") : "episode boundary";
        const meta = document.createElement("div");
        meta.className = "event-meta";
        meta.textContent = `seq ${point.sequence} · ep ${point.episode} · step ${point.step}`;
        item.append(label, meta);
        return item;
      }));
    },
  };
}
