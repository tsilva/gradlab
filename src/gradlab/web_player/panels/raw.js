import { createPanel, renderJson } from "./shared.js";

export function mount({ definition }) {
  const element = createPanel({
    id: definition.id,
    label: definition.label,
    className: "raw-panel",
    body: `
      <details open><summary>Selected transition</summary><pre data-transition class="json-view widget-empty">No data available yet</pre></details>
      <details><summary>Resolved playback configuration</summary><pre data-config>Waiting…</pre></details>
    `,
  });

  return {
    element,
    render(snapshot) {
      element.querySelector("[data-transition]").classList.toggle("widget-empty", !snapshot?.transition);
      renderJson(
        element.querySelector("[data-transition]"),
        snapshot?.transition,
        "No data available yet",
      );
      element.querySelector("[data-config]").textContent = snapshot?.session?.config
        || "No configuration supplied.";
    },
  };
}
