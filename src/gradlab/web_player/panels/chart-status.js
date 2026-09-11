// Eligibility stays with panel presentation, not the request lifecycle.
export function usesChartHistory(definition) {
  return definition.type === "telemetry" && definition.config?.blocks?.some(chartHistoryBlock);
}

export function chartHistoryBlock(block) {
  return ["line", "namespace-explorer", "reward-table"].includes(block.kind);
}

export function chartPoints(history, view) {
  // Null explicitly means this selection has no data yet. It must not fall back
  // to a different retained-history window while the selected range is loading.
  return view?.chartStatus ? view.chartHistory ?? [] : view?.chartHistory ?? history;
}

export function createChartStatus(retry) {
  const element = document.createElement("div");
  element.className = "chart-status";
  element.setAttribute("role", "status");
  element.setAttribute("aria-live", "polite");
  element.hidden = true;
  const message = document.createElement("span");
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = "Retry";
  button.hidden = true;
  button.addEventListener("click", () => retry?.());
  element.append(message, button);
  return {
    element,
    render(chart) {
      element.hidden = !chart || chart.status === "ready" || chart.status === "refreshing";
      button.hidden = chart?.status !== "error";
      message.textContent = chart?.status === "error"
        ? chart.error || "Unable to load episode charts"
        : chart?.status === "recovering" ? "Recovering chart history…"
          : chart?.status === "loading" ? "Loading chart history…"
            : chart?.status === "refreshing" || chart?.status === "ready" ? ""
              : "No recorded chart history";
    },
  };
}
