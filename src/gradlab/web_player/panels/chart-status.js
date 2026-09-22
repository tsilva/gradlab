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

