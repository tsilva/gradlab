export function defaultBlockForKind(kind) {
  if (kind === "stats" || kind === "line") {
    return { kind, metrics: ["reward/shaped"] };
  }
  if (kind === "namespace-explorer") {
    return { kind, namespace: "signal", metric: "" };
  }
  if (kind === "reward-table") return { kind };
  if (kind === "reward-breakdown") return { kind, scope: "episode" };
  return {
    kind,
    metric: kind === "distribution" ? "policy/distribution" : "action/executed",
  };
}

export function editorFieldsForBlock(block) {
  return {
    metric: !["reward-breakdown", "reward-table"].includes(block?.kind),
    namespace: block?.kind === "namespace-explorer",
    scope: block?.kind === "reward-breakdown",
  };
}
