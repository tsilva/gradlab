function number(value, maximumFractionDigits = 2) {
  if (value === null || value === undefined || value === "" || typeof value === "boolean") {
    return "—";
  }
  const numeric = Number(value);
  return Number.isFinite(numeric)
    ? numeric.toLocaleString(undefined, { maximumFractionDigits })
    : "—";
}

function words(value, fallback = "—") {
  const result = String(value || "").trim();
  if (!result) return fallback;
  return result.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function checkpointLabel(checkpointId) {
  const match = String(checkpointId || "").match(/^checkpoint-(\d+)-/);
  return match ? `Checkpoint at step ${Number(match[1]).toLocaleString()}` : "Public checkpoint";
}

export function episodeReport(snapshot) {
  const session = snapshot?.session || {};
  const transition = snapshot?.transition || {};
  const route = snapshot?.app?.route || {};
  const mode = String(session.playback_contract?.mode || "training");
  const semantics = snapshot?.mode === "recording"
    ? "Recorded episode"
    : snapshot?.mode === "dataset"
      ? "Recorded dataset"
      : mode === "evaluation"
        ? "Published evaluation contract"
        : mode === "counterfactual"
          ? "Counterfactual — not evidence"
          : "Training contract";
  const reasons = Array.isArray(transition.boundary_reasons)
    ? transition.boundary_reasons.filter(Boolean).map((reason) => words(reason))
    : [];
  const boundary = transition.boundary
    ? reasons.join(" · ") || "Terminal transition"
    : "Not reached";
  const transitionReturn = transition?.reward?.return;
  const terminalOutcome = transition.outcome && transition.outcome !== "continuing"
    ? transition.outcome
    : transition.terminated
      ? "terminated"
      : transition.truncated ? "truncated" : "terminal";
  return {
    outcome: transition.boundary ? words(terminalOutcome, "Terminal") : "In progress",
    outcomeTone: transition.boundary && /success|win|complete|accepted/i.test(terminalOutcome)
      ? "success"
      : transition.boundary ? "terminal" : "active",
    boundary,
    episodeReturn: number(
      transition.boundary
        ? transitionReturn
        : transitionReturn ?? session.total_reward,
      3,
    ),
    steps: number(transition.step ?? session.step, 0),
    seed: number(transition.seed ?? session.seed, 0),
    semantics,
    source: route.checkpoint_id
      ? checkpointLabel(route.checkpoint_id)
      : snapshot?.mode === "recording"
        ? "Local recording"
        : snapshot?.mode === "dataset" ? "Recorded dataset" : "Local checkpoint",
    disclaimer: "Playback supports interpretation; it is not acceptance or promotion evidence.",
  };
}
