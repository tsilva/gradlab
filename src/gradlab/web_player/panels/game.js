
const FRAME_GAME = 1;

export function gameFramePhase(snapshot) {
  const role = snapshot?.transition?.after?.frame_role;
  if (!snapshot?.transition) return "Initial observation";
  if (role === "terminal_observation") return "Terminal observation";
  if (role === "next_episode_initial_observation") {
    return "Next episode initial observation";
  }
  return "After-action observation";
}

export function gameFrameBoundaryKind(snapshot) {
  const transition = snapshot?.transition;
  if (!transition?.boundary) return "";
  return [
    transition.terminated ? "Terminated" : "",
    transition.truncated ? "Truncated" : "",
  ].filter(Boolean).join(" · ");
}

function displayTerminalFact(value) {
  const label = String(value || "")
    .replaceAll("_", " ")
    .replace(/\s+/g, " ")
    .trim();
  return label ? `${label[0].toUpperCase()}${label.slice(1)}` : "";
}

export function gameFrameTerminationDetail(snapshot) {
  const transition = snapshot?.transition;
  if (transition?.after?.frame_role !== "terminal_observation") return "";
  const events = Array.isArray(transition.events)
    ? transition.events.map(displayTerminalFact).filter(Boolean)
    : [];
  if (events.length) return [...new Set(events)].join(" · ");
  const info = transition.info || {};
  const explicitReason = [
    info.termination_reason,
    info.terminal_reason,
    info.final_info?.termination_reason,
    info.final_info?.terminal_reason,
  ].map(displayTerminalFact).find(Boolean);
  if (explicitReason) return explicitReason;
  const boundaryReason = (
    Array.isArray(transition.boundary_reasons)
      ? transition.boundary_reasons
      : []
  )
    .map(displayTerminalFact)
    .find(Boolean);
  if (boundaryReason) return boundaryReason;
  if (transition.truncated) return "Truncated";
  if (transition.terminated) return "Terminated";
  return displayTerminalFact(
    ["continuing", "neutral", "boundary"].includes(transition.outcome)
      ? ""
      : transition.outcome,
  );
}

export function gameFrameTerminationTone(snapshot) {
  const transition = snapshot?.transition;
  if (transition?.after?.frame_role !== "terminal_observation") return "";
  const outcome = String(transition.outcome || "").toLowerCase();
  return ["success", "failure", "timeout"].includes(outcome) ? outcome : "";
}
