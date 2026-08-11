export function workspaceIsEditable(preset) {
  return preset === "custom";
}

export function playbackSourceTitle(route = {}) {
  const rawEnvironment = String(route.environment_id || "Environment").trim();
  const environment = rawEnvironment
    .replace(/-v\d+$/i, "")
    .replace(/-(nes|snes|genesis|atari\d*)$/i, "")
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .replaceAll("-", " ")
    .replace(/^Vizdoom\b/i, "ViZDoom");
  const checkpoint = String(route.checkpoint_id || "").trim();
  const match = checkpoint.match(/^checkpoint-(\d+)-/);
  if (!match) return environment;
  return `${environment} · ${Number(match[1]).toLocaleString()} steps`;
}

export function statusMessageShouldToast({ status_message: statusMessage = "", session = {} } = {}) {
  const message = String(statusMessage || "").trim();
  if (!message) return false;
  if (/^playing next episode$/i.test(message)) return false;
  if (!session.awaiting_next_episode) return true;
  return /error|expired|unsupported|no configured/i.test(message);
}

export function transportPresentation({
  running = false,
  hasControl = false,
  canReplay = false,
  replaying = false,
  session = {},
  recording = false,
} = {}) {
  if (running) {
    return {
      action: "pause",
      label: replaying ? "Pause replay" : "Pause",
      icon: "player-pause",
      disabled: !hasControl,
      reason: hasControl ? "Pause after the current transition" : "Another window has control",
    };
  }
  if (canReplay) {
    return {
      action: "replay",
      label: "Replay",
      icon: "player-play",
      disabled: !hasControl,
      reason: hasControl ? "Replay from the selected step" : "Another window has control",
    };
  }
  if (session.awaiting_next_episode) {
    const available = Boolean(session.can_start_next_episode) && !recording;
    return {
      action: "next_episode",
      label: "Next episode",
      icon: "player-skip-forward",
      disabled: !hasControl || !available,
      reason: !hasControl
        ? "Another window has control"
        : available
          ? "Start the prepared next episode"
          : "The configured episode limit has been reached",
    };
  }
  return {
    action: "play",
    label: "Play",
    icon: "player-play",
    disabled: !hasControl,
    reason: hasControl ? "Play the current episode" : "Another window has control",
  };
}
