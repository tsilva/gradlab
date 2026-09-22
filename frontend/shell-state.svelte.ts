// Small UI projections only. Image buffers, exact histories and controller
// state remain owned by the existing Playback controllers.
export const shellState = $state({
  savedLayouts: [] as string[],
  loadLayout: (_name: string) => {},
  deleteLayout: (_name: string) => {},
  connection: { label: "Connecting", kind: "warning" },
  toast: { message: "", error: false, visible: false },
  transport: {
    action: "play",
    disabled: true,
    reason: "",
    label: "Play",
    icon: "player-play",
    resetDisabled: true,
    resetTitle: "",
  },
  evidence: { text: "", title: "" },
  timeline: {
    first: 0,
    last: 0,
    selected: 0,
    disabled: true,
    busy: false,
    progress: 0,
    zoom: null as { first: number; last: number } | null,
    label: "EPISODE — · STEP —",
    markerKey: "",
    markers: [] as any[],
  },
});
