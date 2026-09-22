import { eventColor, eventColorFill, eventLabels } from "../event-colors.js";

export function eventAtCursor(point, view) {
  if (Number.isInteger(view.selectedStep)) {
    return point.step === view.selectedStep
      && (view.selectedEpisode == null || point.episode === view.selectedEpisode);
  }
  return view.selectedSequence != null && point.sequence === view.selectedSequence;
}

