import { mountPlaybackSettings } from "../playback-settings.js";
import { createPanel } from "./shared.js";

export function mount({ definition, services }) {
  const element = createPanel({
    id: definition.id,
    label: definition.label,
    tag: "aside",
    className: "control-panel transport",
    headerClass: "control-panel-header",
  });
  const settings = mountPlaybackSettings({ services, idPrefix: "panel-playback" });
  element.append(settings.element);
  return {
    element,
    render: settings.render,
    updateControl: settings.updateControl,
    episodeOptions: settings.episodeOptions,
  };
}
