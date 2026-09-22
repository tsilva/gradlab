import Game from "./components/Game.svelte";
import Observation from "./components/Observation.svelte";
import Attribution from "./components/Attribution.svelte";
import Cnn from "./components/Cnn.svelte";
import Controls from "./components/Controls.svelte";
import Telemetry from "./components/Telemetry.svelte";
import Events from "./components/Events.svelte";
import Raw from "./components/Raw.svelte";
import type { Component } from "svelte";
export const panelComponents: Record<string, Component<any, any>> = {
  game: Game,
  observation: Observation,
  attribution: Attribution,
  cnn: Cnn,
  controls: Controls,
  telemetry: Telemetry,
  events: Events,
  raw: Raw,
};
