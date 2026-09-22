<script lang="ts">
  import Panel from "./Panel.svelte";
  import PlaybackSettings from "./PlaybackSettings.svelte";
  let { definition, services } = $props();
  let settings: ReturnType<typeof PlaybackSettings>;
  export const render = (...args: Parameters<typeof settings.render>) =>
    settings?.render(...args);
  export const updateControl = () => settings?.updateControl();
  export const episodeOptions = () => settings?.episodeOptions();
</script>

<Panel {definition} className="control-panel transport"
  ><PlaybackSettings
    {services}
    idPrefix="panel-playback"
    bind:this={settings}
  /></Panel
>
