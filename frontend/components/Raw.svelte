<script lang="ts">
  import Panel from "./Panel.svelte";
  let { definition } = $props();
  let snapshot = $state.raw<any>(null);
  export function render(next: any) {
    snapshot = next;
  }
  let tokens = $derived.by(() => {
    const source =
      JSON.stringify(snapshot?.transition, null, 2) || "No data available yet";
    const matcher =
      /"(?:\\.|[^"\\])*"(?=\s*:)|"(?:\\.|[^"\\])*"|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|\b(?:true|false|null)\b/g;
    const result = [];
    let cursor = 0;
    for (const match of source.matchAll(matcher)) {
      result.push({ text: source.slice(cursor, match.index), kind: "" });
      const raw = match[0];
      result.push({
        text: raw,
        kind: raw.startsWith('"')
          ? /^\s*:/.test(source.slice(match.index + raw.length))
            ? "json-key"
            : "json-string"
          : raw === "true" || raw === "false"
            ? "json-boolean"
            : raw === "null"
              ? "json-null"
              : "json-number",
      });
      cursor = match.index + raw.length;
    }
    result.push({ text: source.slice(cursor), kind: "" });
    return result;
  });
</script>

<Panel {definition} className="raw-panel"
  ><details open>
    <summary>Selected transition</summary>
    <pre
      data-transition
      class="json-view"
      class:widget-empty={!snapshot?.transition}>{#each tokens as token}{#if token.kind}<span
            class={token.kind}>{token.text}</span
          >{:else}{token.text}{/if}{/each}</pre>
  </details>
  <details>
    <summary>Resolved playback configuration</summary>
    <pre data-config>{snapshot?.session?.config ||
        "No configuration supplied."}</pre>
  </details></Panel
>
