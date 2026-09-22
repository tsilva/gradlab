<script lang="ts">
  import { getContext } from "svelte";
  const active = getContext<() => boolean>("panel-active") ?? (() => true);
  import { atlasTileRect } from "../../src/gradlab/web_player/panels/diagnostic-overlays.js";
  let {
    bitmap,
    atlas,
    tile,
    smooth,
  }: { bitmap: ImageBitmap | null; atlas: any; tile: number; smooth: boolean } =
    $props();
  let canvas: HTMLCanvasElement;
  let lastBitmap: ImageBitmap | null = null,
    lastKey = "";
  $effect(() => {
    if (!active()) return;
    const rect = atlasTileRect(atlas, tile);
    const key = JSON.stringify([rect, smooth]);
    if (bitmap === lastBitmap && key === lastKey) return;
    lastBitmap = bitmap;
    lastKey = key;
    if (!canvas || !bitmap || !rect) {
      canvas?.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }
    if (canvas.width !== rect.width) canvas.width = rect.width;
    if (canvas.height !== rect.height) canvas.height = rect.height;
    const context = canvas.getContext("2d")!;
    context.imageSmoothingEnabled = smooth;
    context.drawImage(
      bitmap,
      rect.x,
      rect.y,
      rect.width,
      rect.height,
      0,
      0,
      rect.width,
      rect.height,
    );
  });
</script>

<canvas
  bind:this={canvas}
  data-kernel={!smooth ? "" : undefined}
  data-activation={smooth ? "" : undefined}
></canvas>
