<script lang="ts">
  let { name }: { name: string } = $props();
</script>

<svg class="icon" aria-hidden="true"
  ><use href={`/assets/tabler-icons.svg#ti-${name}`}></use></svg
>
