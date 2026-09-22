<script lang="ts">
  import {
    rewardInspectionRows,
    formatRewardCell,
  } from "../../src/gradlab/web_player/panels/reward-inspector.js";
  import { selectedPoint } from "../../src/gradlab/web_player/panels/telemetry-panel.js";
  import { chartPoints } from "../../src/gradlab/web_player/panels/chart-status.js";
  let { context, services }: { context: any; services: any } = $props();
  let selected = $derived(
    selectedPoint(context.history, context.snapshot, context.view),
  );
  let reference = $derived(context.view?.rewardReference?.step ?? null);
  let gamma = $derived(
    context.snapshot?.session?.value_discount ??
      context.snapshot?.session?.critic_comparison?.discount,
  );
  let referenceSample = $state.raw<any>(null);
  let rows = $derived.by(() => {
    const sample = context.view?.rewardReference?.sample;
    const retained =
      referenceSample?.step === reference ? referenceSample : null;
    const exact =
      sample?.step === reference
        ? sample
        : selected?.step === reference
          ? selected
          : retained;
    return rewardInspectionRows(
      chartPoints(context.history, context.view),
      exact,
      gamma,
      5,
      reference,
    ).map((point) => ({ ...point, inspected: point.step === selected?.step }));
  });
  $effect(() => {
    const sample = context.view?.rewardReference?.sample;
    if (referenceSample?.step !== reference) referenceSample = null;
    if (sample?.step === reference) referenceSample = sample;
    else if (selected?.step === reference) referenceSample = selected;
  });
  const returnTitle = (point: any) =>
    point.value_comparison_reasons?.join("; ") ||
    (Number.isFinite(point.estimated_return) &&
    !Number.isFinite(point.realized_return)
      ? `Provisional estimate: recorded rewards followed by discounted V(s) at step ${point.return_estimate_step}. Not realized return or calibration evidence.`
      : point.realized_return_bootstrapped
        ? "Bootstrapped return includes the final state value."
        : "Realized discounted return from this state; pending until comparable evidence is available.");
  const values = (point: any) => [
    formatRewardCell(point.reward_provider),
    formatRewardCell(point.reward_shaped),
    point.delay ?? "—",
    formatRewardCell(point.weight, 5),
    formatRewardCell(point.contribution, 5),
    point.value_comparison_reasons?.length
      ? "Incomparable"
      : Number.isFinite(point.realized_return)
        ? `${formatRewardCell(point.realized_return)}${point.realized_return_bootstrapped ? " ⚠" : ""}`
        : Number.isFinite(point.estimated_return)
          ? `${formatRewardCell(point.estimated_return)} ⚠`
          : "Pending",
    formatRewardCell(point.value),
  ];
  const title = (point: any, value: any, index: number) =>
    index === 5
      ? returnTitle(point)
      : index === 6
        ? "Recorded pre-action critic prediction V(s) for this row’s state"
        : value === "—"
          ? point.past && index >= 3
            ? "Past reward: excluded from future contribution"
            : "Unavailable: required recorded data is missing"
          : undefined;
</script>

<section class="telemetry-block reward-table-block">
  <button
    type="button"
    disabled={!Number.isInteger(context.snapshot?.transition?.step)}
    onclick={() =>
      services.setRewardReference(context.snapshot.transition.step)}
    >Set return reference to cursor</button
  >
  <div class="reward-history-scroll">
    <!-- svelte-ignore a11y_role_supports_aria_props_implicit (aria-description is the existing global ARIA description) -->
    <table
      class="reward-history-table"
      aria-label="Recorded rewards from the return reference step"
      aria-description={`Up to five current and future recorded reward samples. Discounted shaped rewards from step ${reference ?? "unavailable"}, gamma ${gamma ?? "unavailable"}. Past rewards are excluded. Discounting does not imply causation.`}
    >
      <thead
        ><tr
          >{#each ["Step", "Native", "Shaped", "Delay", "Weight", "Contribution", "G(s)", "V(s)"] as label}<th
              scope="col">{label}</th
            >{/each}</tr
        ></thead
      >
      <tbody>
        {#each rows as point (point.step)}
          <tr class:is-inspected={point.inspected}>
            <th scope="row"
              ><button
                class="reward-step-button"
                type="button"
                data-reward-step={point.step}
                aria-label={`Inspect step ${point.step}`}
                aria-current={point.inspected ? "step" : undefined}
                onclick={() =>
                  services.inspectStep
                    ? services.inspectStep(point.step)
                    : services.inspectSequence?.(point.sequence)}
                >{point.step}</button
              ></th
            >
            {#each values(point) as value, index}<td
                title={title(point, value, index)}
                aria-label={index === 5 && String(value).includes("⚠")
                  ? `${value}: ${returnTitle(point)}`
                  : undefined}>{value}</td
              >{/each}
          </tr>
        {:else}<tr><td colspan="8">No recorded rewards in this window</td></tr
          >{/each}
      </tbody>
    </table>
  </div>
</section>
