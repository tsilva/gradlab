const COMPONENT_ORDER = Object.freeze([
  "native_reward",
  "cell_novelty_reward",
  "event_reward",
  "progress_reward",
  "score_reward",
  "completion_reward",
  "death_penalty",
  "time_penalty",
]);

const COMPONENT_LABELS = Object.freeze({
  native_reward: "Native reward",
  cell_novelty_reward: "Cell novelty",
  event_reward: "Event reward",
  progress_reward: "Progress",
  score_reward: "Score",
  completion_reward: "Completion",
  death_penalty: "Death penalty",
  time_penalty: "Time penalty",
  unattributed: "Unattributed task reward",
  clip_adjustment: "Clip adjustment",
});

function finite(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function tolerance(gross) {
  return Math.max(1e-9, 1e-6 * Math.max(0, finite(gross) || 0));
}

function selectedPoint(history, snapshot, view) {
  const sequence = view?.selectedSequence ?? snapshot?.transition?.sequence;
  const retained = history.find(
    (point) => Number(point?.sequence) === Number(sequence),
  );
  if (retained) return retained;
  const transition = snapshot?.transition;
  if (!transition || Number(transition.sequence) !== Number(sequence)) return null;
  return {
    sequence: transition.sequence,
    episode: transition.episode,
    step: transition.step,
    reward_raw: transition.reward?.raw,
    reward_shaped: transition.reward?.shaped,
    reward_accounting_error: transition.reward?.accounting_error,
    return: transition.reward?.return,
    components: transition.reward?.components || {},
  };
}

function pointLedger(point, contract) {
  if (point?.reward_accounting_error) {
    return { error: String(point.reward_accounting_error) };
  }
  const raw = finite(point?.reward_raw);
  const final = finite(point?.reward_shaped);
  const scale = finite(contract?.reward_scale);
  if (raw === null) return { error: "Pre-transform reward is missing or non-finite." };
  if (final === null) return { error: "Final reward is missing or non-finite." };
  if (scale === null || scale < 0 || scale > 1) return { error: "Reward scale is invalid." };
  const values = point?.components;
  if (!values || typeof values !== "object" || Array.isArray(values)) {
    return { error: "Reward components are malformed." };
  }
  const components = {};
  for (const [id, value] of Object.entries(values)) {
    if (!COMPONENT_ORDER.includes(id)) {
      return { error: `Unknown reward component “${id}”.` };
    }
    const number = finite(value);
    if (number === null) return { error: `Reward component “${id}” is non-finite.` };
    components[id] = number;
  }
  const clip = contract.clip_bounds;
  if (
    clip !== null
    && (!Array.isArray(clip) || clip.length !== 2
      || finite(clip[0]) === null || finite(clip[1]) === null
      || Number(clip[0]) > Number(clip[1]))
  ) return { error: "Reward clip bounds are invalid." };

  const preclip = raw * scale;
  const expectedFinal = clip === null
    ? preclip
    : Math.min(Number(clip[1]), Math.max(Number(clip[0]), preclip));
  const gross = Math.abs(raw) + Math.abs(preclip) + Math.abs(final);
  if (Math.abs(final - expectedFinal) > tolerance(gross)) {
    return {
      error: `Final reward ${final} does not match scale-then-clip accounting ${expectedFinal}.`,
    };
  }
  const entries = COMPONENT_ORDER
    .filter((id) => Object.hasOwn(components, id))
    .map((id) => ({
      id,
      label: COMPONENT_LABELS[id],
      kind: "component",
      raw: components[id],
      impact: components[id] * scale,
    }));
  const attributedRaw = entries.reduce((sum, entry) => sum + entry.raw, 0);
  const unattributedRaw = raw - attributedRaw;
  entries.push({
    id: "unattributed",
    label: COMPONENT_LABELS.unattributed,
    kind: "residual",
    raw: unattributedRaw,
    impact: unattributedRaw * scale,
  });
  entries.push({
    id: "clip_adjustment",
    label: COMPONENT_LABELS.clip_adjustment,
    kind: "transform",
    raw: null,
    impact: final - preclip,
  });
  const reconciled = entries.reduce((sum, entry) => sum + entry.impact, 0);
  const entryGross = entries.reduce((sum, entry) => sum + Math.abs(entry.impact), 0);
  if (Math.abs(reconciled - final) > tolerance(entryGross)) {
    return { error: "Reward components do not reconcile to final reward." };
  }
  return { raw, preclip, final, entries };
}

function scopePoints(scope, history, selected) {
  if (scope === "step") return { points: [selected] };
  const episode = Number(selected.episode);
  const selectedSequence = Number(selected.sequence);
  const points = history
    .filter((point) => (
      Number(point?.episode) === episode
      && Number(point?.sequence) <= selectedSequence
    ))
    .sort((left, right) => Number(left.sequence) - Number(right.sequence));
  if (!points.some((point) => Number(point.sequence) === selectedSequence)) {
    return { error: "The selected transition is not retained in episode history." };
  }
  if (Number(points[0]?.step) !== 1) {
    return { partial: "Episode accounting is unavailable because history no longer starts at step 1." };
  }
  for (let index = 1; index < points.length; index += 1) {
    if (Number(points[index].step) !== Number(points[index - 1].step) + 1) {
      return { partial: "Episode accounting is unavailable because retained history has a step gap." };
    }
  }
  if (Number(points.at(-1)?.step) !== Number(selected.step)) {
    return { partial: "Episode accounting is unavailable because history stops before the selected step." };
  }
  return { points };
}

function aggregateLedgers(ledgers, selected, scope) {
  const rows = new Map();
  let raw = 0;
  let preclip = 0;
  let final = 0;
  let positive = 0;
  let negative = 0;
  for (const ledger of ledgers) {
    raw += ledger.raw;
    preclip += ledger.preclip;
    final += ledger.final;
    for (const entry of ledger.entries) {
      if (entry.impact > 0) positive += entry.impact;
      if (entry.impact < 0) negative += entry.impact;
      const current = rows.get(entry.id) || {
        id: entry.id,
        label: entry.label,
        kind: entry.kind,
        raw: entry.raw === null ? null : 0,
        impact: 0,
        magnitude: 0,
      };
      if (entry.raw !== null) current.raw += entry.raw;
      current.impact += entry.impact;
      current.magnitude += Math.abs(entry.impact);
      rows.set(entry.id, current);
    }
  }
  const gross = positive + Math.abs(negative);
  const epsilon = tolerance(gross);
  const signedAvailable = Math.abs(final) > epsilon;
  const magnitudeAvailable = gross > epsilon;
  const orderedRows = [
    ...COMPONENT_ORDER,
    "unattributed",
    "clip_adjustment",
  ].map((id) => rows.get(id)).filter(Boolean).filter((row) => (
    COMPONENT_ORDER.includes(row.id)
    || Math.abs(row.impact) > epsilon
    || row.magnitude > epsilon
  ));
  orderedRows.forEach((row) => {
    row.signedContribution = signedAvailable ? (100 * row.impact) / Math.abs(final) : null;
    row.magnitudeShare = magnitudeAvailable ? (100 * row.magnitude) / gross : null;
  });
  const authoritativeReturn = finite(selected?.return);
  if (
    scope === "episode"
    && (authoritativeReturn === null
      || Math.abs(final - authoritativeReturn) > tolerance(gross + Math.abs(authoritativeReturn || 0)))
  ) {
    return {
      error: authoritativeReturn === null
        ? "Authoritative episode return is missing or non-finite."
        : "Retained shaped rewards do not reconcile to the authoritative episode return.",
    };
  }
  return {
    raw,
    preclip,
    final,
    positive,
    negative,
    gross,
    rows: orderedRows,
    signedAvailable,
    magnitudeAvailable,
    authoritativeReturn,
  };
}

export function rewardBreakdownPresentation({
  snapshot,
  history = [],
  view = {},
  scope = "episode",
} = {}) {
  const normalizedScope = scope === "step" ? "step" : "episode";
  const contract = snapshot?.session?.reward_accounting;
  if (contract?.status === "unavailable") {
    return {
      status: "unavailable",
      scope: normalizedScope,
      message: `Reward accounting unavailable: ${contract.reason || "the source did not provide it"}.`,
    };
  }
  if (Number(snapshot?.protocol) < 8 || contract?.status !== "available") {
    return {
      status: "protocol-error",
      scope: normalizedScope,
      message: "Reward accounting requires playback protocol v8+ and an available transform contract.",
    };
  }
  const selected = selectedPoint(history, snapshot, view);
  if (!selected) {
    return {
      status: "not-yet-observed",
      scope: normalizedScope,
      message: "No transition has been observed at the selected cursor.",
    };
  }
  const scoped = scopePoints(normalizedScope, history, selected);
  if (scoped.partial) {
    return { status: "partial-history", scope: normalizedScope, message: scoped.partial };
  }
  if (scoped.error) {
    return { status: "protocol-error", scope: normalizedScope, message: scoped.error };
  }
  const ledgers = [];
  for (const point of scoped.points) {
    const ledger = pointLedger(point, contract);
    if (ledger.error) {
      return {
        status: "protocol-error",
        scope: normalizedScope,
        message: `Step ${point?.step ?? "?"}: ${ledger.error}`,
      };
    }
    ledgers.push(ledger);
  }
  const aggregate = aggregateLedgers(ledgers, selected, normalizedScope);
  if (aggregate.error) {
    return {
      status: "protocol-error",
      scope: normalizedScope,
      message: aggregate.error,
    };
  }
  return {
    status: "available",
    scope: normalizedScope,
    step: Number(selected.step),
    count: scoped.points.length,
    contract: {
      rewardScale: Number(contract.reward_scale),
      clipBounds: contract.clip_bounds,
    },
    ...aggregate,
  };
}

export function signedContributionLabel(value) {
  return value === null || value === undefined
    ? "N/A"
    : `${value >= 0 ? "+" : ""}${value.toFixed(1)}%`;
}

export function magnitudeShareLabel(value) {
  return value === null || value === undefined ? "N/A" : `${value.toFixed(1)}%`;
}
