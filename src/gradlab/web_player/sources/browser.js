const ICONS = "/assets/tabler-icons.svg";

function icon(name) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.classList.add("icon");
  svg.setAttribute("aria-hidden", "true");
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", `${ICONS}#ti-${name}`);
  svg.append(use);
  return svg;
}

function button(label, { iconName = null, quiet = false, primary = false } = {}) {
  const element = document.createElement("button");
  element.type = "button";
  element.className = [
    iconName ? "button-with-icon" : "",
    quiet ? "quiet" : "",
    primary ? "primary" : "",
  ].filter(Boolean).join(" ");
  if (iconName) element.append(icon(iconName));
  const text = document.createElement("span");
  text.textContent = label;
  element.append(text);
  return element;
}

export function formatDate(value, nowValue = Date.now()) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  const now = new Date(nowValue);
  if (
    !Number.isNaN(now.getTime())
    && date.getFullYear() === now.getFullYear()
    && date.getMonth() === now.getMonth()
    && date.getDate() === now.getDate()
  ) {
    const elapsedMilliseconds = now.getTime() - date.getTime();
    const absoluteMilliseconds = Math.abs(elapsedMilliseconds);
    const [amount, unit] = absoluteMilliseconds < 60_000
      ? [Math.floor(absoluteMilliseconds / 1_000), "second"]
      : absoluteMilliseconds < 3_600_000
        ? [Math.floor(absoluteMilliseconds / 60_000), "minute"]
        : [Math.floor(absoluteMilliseconds / 3_600_000), "hour"];
    const label = `${unit}${amount === 1 ? "" : "s"}`;
    return elapsedMilliseconds >= 0 ? `${amount} ${label} ago` : `in ${amount} ${label}`;
  }
  return date.toLocaleString();
}

export function formatCalendarDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  const months = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
  ];
  return `${date.getUTCDate()} ${months[date.getUTCMonth()]} ${date.getUTCFullYear()}`;
}

function formatGoalConfigurationDate(value, nowValue = Date.now()) {
  const date = new Date(value);
  const now = new Date(nowValue);
  if (Number.isNaN(date.getTime())) return "—";
  if (
    !Number.isNaN(now.getTime())
    && date.getFullYear() === now.getFullYear()
    && date.getMonth() === now.getMonth()
    && date.getDate() === now.getDate()
  ) {
    return formatDate(value, nowValue);
  }
  return formatCalendarDate(value);
}

const GOAL_CONFIGURATION_KINDS = {
  current_default: {
    label: "Current default",
    sourceLabel: "Current",
    behaviorLabel: "Default",
  },
  current_modified: {
    label: "Current modified",
    sourceLabel: "Current",
    behaviorLabel: "Launch override",
  },
  previous_default: {
    label: "Previous default",
    sourceLabel: "",
    behaviorLabel: "Default",
  },
  previous_modified: {
    label: "Previous modified",
    sourceLabel: "",
    behaviorLabel: "Launch override",
  },
};

const SUCCESS_BADGES = ["train/success", "eval/success"];

export function successBadgeLabels(item) {
  const badges = Array.isArray(item?.success_badges) ? item.success_badges : [];
  const present = new Set(badges.map((badge) => String(badge)));
  return SUCCESS_BADGES.filter((badge) => present.has(badge));
}

function renderSuccessBadges(item) {
  const labels = successBadgeLabels(item);
  if (!labels.length) return null;
  const badges = document.createElement("span");
  badges.className = "success-badges";
  labels.forEach((label) => {
    const badge = document.createElement("span");
    badge.className = `success-badge ${label.startsWith("train/") ? "training" : "evaluation"}`;
    badge.textContent = label;
    badge.title = label === "train/success"
      ? "A run reached this training goal's success condition"
      : "A verified evaluation reached this goal's acceptance condition";
    badges.append(badge);
  });
  return badges;
}

export function goalConfigurationPresentation(item, nowValue = Date.now()) {
  const kind = String(item?.configuration_kind || "previous_default");
  const kindPresentation = GOAL_CONFIGURATION_KINDS[kind] || {
    label: "Previous configuration",
    sourceLabel: "",
    behaviorLabel: "Configuration",
  };
  const runCount = Math.max(0, Number(item?.run_count) || 0);
  const runLabel = `${runCount.toLocaleString()} ${runCount === 1 ? "run" : "runs"}`;
  const firstUsed = item?.first_used_at
    ? formatGoalConfigurationDate(item.first_used_at, nowValue)
    : "—";
  const lastActivity = item?.last_activity_at
    ? formatGoalConfigurationDate(item.last_activity_at, nowValue)
    : "—";
  const comparisonAvailable = Boolean(item?.comparison_available);
  const rawDifferenceCount = item?.current_diff_count;
  const differenceCount = rawDifferenceCount === null || rawDifferenceCount === undefined
    ? null
    : Math.max(0, Number(rawDifferenceCount) || 0);
  const differenceCountExact = Boolean(item?.current_diff_count_exact);
  const differenceLabel = !comparisonAvailable
    ? "Exact diff unavailable"
    : differenceCount === null
      ? "Exact count unavailable"
      : `${differenceCount.toLocaleString()}${differenceCountExact ? "" : "+"} ${differenceCount === 1 ? "change" : "changes"}`;
  return {
    kind,
    kindLabel: kindPresentation.label,
    sourceLabel: kindPresentation.sourceLabel,
    behaviorLabel: kindPresentation.behaviorLabel,
    differenceCount,
    differenceCountExact,
    differenceLabel,
    comparisonAvailable,
    runCount,
    runLabel,
    firstUsedDate: firstUsed,
    lastActivityDate: lastActivity,
  };
}

export function formatGoalDiffValue(value, { unavailable = false } = {}) {
  if (unavailable) return "—";
  if (value === undefined) return "—";
  const rendered = JSON.stringify(value);
  return rendered === undefined ? String(value) : rendered;
}

function formatBytes(value) {
  const bytes = Number(value);
  if (!Number.isFinite(bytes) || bytes <= 0) return "—";
  const units = ["B", "KiB", "MiB", "GiB"];
  const unit = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
  return `${(bytes / (1024 ** unit)).toFixed(unit ? 1 : 0)} ${units[unit]}`;
}

export function runStatePresentation(item) {
  const state = String(item?.state || "").trim().toLowerCase();
  const stopReason = String(item?.stop_reason || "").trim();
  const earlyStopTrigger = String(item?.early_stop?.trigger || "").trim();
  if (
    (
      (state === "stopped" && stopReason.startsWith("early_stop_neutral:"))
      || (state === "failed" && stopReason.startsWith("early_stop_failure:"))
    )
    && earlyStopTrigger === "no_improvement"
  ) {
    return {
      iconName: "player-pause",
      tone: "stopped",
      label: "Training stalled",
    };
  }
  if (state === "finished" || state === "succeeded") {
    return { iconName: "check", tone: "finished", label: "Finished" };
  }
  if (state === "running") {
    return { iconName: "activity-heartbeat", tone: "running", label: "Running" };
  }
  if (["failed", "crashed", "resumable_failure"].includes(state)) {
    return {
      iconName: "x",
      tone: "failed",
      label: state === "crashed" ? "Crashed" : "Failed",
    };
  }
  if (["pending", "queued", "starting"].includes(state)) {
    return {
      iconName: "player-pause",
      tone: "pending",
      label: state ? humanizeMetricPart(state) : "Pending",
    };
  }
  if (
    ["killed", "cancelled", "canceled", "preempted", "interrupted"].includes(state)
  ) {
    return { iconName: "x", tone: "stopped", label: humanizeMetricPart(state) };
  }
  return {
    iconName: "activity-heartbeat",
    tone: "unknown",
    label: state ? humanizeMetricPart(state) : "Unknown",
  };
}

function humanizeMetricPart(value) {
  return String(value || "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function comparisonOperatorLabel(value) {
  return {
    ">": ">",
    ">=": "≥",
    "<": "<",
    "<=": "≤",
  }[String(value || "")] || String(value || "");
}

export function runFinishPresentation(item) {
  const reason = String(item?.stop_reason || "").trim();
  const finalStep = Number(item?.final_step);
  const stepDetail = Number.isSafeInteger(finalStep) && finalStep >= 0
    ? `Stopped at ${finalStep.toLocaleString()} steps`
    : "";
  const earlyStop = item?.early_stop && typeof item.early_stop === "object"
    ? item.early_stop
    : {};
  const condition = earlyStop.condition && typeof earlyStop.condition === "object"
    ? earlyStop.condition
    : {};
  const earlyStopMetric = String(earlyStop.metric || condition.metric || "").trim();
  const earlyStopOperator = String(condition.operator || "").trim();
  const earlyStopThreshold = condition.threshold;
  const hasThreshold = (
    earlyStopMetric
    && [">", ">=", "<", "<="].includes(earlyStopOperator)
    && earlyStopThreshold !== null
    && earlyStopThreshold !== undefined
    && Number.isFinite(Number(earlyStopThreshold))
  );
  const earlyStopCriterion = hasThreshold
    ? `${metricLabel(earlyStopMetric)} ${comparisonOperatorLabel(earlyStopOperator)} ${
        formatMetricValue(earlyStopMetric, earlyStopThreshold)
      }`
    : "";
  const observedValue = earlyStop.value;
  const observedDetail = (
    earlyStopCriterion
    && observedValue !== null
    && observedValue !== undefined
    && Number.isFinite(Number(observedValue))
  )
    ? `observed ${formatMetricValue(earlyStopMetric, observedValue)}`
    : "";
  const conditionId = String(earlyStop.condition_id || "").trim();
  const earlyStopDetail = [
    earlyStopCriterion || (conditionId ? humanizeMetricPart(conditionId) : ""),
    observedDetail,
    stepDetail,
  ].filter(Boolean).join(" · ");
  if (!reason) {
    const state = String(item?.state || "").trim().toLowerCase();
    return ["finished", "failed", "stopped", "crashed", "canceled", "cancelled", "killed"]
      .includes(state)
      ? {
          label: "Reason unavailable",
          detail: "This run has no projected terminal receipt.",
          tone: "unknown",
        }
      : { label: "—", detail: "", tone: "none" };
  }
  if (["eval_acceptance", "completed_after_eval_acceptance"].includes(reason)) {
    return {
      label: "Evaluation criteria met",
      detail: stepDetail,
      tone: "success",
    };
  }
  if (["deterministic_training_acceptance", "first_completion"].includes(reason)) {
    return {
      label: "Training success criterion met",
      detail: stepDetail,
      tone: "success",
    };
  }
  if (reason.startsWith("early_stop_success:")) {
    return {
      label: "Training target met",
      detail: earlyStopDetail || [
        humanizeMetricPart(reason.split(":", 2)[1]),
        stepDetail,
      ].filter(Boolean).join(" · "),
      evidence: hasThreshold
        ? {
            metric: metricLabel(earlyStopMetric),
            observed: (
              observedValue !== null
              && observedValue !== undefined
              && Number.isFinite(Number(observedValue))
            )
              ? formatMetricValue(earlyStopMetric, observedValue)
              : "—",
            required: (
              `${comparisonOperatorLabel(earlyStopOperator)} `
              + formatMetricValue(earlyStopMetric, earlyStopThreshold)
            ),
            step: stepDetail,
          }
        : null,
      tone: "success",
    };
  }
  if (reason.startsWith("early_stop_failure:")) {
    const stalled = String(item?.early_stop?.trigger || "") === "no_improvement";
    return {
      label: stalled ? "Training stalled" : "Training stop criterion met",
      detail: [humanizeMetricPart(reason.split(":", 2)[1]), stepDetail]
        .filter(Boolean)
        .join(" · "),
      tone: stalled ? "neutral" : "failure",
    };
  }
  if (reason.startsWith("early_stop_neutral:")) {
    return {
      label: "Training stalled",
      detail: [humanizeMetricPart(reason.split(":", 2)[1]), stepDetail]
        .filter(Boolean)
        .join(" · "),
      tone: "neutral",
    };
  }
  if (reason.startsWith("early_stop_success_without_acceptance:")) {
    return {
      label: "Training target met; evaluation not accepted",
      detail: [humanizeMetricPart(reason.split(":", 2)[1]), stepDetail]
        .filter(Boolean)
        .join(" · "),
      tone: "failure",
    };
  }
  if (reason === "training_cap_complete") {
    return {
      label: "Maximum timesteps reached",
      detail: stepDetail,
      tone: "neutral",
    };
  }
  if (reason === "training_cap_without_acceptance") {
    return {
      label: "Maximum timesteps; evaluation not accepted",
      detail: stepDetail,
      tone: "failure",
    };
  }
  if (reason === "canceled") {
    return { label: "Canceled", detail: stepDetail, tone: "neutral" };
  }
  const known = {
    evaluation_evidence_incomplete: "Evaluation evidence incomplete",
    pre_submit_failure: "Submission failed",
    supervisor_startup_failure: "Supervisor failed to start",
    supervisor_failure: "Supervisor failed",
    scratch_storage_above_80_percent: "Scratch storage limit reached",
  };
  return {
    label: known[reason] || humanizeMetricPart(reason),
    detail: stepDetail,
    tone: "failure",
  };
}

export function metricLabel(metric) {
  const name = String(metric || "");
  const known = {
    "leader/checkpoint/step": "Checkpoint step",
    "train/global_step": "Global step",
    "train/episode/return/shaped/origin/target/rolling/mean": "Recent target return mean",
    "train/outcome/success/starts/all/rolling/rate/min": "Recent all-start success rate min",
    "train/outcome/success/starts/all/rolling/rate/mean": "Recent all-start success rate mean",
    "eval/full/outcome/success/starts/rate/min": "Full-eval start success rate min",
    "eval/full/outcome/success/starts/rate/mean": "Full-eval start success rate mean",
    "eval/full/episode/return/shaped/mean": "Mean return",
    "eval/full/episode/return/shaped/max": "Best return",
  };
  if (known[name]) return known[name];
  const reason = name.match(/^eval\/full\/outcome\/reason\/([^/]+)\/rate$/);
  if (reason) return `${humanizeMetricPart(reason[1])} failure rate`;
  const progress = name.match(/^eval\/full\/progress\/([^/]+)\/(mean|max)$/);
  if (progress) {
    return `${humanizeMetricPart(progress[1])} ${progress[2]}`;
  }
  const trainingProgress = name.match(
    /^train\/progress\/([^/]+)\/origin\/target\/rolling\/mean$/,
  );
  if (trainingProgress) {
    return `Recent target ${humanizeMetricPart(trainingProgress[1]).toLowerCase()} mean`;
  }
  return name
    .replace(/^(eval\/full|leader|train)\//, "")
    .split("/")
    .map(humanizeMetricPart)
    .join(" · ");
}

export function formatMetricValue(metric, value) {
  if (value === null || value === undefined || value === "") return "—";
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "—";
  if (String(metric).includes("/rate/") || String(metric).endsWith("/rate")) {
    return `${(numeric * 100).toLocaleString(undefined, { maximumFractionDigits: 2 })}%`;
  }
  return numeric.toLocaleString(undefined, { maximumFractionDigits: 3 });
}

function compactRecipeOverride(value) {
  const text = String(value || "").trim();
  const separator = text.indexOf("=");
  if (separator < 0) return text;
  let key = text.slice(0, separator).trim();
  const overrideValue = text.slice(separator + 1).trim();
  for (const prefix of [
    "train.backend.config.",
    "train.environment.env_config.",
    "train.environment.task.",
    "train.",
  ]) {
    if (key.startsWith(prefix)) {
      key = key.slice(prefix.length);
      break;
    }
  }
  return `${key}=${overrideValue}`;
}

export function recipeVariantPresentation(item) {
  const overrides = Array.isArray(item?.recipe_overrides)
    ? item.recipe_overrides.map((value) => String(value || "").trim()).filter(Boolean)
    : [];
  const variantId = String(item?.recipe_variant_id || "").trim();
  if (!overrides.length) {
    if (variantId === "base") {
      return {
        summary: "base",
        detail: "Checked-in recipe with no launch-time configuration overrides.",
      };
    }
    return {
      summary: "variation unknown",
      detail: "This run predates explicit recipe-variation metadata.",
    };
  }
  const visible = overrides.slice(0, 2).map(compactRecipeOverride);
  if (overrides.length > visible.length) visible.push(`+${overrides.length - visible.length}`);
  if (variantId) visible.push(variantId);
  return {
    summary: visible.join(" · "),
    detail: [
      variantId ? `Recipe variant ${variantId}` : "Launch-time recipe overrides",
      ...overrides,
    ].join("\n"),
  };
}

export function sortRunItems(items, sort) {
  const metric = String(sort?.metric || "");
  const direction = sort?.direction === "ascending" ? "ascending" : "descending";
  if (!metric) return [...items];
  return items
    .map((item, index) => ({ item, index }))
    .sort((left, right) => {
      const leftRaw = left.item?.metrics?.[metric];
      const rightRaw = right.item?.metrics?.[metric];
      const leftValue = Number(leftRaw);
      const rightValue = Number(rightRaw);
      const leftMissing = leftRaw === null
        || leftRaw === undefined
        || !Number.isFinite(leftValue);
      const rightMissing = rightRaw === null
        || rightRaw === undefined
        || !Number.isFinite(rightValue);
      if (leftMissing !== rightMissing) return leftMissing ? 1 : -1;
      if (leftMissing) return left.index - right.index;
      const difference = direction === "ascending"
        ? leftValue - rightValue
        : rightValue - leftValue;
      return difference || left.index - right.index;
    })
    .map(({ item }) => item);
}

function hasCompleteRunMetrics(item, columns) {
  return columns.length > 0 && columns.every((column) => {
    const raw = item?.metrics?.[column.metric];
    return raw !== null && raw !== undefined && Number.isFinite(Number(raw));
  });
}

export function activeRunMetricColumns(items, primaryColumns, fallbackColumns = []) {
  const primary = Array.isArray(primaryColumns) ? primaryColumns : [];
  const fallback = Array.isArray(fallbackColumns) ? fallbackColumns : [];
  if (primary.some(Boolean) && items.some((item) => hasCompleteRunMetrics(item, primary))) {
    return primary;
  }
  return fallback.length ? fallback : primary;
}

export function availableRunMetricColumns(items, columns) {
  return (Array.isArray(columns) ? columns : []).filter((column) => (
    items.some((item) => {
      const value = item?.metrics?.[column.metric];
      return value !== null && value !== undefined && Number.isFinite(Number(value));
    })
  ));
}

export function rankRunItems(items, columns) {
  const criteria = Array.isArray(columns) ? columns : [];
  if (!criteria.length) return [...items];
  return items
    .map((item, index) => ({ item, index }))
    .sort((left, right) => {
      const leftComplete = hasCompleteRunMetrics(left.item, criteria);
      const rightComplete = hasCompleteRunMetrics(right.item, criteria);
      if (leftComplete !== rightComplete) return leftComplete ? -1 : 1;
      if (!leftComplete) return left.index - right.index;
      for (const criterion of criteria) {
        const leftValue = Number(left.item.metrics[criterion.metric]);
        const rightValue = Number(right.item.metrics[criterion.metric]);
        const difference = criterion.direction === "min"
          ? leftValue - rightValue
          : rightValue - leftValue;
        if (difference) return difference;
      }
      return left.index - right.index;
    })
    .map(({ item }) => item);
}

export function bestRunEfficiency(items, primaryColumns, fallbackColumns = []) {
  const columns = activeRunMetricColumns(items, primaryColumns, fallbackColumns);
  const ranked = rankRunItems(items, columns);
  if (!ranked.length || !hasCompleteRunMetrics(ranked[0], columns)) return null;
  const usesPrimary = columns === primaryColumns;
  const evaluated = usesPrimary && columns.some(
    (column) => /^(eval\/|leader\/)/.test(String(column.metric || "")),
  );
  return {
    item: ranked[0],
    columns,
    evidence: evaluated ? "evaluation" : "training",
  };
}

export function checkpointPlaybackSeed(item) {
  const value = item?.playback_seed;
  if (value === null || value === undefined || value === "") return null;
  const seed = Number(value);
  return Number.isSafeInteger(seed) && seed >= 0 ? seed : null;
}

export function checkpointCanEvaluate(item) {
  const queueState = String(item?.evaluation_queue?.state || "");
  const evaluation = item?.evaluation;
  const hasAcceptedEvaluation = (
    evaluation
    && typeof evaluation === "object"
    && (
      evaluation.pass === true
      || String(evaluation.status || "") === "accepted"
    )
  );
  return (
    !hasAcceptedEvaluation
    && ![
      "queued",
      "running",
      "retry_wait",
      "waiting_for_training_terminal",
      "waiting_for_run_lease",
      "submitted",
      "submission_uncertain",
      "awaiting_projection",
      "flusher_unavailable",
      "accepted",
      "rejected",
      "blocked",
      "failed",
      "expired",
      "canceled",
    ].includes(queueState)
  );
}

export function checkpointMetricIsBest(item, metric) {
  return (
    Array.isArray(item?.best_metrics)
    && item.best_metrics.includes(metric)
  );
}

export function checkpointMetricRoleLabel(column) {
  const roles = new Set(Array.isArray(column?.roles) ? column.roles : []);
  if (roles.has("objective") && roles.has("acceptance")) return "Objective · gate";
  if (roles.has("objective")) return "Objective";
  if (roles.has("tie_breaker")) return "Tie-breaker";
  if (roles.has("acceptance")) return "Acceptance";
  if (roles.has("training_proxy")) return "Training proxy";
  if (roles.has("optimization")) return "Optimization";
  return "";
}

export function checkpointMetricHeaderLabel(column) {
  const metric = String(column?.metric || "");
  const evidence = (
    column?.evidence === "evaluation"
    || /^(eval\/|leader\/)/.test(metric)
  ) ? "Eval" : "Train";
  if (/\/outcome\/success\//.test(metric)) return `${evidence} success`;
  if (/\/episode\/return\//.test(metric)) return `${evidence} return`;
  const reason = metric.match(/\/outcome\/reason\/([^/]+)\/rate$/);
  if (reason) return `${evidence} ${humanizeMetricPart(reason[1]).toLowerCase()}`;
  const progress = metric.match(/\/progress\/([^/]+)\//);
  if (progress) return `${evidence} ${humanizeMetricPart(progress[1]).toLowerCase()}`;
  if (metric === "leader/checkpoint/step") return "Checkpoint step";
  return column?.label || metricLabel(metric);
}

export function checkpointMetricDescription(column) {
  const role = checkpointMetricRoleLabel(column);
  const evidence = column?.evidence === "evaluation"
    ? "Frozen checkpoint-evaluation evidence"
    : Array.isArray(column?.roles) && column.roles.includes("training_proxy")
      ? "Diagnostic online training proxy; checkpoint evaluation remains authoritative"
      : "Diagnostic online training evidence";
  const direction = column?.direction === "min"
    ? "Lower is better"
    : column?.direction === "max"
      ? "Higher is better"
      : "No single better direction";
  return [role, evidence, direction].filter(Boolean).join(" · ");
}

export function checkpointMetricBestBadge() {
  return "Best";
}

export function checkpointEvidencePresentation(item) {
  const evaluation = item?.evaluation || {};
  const accepted = evaluation.pass === true || String(evaluation.status || "") === "accepted";
  const best = Array.isArray(item?.best_metrics) ? item.best_metrics : [];
  const hasEvaluationLead = best.some((metric) => /^(eval\/|leader\/)/.test(String(metric)));
  const hasTrainingLead = best.some((metric) => String(metric).startsWith("train/"));
  if (item?.promoted) {
    return { label: "Promoted", detail: "Authoritative promoted checkpoint", tone: "evaluation", rank: 0 };
  }
  if (accepted) {
    return { label: "Accepted", detail: "Accepted checkpoint evaluation", tone: "evaluation", rank: 1 };
  }
  if (hasEvaluationLead) {
    return { label: "Evaluation lead", detail: "Best available frozen evaluation evidence", tone: "evaluation", rank: 2 };
  }
  if (hasTrainingLead) {
    return { label: "Training lead", detail: "Diagnostic training evidence; evaluation remains authoritative", tone: "training", rank: 3 };
  }
  if (/final/i.test(String(item?.purpose || ""))) {
    return { label: "Final checkpoint", detail: "Final checkpoint with no stronger evaluation signal", tone: "neutral", rank: 4 };
  }
  return { label: "Available", detail: "Playable checkpoint; no acceptance claim", tone: "neutral", rank: 5 };
}

export function checkpointRecommendation(items) {
  const ranked = (Array.isArray(items) ? items : [])
    .map((item, index) => ({ item, index, evidence: checkpointEvidencePresentation(item) }))
    .sort((left, right) => (
      left.evidence.rank - right.evidence.rank
      || (Number(right.item?.step) || 0) - (Number(left.item?.step) || 0)
      || left.index - right.index
    ));
  return ranked[0] || null;
}

export function checkpointRecommendationMetrics(item, columns) {
  return (Array.isArray(columns) ? columns : [])
    .filter((column) => {
      const value = item?.metrics?.[column.metric];
      return value !== null && value !== undefined && Number.isFinite(Number(value));
    })
    .map((column) => ({
      metric: String(column.metric),
      label: column.label || metricLabel(column.metric),
      role: checkpointMetricRoleLabel(column),
      tone: column.evidence || "training",
      value: formatMetricValue(column.metric, item.metrics[column.metric]),
    }));
}

export function humanSourceLabel(value, kind) {
  const raw = String(value || "").trim();
  if (!raw) return "";
  if (kind === "environment") {
    const base = raw
      .replace(/-v\d+$/i, "")
      .replace(/-(nes|snes|genesis|atari\d*)$/i, "");
    if (/^vizdoom$/i.test(base)) return "ViZDoom";
    return base
      .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
      .replaceAll("-", " ");
  }
  if (kind === "goal") {
    return raw
      .replace(/-v\d+$/i, "")
      .replace(/^Level(?=\d)/i, "Level ")
      .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
      .replaceAll("_", " ");
  }
  if (kind === "variant") return "Goal configuration";
  if (kind === "run") return "Run";
  if (kind === "checkpoint") {
    const match = raw.match(/^checkpoint-(\d+)-/);
    return match ? `Checkpoint · ${Number(match[1]).toLocaleString()} steps` : "Checkpoint";
  }
  return raw;
}

export function catalogItemMatchesSearch(item, query) {
  const normalized = String(query || "").trim().toLocaleLowerCase();
  if (!normalized) return true;
  return JSON.stringify(item ?? {}).toLocaleLowerCase().includes(normalized);
}

export function environmentEvidenceRank(item) {
  const badges = successBadgeLabels(item);
  if (badges.includes("eval/success")) return 0;
  if (badges.includes("train/success")) return 1;
  return 2;
}

function decodePathPart(value) {
  try {
    return decodeURIComponent(value);
  } catch {
    return "";
  }
}

export function sourceRouteFromPath(pathname = location.pathname) {
  const parts = String(pathname || "").split("/").filter(Boolean);
  if (!parts.length) {
    return {
      level: "environments",
      environment_id: "",
      goal_id: "",
      goal_variant_id: "",
      run_id: "",
      checkpoint_id: "",
    };
  }
  if (parts[0] !== "environments" || !parts[1]) return null;
  const environment_id = decodePathPart(parts[1]);
  if (!environment_id) return null;
  if (parts.length === 2) {
    return {
      level: "goals",
      environment_id,
      goal_id: "",
      goal_variant_id: "",
      run_id: "",
      checkpoint_id: "",
    };
  }
  if (parts[2] !== "goals" || !parts[3]) return null;
  const goal_id = decodePathPart(parts[3]);
  if (!goal_id) return null;
  if (parts.length === 4) {
    return {
      level: "goal_variants",
      environment_id,
      goal_id,
      goal_variant_id: "",
      run_id: "",
      checkpoint_id: "",
    };
  }
  if (parts[4] !== "variants" || !parts[5]) return null;
  const goal_variant_id = decodePathPart(parts[5]);
  if (!goal_variant_id) return null;
  if (parts.length === 6) {
    return {
      level: "runs",
      environment_id,
      goal_id,
      goal_variant_id,
      run_id: "",
      checkpoint_id: "",
    };
  }
  if (parts[6] !== "runs" || !parts[7]) return null;
  const run_id = decodePathPart(parts[7]);
  if (!run_id) return null;
  if (parts.length === 8) {
    return {
      level: "runs",
      environment_id,
      goal_id,
      goal_variant_id,
      run_id,
      checkpoint_id: "",
    };
  }
  if (parts.length !== 10 || parts[8] !== "checkpoints" || !parts[9]) return null;
  const checkpoint_id = decodePathPart(parts[9]);
  if (!checkpoint_id) return null;
  return {
    level: "runs",
    environment_id,
    goal_id,
    goal_variant_id,
    run_id,
    checkpoint_id,
  };
}

export function sourceRoutePath(route) {
  const environmentId = String(route?.environment_id || "").trim();
  const goalId = String(route?.goal_id || "").trim();
  const goalVariantId = String(route?.goal_variant_id || "").trim();
  const runId = String(route?.run_id || "").trim();
  const checkpointId = String(route?.checkpoint_id || "").trim();
  if (!environmentId) return "/";
  let path = `/environments/${encodeURIComponent(environmentId)}`;
  if (!goalId) return path;
  path += `/goals/${encodeURIComponent(goalId)}`;
  if (!goalVariantId) return path;
  path += `/variants/${encodeURIComponent(goalVariantId)}`;
  if (!runId) return path;
  path += `/runs/${encodeURIComponent(runId)}`;
  if (!checkpointId) return path;
  return `${path}/checkpoints/${encodeURIComponent(checkpointId)}`;
}

function routeSignature(route) {
  return JSON.stringify({
    level: route?.level || "environments",
    environment_id: route?.environment_id || "",
    goal_id: route?.goal_id || "",
    goal_variant_id: route?.goal_variant_id || "",
    run_id: route?.run_id || "",
    checkpoint_id: route?.checkpoint_id || "",
  });
}

function activeCheckpointCacheKey(route) {
  return JSON.stringify({
    run_id: route?.run_id || "",
    goal_variant_id: route?.goal_variant_id || "",
  });
}

export function checkpointNavigationPresentation(items, checkpointId) {
  const checkpoints = [...new Map(
    (Array.isArray(items) ? items : [])
      .filter((item) => item && String(item.checkpoint_id || ""))
      .map((item) => [String(item.checkpoint_id), item]),
  ).values()].sort((left, right) => (
    (Number(left.step) || 0) - (Number(right.step) || 0)
    || String(left.sha256 || "").localeCompare(String(right.sha256 || ""))
    || String(left.checkpoint_id || "").localeCompare(String(right.checkpoint_id || ""))
  ));
  const currentIndex = checkpoints.findIndex(
    (item) => String(item.checkpoint_id) === String(checkpointId || ""),
  );
  return {
    count: checkpoints.length,
    position: currentIndex < 0 ? null : currentIndex + 1,
    current: currentIndex < 0 ? null : checkpoints[currentIndex],
    previous: currentIndex > 0 ? checkpoints[currentIndex - 1] : null,
    next: currentIndex >= 0 && currentIndex + 1 < checkpoints.length
      ? checkpoints[currentIndex + 1]
      : null,
  };
}

export function checkpointPrefetchSources(items, checkpointId) {
  const presentation = checkpointNavigationPresentation(items, checkpointId);
  return [presentation.previous, presentation.next]
    .filter((item) => item && String(item.manifest_url || ""))
    .map((item) => ({
      kind: "public_run",
      value: String(item.manifest_url),
      run_id: String(item.run_id || ""),
      checkpoint_id: String(item.checkpoint_id || ""),
    }));
}

export function sourceBreadcrumbItems(route) {
  const hasActiveCheckpoint = Boolean(route?.checkpoint_id);
  const items = [{
    label: "Environments",
    current: route?.level === "environments" && !hasActiveCheckpoint,
    route: {
      level: "environments",
      environment_id: "",
      goal_id: "",
      goal_variant_id: "",
      run_id: "",
      checkpoint_id: "",
    },
  }];
  if (route?.environment_id) {
    items.push({
      label: humanSourceLabel(route.environment_id, "environment"),
      title: route.environment_id,
      current: route.level === "goals" && !hasActiveCheckpoint,
      route: {
        level: "goals",
        goal_id: "",
        goal_variant_id: "",
        run_id: "",
        checkpoint_id: "",
      },
    });
  }
  if (route?.goal_id) {
    items.push({
      label: humanSourceLabel(route.goal_id, "goal"),
      title: route.goal_id,
      current: route.level === "goal_variants" && !hasActiveCheckpoint,
      route: {
        level: "goal_variants",
        goal_variant_id: "",
        run_id: "",
        checkpoint_id: "",
      },
    });
  }
  if (route?.goal_variant_id) {
    items.push({
      label: humanSourceLabel(route.goal_variant_id, "variant"),
      title: route.goal_variant_id,
      current: route.level === "runs" && !route.run_id && !hasActiveCheckpoint,
      route: {
        level: "runs",
        run_id: "",
        checkpoint_id: "",
      },
    });
  }
  if (route?.run_id) {
    items.push({
      label: humanSourceLabel(route.run_id, "run"),
      title: route.run_id,
      current: !route.checkpoint_id,
      route: {
        level: "runs",
        checkpoint_id: "",
      },
    });
  }
  if (route?.checkpoint_id) {
    items.push({
      label: humanSourceLabel(route.checkpoint_id, "checkpoint"),
      title: route.checkpoint_id,
      current: true,
      route: null,
    });
  }
  return items;
}

export class SourceBrowser {
  constructor(
    root,
    breadcrumbsRoot,
    {
      token,
      command,
      getState,
      showToast,
      checkpointNavigationRoot = null,
      beginCheckpointLoad = null,
      openInspection,
      openSourceRoute,
      resumePlayback = null,
      catalogRequestTimeoutMs = 30_000,
    },
  ) {
    this.root = root;
    this.breadcrumbsRoot = breadcrumbsRoot;
    this.token = token;
    this.command = command;
    this.getState = getState;
    this.showToast = showToast;
    this.checkpointNavigationRoot = checkpointNavigationRoot;
    this.checkpointPrevious = checkpointNavigationRoot?.querySelector(
      "[data-checkpoint-previous]",
    ) || null;
    this.checkpointNext = checkpointNavigationRoot?.querySelector(
      "[data-checkpoint-next]",
    ) || null;
    this.checkpointPosition = checkpointNavigationRoot?.querySelector(
      "[data-checkpoint-position]",
    ) || null;
    this.beginCheckpointLoad = beginCheckpointLoad;
    this.openInspection = openInspection;
    this.openSourceRoute = openSourceRoute;
    this.resumePlayback = resumePlayback;
    this.route = {
      level: "environments",
      environment_id: "",
      goal_id: "",
      goal_variant_id: "",
      run_id: "",
      checkpoint_id: "",
    };
    this.query = "";
    this.items = [];
    this.sourceItems = [];
    this.metricColumns = [];
    this.fallbackMetricColumns = [];
    this.sort = { metric: "", direction: "" };
    this.nextCursor = null;
    this.freshness = "fresh";
    this.catalogWarnings = [];
    this.catalogSource = null;
    this.generatedAt = null;
    this.selectionFence = "";
    this.loading = false;
    this.error = "";
    this.app = { phase: "selecting" };
    this.lastAppRoute = "";
    this.searchOpen = false;
    this.loadedKey = "";
    this.requestSerial = 0;
    this.requestController = null;
    this.checkpointTrainingController = null;
    this.checkpointTrainingSerial = 0;
    this.loadingKey = "";
    this.catalogRequestTimeoutMs = catalogRequestTimeoutMs;
    this.searchTimer = null;
    this.pollTimer = null;
    this.selectedCheckpoints = new Set();
    this.evaluating = false;
    this.selectedGoalVariantId = "";
    this.goalConfigurationsExpanded = false;
    this.goalVariantDiff = null;
    this.goalVariantDiffController = null;
    this.goalVariantDiffSerial = 0;
    this.goalVariantRunPages = new Map();
    this.goalVariantRunSort = new Map();
    this.activityRevision = "";
    this.activityHasActiveRuns = false;
    this.autoSelectedRoute = "";
    this.playbackRoute = null;
    this.activeBreadcrumbRoute = "";
    this.activeCheckpointCache = new Map();
    this.activeCheckpointController = null;
    this.activeCheckpointRequestSerial = 0;
    this.activeCheckpointError = "";
    this.activeCheckpointPendingId = "";
    this.adjacentPrefetchKey = "";
    this.initialEnvironmentCatalog = null;
    this.initialCatalogConsumed = false;
    this.historyEnabled = (
      location.pathname === "/"
      || location.pathname.startsWith("/environments/")
    );
    this.pendingLocationRoute = this.historyEnabled
      ? sourceRouteFromPath(location.pathname)
      : null;
    this.onPopState = () => {
      const parsedRoute = sourceRouteFromPath(location.pathname);
      if (!parsedRoute) return;
      this.navigate(parsedRoute, { historyMode: null });
    };
    this.checkpointPrevious?.addEventListener("click", () => {
      this.selectAdjacentCheckpoint("previous");
    });
    this.checkpointNext?.addEventListener("click", () => {
      this.selectAdjacentCheckpoint("next");
    });
    if (this.historyEnabled) window.addEventListener("popstate", this.onPopState);
  }

  render(snapshot) {
    this.hideActiveCheckpointNavigation();
    this.app = snapshot?.app || { phase: "active" };
    if (
      !this.initialCatalogConsumed
      && this.app.catalog
      && typeof this.app.catalog === "object"
    ) {
      this.initialEnvironmentCatalog = this.app.catalog;
    }
    const appRoute = this.app.route || {};
    if (
      this.pendingLocationRoute
      && location.pathname !== "/"
      && this.app.phase === "selecting"
      && !this.app.source
    ) {
      const pending = { ...this.pendingLocationRoute };
      this.pendingLocationRoute = null;
      this.lastAppRoute = routeSignature(appRoute);
      this.applyRoute(pending);
      this.command("browse_sources", { route: { ...pending } });
    } else {
      this.pendingLocationRoute = null;
    }
    const signature = routeSignature(appRoute);
    if (signature !== this.lastAppRoute) {
      this.lastAppRoute = signature;
      this.route = {
        level: appRoute.level || "environments",
        environment_id: appRoute.environment_id || "",
        goal_id: appRoute.goal_id || "",
        goal_variant_id: appRoute.goal_variant_id || "",
        run_id: appRoute.run_id || "",
        checkpoint_id: appRoute.checkpoint_id || "",
      };
      this.query = "";
      this.searchOpen = false;
      this.items = [];
      this.sourceItems = [];
      this.metricColumns = [];
      this.fallbackMetricColumns = [];
      this.sort = { metric: "", direction: "" };
      this.nextCursor = null;
      this.freshness = "fresh";
      this.catalogWarnings = [];
      this.catalogSource = null;
      this.generatedAt = null;
      this.selectionFence = "";
      this.checkpointTrainingController?.abort();
      this.checkpointTrainingController = null;
      this.checkpointTrainingSerial += 1;
      this.loadedKey = "";
      this.error = "";
      this.selectedCheckpoints.clear();
      this.resetGoalVariantDetail();
      this.autoSelectedRoute = "";
      this.syncUrl("replace");
    }
    this.hydrateInitialEnvironments();
    this.renderView();
    if (this.app.phase === "selecting") this.ensureLoaded();
    this.updatePolling();
  }

  stop({ preserveBreadcrumbs = false, preserveCheckpointNavigation = false } = {}) {
    clearTimeout(this.searchTimer);
    clearInterval(this.pollTimer);
    this.pollTimer = null;
    this.requestController?.abort();
    this.requestController = null;
    this.requestSerial += 1;
    this.checkpointTrainingController?.abort();
    this.checkpointTrainingController = null;
    this.checkpointTrainingSerial += 1;
    this.goalVariantDiffController?.abort();
    this.goalVariantDiffController = null;
    this.goalVariantDiffSerial += 1;
    this.loading = false;
    this.loadingKey = "";
    if (!preserveCheckpointNavigation) {
      this.activeCheckpointController?.abort();
      this.activeCheckpointController = null;
      this.activeCheckpointRequestSerial += 1;
      this.hideActiveCheckpointNavigation();
    }
    if (!preserveBreadcrumbs) {
      this.activeBreadcrumbRoute = "";
      this.breadcrumbsRoot.replaceChildren();
      this.breadcrumbsRoot.hidden = true;
    }
  }

  renderActiveBreadcrumbs(snapshot) {
    const app = snapshot?.app || {};
    const route = app.route || {};
    const signature = routeSignature(route);
    if (
      route.checkpoint_id
      && signature === this.activeBreadcrumbRoute
    ) {
      this.renderActiveCheckpointNavigation(route);
      return;
    }
    this.stop({ preserveBreadcrumbs: true, preserveCheckpointNavigation: true });
    this.app = app;
    if (!route.checkpoint_id) {
      this.activeBreadcrumbRoute = "";
      this.breadcrumbsRoot.replaceChildren();
      this.breadcrumbsRoot.hidden = true;
      this.hideActiveCheckpointNavigation();
      return;
    }
    this.route = {
      level: route.level || "runs",
      environment_id: route.environment_id || "",
      goal_id: route.goal_id || "",
      goal_variant_id: route.goal_variant_id || "",
      run_id: route.run_id || "",
      checkpoint_id: route.checkpoint_id || "",
    };
    this.playbackRoute = { ...this.route };
    this.activeCheckpointPendingId = "";
    this.activeCheckpointError = "";
    this.activeBreadcrumbRoute = signature;
    this.renderBreadcrumbs(this.breadcrumbsRoot);
    this.breadcrumbsRoot.hidden = false;
    this.renderActiveCheckpointNavigation(this.route);
    this.syncUrl("replace");
    void this.loadActiveCheckpointNavigation(this.route, signature);
  }

  hideActiveCheckpointNavigation() {
    if (this.checkpointNavigationRoot) this.checkpointNavigationRoot.hidden = true;
  }

  activeCheckpointItems(route = this.route) {
    return this.activeCheckpointCache.get(activeCheckpointCacheKey(route)) || null;
  }

  renderActiveCheckpointNavigation(route = this.route) {
    const root = this.checkpointNavigationRoot;
    if (!root || !route?.checkpoint_id || !route?.run_id) {
      this.hideActiveCheckpointNavigation();
      return;
    }
    root.hidden = false;
    const items = this.activeCheckpointItems(route);
    const presentation = checkpointNavigationPresentation(items, route.checkpoint_id);
    const pending = Boolean(this.activeCheckpointPendingId);
    const loading = items === null && !this.activeCheckpointError;
    const unavailable = !loading && presentation.position === null;
    root.classList.toggle("warning", Boolean(this.activeCheckpointError || unavailable));
    if (this.checkpointPosition) {
      this.checkpointPosition.textContent = loading
        ? "… / …"
        : presentation.position === null
          ? `— / ${presentation.count.toLocaleString()}`
          : `${presentation.position.toLocaleString()} / ${presentation.count.toLocaleString()}`;
      this.checkpointPosition.title = this.activeCheckpointError
        || String(presentation.current?.checkpoint_id || route.checkpoint_id);
    }
    const canChange = this.hasControl() && !pending;
    if (this.checkpointPrevious) {
      this.checkpointPrevious.disabled = !canChange || !presentation.previous;
      this.checkpointPrevious.title = presentation.previous
        ? `Previous checkpoint · step ${Number(presentation.previous.step).toLocaleString()}`
        : loading ? "Loading checkpoints" : "This is the first checkpoint";
    }
    if (this.checkpointNext) {
      this.checkpointNext.disabled = !canChange || !presentation.next;
      this.checkpointNext.title = presentation.next
        ? `Next checkpoint · step ${Number(presentation.next.step).toLocaleString()}`
        : loading ? "Loading checkpoints" : "This is the latest checkpoint";
    }
  }

  async loadActiveCheckpointNavigation(route, expectedSignature) {
    const requestRoute = { ...route };
    const cacheKey = activeCheckpointCacheKey(requestRoute);
    this.activeCheckpointController?.abort();
    const controller = new AbortController();
    this.activeCheckpointController = controller;
    const serial = ++this.activeCheckpointRequestSerial;
    const query = new URLSearchParams();
    if (requestRoute.goal_variant_id) {
      query.set("goal_variant_id", requestRoute.goal_variant_id);
    }
    try {
      const response = await fetch(
        `/api/catalog/runs/${encodeURIComponent(requestRoute.run_id)}/checkpoints?${query}`,
        {
          headers: { Authorization: `Bearer ${this.token}` },
          cache: "no-store",
          signal: controller.signal,
        },
      );
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload.error || `Checkpoint navigation failed (${response.status})`);
      }
      if (serial !== this.activeCheckpointRequestSerial) return;
      this.activeCheckpointCache.set(
        cacheKey,
        Array.isArray(payload.items) ? payload.items : [],
      );
      this.activeCheckpointError = "";
      this.prefetchAdjacentCheckpoints(requestRoute);
    } catch (error) {
      if (controller.signal.aborted || serial !== this.activeCheckpointRequestSerial) return;
      this.activeCheckpointError = String(error?.message || error);
    } finally {
      if (serial === this.activeCheckpointRequestSerial) {
        this.activeCheckpointController = null;
        if (expectedSignature === this.activeBreadcrumbRoute) {
          this.renderActiveCheckpointNavigation(this.route);
        }
      }
    }
  }

  selectAdjacentCheckpoint(direction) {
    const presentation = checkpointNavigationPresentation(
      this.activeCheckpointItems(this.route),
      this.route.checkpoint_id,
    );
    const item = direction === "previous" ? presentation.previous : presentation.next;
    if (!item || this.activeCheckpointPendingId) return false;
    const commandId = this.selectCheckpoint(item);
    if (!commandId) return false;
    this.activeCheckpointPendingId = String(item.checkpoint_id || "");
    this.beginCheckpointLoad?.({
      commandId,
      checkpointId: this.activeCheckpointPendingId,
    });
    this.renderActiveCheckpointNavigation(this.route);
    return true;
  }

  prefetchAdjacentCheckpoints(route = this.route) {
    if (!this.hasControl()) return false;
    const sources = checkpointPrefetchSources(
      this.activeCheckpointItems(route),
      route?.checkpoint_id,
    );
    if (!sources.length) return false;
    const key = JSON.stringify(sources.map((source) => source.value));
    if (key === this.adjacentPrefetchKey) return false;
    const commandId = this.command("prefetch_sources", { sources });
    if (commandId === null) return false;
    this.adjacentPrefetchKey = key;
    return true;
  }

  hasControl() {
    return Boolean(this.getState()?.hasControl);
  }

  currentPlaybackRoute() {
    const retainedRoute = this.getState()?.backgroundPlaybackSnapshot?.app?.route;
    const route = [retainedRoute, this.playbackRoute, this.app?.route]
      .find((candidate) => candidate?.checkpoint_id);
    return route ? { ...route } : null;
  }

  resetGoalVariantDetail() {
    this.goalVariantDiffController?.abort();
    this.goalVariantDiffController = null;
    this.goalVariantDiffSerial += 1;
    this.selectedGoalVariantId = "";
    this.goalConfigurationsExpanded = false;
    this.goalVariantDiff = null;
    this.goalVariantRunPages.clear();
    this.goalVariantRunSort.clear();
    this.activityRevision = "";
    this.activityHasActiveRuns = false;
  }

  inspectGoal(goal) {
    const base = (
      `/api/catalog/environments/${encodeURIComponent(this.route.environment_id)}`
      + `/goals/${encodeURIComponent(goal.goal_id)}`
    );
    return this.openInspection(`${base}/inspection`, {
      preferredDocument: "goal",
      recipesEndpoint: `${base}/recipes`,
      recipeEndpoint: (recipeId) => (
        `${base}/recipes/${encodeURIComponent(recipeId)}/inspection`
      ),
    });
  }

  inspectGoalVariant(variant) {
    return this.openInspection(
      `/api/catalog/environments/${encodeURIComponent(this.route.environment_id)}`
      + `/goals/${encodeURIComponent(this.route.goal_id)}`
      + `/variants/${encodeURIComponent(variant.variant_id)}/inspection`,
      { preferredDocument: "goal" },
    );
  }

  goalVariantDiffFromActivity(variant) {
    const variantId = String(variant?.variant_id || "");
    if (!variant?.comparison_available) {
      return {
        variantId,
        state: "ready",
        availability: "unavailable",
        changeCount: null,
        entries: [],
        message: "The live goal activity has no exact comparison proof.",
      };
    }
    const entries = Array.isArray(variant.current_diff) ? variant.current_diff : [];
    const count = Number(variant.current_diff_count);
    return {
      variantId,
      state: "ready",
      availability: "exact",
      changeCount: variant.current_diff_count_exact && Number.isFinite(count)
        ? Math.max(0, count)
        : entries.length,
      entries,
      message: variant.current_diff_truncated
        ? "The live activity response contains the first exact differences only. View YAML for the complete contract."
        : "",
    };
  }

  selectGoalVariant(variant) {
    const variantId = String(variant?.variant_id || "");
    if (!variantId) return;
    if (this.selectedGoalVariantId !== variantId) {
      this.goalVariantDiffController?.abort();
      this.goalVariantDiffController = null;
      this.goalVariantDiffSerial += 1;
      this.selectedGoalVariantId = variantId;
      this.goalVariantDiff = this.goalVariantDiffFromActivity(variant);
    }
    this.goalConfigurationsExpanded = false;
    this.renderView();
  }

  inspectRun(runId = this.route.run_id) {
    return this.openInspection(
      `/api/catalog/runs/${encodeURIComponent(runId)}/inspection`,
      { preferredDocument: "recipe" },
    );
  }

  routeKey() {
    return `${routeSignature(this.route)}:${this.query.trim().toLocaleLowerCase()}`;
  }

  hydrateInitialEnvironments() {
    const catalog = this.initialEnvironmentCatalog;
    if (
      !catalog
      || this.route.level !== "environments"
      || this.query.trim()
      || this.loadedKey
    ) {
      return false;
    }
    this.initialCatalogConsumed = true;
    this.initialEnvironmentCatalog = null;
    this.sourceItems = Array.isArray(catalog.items) ? [...catalog.items] : [];
    this.items = [...this.sourceItems];
    this.metricColumns = Array.isArray(catalog.metric_columns)
      ? [...catalog.metric_columns]
      : [];
    this.fallbackMetricColumns = Array.isArray(catalog.fallback_metric_columns)
      ? [...catalog.fallback_metric_columns]
      : [];
    this.nextCursor = catalog.next_cursor || null;
    this.freshness = "partial";
    this.catalogWarnings = Array.isArray(catalog.warnings) ? catalog.warnings : [];
    this.error = "";
    return true;
  }

  endpoint(cursor = null, { force = false } = {}) {
    const query = new URLSearchParams();
    if (this.query.trim()) query.set("q", this.query.trim());
    if (cursor) query.set("cursor", cursor);
    if (force) query.set("refresh", "1");
    if (this.route.level === "goals") {
      return `/api/catalog/environments/${encodeURIComponent(this.route.environment_id)}/goals?${query}`;
    }
    if (this.route.level === "goal_variants") {
      return `/api/catalog/environments/${encodeURIComponent(this.route.environment_id)}/goals/${encodeURIComponent(this.route.goal_id)}/activity?${query}`;
    }
    if (this.route.level === "runs" && this.route.run_id) {
      if (this.route.goal_variant_id) {
        query.set("goal_variant_id", this.route.goal_variant_id);
      }
      return `/api/catalog/runs/${encodeURIComponent(this.route.run_id)}/checkpoints?${query}`;
    }
    if (this.route.level === "runs") {
      return `/api/catalog/environments/${encodeURIComponent(this.route.environment_id)}/goals/${encodeURIComponent(this.route.goal_id)}/variants/${encodeURIComponent(this.route.goal_variant_id)}/runs?${query}`;
    }
    return `/api/catalog/environments?${query}`;
  }

  async ensureLoaded() {
    const key = this.routeKey();
    if ((this.loading && this.loadingKey === key) || this.loadedKey === key) return;
    await this.load();
  }

  async load({ append = false, quiet = false, force = false } = {}) {
    const key = this.routeKey();
    if (this.loading && this.loadingKey === key) return;
    this.requestController?.abort();
    const controller = new AbortController();
    this.requestController = controller;
    const cursor = append ? this.nextCursor : null;
    const serial = ++this.requestSerial;
    this.loading = true;
    this.loadingKey = key;
    let timedOut = false;
    const timeout = setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, this.catalogRequestTimeoutMs);
    if (!quiet) this.error = "";
    this.renderView();
    try {
      const headers = { Authorization: `Bearer ${this.token}` };
      const response = await fetch(this.endpoint(cursor, { force }), {
        headers,
        cache: "no-store",
        signal: controller.signal,
      });
      const payload = await response.json().catch(() => ({}));
      if (response.status === 409 && append) {
        this.items = [];
        this.sourceItems = [];
        this.nextCursor = null;
        this.loadedKey = "";
        this.showToast("Catalog changed; reloading the first page.", false);
        queueMicrotask(() => this.load({ force: true }));
        return;
      }
      if (!response.ok) throw new Error(payload.error || `Catalog request failed (${response.status})`);
      if (serial !== this.requestSerial || key !== this.routeKey()) return;
      const received = Array.isArray(payload.items) ? payload.items : [];
      this.sourceItems = append ? [...this.sourceItems, ...received] : received;
      this.items = [...this.sourceItems];
      const visibleEligible = new Set(
        this.items
          .filter(checkpointCanEvaluate)
          .map((item) => String(item.checkpoint_id || "")),
      );
      this.selectedCheckpoints = new Set(
        [...this.selectedCheckpoints].filter((checkpointId) => (
          visibleEligible.has(checkpointId)
        )),
      );
      if (!append) {
        this.freshness = ["fresh", "stale", "partial"].includes(payload.freshness)
          ? payload.freshness
          : "fresh";
        this.catalogWarnings = Array.isArray(payload.warnings)
          ? payload.warnings
          : [];
        this.catalogSource = payload.source && typeof payload.source === "object"
          ? payload.source
          : null;
        this.generatedAt = Number.isFinite(Number(payload.generated_at))
          ? Number(payload.generated_at)
          : null;
        this.selectionFence = typeof payload.selection_fence === "string"
          ? payload.selection_fence
          : "";
        if (this.route.level === "goal_variants") {
          this.activityRevision = String(payload.revision || "");
          this.activityHasActiveRuns = Boolean(payload.has_active_runs);
        }
        this.metricColumns = Array.isArray(payload.metric_columns)
          ? payload.metric_columns
          : [];
        this.fallbackMetricColumns = Array.isArray(payload.fallback_metric_columns)
          ? payload.fallback_metric_columns
          : [];
        if (
          this.sort.metric
          && ![...this.metricColumns, ...this.fallbackMetricColumns]
            .some((column) => column.metric === this.sort.metric)
        ) {
          this.sort = { metric: "", direction: "" };
        }
      }
        this.nextCursor = payload.next_cursor || null;
      this.loadedKey = this.routeKey();
      this.error = "";
      if (
        !append
        && payload.training_enrichment === "pending"
        && this.route.level === "runs"
        && this.route.run_id
      ) {
        queueMicrotask(() => this.loadCheckpointTraining(key));
      }
      if (this.route.checkpoint_id && !append) {
        const selected = received.find(
          (item) => item.checkpoint_id === this.route.checkpoint_id,
        );
        if (!selected) {
          throw new Error(`Checkpoint ${this.route.checkpoint_id} was not found`);
        }
        if (this.autoSelectedRoute !== key) {
          this.autoSelectedRoute = key;
          this.selectCheckpoint(selected, { historyMode: "replace" });
        }
      }
    } catch (error) {
      if (serial !== this.requestSerial || key !== this.routeKey()) return;
      this.error = timedOut
        ? "Catalog request timed out. Try Refresh."
        : String(error?.message || error);
      if (quiet) this.showToast(this.error, true);
    } finally {
      clearTimeout(timeout);
      if (serial === this.requestSerial) {
        this.requestController = null;
        this.loading = false;
        this.loadingKey = "";
        this.renderView();
        this.updatePolling();
        if (
          key !== this.routeKey()
          && this.loadedKey !== this.routeKey()
          && this.app.phase === "selecting"
        ) {
          this.ensureLoaded();
        }
      }
    }
  }

  async loadCheckpointTraining(expectedKey) {
    const runId = String(this.route.run_id || "");
    if (!runId || expectedKey !== this.routeKey()) return;
    this.checkpointTrainingController?.abort();
    const controller = new AbortController();
    this.checkpointTrainingController = controller;
    const serial = ++this.checkpointTrainingSerial;
    const query = new URLSearchParams();
    if (this.query.trim()) query.set("q", this.query.trim());
    if (this.route.goal_variant_id) {
      query.set("goal_variant_id", this.route.goal_variant_id);
    }
    const timeout = setTimeout(() => controller.abort(), this.catalogRequestTimeoutMs);
    try {
      const response = await fetch(
        `/api/catalog/runs/${encodeURIComponent(runId)}/checkpoint-training?${query}`,
        {
          headers: { Authorization: `Bearer ${this.token}` },
          cache: "no-store",
          signal: controller.signal,
        },
      );
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload.error || `Training evidence request failed (${response.status})`);
      }
      if (
        serial !== this.checkpointTrainingSerial
        || expectedKey !== this.routeKey()
        || runId !== this.route.run_id
      ) return;
      this.sourceItems = Array.isArray(payload.items) ? payload.items : [];
      this.items = [...this.sourceItems];
      this.metricColumns = Array.isArray(payload.metric_columns)
        ? payload.metric_columns
        : this.metricColumns;
      this.selectionFence = typeof payload.selection_fence === "string"
        ? payload.selection_fence
        : this.selectionFence;
      this.catalogWarnings = Array.isArray(payload.warnings) ? payload.warnings : [];
      this.freshness = this.catalogWarnings.length ? "partial" : "fresh";
    } catch (error) {
      if (
        controller.signal.aborted
        || serial !== this.checkpointTrainingSerial
        || expectedKey !== this.routeKey()
      ) return;
      this.freshness = "partial";
      this.catalogWarnings = [
        ...this.catalogWarnings.filter((warning) => warning?.code !== "wandb_enrichment_pending"),
        {
          code: "wandb_enrichment_unavailable",
          message: `Live W&B training evidence is unavailable: ${String(error?.message || error)}`,
          retryable: true,
          source: "wandb",
        },
      ];
    } finally {
      clearTimeout(timeout);
      if (serial === this.checkpointTrainingSerial) {
        this.checkpointTrainingController = null;
        this.renderView();
      }
    }
  }

  updatePolling() {
    const shouldPoll = (
      this.app.phase === "selecting"
      && this.route.level === "goal_variants"
      && this.activityHasActiveRuns
    );
    if (shouldPoll && this.pollTimer === null) {
      this.pollTimer = window.setInterval(() => {
        this.loadedKey = "";
        this.load({ quiet: true });
      }, 5000);
    } else if (!shouldPoll && this.pollTimer !== null) {
      clearInterval(this.pollTimer);
      this.pollTimer = null;
    }
  }

  applyRoute(route, { seedItems = null } = {}) {
    this.route = { ...this.route, ...route };
    this.query = "";
    this.searchOpen = false;
    this.items = [];
    this.sourceItems = [];
    this.metricColumns = [];
    this.fallbackMetricColumns = [];
    this.sort = { metric: "", direction: "" };
    this.nextCursor = null;
    this.freshness = "fresh";
    this.catalogWarnings = [];
    this.catalogSource = null;
    this.generatedAt = null;
    this.selectionFence = "";
    this.loadedKey = "";
    this.error = "";
    this.checkpointTrainingController?.abort();
    this.checkpointTrainingController = null;
    this.checkpointTrainingSerial += 1;
    this.selectedCheckpoints.clear();
    this.resetGoalVariantDetail();
    this.autoSelectedRoute = "";
    if (Array.isArray(seedItems) && seedItems.length) {
      this.sourceItems = seedItems.map((item) => ({ ...item }));
      this.items = [...this.sourceItems];
      this.freshness = "partial";
    }
    this.hydrateInitialEnvironments();
    this.renderView();
    this.ensureLoaded();
    this.updatePolling();
  }

  syncUrl(mode = "push") {
    if (!this.historyEnabled) return;
    const path = sourceRoutePath(this.route);
    const target = `${path}${location.search}${location.hash}`;
    const current = `${location.pathname}${location.search}${location.hash}`;
    if (target === current) return;
    if (mode === "replace") history.replaceState(null, "", target);
    else history.pushState(null, "", target);
  }

  navigate(route, { historyMode = "push", seedItems = null } = {}) {
    const nextRoute = { ...this.route, ...route };
    if (!this.app?.has_active_runner) {
      const commandId = this.command("browse_sources", { route: nextRoute });
      if (commandId === null) return false;
    }
    this.applyRoute(route, { seedItems });
    if (historyMode) this.syncUrl(historyMode);
    this.openSourceRoute?.({ ...this.route });
    return true;
  }

  browseCurrentSource() {
    const route = this.app.route || this.route;
    const next = route.run_id
      ? { ...route, level: "runs", checkpoint_id: "" }
      : {
          level: "environments",
          environment_id: "",
          goal_id: "",
          goal_variant_id: "",
          run_id: "",
          checkpoint_id: "",
        };
    this.navigate(next);
  }

  goHome() {
    this.navigate({
      level: "environments",
      environment_id: "",
      goal_id: "",
      goal_variant_id: "",
      run_id: "",
      checkpoint_id: "",
    });
  }

  selectCheckpoint(item, { historyMode = "push" } = {}) {
    const route = {
      ...this.route,
      level: "runs",
      checkpoint_id: item.checkpoint_id,
    };
    const commandId = this.command("select_source", {
      source: {
        kind: "public_run",
        value: item.manifest_url,
        run_id: item.run_id,
        checkpoint_id: item.checkpoint_id,
        seed: checkpointPlaybackSeed(item),
      },
      route: { ...route },
    });
    if (commandId === null) return false;
    this.route = route;
    this.syncUrl(historyMode);
    return commandId;
  }

  back() {
    if (this.route.level === "runs" && this.route.run_id) {
      this.navigate({ level: "runs", run_id: "", checkpoint_id: "" });
    } else if (this.route.level === "runs") {
      this.navigate({
        level: "goal_variants",
        goal_variant_id: "",
        run_id: "",
        checkpoint_id: "",
      });
    } else if (this.route.level === "goal_variants") {
      this.navigate({
        level: "goals",
        goal_id: "",
        goal_variant_id: "",
        run_id: "",
        checkpoint_id: "",
      });
    } else if (this.route.level === "goals") {
      this.navigate({
        level: "environments",
        environment_id: "",
        goal_id: "",
        goal_variant_id: "",
        run_id: "",
        checkpoint_id: "",
      });
    }
  }

  setSearch(value) {
    this.query = value;
    clearTimeout(this.searchTimer);
    this.checkpointTrainingController?.abort();
    this.checkpointTrainingController = null;
    this.checkpointTrainingSerial += 1;
    this.items = this.sourceItems.filter((item) => catalogItemMatchesSearch(item, value));
    this.nextCursor = null;
    this.loadedKey = "";
    this.renderView();
    this.searchTimer = window.setTimeout(() => {
      this.load();
    }, 80);
  }

  renderView() {
    const activeElement = document.activeElement;
    const restoreSearchFocus = (
      activeElement instanceof HTMLInputElement
      && activeElement.type === "search"
      && this.root.contains(activeElement)
    );
    const selectionStart = restoreSearchFocus ? activeElement.selectionStart : null;
    const selectionEnd = restoreSearchFocus ? activeElement.selectionEnd : null;
    const shell = document.createElement("section");
    shell.className = "source-shell";
    const head = document.createElement("div");
    head.className = "source-head";
    const titleBlock = document.createElement("div");
    titleBlock.className = "source-title";
    const eyebrow = document.createElement("span");
    eyebrow.className = "eyebrow";
    eyebrow.textContent = "PLAYBACK SOURCE";
    const heading = document.createElement("h2");
    heading.textContent = this.heading();
    titleBlock.append(eyebrow, heading);
    head.append(titleBlock);

    if (this.app.phase === "selecting") {
      const refresh = button("Refresh", { iconName: "refresh", quiet: true });
      refresh.title = "Refresh this list";
      refresh.disabled = this.loading;
      refresh.addEventListener("click", () => {
        this.loadedKey = "";
        this.load({ force: true });
      });
      head.append(refresh);
    }
    shell.append(head);

    if (!this.hasControl()) {
      const observer = document.createElement("p");
      observer.className = "source-notice";
      observer.textContent = "Observer window — choose Control here to change the shared run.";
      shell.append(observer);
    }

    const showsCatalog = ![
      "error",
      "resolving",
      "verifying",
      "loading",
    ].includes(this.app.phase);
    if (showsCatalog) {
      this.renderBreadcrumbs(this.breadcrumbsRoot);
      this.breadcrumbsRoot.hidden = false;
    } else {
      this.breadcrumbsRoot.replaceChildren();
      this.breadcrumbsRoot.hidden = true;
    }

    if (this.app.phase === "error") {
      shell.append(this.renderFailure());
    } else if (["resolving", "verifying", "loading"].includes(this.app.phase)) {
      shell.append(this.renderProgress());
    } else {
      if (this.route.level === "goal_variants") {
        const description = document.createElement("p");
        description.className = "source-description";
        description.textContent = (
          "Each configuration groups runs with the same resolved goal behavior. "
          + "Select one to inspect its exact differences from the current checked-in goal."
        );
        shell.append(description);
      }
      if (this.app.has_active_runner) shell.append(this.renderContinuePlayback());
      shell.append(this.renderSearch());
      if (this.route.level === "runs" && this.route.run_id) {
        shell.append(this.renderCheckpointRecommendation());
      }
      shell.append(this.renderResults());
    }
    this.root.replaceChildren(shell);
    if (restoreSearchFocus) {
      const search = this.root.querySelector('input[type="search"]');
      search?.focus({ preventScroll: true });
      if (search && selectionStart !== null && selectionEnd !== null) {
        search.setSelectionRange(selectionStart, selectionEnd);
      }
    }
  }

  heading() {
    if (this.app.phase === "error") return "Could not open checkpoint";
    if (["resolving", "verifying", "loading"].includes(this.app.phase)) return "Opening checkpoint";
    if (this.route.level === "runs" && this.route.run_id) {
      return "Runs · choose a checkpoint";
    }
    if (this.route.level === "runs") return "Choose a run";
    if (this.route.level === "goal_variants") {
      return "Choose a goal version";
    }
    if (this.route.level === "goals") return "Choose a goal";
    return "Choose an environment";
  }

  renderBreadcrumbs(nav) {
    nav.replaceChildren();
    sourceBreadcrumbItems(this.route).forEach((item) => {
      const crumb = button(item.label, { quiet: true });
      if (item.title) crumb.title = item.title;
      crumb.disabled = item.current;
      if (item.route) {
        crumb.addEventListener("click", () => this.navigate(item.route));
      }
      nav.append(crumb);
    });
    return nav;
  }

  renderSearch() {
    const wrap = document.createElement("label");
    wrap.className = "source-search";
    wrap.append(icon("search"));
    const input = document.createElement("input");
    input.type = "search";
    input.value = this.query;
    input.autocomplete = "off";
    input.placeholder = this.route.level === "environments"
      ? "Search environments"
      : this.route.level === "goals"
        ? "Search goals"
        : this.route.level === "goal_variants"
          ? "Search configuration, difference, status, date, or contract hash"
          : this.route.level === "runs" && !this.route.run_id
          ? "Search run, finish reason, description, recipe, variant, override, or seed"
          : "Search checkpoint, step, hash, purpose, or evaluation";
    input.setAttribute("aria-label", input.placeholder);
    input.addEventListener("input", (event) => this.setSearch(event.target.value));
    wrap.append(input);
    const disclosure = document.createElement("details");
    disclosure.className = "source-search-disclosure";
    disclosure.open = this.searchOpen;
    const summary = document.createElement("summary");
    summary.append(icon("search"), document.createTextNode("Search"));
    disclosure.append(summary, wrap);
    disclosure.addEventListener("toggle", () => {
      this.searchOpen = disclosure.open;
      if (disclosure.open) {
        requestAnimationFrame(() => input.focus({ preventScroll: true }));
      }
    });
    return disclosure;
  }

  renderContinuePlayback() {
    const card = document.createElement("section");
    card.className = "continue-playback-card";
    const copy = document.createElement("div");
    const heading = document.createElement("strong");
    heading.textContent = "Continue current playback";
    copy.append(heading);
    const playbackRoute = this.currentPlaybackRoute();
    const breadcrumbItems = playbackRoute
      ? sourceBreadcrumbItems(playbackRoute).slice(1)
      : [];
    if (breadcrumbItems.length) {
      const breadcrumb = document.createElement("nav");
      breadcrumb.className = "continue-playback-breadcrumb";
      breadcrumb.setAttribute("aria-label", "Current playback source");
      breadcrumbItems.forEach((item) => {
        const crumb = document.createElement("span");
        crumb.textContent = item.label;
        if (item.title) crumb.title = item.title;
        breadcrumb.append(crumb);
      });
      copy.append(breadcrumb);
    } else {
      const detail = document.createElement("p");
      detail.textContent = "Loaded checkpoint";
      copy.append(detail);
    }
    const resume = button("Continue watching", { iconName: "eye", primary: true });
    resume.disabled = !this.hasControl();
    resume.addEventListener("click", () => {
      if (!this.resumePlayback?.()) this.command("cancel_source");
    });
    card.append(copy, resume);
    return card;
  }

  renderEvaluationActions() {
    const actions = document.createElement("div");
    actions.className = "source-evaluation-actions";
    const selected = this.selectedCheckpoints.size;
    const summary = document.createElement("span");
    summary.textContent = selected
      ? `${selected.toLocaleString()} selected`
      : "Select checkpoints to evaluate";
    const evaluate = button(
      this.evaluating
        ? "Adding to queue…"
        : selected
          ? `Evaluate ${selected.toLocaleString()}`
          : "Evaluate selected",
      { iconName: "player-play", primary: true },
    );
    evaluate.disabled = !selected || this.evaluating;
    evaluate.addEventListener("click", () => this.evaluateSelected());
    const inspect = button("Inspect run YAML", { iconName: "code", quiet: true });
    inspect.addEventListener("click", () => {
      void this.inspectRun().catch(
        (error) => this.showToast(String(error?.message || error), true),
      );
    });
    actions.append(summary, inspect, evaluate);
    return actions;
  }

  renderCheckpointRecommendation() {
    const recommendation = checkpointRecommendation(this.items);
    const card = document.createElement("section");
    card.className = "checkpoint-recommendation";
    if (!recommendation) {
      card.hidden = true;
      return card;
    }
    const copy = document.createElement("div");
    const eyebrow = document.createElement("span");
    eyebrow.className = "eyebrow";
    eyebrow.textContent = "BEST AVAILABLE TO WATCH";
    const heading = document.createElement("strong");
    heading.textContent = `${recommendation.evidence.label} · step ${Number(recommendation.item.step).toLocaleString()}`;
    const detail = document.createElement("p");
    detail.textContent = recommendation.evidence.detail;
    copy.append(eyebrow, heading, detail);
    const metrics = checkpointRecommendationMetrics(
      recommendation.item,
      this.metricColumns,
    );
    if (metrics.length) {
      const evidence = document.createElement("dl");
      evidence.className = "checkpoint-recommendation-metrics";
      metrics.forEach((metric) => {
        const term = document.createElement("dt");
        term.className = metric.tone;
        term.textContent = `${metric.label} · ${metric.role}`;
        const value = document.createElement("dd");
        value.textContent = metric.value;
        evidence.append(term, value);
      });
      copy.append(evidence);
    }
    const play = button("Watch checkpoint", { iconName: "player-play", primary: true });
    play.disabled = !this.hasControl();
    play.addEventListener("click", () => this.selectCheckpoint(recommendation.item));
    card.classList.add(recommendation.evidence.tone);
    card.append(copy, play);
    return card;
  }

  async evaluateSelected() {
    const checkpointIds = [...this.selectedCheckpoints];
    if (!checkpointIds.length || this.evaluating) return;
    this.evaluating = true;
    this.renderView();
    try {
      const response = await fetch(
        `/api/catalog/runs/${encodeURIComponent(this.route.run_id)}/evaluations`,
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${this.token}`,
            "Content-Type": "application/json",
          },
          cache: "no-store",
          body: JSON.stringify({
            checkpoint_ids: checkpointIds,
            selection_fence: this.selectionFence,
          }),
        },
      );
      const payload = await response.json().catch(() => ({}));
      if (response.status === 409 && payload.code === "checkpoint_catalog_changed") {
        this.selectedCheckpoints.clear();
        this.loadedKey = "";
        await this.load({ force: true });
      }
      if (!response.ok) {
        throw new Error(payload.error || `Evaluation request failed (${response.status})`);
      }
      const statuses = new Map(
        (Array.isArray(payload.items) ? payload.items : [])
          .map((item) => [String(item.checkpoint_id || ""), item]),
      );
      this.items = this.items.map((item) => {
        const status = statuses.get(String(item.checkpoint_id || ""));
        return status
          ? {
              ...item,
              evaluation: status.evaluation || item.evaluation,
              evaluation_queue: status,
            }
          : item;
      });
      this.selectedCheckpoints.clear();
      const admitted = [...statuses.values()].filter(
        (item) => [
          "queued",
          "running",
          "retry_wait",
          "waiting_for_training_terminal",
          "waiting_for_run_lease",
          "submitted",
          "submission_uncertain",
          "awaiting_projection",
        ].includes(String(item.state || "")),
      ).length;
      const workerWarning = payload?.worker?.state === "start_failed"
        ? String(payload.worker.message || "The local evaluation flusher could not start.")
        : "";
      this.showToast(
        workerWarning || (
          admitted
          ? `${admitted.toLocaleString()} checkpoint${admitted === 1 ? "" : "s"} queued for evaluation.`
          : "The selected checkpoints already have evaluation state."
        ),
        Boolean(workerWarning),
      );
    } catch (error) {
      this.showToast(String(error?.message || error), true);
    } finally {
      this.evaluating = false;
      this.renderView();
    }
  }

  renderResults() {
    const body = document.createElement("div");
    body.className = "source-results";
    if (this.error) {
      const error = document.createElement("div");
      error.className = "source-inline-error";
      const message = document.createElement("p");
      message.textContent = this.error;
      const retry = button("Retry", { iconName: "refresh" });
      retry.addEventListener("click", () => {
        this.loadedKey = "";
        this.load();
      });
      error.append(message, retry);
      body.append(error);
      return body;
    }
    if (this.loading) {
      body.classList.add("loading");
      if (!this.items.length) body.classList.add("loading-empty");
      const indicator = document.createElement("div");
      indicator.className = "source-list-loading-indicator";
      indicator.setAttribute("role", "status");
      indicator.setAttribute("aria-label", "Refreshing list");
      const spinner = document.createElement("span");
      spinner.className = "spinner";
      spinner.setAttribute("aria-hidden", "true");
      indicator.append(spinner);
      body.append(indicator);
    }
    if (this.loading && !this.items.length) return body;
    if (!this.items.length) {
      const empty = document.createElement("div");
      empty.className = "source-empty";
      const heading = document.createElement("strong");
      heading.textContent = this.route.level === "runs" && this.route.run_id
        ? "No public checkpoints yet"
        : "No matching results";
      const detail = document.createElement("p");
      detail.textContent = this.route.level === "runs" && this.route.run_id
        ? "This run has not published a playable checkpoint to public model storage."
        : "Try a broader search.";
      empty.append(heading, detail);
      body.append(empty);
      return body;
    }
    const results = this.route.level === "environments"
      ? this.renderEnvironments()
      : this.route.level === "goals"
        ? this.renderGoals()
        : this.route.level === "goal_variants"
          ? this.renderGoalVariants()
          : this.route.level === "runs"
            ? this.renderRunResults()
            : this.renderTable();
    if (this.route.level === "runs" && this.route.run_id) {
      body.classList.add("source-checkpoint-results");
      body.append(this.renderEvaluationActions(), results);
    } else {
      body.append(results);
    }
    if (this.nextCursor) {
      const more = button(this.loading ? "Loading…" : "Load more");
      more.classList.add("source-load-more");
      more.disabled = this.loading;
      more.addEventListener("click", () => this.load({ append: true }));
      body.append(more);
    }
    return body;
  }

  renderEnvironments() {
    const list = document.createElement("div");
    list.className = "environment-list";
    [...this.items].sort((left, right) => (
      environmentEvidenceRank(left) - environmentEvidenceRank(right)
    )).forEach((environment) => {
      const row = document.createElement("button");
      row.type = "button";
      row.className = "environment-row";
      row.disabled = !this.hasControl();
      const identity = document.createElement("span");
      identity.className = "environment-row-identity";
      const name = document.createElement("strong");
      name.textContent = environment.name;
      identity.append(name);
      const success = renderSuccessBadges(environment);
      if (success) identity.append(success);
      const meta = document.createElement("span");
      const goalLabel = Number(environment.goal_count) === 1 ? "goal" : "goals";
      meta.textContent = `${Number(environment.goal_count).toLocaleString()} ${goalLabel}`;
      row.append(identity, meta);
      row.addEventListener("click", () => this.navigate({
        level: "goals",
        environment_id: environment.name,
        goal_id: "",
        goal_variant_id: "",
        run_id: "",
        checkpoint_id: "",
      }));
      list.append(row);
    });
    return list;
  }

  renderGoals() {
    const list = document.createElement("div");
    list.className = "goal-list";
    this.items.forEach((goal) => {
      const row = document.createElement("div");
      row.className = "goal-row";
      const navigate = document.createElement("button");
      navigate.type = "button";
      navigate.className = "goal-row-navigation";
      navigate.disabled = !this.hasControl();
      const identity = document.createElement("div");
      identity.className = "goal-row-identity";
      const heading = document.createElement("div");
      heading.className = "goal-row-heading";
      const name = document.createElement("strong");
      name.textContent = goal.goal_id;
      heading.append(name);
      const success = renderSuccessBadges(goal);
      if (success) heading.append(success);
      identity.append(heading);
      const meta = document.createElement("span");
      const recipeLabel = Number(goal.recipe_count) === 1 ? "recipe" : "recipes";
      meta.textContent = `${goal.title || goal.goal_slug} · ${Number(goal.recipe_count).toLocaleString()} ${recipeLabel}`;
      identity.append(meta);
      if (goal.goal_slug && goal.goal_slug !== goal.goal_id) {
        navigate.title = `Goal slug: ${goal.goal_slug}`;
      }
      navigate.append(identity);
      navigate.addEventListener("click", () => this.navigate({
        level: "goal_variants",
        goal_id: goal.goal_id,
        goal_variant_id: "",
        run_id: "",
        checkpoint_id: "",
      }));
      const inspect = button("", { iconName: "code", quiet: true });
      inspect.classList.add("goal-row-inspect", "icon-only");
      inspect.title = `Inspect ${goal.goal_id} YAML`;
      inspect.setAttribute("aria-label", inspect.title);
      inspect.addEventListener("click", () => {
        void this.inspectGoal(goal).catch(
          (error) => this.showToast(String(error?.message || error), true),
        );
      });
      row.append(navigate, inspect);
      list.append(row);
    });
    return list;
  }

  renderGoalVariants() {
    const container = document.createElement("div");
    container.className = "goal-configuration-browser";
    const variants = this.items.map((variant) => ({
      variant,
      presentation: goalConfigurationPresentation(variant),
    }));
    const selected = variants.find(
      ({ variant }) => variant.variant_id === this.selectedGoalVariantId,
    ) || variants.find(({ presentation }) => presentation.kind === "current_default")
      || variants[0];
    if (!selected) return container;
    this.selectedGoalVariantId = String(selected.variant.variant_id || "");
    if (this.goalVariantDiff?.variantId !== this.selectedGoalVariantId) {
      this.goalVariantDiff = this.goalVariantDiffFromActivity(selected.variant);
    }

    const tableScroll = document.createElement("div");
    tableScroll.className = "goal-configuration-table-scroll";
    const table = document.createElement("table");
    table.className = "goal-configuration-table";
    const columns = document.createElement("colgroup");
    ["configuration", "differences", "runs", "first-used", "last-activity"].forEach((name) => {
      const column = document.createElement("col");
      column.className = `goal-configuration-column ${name}`;
      columns.append(column);
    });
    const head = document.createElement("thead");
    const headerRow = document.createElement("tr");
    ["Configuration", "Differences", "Runs", "First used", "Last activity"].forEach((label) => {
      const cell = document.createElement("th");
      cell.scope = "col";
      if (label === "Configuration" && variants.length > 1) {
        cell.className = "goal-configuration-heading";
        const heading = document.createElement("span");
        heading.textContent = label;
        const otherCount = variants.length - 1;
        const toggle = button(
          this.goalConfigurationsExpanded
            ? "Show selected only"
            : `Show ${otherCount.toLocaleString()} more`,
          { quiet: true },
        );
        toggle.classList.add("goal-configuration-toggle");
        toggle.setAttribute("aria-expanded", String(this.goalConfigurationsExpanded));
        toggle.setAttribute(
          "aria-label",
          this.goalConfigurationsExpanded
            ? "Show only the selected goal configuration"
            : `Show ${otherCount.toLocaleString()} other goal configurations`,
        );
        toggle.addEventListener("click", () => {
          this.goalConfigurationsExpanded = !this.goalConfigurationsExpanded;
          this.renderView();
        });
        cell.append(heading, toggle);
      } else {
        cell.textContent = label;
      }
      headerRow.append(cell);
    });
    head.append(headerRow);
    const body = document.createElement("tbody");

    const displayedVariants = this.goalConfigurationsExpanded ? variants : [selected];
    displayedVariants.forEach(({ variant, presentation }) => {
      const row = document.createElement("tr");
      const isSelected = variant.variant_id === selected.variant.variant_id;
      row.className = `goal-configuration-row${isSelected ? " selected" : ""}`;
      const configuration = document.createElement("td");
      const choice = document.createElement("label");
      choice.className = "goal-configuration-choice";
      const radio = document.createElement("input");
      radio.type = "radio";
      radio.name = "goal-configuration";
      radio.value = String(variant.variant_id || "");
      radio.checked = isSelected;
      radio.addEventListener("change", () => this.selectGoalVariant(variant));
      const badges = document.createElement("span");
      badges.className = "goal-configuration-badges";
      if (presentation.sourceLabel) {
        const sourceBadge = document.createElement("span");
        sourceBadge.className = "goal-configuration-badge current";
        sourceBadge.textContent = presentation.sourceLabel;
        badges.append(sourceBadge);
      }
      const behaviorBadge = document.createElement("span");
      behaviorBadge.className = "goal-configuration-badge behavior";
      behaviorBadge.textContent = presentation.behaviorLabel;
      badges.append(behaviorBadge);
      const success = renderSuccessBadges(variant);
      if (success) badges.append(success);
      choice.append(radio, badges);
      configuration.append(choice);

      const differences = document.createElement("td");
      differences.className = `goal-configuration-difference${presentation.comparisonAvailable ? "" : " unavailable"}`;
      differences.textContent = presentation.differenceLabel;
      const runs = document.createElement("td");
      runs.className = "goal-configuration-number";
      runs.textContent = presentation.runCount.toLocaleString();
      const firstUsed = document.createElement("td");
      firstUsed.className = "goal-configuration-date";
      firstUsed.textContent = presentation.firstUsedDate;
      const lastActivity = document.createElement("td");
      lastActivity.className = "goal-configuration-date";
      lastActivity.textContent = presentation.lastActivityDate;
      row.append(configuration, differences, runs, firstUsed, lastActivity);
      body.append(row);
    });
    table.append(columns, head, body);
    tableScroll.append(table);
    container.append(tableScroll, this.renderGoalVariantDetail(selected));

    return container;
  }

  renderGoalVariantDetail({ variant, presentation }) {
    const section = document.createElement("section");
    section.className = "goal-configuration-detail";
    const header = document.createElement("div");
    header.className = "goal-configuration-detail-header";
    const identity = document.createElement("div");
    const heading = document.createElement("h3");
    heading.textContent = "Selected goal version";
    const baseline = document.createElement("p");
    baseline.textContent = `${presentation.kindLabel} · ${presentation.runLabel}`;
    identity.append(heading, baseline);
    const actions = document.createElement("div");
    actions.className = "goal-configuration-detail-actions";
    const inspect = button("View YAML", { iconName: "code", quiet: true });
    inspect.addEventListener("click", () => {
      const inspection = presentation.kind === "current_default"
        ? this.inspectGoal({ goal_id: this.route.goal_id })
        : this.inspectGoalVariant(variant);
      void inspection.catch(
        (error) => this.showToast(String(error?.message || error), true),
      );
    });
    const viewRuns = button(
      presentation.runCount === 1 ? "View run" : `View ${presentation.runCount.toLocaleString()} runs`,
      { iconName: "external-link", primary: true },
    );
    viewRuns.disabled = !this.hasControl() || presentation.runCount === 0;
    viewRuns.addEventListener("click", () => this.navigate(
      {
        level: "runs",
        goal_variant_id: variant.variant_id,
        run_id: "",
        checkpoint_id: "",
      },
      { seedItems: Array.isArray(variant.recent_runs) ? variant.recent_runs : [] },
    ));
    actions.append(inspect, viewRuns);
    header.append(identity, actions);
    section.append(header, this.renderEmbeddedGoalRuns(variant));

    const differences = document.createElement("details");
    differences.className = "goal-configuration-differences";
    const differencesSummary = document.createElement("summary");
    differencesSummary.textContent = `Contract differences · ${presentation.differenceLabel}`;
    const differencesIntro = document.createElement("p");
    differencesIntro.textContent = (
      "Baseline: current checked-in goal · Exact JSON-Pointer paths and typed values."
    );
    differences.append(differencesSummary, differencesIntro);
    const finishDifferences = (content) => {
      differences.append(content);
      section.append(differences);
      return section;
    };

    if (presentation.kind === "current_default") {
      return finishDifferences(this.goalVariantDiffEmpty(
        "This configuration exactly matches the current checked-in goal.",
      ));
    }
    if (!presentation.comparisonAvailable) {
      return finishDifferences(this.goalVariantDiffEmpty(
        "The exact historical contract is not sufficiently proven, so no field-level comparison is shown.",
        { warning: true },
      ));
    }

    const state = this.goalVariantDiff?.variantId === variant.variant_id
      ? this.goalVariantDiff
      : null;
    if (!state || state.state === "loading") {
      return finishDifferences(this.loadingState("Loading exact contract differences…"));
    }
    if (state.state === "error") {
      return finishDifferences(this.goalVariantDiffEmpty(state.message, { warning: true }));
    }
    if (state.availability !== "exact") {
      return finishDifferences(this.goalVariantDiffEmpty(
        state.message || "An exact field-level comparison is unavailable.",
        { warning: true },
      ));
    }
    if (!state.entries.length) {
      return finishDifferences(this.goalVariantDiffEmpty(
        "This configuration has no behavioral differences from the current checked-in goal.",
      ));
    }

    const count = document.createElement("span");
    count.className = "goal-configuration-detail-count";
    const changeCount = state.changeCount ?? state.entries.length;
    count.textContent = `${changeCount.toLocaleString()} ${changeCount === 1 ? "changed key" : "changed keys"}`;
    differencesSummary.append(" · ", count);
    const scroll = document.createElement("div");
    scroll.className = "goal-configuration-diff-scroll";
    const table = document.createElement("table");
    table.className = "goal-configuration-diff-table";
    const head = document.createElement("thead");
    const headerRow = document.createElement("tr");
    ["Operation", "Exact contract path", "Before", "After"].forEach((label) => {
      const cell = document.createElement("th");
      cell.scope = "col";
      cell.textContent = label;
      headerRow.append(cell);
    });
    head.append(headerRow);
    const body = document.createElement("tbody");
    state.entries.forEach((entry) => {
      const row = document.createElement("tr");
      const kind = ["added", "removed", "changed"].includes(String(entry?.kind))
        ? String(entry.kind)
        : "changed";
      const operation = document.createElement("td");
      operation.className = `goal-configuration-operation ${kind}`;
      operation.textContent = kind[0].toUpperCase() + kind.slice(1);
      const path = document.createElement("td");
      path.className = "goal-configuration-path";
      const pathCode = document.createElement("code");
      pathCode.textContent = String(entry?.path || "");
      path.append(pathCode);
      const before = document.createElement("td");
      before.className = "goal-configuration-value";
      const beforeCode = document.createElement("code");
      beforeCode.textContent = formatGoalDiffValue(entry?.before, { unavailable: kind === "added" });
      before.append(beforeCode);
      const after = document.createElement("td");
      after.className = `goal-configuration-value goal-configuration-after ${kind}`;
      const afterCode = document.createElement("code");
      afterCode.textContent = formatGoalDiffValue(entry?.after, { unavailable: kind === "removed" });
      after.append(afterCode);
      row.append(operation, path, before, after);
      body.append(row);
    });
    table.append(head, body);
    scroll.append(table);
    return finishDifferences(scroll);
  }

  async loadEmbeddedGoalRuns(variant, { append = false } = {}) {
    const variantId = String(variant?.variant_id || "");
    if (!variantId) return;
    const current = this.goalVariantRunPages.get(variantId) || {
      items: [],
      nextCursor: null,
      loading: false,
      error: "",
    };
    if (current.loading || (append && !current.nextCursor)) return;
    const next = { ...current, loading: true, error: "" };
    this.goalVariantRunPages.set(variantId, next);
    this.renderView();
    const query = new URLSearchParams();
    if (append && current.nextCursor) query.set("cursor", current.nextCursor);
    try {
      const endpoint = (
        `/api/catalog/environments/${encodeURIComponent(this.route.environment_id)}`
        + `/goals/${encodeURIComponent(this.route.goal_id)}`
        + `/variants/${encodeURIComponent(variantId)}/runs?${query}`
      );
      const response = await fetch(endpoint, {
        headers: { Authorization: `Bearer ${this.token}` },
        cache: "no-store",
      });
      const payload = await response.json().catch(() => ({}));
      if (response.status === 409) {
        this.goalVariantRunPages.delete(variantId);
        this.loadedKey = "";
        await this.load({ force: true });
        return;
      }
      if (!response.ok) throw new Error(payload.error || `Run request failed (${response.status})`);
      const received = Array.isArray(payload.items) ? payload.items : [];
      this.goalVariantRunPages.set(variantId, {
        items: append ? [...current.items, ...received] : received,
        nextCursor: payload.next_cursor || null,
        loading: false,
        error: "",
      });
    } catch (error) {
      this.goalVariantRunPages.set(variantId, {
        ...current,
        loading: false,
        error: String(error?.message || error),
      });
    } finally {
      this.renderView();
    }
  }

  renderEmbeddedGoalRuns(variant) {
    const section = document.createElement("section");
    section.className = "goal-configuration-runs";
    const header = document.createElement("div");
    header.className = "goal-configuration-runs-header";
    const heading = document.createElement("h3");
    heading.textContent = "Runs using this configuration";
    const sort = document.createElement("div");
    sort.className = "goal-configuration-run-sort";
    const variantId = String(variant?.variant_id || "");
    const sortMode = this.goalVariantRunSort.get(variantId) || "recent";
    [
      ["recent", "Recent"],
      ["best", "Best"],
    ].forEach(([mode, label]) => {
      const control = button(label, { quiet: mode !== sortMode });
      control.classList.toggle("active", mode === sortMode);
      control.addEventListener("click", () => {
        this.goalVariantRunSort.set(variantId, mode);
        this.renderView();
      });
      sort.append(control);
    });
    header.append(heading, sort);
    section.append(header);

    const page = this.goalVariantRunPages.get(variantId);
    const baseItems = page?.items?.length
      ? page.items
      : sortMode === "best"
        ? variant.best_runs
        : variant.recent_runs;
    const items = Array.isArray(baseItems) ? baseItems : [];
    if (!items.length) {
      const empty = document.createElement("p");
      empty.className = "goal-configuration-runs-empty";
      empty.textContent = "No runs use this configuration yet.";
      section.append(empty);
      return section;
    }
    const list = document.createElement("div");
    list.className = "goal-configuration-run-list";
    items.forEach((run) => {
      const row = document.createElement("button");
      row.type = "button";
      row.className = "goal-configuration-run-row";
      const identity = document.createElement("span");
      identity.className = "goal-configuration-run-identity";
      const name = document.createElement("strong");
      name.textContent = String(run?.name || run?.run_id || "Run");
      const description = document.createElement("small");
      description.textContent = String(run?.description || run?.run_id || "");
      identity.append(name, description);
      const success = renderSuccessBadges(run);
      if (success) identity.append(success);
      const state = document.createElement("span");
      state.className = `goal-configuration-run-state ${String(run?.state || "unknown")}`;
      state.textContent = String(run?.state || "unknown");
      const updated = document.createElement("span");
      updated.className = "goal-configuration-run-updated";
      updated.textContent = run?.updated_at ? formatDate(run.updated_at) : "—";
      row.append(identity, state, updated);
      row.addEventListener("click", () => this.navigate({
        level: "runs",
        goal_variant_id: variantId,
        run_id: String(run.run_id || ""),
        checkpoint_id: "",
      }));
      list.append(row);
    });
    section.append(list);
    if (page?.error) {
      const error = document.createElement("p");
      error.className = "source-inline-error";
      error.textContent = page.error;
      section.append(error);
    }
    if (variant.has_more_runs || page?.nextCursor) {
      const load = button(page?.nextCursor ? "Load more" : "Load older runs", {
        iconName: "refresh",
        quiet: true,
      });
      load.disabled = Boolean(page?.loading);
      load.addEventListener("click", () => this.loadEmbeddedGoalRuns(variant, {
        append: Boolean(page?.items?.length),
      }));
      section.append(load);
    }
    return section;
  }

  goalVariantDiffEmpty(message, { warning = false } = {}) {
    const empty = document.createElement("div");
    empty.className = `goal-configuration-diff-empty${warning ? " warning" : ""}`;
    empty.textContent = message;
    return empty;
  }

  activeRunMetricColumns() {
    return activeRunMetricColumns(
      this.items,
      this.metricColumns,
      this.fallbackMetricColumns,
    );
  }

  runEfficiency() {
    return bestRunEfficiency(
      this.items,
      this.metricColumns,
      this.fallbackMetricColumns,
    );
  }

  renderRunResults() {
    if (this.route.run_id) return this.renderTable();
    const efficiency = this.runEfficiency();
    return this.renderTable(efficiency);
  }

  renderTable(efficiency = null) {
    const scroll = document.createElement("div");
    scroll.className = "source-table-scroll";
    if (efficiency?.evidence === "training") {
      scroll.classList.add("training-leader");
    }
    const table = document.createElement("table");
    table.className = "source-table";
    const head = document.createElement("thead");
    const headerRow = document.createElement("tr");
    const showingRuns = this.route.level === "runs" && !this.route.run_id;
    const showingCheckpoints = this.route.level === "runs" && Boolean(this.route.run_id);
    if (showingCheckpoints) table.classList.add("checkpoint-table");
    const runRankingColumns = showingRuns ? this.activeRunMetricColumns() : [];
    const runMetricColumns = availableRunMetricColumns(this.items, runRankingColumns);
    const checkpointMetricColumns = showingCheckpoints ? this.metricColumns : [];
    const columns = showingRuns
      ? [
          { label: "Run" },
          { label: "Recipe / variant" },
          { label: "Seed" },
          { label: "Training result" },
          ...runMetricColumns.map((column) => ({
            ...column,
            label: metricLabel(column.metric),
          })),
          { label: "Updated" },
          { label: "Contract" },
        ]
      : [
          ...(showingCheckpoints ? [{ label: "", selection: true }] : []),
          { label: "Checkpoint" },
          ...(showingCheckpoints ? [{ label: "Purpose" }] : []),
          { label: "Step" },
          ...checkpointMetricColumns.map((column) => ({
            ...column,
            fullLabel: column.label || metricLabel(column.metric),
            label: checkpointMetricHeaderLabel(column),
          })),
          ...(showingCheckpoints ? [{ label: "Size" }, { label: "Created" }] : []),
        ];
    columns.forEach((column) => {
      const cell = document.createElement("th");
      cell.scope = "col";
      if (column.selection) {
        cell.className = "source-selection-cell";
        const eligible = this.items.filter(checkpointCanEvaluate);
        const allSelected = (
          eligible.length > 0
          && eligible.every((item) => this.selectedCheckpoints.has(item.checkpoint_id))
        );
        const selectAll = document.createElement("input");
        selectAll.type = "checkbox";
        selectAll.checked = allSelected;
        selectAll.disabled = !eligible.length || this.evaluating;
        selectAll.setAttribute("aria-label", "Select all eligible checkpoints");
        selectAll.addEventListener("change", () => {
          if (selectAll.checked) {
            eligible.forEach((item) => this.selectedCheckpoints.add(item.checkpoint_id));
          } else {
            eligible.forEach((item) => this.selectedCheckpoints.delete(item.checkpoint_id));
          }
          this.renderView();
        });
        cell.append(selectAll);
      } else if (column.metric) {
        const fullLabel = column.fullLabel || column.label;
        const active = this.sort.metric === column.metric;
        const defaultDirection = column.direction === "min" ? "ascending" : "descending";
        const nextDirection = active && this.sort.direction === "ascending"
          ? "descending"
          : active && this.sort.direction === "descending"
            ? "ascending"
            : defaultDirection;
        cell.setAttribute("aria-sort", active ? this.sort.direction : "none");
        const sortButton = document.createElement("button");
        sortButton.type = "button";
        sortButton.className = "source-sort";
        sortButton.title = `${fullLabel} · ${
          showingCheckpoints
            ? checkpointMetricDescription(column)
            : column.direction === "min" ? "Lower is better" : "Higher is better"
        }`;
        sortButton.setAttribute(
          "aria-label",
          `Sort by ${fullLabel}, ${nextDirection}`,
        );
        const labelGroup = document.createElement("span");
        labelGroup.className = "source-sort-label";
        const label = document.createElement("span");
        label.textContent = column.label;
        labelGroup.append(label);
        const indicator = document.createElement("span");
        indicator.className = "source-sort-indicator";
        indicator.setAttribute("aria-hidden", "true");
        indicator.hidden = showingCheckpoints && !active;
        indicator.textContent = active
          ? this.sort.direction === "ascending" ? "↑" : "↓"
          : "↕";
        sortButton.append(labelGroup, indicator);
        sortButton.addEventListener("click", () => {
          this.sort = { metric: column.metric, direction: nextDirection };
          this.renderView();
        });
        cell.append(sortButton);
      } else {
        cell.textContent = column.label;
      }
      headerRow.append(cell);
    });
    head.append(headerRow);
    const body = document.createElement("tbody");
    const items = showingRuns
      ? this.sort.metric
        ? sortRunItems(this.items, this.sort)
        : rankRunItems(this.items, runRankingColumns)
      : showingCheckpoints && this.sort.metric
        ? sortRunItems(this.items, this.sort)
      : this.items;
    items.forEach((item) => {
      const row = document.createElement("tr");
      const isEfficiencyLeader = (
        showingRuns
        && efficiency?.item?.run_id === item.run_id
      );
      if (isEfficiencyLeader) row.classList.add("efficiency-leader");
      row.tabIndex = this.hasControl() ? 0 : -1;
      row.setAttribute("role", "button");
      row.setAttribute("aria-disabled", String(!this.hasControl()));
      const finish = showingRuns ? runFinishPresentation(item) : null;
      const values = showingRuns
        ? [
            [item.description || item.name || item.run_id, item.run_id, "run-cell"],
            [
              item.recipe || "—",
              "",
              "recipe-cell",
            ],
            [item.seed ?? "—", "", "data-cell"],
            [
              finish.label,
              finish.detail,
              `finish-reason ${finish.tone}`,
              finish.evidence,
            ],
            ...runMetricColumns.map((column) => [
              formatMetricValue(column.metric, item.metrics?.[column.metric]),
              "",
              "data-cell",
            ]),
            [formatDate(item.updated_at || item.created_at), "", "data-cell"],
            [null, "", "inspection-cell"],
          ]
        : (() => {
            const checkpointName = /final/i.test(String(item.purpose || ""))
              ? "Final checkpoint"
              : `Checkpoint at ${Number(item.step).toLocaleString()} steps`;
            return [
              ...(showingCheckpoints ? [[null, "", "source-selection-cell"]] : []),
              [
                checkpointName,
                showingCheckpoints
                  ? [item.checkpoint_id, item.sha256].filter(Boolean).join(" · ")
                  : "",
                "checkpoint-cell",
              ],
              ...(showingCheckpoints ? [[item.purpose || "—"]] : []),
              [Number(item.step).toLocaleString(), "", "data-cell"],
              ...checkpointMetricColumns.map((column) => [
                formatMetricValue(column.metric, item.metrics?.[column.metric]),
                "",
                "checkpoint-metric-cell",
                null,
                {
                  isBest: checkpointMetricIsBest(item, column.metric),
                  label: column.label || metricLabel(column.metric),
                  column,
                },
              ]),
              ...(showingCheckpoints
                ? [
                    [formatBytes(item.size_bytes), "", "data-cell"],
                    [formatDate(item.created_at), "", "data-cell"],
                  ]
                : []),
            ];
          })();
      values.forEach(([
        primary,
        secondary = "",
        className = "",
        evidence = null,
        metadata = null,
      ]) => {
        const cell = document.createElement("td");
        if (className) cell.className = className;
        if (className.includes("source-selection-cell")) {
          const selectable = showingCheckpoints && checkpointCanEvaluate(item);
          cell.addEventListener("click", (event) => event.stopPropagation());
          const checkbox = document.createElement("input");
          checkbox.type = "checkbox";
          checkbox.checked = this.selectedCheckpoints.has(item.checkpoint_id);
          checkbox.disabled = !selectable || this.evaluating;
          checkbox.setAttribute(
            "aria-label",
            selectable
              ? `Select ${item.checkpoint_id} for evaluation`
              : `${item.checkpoint_id} cannot be evaluated again`,
          );
          checkbox.addEventListener("change", () => {
            if (checkbox.checked) this.selectedCheckpoints.add(item.checkpoint_id);
            else this.selectedCheckpoints.delete(item.checkpoint_id);
            this.renderView();
          });
          cell.append(checkbox);
          row.append(cell);
          return;
        }
        if (className.includes("inspection-cell")) {
          const inspect = button("Inspect", { iconName: "code", quiet: true });
          inspect.addEventListener("click", (event) => {
            event.stopPropagation();
            void this.inspectRun(item.run_id).catch(
              (error) => this.showToast(String(error?.message || error), true),
            );
          });
          cell.append(inspect);
          row.append(cell);
          return;
        }
        if (className.includes("finish-reason") && evidence) {
          const status = document.createElement("span");
          status.className = "finish-status";
          status.textContent = String(primary);

          const comparison = document.createElement("div");
          comparison.className = "finish-evidence";
          [
            ["Observed", evidence.observed],
            ["Required", evidence.required],
          ].forEach(([labelText, valueText]) => {
            const item = document.createElement("div");
            item.className = "finish-evidence-item";
            const value = document.createElement("strong");
            value.className = "finish-evidence-value";
            value.textContent = String(valueText);
            const label = document.createElement("span");
            label.className = "finish-evidence-label";
            label.textContent = labelText;
            item.append(value, label);
            comparison.append(item);
          });

          const metric = document.createElement("small");
          metric.className = "finish-evidence-metric";
          metric.textContent = String(evidence.metric);
          cell.append(status, comparison, metric);
          if (evidence.step) {
            const step = document.createElement("small");
            step.className = "finish-evidence-step";
            step.textContent = String(evidence.step);
            cell.append(step);
          }
          row.append(cell);
          return;
        }
        const main = document.createElement("span");
        main.textContent = String(primary);
        if (className.includes("checkpoint-cell")) {
          main.title = String(item.checkpoint_id || "");
        }
        if (className.includes("run-cell")) {
          const presentation = runStatePresentation(item);
          const identity = document.createElement("button");
          identity.type = "button";
          identity.className = "run-identity";
          identity.disabled = !this.hasControl();
          const state = document.createElement("span");
          state.className = `run-state ${presentation.tone}`;
          state.title = `Run state: ${presentation.label}`;
          state.setAttribute("aria-label", `Run state: ${presentation.label}`);
          state.append(icon(presentation.iconName));
          const text = document.createElement("div");
          text.className = "run-identity-text";
          main.className = "run-name";
          text.append(main);
          if (secondary) {
            const small = document.createElement("small");
            small.textContent = String(secondary);
            text.append(small);
          }
          identity.append(state, text);
          cell.append(identity);
          const success = renderSuccessBadges(item);
          if (success) cell.append(success);
        } else {
          cell.append(main);
        }
        if (className.includes("checkpoint-metric-cell") && metadata?.isBest) {
          const badge = document.createElement("span");
          const badgeLabel = checkpointMetricBestBadge(metadata.column);
          const description = `${badgeLabel}: ${String(metadata.label).toLowerCase()}`;
          badge.className = "checkpoint-best-badge";
          badge.textContent = badgeLabel;
          badge.title = description;
          badge.setAttribute("aria-label", description);
          cell.append(badge);
        }
        if (className.includes("run-cell") && isEfficiencyLeader) {
          const badge = document.createElement("span");
          badge.className = "source-leader-badge";
          badge.textContent = efficiency.evidence === "evaluation"
            ? "Most efficient"
            : "Training lead";
          cell.append(badge);
        }
        if (className.includes("recipe-cell")) {
          const variant = recipeVariantPresentation(item);
          const variation = document.createElement("small");
          variation.className = "recipe-variant";
          variation.textContent = variant.summary;
          variation.title = variant.detail;
          cell.append(variation);
          if (item.recipe_sha256) {
            const revision = document.createElement("small");
            revision.className = "recipe-revision";
            revision.textContent = `rev ${String(item.recipe_sha256).slice(0, 12)}`;
            revision.title = `Recipe SHA-256: ${item.recipe_sha256}`;
            cell.append(revision);
          }
        }
        if (secondary && !className.includes("run-cell")) {
          const small = document.createElement("small");
          small.textContent = String(secondary);
          cell.append(small);
        }
        row.append(cell);
      });
      if (this.selectedCheckpoints.has(item.checkpoint_id)) {
        row.classList.add("selected");
      }
      const activate = () => {
        if (!this.hasControl()) return;
        if (showingRuns) {
          this.navigate({
            level: "runs",
            run_id: item.run_id,
            checkpoint_id: "",
          });
        } else {
          this.selectCheckpoint(item);
        }
      };
      row.addEventListener("click", activate);
      row.addEventListener("keydown", (event) => {
        if (event.target !== row) return;
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          activate();
        }
      });
      body.append(row);
    });
    table.append(head, body);
    scroll.append(table);
    return scroll;
  }

  loadingState(message) {
    const state = document.createElement("div");
    state.className = "source-loading";
    const spinner = document.createElement("span");
    spinner.className = "spinner";
    spinner.setAttribute("aria-hidden", "true");
    const text = document.createElement("p");
    text.textContent = message;
    state.append(spinner, text);
    return state;
  }

  renderProgress() {
    const wrap = document.createElement("div");
    wrap.className = "source-centered";
    wrap.append(this.loadingState(this.app.message || "Preparing playback…"));
    if (this.app.has_active_runner) {
      const cancel = button("Back to current run", { iconName: "arrow-left", quiet: true });
      cancel.disabled = !this.hasControl();
      cancel.addEventListener("click", () => this.command("cancel_source"));
      wrap.append(cancel);
    }
    return wrap;
  }

  renderFailure() {
    const wrap = document.createElement("div");
    wrap.className = "source-centered source-failure";
    const message = document.createElement("p");
    message.textContent = this.app.error || "The checkpoint could not be opened.";
    const actions = document.createElement("div");
    actions.className = "source-actions";
    const retry = button("Retry", { iconName: "refresh", primary: true });
    retry.disabled = !this.hasControl();
    retry.addEventListener("click", () => this.command("retry_source"));
    const choose = button("Choose another", { iconName: "folder-search", quiet: true });
    choose.disabled = !this.hasControl();
    choose.addEventListener("click", () => this.browseCurrentSource());
    actions.append(retry, choose);
    if (this.app.has_active_runner) {
      const cancel = button("Back to current run", { iconName: "arrow-left", quiet: true });
      cancel.disabled = !this.hasControl();
      cancel.addEventListener("click", () => this.command("cancel_source"));
      actions.append(cancel);
    }
    wrap.append(message, actions);
    return wrap;
  }

}
