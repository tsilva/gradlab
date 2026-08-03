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

const GOAL_CONFIGURATION_KINDS = {
  current_default: { label: "Current default", group: "current" },
  current_modified: { label: "Current modified", group: "current" },
  previous_default: { label: "Previous default", group: "previous" },
  previous_modified: { label: "Previous modified", group: "previous" },
};

export function goalConfigurationPresentation(item, nowValue = Date.now()) {
  const kind = String(item?.configuration_kind || "previous_default");
  const kindPresentation = GOAL_CONFIGURATION_KINDS[kind] || {
    label: "Previous configuration",
    group: "previous",
  };
  const runCount = Math.max(0, Number(item?.run_count) || 0);
  const runLabel = `${runCount.toLocaleString()} ${runCount === 1 ? "run" : "runs"}`;
  const firstUsed = item?.first_used_at ? formatCalendarDate(item.first_used_at) : "—";
  const lastActivity = item?.last_activity_at
    ? formatCalendarDate(item.last_activity_at)
    : "—";
  const comparisonAvailable = Boolean(item?.comparison_available);
  const hasExactDefinition = Boolean(item?.exact_resolution_run_id);
  return {
    kind,
    kindLabel: kindPresentation.label,
    group: kindPresentation.group,
    displayLabel: String(
      item?.display_label || "Behavioral difference unavailable — no exact goal proof",
    ),
    activity: runCount
      ? `${runLabel} · First used ${firstUsed} · Last activity ${lastActivity}`
      : "No runs yet",
    actionLabel: kind === "current_default" || !comparisonAvailable || !hasExactDefinition
      ? "View definition"
      : "Compare",
  };
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
    "train/episode/return/shaped/from/target/rolling_up_to_100/mean": "Mean target-start return (up to 100)",
    "train/episode/return/shaped/from/target/window_100/mean": "Mean target-start return (last 100)",
    "train/outcome/success/across_starts/window_100/rate/min": "Min success (last 100)",
    "eval/full/outcome/success/across_starts/rate/min": "Min success",
    "eval/full/outcome/success/across_starts/rate/mean": "Mean success",
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

export function checkpointEvaluationCell(item) {
  const evaluation = item?.evaluation;
  const queue = item?.evaluation_queue;
  const state = String(queue?.state || "");
  const presentation = {
    queued: ["Queued", "Waiting to be submitted"],
    running: ["Starting", "The local evaluation supervisor is preparing this checkpoint"],
    retry_wait: ["Retrying", queue?.message || "Waiting for the next safe retry"],
    waiting_for_training_terminal: [
      "Waiting for training",
      queue?.message || "Evaluation starts after training is terminal",
    ],
    waiting_for_run_lease: [
      "Waiting for writer",
      queue?.message || "Waiting for exclusive run-writer authority",
    ],
    submitted: ["Running", "Submitted to the evaluation worker"],
    submission_uncertain: ["Reconciling", queue?.message || "Checking submission state"],
    awaiting_projection: ["Syncing", queue?.message || "Publishing verified evidence"],
    flusher_unavailable: [
      "Flusher unavailable",
      queue?.message || "The durable request will resume on the next startup attempt",
    ],
    blocked: ["Blocked", queue?.message || "Operator action is required"],
    failed: ["Failed", queue?.message || "Evaluation could not be completed"],
    expired: ["Expired", queue?.message || "Evaluation did not complete"],
    canceled: ["Canceled", queue?.message || "Canceled by the operator"],
  }[state];
  if (presentation) {
    return [presentation[0], presentation[1], `evaluation-cell ${state}`];
  }
  if (!evaluation || typeof evaluation !== "object") return ["—"];
  const details = [];
  const completed = Number(evaluation.episodes_completed);
  const planned = Number(evaluation.episodes_planned);
  if (Number.isFinite(completed) && Number.isFinite(planned)) {
    details.push(`${completed.toLocaleString()} / ${planned.toLocaleString()} episodes`);
  }
  (Array.isArray(evaluation.criteria) ? evaluation.criteria : []).forEach((criterion) => {
    const label = metricLabel(criterion.metric);
    const threshold = formatMetricValue(criterion.metric, criterion.threshold);
    if (
      criterion.value !== null
      && criterion.value !== undefined
      && criterion.value !== ""
      && Number.isFinite(Number(criterion.value))
    ) {
      const value = formatMetricValue(criterion.metric, criterion.value);
      details.push(`${label}: ${value} ${criterion.operator} ${threshold}`);
    } else {
      details.push(`${label} ${criterion.operator} ${threshold}`);
    }
  });
  const failureCount = Number(evaluation.failure_count);
  if (Number.isFinite(failureCount) && failureCount > 0) {
    details.push(`${failureCount.toLocaleString()} failed`);
  }
  return [
    evaluation.pass ? "Passed" : "Failed",
    details.join(" · "),
    `evaluation-cell ${evaluation.pass ? "accepted" : "rejected"}`,
  ];
}

export function checkpointRankTags(item) {
  const tags = [];
  if (item?.best_training) tags.push("Best training");
  if (item?.best_evaluation) tags.push("Best eval");
  return tags;
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
      label: route.environment_id,
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
      label: route.goal_id,
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
      label: route.goal_variant_id,
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
      label: route.run_id,
      current: !route.checkpoint_id,
      route: {
        level: "runs",
        checkpoint_id: "",
      },
    });
  }
  if (route?.checkpoint_id) {
    items.push({
      label: route.checkpoint_id,
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
      openInspection,
      openSourceRoute,
      catalogRequestTimeoutMs = 30_000,
    },
  ) {
    this.root = root;
    this.breadcrumbsRoot = breadcrumbsRoot;
    this.token = token;
    this.command = command;
    this.getState = getState;
    this.showToast = showToast;
    this.openInspection = openInspection;
    this.openSourceRoute = openSourceRoute;
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
    this.loadedKey = "";
    this.requestSerial = 0;
    this.requestController = null;
    this.loadingKey = "";
    this.catalogRequestTimeoutMs = catalogRequestTimeoutMs;
    this.searchTimer = null;
    this.pollTimer = null;
    this.selectedCheckpoints = new Set();
    this.evaluating = false;
    this.autoSelectedRoute = "";
    this.activeBreadcrumbRoute = "";
    this.initialEnvironmentCatalog = null;
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
    if (this.historyEnabled) window.addEventListener("popstate", this.onPopState);
  }

  render(snapshot) {
    this.app = snapshot?.app || { phase: "active" };
    if (this.app.catalog && typeof this.app.catalog === "object") {
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
      this.items = [];
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
      this.selectedCheckpoints.clear();
      this.autoSelectedRoute = "";
      this.syncUrl("replace");
    }
    this.hydrateInitialEnvironments();
    this.renderView();
    if (this.app.phase === "selecting") this.ensureLoaded();
    this.updatePolling();
  }

  stop({ preserveBreadcrumbs = false } = {}) {
    clearTimeout(this.searchTimer);
    clearInterval(this.pollTimer);
    this.pollTimer = null;
    this.requestController?.abort();
    this.requestController = null;
    this.requestSerial += 1;
    this.loading = false;
    this.loadingKey = "";
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
      && !this.breadcrumbsRoot.hidden
    ) {
      return;
    }
    this.stop({ preserveBreadcrumbs: true });
    this.app = app;
    if (!route.checkpoint_id) {
      this.activeBreadcrumbRoute = "";
      this.breadcrumbsRoot.replaceChildren();
      this.breadcrumbsRoot.hidden = true;
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
    this.activeBreadcrumbRoute = signature;
    this.renderBreadcrumbs(this.breadcrumbsRoot);
    this.breadcrumbsRoot.hidden = false;
    this.syncUrl("replace");
  }

  hasControl() {
    return Boolean(this.getState()?.hasControl);
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
    this.items = Array.isArray(catalog.items) ? [...catalog.items] : [];
    this.metricColumns = Array.isArray(catalog.metric_columns)
      ? [...catalog.metric_columns]
      : [];
    this.fallbackMetricColumns = Array.isArray(catalog.fallback_metric_columns)
      ? [...catalog.fallback_metric_columns]
      : [];
    this.nextCursor = catalog.next_cursor || null;
    this.loadedKey = this.routeKey();
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
      return `/api/catalog/environments/${encodeURIComponent(this.route.environment_id)}/goals/${encodeURIComponent(this.route.goal_id)}/variants?${query}`;
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
      const response = await fetch(this.endpoint(cursor, { force }), {
        headers: { Authorization: `Bearer ${this.token}` },
        cache: "no-store",
        signal: controller.signal,
      });
      const payload = await response.json().catch(() => ({}));
      if (response.status === 409 && append) {
        this.items = [];
        this.nextCursor = null;
        this.loadedKey = "";
        this.showToast("Catalog changed; reloading the first page.", false);
        queueMicrotask(() => this.load({ force: true }));
        return;
      }
      if (!response.ok) throw new Error(payload.error || `Catalog request failed (${response.status})`);
      if (serial !== this.requestSerial || key !== this.routeKey()) return;
      const received = Array.isArray(payload.items) ? payload.items : [];
      this.items = append ? [...this.items, ...received] : received;
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

  updatePolling() {
    const shouldPoll = (
      this.app.phase === "selecting"
      && this.route.level === "runs"
      && Boolean(this.route.run_id)
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

  applyRoute(route) {
    this.route = { ...this.route, ...route };
    this.query = "";
    this.items = [];
    this.metricColumns = [];
    this.fallbackMetricColumns = [];
    this.sort = { metric: "", direction: "" };
    this.nextCursor = null;
    this.loadedKey = "";
    this.error = "";
    this.selectedCheckpoints.clear();
    this.autoSelectedRoute = "";
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

  navigate(route, { historyMode = "push" } = {}) {
    const nextRoute = { ...this.route, ...route };
    const commandId = this.command("browse_sources", { route: nextRoute });
    if (commandId === null) return false;
    this.applyRoute(route);
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
    this.route = route;
    this.syncUrl(historyMode);
    this.command("select_source", {
      source: {
        kind: "manifest",
        value: item.manifest_url,
        run_id: item.run_id,
        checkpoint_id: item.checkpoint_id,
        seed: checkpointPlaybackSeed(item),
      },
      route: { ...route },
    });
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
    this.searchTimer = window.setTimeout(() => {
      this.items = [];
      this.nextCursor = null;
      this.loadedKey = "";
      this.load();
    }, 220);
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

    if (this.freshness !== "fresh" || this.catalogWarnings.length) {
      const notice = document.createElement("div");
      notice.className = `source-notice catalog-${this.freshness}`;
      const label = this.freshness === "stale"
        ? "Showing stale catalog data."
        : this.freshness === "partial"
          ? "Some catalog evidence is unavailable."
          : "Catalog warning.";
      const messages = this.catalogWarnings
        .map((warning) => String(warning?.message || "").trim())
        .filter(Boolean);
      notice.textContent = [label, ...messages].join(" ");
      shell.append(notice);
    }

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
          "A configuration groups runs that used the same resolved goal behavior. "
          + "Previous configurations remain available for reproducible playback."
        );
        shell.append(description);
      }
      shell.append(this.renderSearch());
      if (this.route.level === "runs" && this.route.run_id) {
        shell.append(this.renderEvaluationActions());
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
    if (this.route.level === "goal_variants") return "Choose a goal configuration";
    if (this.route.level === "goals") return "Choose a goal";
    return "Choose an environment";
  }

  renderBreadcrumbs(nav) {
    nav.replaceChildren();
    sourceBreadcrumbItems(this.route).forEach((item) => {
      const crumb = button(item.label, { quiet: true });
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
    return wrap;
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
    if (this.loading && !this.items.length) {
      body.append(this.loadingState("Loading catalog…"));
      return body;
    }
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
    body.append(
      this.route.level === "environments"
        ? this.renderEnvironments()
        : this.route.level === "goals"
          ? this.renderGoals()
          : this.route.level === "goal_variants"
            ? this.renderGoalVariants()
          : this.route.level === "runs"
            ? this.renderRunResults()
            : this.renderTable(),
    );
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
    this.items.forEach((environment) => {
      const row = document.createElement("button");
      row.type = "button";
      row.className = "environment-row";
      row.disabled = !this.hasControl();
      const name = document.createElement("strong");
      name.textContent = environment.name;
      const meta = document.createElement("span");
      const goalLabel = Number(environment.goal_count) === 1 ? "goal" : "goals";
      meta.textContent = `${Number(environment.goal_count).toLocaleString()} ${goalLabel}`;
      row.append(name, meta);
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
      const name = document.createElement("strong");
      name.textContent = goal.goal_id;
      identity.append(name);
      const meta = document.createElement("span");
      const recipeLabel = Number(goal.recipe_count) === 1 ? "recipe" : "recipes";
      meta.textContent = `${goal.title || goal.goal_slug} · ${Number(goal.recipe_count).toLocaleString()} ${recipeLabel}`;
      identity.append(meta);
      if (goal.goal_slug && goal.goal_slug !== goal.goal_id) {
        const slug = document.createElement("small");
        slug.textContent = goal.goal_slug;
        identity.append(slug);
      }
      navigate.append(identity);
      navigate.addEventListener("click", () => this.navigate({
        level: "goal_variants",
        goal_id: goal.goal_id,
        goal_variant_id: "",
        run_id: "",
        checkpoint_id: "",
      }));
      const inspect = button("Inspect", { iconName: "code", quiet: true });
      inspect.classList.add("goal-row-inspect");
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
    container.className = "goal-configuration-groups";
    [
      { id: "current", label: "Current" },
      { id: "previous", label: "Previous configurations" },
    ].forEach((group) => {
      const variants = this.items
        .map((variant) => ({
          variant,
          presentation: goalConfigurationPresentation(variant),
        }))
        .filter(({ presentation }) => presentation.group === group.id);
      if (!variants.length) return;

      const section = document.createElement("section");
      section.className = "goal-configuration-group";
      const heading = document.createElement("h3");
      heading.textContent = group.label;
      const list = document.createElement("div");
      list.className = "goal-list goal-variant-list";

      variants.forEach(({ variant, presentation }) => {
        const row = document.createElement("div");
        row.className = "goal-row goal-variant-row";
        const navigate = document.createElement("button");
        navigate.type = "button";
        navigate.className = "goal-row-navigation";
        navigate.disabled = !this.hasControl();
        const identity = document.createElement("div");
        const title = document.createElement("div");
        title.className = "goal-configuration-title";
        const badge = document.createElement("span");
        badge.className = `goal-configuration-badge ${presentation.kind}`;
        badge.textContent = presentation.kindLabel;
        const name = document.createElement("strong");
        name.textContent = presentation.displayLabel;
        title.append(badge, name);
        identity.append(title);
        const meta = document.createElement("span");
        meta.className = "goal-configuration-activity";
        meta.textContent = presentation.activity;
        identity.append(meta);
        if (!variant.comparison_available) {
          const detail = document.createElement("small");
          detail.className = "goal-configuration-warning";
          detail.textContent = (
            "A verified comparison is unavailable; technical identity is in the definition."
          );
          identity.append(detail);
        }
        navigate.append(identity);
        navigate.setAttribute(
          "aria-label",
          `${presentation.kindLabel}: ${presentation.displayLabel}. ${presentation.activity}`,
        );
        navigate.addEventListener("click", () => this.navigate({
          level: "runs",
          goal_variant_id: variant.variant_id,
          run_id: "",
          checkpoint_id: "",
        }));
        const inspect = button(presentation.actionLabel, { iconName: "code", quiet: true });
        inspect.classList.add("goal-row-inspect");
        inspect.setAttribute(
          "aria-label",
          `${presentation.actionLabel} for ${presentation.kindLabel.toLowerCase()}: ${presentation.displayLabel}`,
        );
        inspect.addEventListener("click", () => {
          void this.inspectGoalVariant(variant).catch(
            (error) => this.showToast(String(error?.message || error), true),
          );
        });
        row.append(navigate, inspect);
        list.append(row);
      });
      section.append(heading, list);
      container.append(section);
    });
    return container;
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
          { label: "", selection: true },
          { label: "Checkpoint" },
          { label: "Purpose" },
          { label: "Step" },
          ...checkpointMetricColumns.map((column) => ({
            ...column,
            label: metricLabel(column.metric),
          })),
          { label: "Evaluation" },
          { label: "Size" },
          { label: "Created" },
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
        sortButton.title = `${column.label} · ${
          column.direction === "min" ? "lower is better" : "higher is better"
        }`;
        sortButton.setAttribute(
          "aria-label",
          `Sort by ${column.label}, ${nextDirection}`,
        );
        const label = document.createElement("span");
        label.textContent = column.label;
        const indicator = document.createElement("span");
        indicator.className = "source-sort-indicator";
        indicator.setAttribute("aria-hidden", "true");
        indicator.textContent = active
          ? this.sort.direction === "ascending" ? "↑" : "↓"
          : "↕";
        sortButton.append(label, indicator);
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
            [item.seed ?? "—"],
            [
              finish.label,
              finish.detail,
              `finish-reason ${finish.tone}`,
              finish.evidence,
            ],
            ...runMetricColumns.map((column) => [
              formatMetricValue(column.metric, item.metrics?.[column.metric]),
            ]),
            [formatDate(item.updated_at || item.created_at)],
            [null, "", "inspection-cell"],
          ]
        : [
            [null, "", "source-selection-cell"],
            [
              item.promoted ? `${item.checkpoint_id} · promoted` : item.checkpoint_id,
              item.sha256,
              "checkpoint-cell",
            ],
            [item.purpose],
            [Number(item.step).toLocaleString()],
            ...checkpointMetricColumns.map((column) => [
              formatMetricValue(column.metric, item.metrics?.[column.metric]),
            ]),
            checkpointEvaluationCell(item),
            [formatBytes(item.size_bytes)],
            [formatDate(item.created_at)],
          ];
      values.forEach(([primary, secondary = "", className = "", evidence = null]) => {
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
        if (className.includes("evaluation-cell")) main.className = "evaluation-verdict";
        main.textContent = String(primary);
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
        } else {
          cell.append(main);
        }
        if (className.includes("checkpoint-cell")) {
          const tags = checkpointRankTags(item);
          if (tags.length) {
            const badges = document.createElement("div");
            badges.className = "checkpoint-rank-badges";
            tags.forEach((label) => {
              const badge = document.createElement("span");
              badge.className = label === "Best eval"
                ? "checkpoint-rank-badge evaluation"
                : "checkpoint-rank-badge training";
              badge.textContent = label;
              badges.append(badge);
            });
            cell.append(badges);
          }
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
