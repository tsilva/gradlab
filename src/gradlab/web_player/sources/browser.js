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

export function formatGoalConfigurationDate(value, nowValue = Date.now()) {
  if (!value) return "—";
  const date = new Date(value);
  const now = new Date(nowValue);
  if (Number.isNaN(date.getTime())) return "—";
  const elapsed = now.getTime() - date.getTime();
  const age = Math.abs(elapsed);
  if (Number.isFinite(age) && age < 30 * 86_400_000) {
    if (age < 1_000) return "just now";
    const [duration, unit] = age < 60_000
      ? [1_000, "second"]
      : age < 3_600_000
        ? [60_000, "minute"]
        : age < 86_400_000
          ? [3_600_000, "hour"]
          : [86_400_000, "day"];
    const amount = Math.floor(age / duration) * (elapsed >= 0 ? -1 : 1);
    return new Intl.RelativeTimeFormat("en", { numeric: "always" }).format(amount, unit);
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
const ENVIRONMENT_FAVORITES_STORAGE_KEY = "gradlab.playback.favorite-environments.v1";
const ENVIRONMENT_FAVORITES_COOKIE_KEY = "gradlab_playback_favorite_environments_v1";
const ENVIRONMENT_FAVORITES_COOKIE_MAX_AGE_SECONDS = 60 * 60 * 24 * 365 * 10;

function normalizedEnvironmentFavorites(value) {
  if (!Array.isArray(value)) return null;
  return new Set(value.map((name) => String(name).trim()).filter(Boolean));
}

function environmentFavoritesFromCookie(cookieHeader) {
  const prefix = `${ENVIRONMENT_FAVORITES_COOKIE_KEY}=`;
  const entry = String(cookieHeader || "")
    .split(";")
    .map((part) => part.trim())
    .find((part) => part.startsWith(prefix));
  if (!entry) return null;
  try {
    return normalizedEnvironmentFavorites(
      JSON.parse(decodeURIComponent(entry.slice(prefix.length))),
    );
  } catch {
    return null;
  }
}

export function readEnvironmentFavorites(storage, cookieHeader) {
  const cookieFavorites = environmentFavoritesFromCookie(
    cookieHeader === undefined ? globalThis.document?.cookie : cookieHeader,
  );
  if (cookieFavorites !== null) return cookieFavorites;
  try {
    const target = storage === undefined ? globalThis.window?.localStorage : storage;
    return normalizedEnvironmentFavorites(
      JSON.parse(target?.getItem(ENVIRONMENT_FAVORITES_STORAGE_KEY) || "[]"),
    ) || new Set();
  } catch {
    return new Set();
  }
}

export function writeEnvironmentFavorites(favorites, storage, cookieTarget) {
  const names = [...favorites]
    .map((name) => String(name).trim())
    .filter(Boolean)
    .sort((left, right) => left.localeCompare(right));
  let wrote = false;
  try {
    const target = storage === undefined ? globalThis.window?.localStorage : storage;
    if (target) {
      target.setItem(ENVIRONMENT_FAVORITES_STORAGE_KEY, JSON.stringify(names));
      wrote = true;
    }
  } catch {
    // Private browsing and hardened browser settings may reject local storage.
  }
  try {
    const target = cookieTarget === undefined ? globalThis.document : cookieTarget;
    if (target) {
      target.cookie = [
        `${ENVIRONMENT_FAVORITES_COOKIE_KEY}=${encodeURIComponent(JSON.stringify(names))}`,
        "Path=/",
        `Max-Age=${ENVIRONMENT_FAVORITES_COOKIE_MAX_AGE_SECONDS}`,
        "SameSite=Strict",
      ].join("; ");
      wrote = true;
    }
  } catch {
    // Keep favorites usable for this launch when persistent cookies are unavailable.
  }
  return wrote;
}

export function toggleEnvironmentFavorite(favorites, name) {
  const next = new Set(favorites || []);
  if (next.has(name)) next.delete(name);
  else next.add(name);
  return next;
}

export function sortEnvironmentItems(items, favorites) {
  const favoriteNames = favorites instanceof Set ? favorites : new Set(favorites || []);
  return [...items].sort((left, right) => {
    const leftFavorite = favoriteNames.has(String(left?.name || ""));
    const rightFavorite = favoriteNames.has(String(right?.name || ""));
    if (leftFavorite !== rightFavorite) return leftFavorite ? -1 : 1;
    if (leftFavorite) {
      return String(left?.name || "").localeCompare(String(right?.name || ""));
    }
    return environmentEvidenceRank(left) - environmentEvidenceRank(right);
  });
}

export function successBadgeLabels(item) {
  const badges = Array.isArray(item?.success_badges) ? item.success_badges : [];
  const present = new Set(badges.map((badge) => String(badge)));
  return SUCCESS_BADGES.filter((badge) => present.has(badge));
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

export function goalConfigurationSummary(item, presentation) {
  if (presentation.kind === "current_default") return "Matches checked-in goal";
  const label = String(item?.display_label || "").trim();
  const readableParts = label
    .split(" · ")
    .map((part) => part.trim())
    .filter((part) => (
      part
      && part.length <= 96
      && !/[\[{]/.test(part)
      && !/^\+\d+ more$/i.test(part)
    ));
  if (readableParts.length) {
    const summary = readableParts.slice(0, 2).join(" · ");
    const count = presentation.differenceCount;
    if (count !== null && count > readableParts.length) {
      return `${summary} · ${count.toLocaleString()} changes total`;
    }
    return summary;
  }
  return `${presentation.differenceLabel} from current goal`;
}

export function groupGoalConfigurations(items) {
  const variants = Array.isArray(items) ? items : [];
  const groups = [];
  const byRevision = new Map();
  const revisionId = (variant) => String(
    variant?.goal_contract_sha256
    || variant?.variant_id
    || "unknown-goal-revision"
  );
  const ensureGroup = (variant) => {
    const id = revisionId(variant);
    let group = byRevision.get(id);
    if (!group) {
      group = {
        revisionId: id,
        defaultVariant: null,
        overrides: [],
        current: false,
      };
      byRevision.set(id, group);
      groups.push(group);
    }
    if (String(variant?.configuration_kind || "").startsWith("current_")) {
      group.current = true;
    }
    return group;
  };

  const isDefault = (variant) => (
    variant?.source_relation === "canonical"
    || String(variant?.configuration_kind || "").endsWith("_default")
  );

  variants
    .filter(isDefault)
    .forEach((variant) => {
      ensureGroup(variant).defaultVariant = variant;
    });
  variants
    .filter((variant) => !isDefault(variant))
    .forEach((variant) => {
      ensureGroup(variant).overrides.push(variant);
    });
  return groups.sort((left, right) => Number(right.current) - Number(left.current));
}

export function formatGoalDiffValue(value, { unavailable = false } = {}) {
  if (unavailable) return "—";
  if (value === undefined) return "—";
  const rendered = JSON.stringify(value);
  return rendered === undefined ? String(value) : rendered;
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
    scratch_storage_above_95_percent: "Scratch storage limit reached",
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
    "leader/step": "Checkpoint step",
    "train/step": "Global step",
    "train/return/mean": "Recent target return mean",
    "train/success/min": "Recent all-start success rate min",
    "train/success/mean": "Recent all-start success rate mean",
    "eval/success/min": "Full-eval start success rate min",
    "eval/success/mean": "Full-eval start success rate mean",
    "eval/return/mean": "Mean return",
    "eval/return/max": "Best return",
  };
  if (known[name]) return known[name];
  const progress = name.match(/^eval\/progress\/([^/]+)\/(mean|max)$/);
  if (progress) {
    return `${humanizeMetricPart(progress[1])} ${progress[2]}`;
  }
  const trainingProgress = name.match(
    /^train\/progress\/([^/]+)\/(mean|max|min)$/,
  );
  if (trainingProgress) {
    return `Recent target ${humanizeMetricPart(trainingProgress[1]).toLowerCase()} ${trainingProgress[2]}`;
  }
  return name
    .replace(/^(eval|leader|train)\//, "")
    .split("/")
    .map(humanizeMetricPart)
    .join(" · ");
}

export function formatMetricValue(metric, value) {
  if (value === null || value === undefined || value === "") return "—";
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "—";
  if (String(metric).includes("/rate/") || String(metric).endsWith("/rate") || /\/success\/(?:min|mean|max)$/.test(String(metric))) {
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

export function checkpointMetricRoleLabel(column) {
  const roles = new Set(Array.isArray(column?.roles) ? column.roles : []);
  if (roles.has("objective") && roles.has("acceptance")) return "Objective · gate";
  if (roles.has("objective")) return "Objective";
  if (roles.has("tie_breaker")) return "Tie-breaker";
  if (roles.has("acceptance")) return "Acceptance";
  if (roles.has("training_proxy")) return "Training proxy";
  if (roles.has("optimization")) return "Optimization";
  if (roles.has("observation")) return "Observation";
  return "";
}

export function checkpointMetricHeaderLabel(column) {
  const metric = String(column?.metric || "");
  const prefix = column?.evidence === "training" ? "train/"
    : column?.evidence === "evaluation" ? "eval/" : "";
  return prefix && metric.startsWith(prefix) ? metric.slice(prefix.length) : metric;
}

export function checkpointMetricIsLoading(item, column) {
  if (!item?.training_pending || !column || item.metrics?.[column.metric] != null) {
    return false;
  }
  return column.evidence === "evaluation"
    || (column.evidence === "training"
      && !item.training_loaded_metrics?.includes(column.metric));
}

export function checkpointEvaluationPresentation(item) {
  const queue = item?.evaluation_queue;
  const state = String(queue?.state || "").toLowerCase();
  const evaluation = queue?.evaluation || item?.evaluation;
  const completed = Number(evaluation?.episodes_completed);
  const planned = Number(evaluation?.episodes_planned);
  const progressCount = Number.isFinite(completed) && Number.isFinite(planned) && planned > 0
    ? `${completed}/${planned}` : "";
  const progress = progressCount ? ` · ${progressCount}` : "";
  if (["queued", "retry_wait", "waiting_for_training_terminal", "waiting_for_run_lease", "submitted", "submission_uncertain", "awaiting_projection", "flusher_unavailable"].includes(state)) {
    return { label: "Queued", tone: "pending" };
  }
  if (state === "running") return { label: `Running${progress}`, tone: "running" };
  if (["failed", "blocked", "expired", "canceled"].includes(state)) {
    return { label: humanizeMetricPart(state), tone: "failed" };
  }
  if (evaluation?.source === "monitoring") {
    const title = "Checkpoint Monitoring · observational";
    if (evaluation.status === "queued") return { label: "Queued", tone: "pending", title };
    if (evaluation.status === "running") return { label: `Running${progress}`, tone: "running", title };
    if (evaluation.status === "finalizing") return { label: `Finalizing${progress}`, tone: "pending", title };
    if (evaluation.status === "failed") return { label: "Failed", tone: "failed", title };
    if (evaluation.status === "canceled") return { label: "Canceled", tone: "failed", title };
  }
  if (evaluation?.source === "monitoring" && evaluation.status === "verified") {
    return {
      label: "FINISHED", progress: progressCount, tone: "verified",
      title: "Checkpoint Monitoring · observational",
    };
  }
  if (["accepted", "rejected"].includes(evaluation?.status) || ["accepted", "rejected"].includes(state)) {
    return {
      label: "FINISHED", progress: progressCount, tone: "verified",
      title: evaluation?.manual ? "Manual Acceptance evaluation" : "Acceptance evaluation",
    };
  }
  if (evaluation?.status === "verified") {
    return { label: "FINISHED", progress: progressCount, tone: "verified" };
  }
  return { label: "Not evaluated", tone: "absent" };
}

export function checkpointMetricDescription(column) {
  const role = checkpointMetricRoleLabel(column);
  const evidence = column?.evidence === "evaluation"
    ? Array.isArray(column?.roles) && column.roles.includes("observation")
      ? "Verified observational checkpoint evaluation; no Acceptance authority"
      : "Frozen checkpoint-evaluation evidence"
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

export function environmentSuccessStatus(item, badge) {
  if (["pending", "unavailable"].includes(item?.evidence_status)) {
    const pending = item.evidence_status === "pending";
    return {
      label: pending ? "Loading…" : "Unavailable",
      className: "not-applicable",
      description: pending ? "Loading success evidence" : "Success evidence unavailable; refresh to retry",
    };
  }
  if (successBadgeLabels(item).includes(badge)) {
    return {
      label: "✅",
      className: "satisfied",
      description: badge === "train/success"
        ? "Training success satisfied"
        : "Evaluation success satisfied",
    };
  }
  if (Math.max(0, Number(item?.run_count) || 0) > 0) {
    return {
      label: "❌",
      className: "unsatisfied",
      description: badge === "train/success"
        ? "Training runs exist, but training success is not satisfied"
        : "Training runs exist, but evaluation success is not satisfied",
    };
  }
  return {
    label: "∅",
    className: "not-applicable",
    description: "No results recorded",
  };
}

export function runTrainingEvidenceStatus(item) {
  const trainingSuccess = item?.training_success;
  const criterion = trainingSuccess?.criterion;
  const best = trainingSuccess?.best;
  const criterionDetail = criterion
    ? `${criterion.metric} ${criterion.operator} ${criterion.threshold}`
    : "the declared Training Success criterion";
  if (successBadgeLabels(item).includes("train/success")) {
    return {
      label: "✅",
      className: "met",
      description: trainingSuccess?.status === "met"
        ? `Training Success reached: ${criterionDetail}`
        : "The Run met its declared Training Success proxy",
    };
  }
  const state = String(item?.state || "").trim().toLowerCase();
  if (["pending", "queued", "starting", "running"].includes(state)) {
    return {
      label: "⏳",
      className: "in-progress",
      description: "The Run is still in progress",
    };
  }
  if (trainingSuccess?.status === "unavailable") {
    return {
      label: "∅",
      className: "not-applicable",
      description: `No training samples are available for ${criterionDetail}`,
    };
  }
  return {
    label: "❌",
    className: "not-met",
    description: trainingSuccess?.status === "not_met" && best
      ? `${criterionDetail} was not reached; best observed value ${best.value} at step ${best.step}`
      : "The Run did not meet its declared Training Success proxy",
  };
}

export function runEvaluationEvidenceStatus(item) {
  if (successBadgeLabels(item).includes("eval/success")) {
    return {
      label: "✅",
      className: "accepted",
      description: "Verified evaluation evidence satisfied Acceptance",
    };
  }
  const projected = String(item?.evaluation_status || "").trim().toLowerCase();
  if (projected === "in_progress") {
    return {
      label: "⏳",
      className: "in-progress",
      description: "Evaluation is in progress",
    };
  }
  if (projected === "not_accepted") {
    return {
      label: "❌",
      className: "not-accepted",
      description: "Available evaluation evidence did not satisfy Acceptance",
    };
  }
  if (projected === "not_evaluated") {
    return {
      label: "∅",
      className: "not-evaluated",
      description: "No evaluation evidence is available for this Run",
    };
  }
  const evaluations = item?.evaluations;
  const records = [
    ...(item?.evaluation && typeof item.evaluation === "object" ? [item.evaluation] : []),
    ...(evaluations && typeof evaluations === "object" ? Object.values(evaluations) : []),
  ].filter((record) => record && typeof record === "object");
  const statuses = records.map((record) => String(record.status || "").trim().toLowerCase());
  if (statuses.some((status) => ["pending", "queued", "submitted", "running", "evaluating"].includes(status))) {
    return {
      label: "⏳",
      className: "in-progress",
      description: "Evaluation is in progress",
    };
  }
  if (records.length) {
    return {
      label: "❌",
      className: "not-accepted",
      description: "Available evaluation evidence did not satisfy Acceptance",
    };
  }
  return {
    label: "∅",
    className: "not-evaluated",
    description: "No evaluation evidence is available for this Run",
  };
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
  if (parts.length === 6) return null;
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
  if (!goalVariantId || !runId) return path;
  path += `/variants/${encodeURIComponent(goalVariantId)}`;
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

function canonicalSourceRoute(route) {
  if (route?.level !== "runs" || route?.run_id) return route;
  let level = "environments";
  if (route?.goal_id) level = "goal_variants";
  else if (route?.environment_id) level = "goals";
  return {
    ...route,
    level,
    goal_variant_id: "",
    run_id: "",
    checkpoint_id: "",
  };
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

export function sourceTablePresentation(view) {
  const { items, metricColumns, fallbackMetricColumns, sort, loading, route } = view;
  const showingRuns = route.level === "runs" && !route.run_id;
  const ranking = showingRuns ? activeRunMetricColumns(items, metricColumns, fallbackMetricColumns) : [];
  const metrics = availableRunMetricColumns(items, ranking).map((c) => ({ ...c, label: metricLabel(c.metric) }));
  const checkpointColumns = (evidence) => metricColumns.filter((c) => c.evidence === evidence).map((c) => ({
    ...c, fullLabel: c.label || metricLabel(c.metric), label: checkpointMetricHeaderLabel(c),
  }));
  const training = checkpointColumns("training");
  const evaluation = checkpointColumns("evaluation");
  const trainingPlaceholder = loading && !items.length && !training.length;
  const columns = showingRuns ? [
    { label: "Run" }, { label: "Recipe / variant" }, { label: "Seed" }, { label: "Training result" },
    ...metrics, { label: "Updated" }, { label: "Contract" },
  ] : [
    { label: "", selection: true }, { label: "Step" }, ...training,
    ...(trainingPlaceholder ? [{ label: "", trainingPlaceholder: true }] : []),
    { label: "Status", evaluationStatus: true }, ...evaluation,
  ];
  const widths = columns.map((c) => {
    if (c.selection) return 3;
    if (c.label === "Step") return 15;
    if (c.evaluationStatus) return Math.max(8, Math.min(15, Math.max(0, ...items.map((i) => checkpointEvaluationPresentation(i).label.length)) + 1));
    if (c.trainingPlaceholder) return 8;
    const header = Math.max(...String(c.label || "").split("/").map((p) => p.length));
    const value = Math.max(0, ...items.map((i) => formatMetricValue(c.metric, i.metrics?.[c.metric]).length));
    return Math.max(8, Math.min(18, header + 1), Math.min(18, value + 1));
  });
  return { training, evaluation, trainingPlaceholder, columns, widths,
    totalWidth: widths.reduce((total, width) => total + width, 0), metrics,
    items: sort.metric ? sortRunItems(items, sort) : showingRuns ? rankRunItems(items, ranking) : items,
    efficiency: showingRuns ? bestRunEfficiency(items, metricColumns, fallbackMetricColumns) : null,
  };
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
      selection,
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
    this.checkpointNavigationRoot = checkpointNavigationRoot;
    this.selection = selection;
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
    this.runStatus = null;
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
    this.selectedCheckpoints = new Set();
    this.evaluating = false;
    this.selectedGoalVariantId = "";
    this.goalVariantDiff = null;
    this.goalVariantDiffController = null;
    this.goalVariantDiffSerial = 0;
    this.goalVariantRunPages = new Map();
    this.activityRevision = "";
    this.autoSelectedRoute = "";
    this.activeBreadcrumbRoute = "";
    this.activeCheckpointCache = new Map();
    this.activeCheckpointController = null;
    this.activeCheckpointRequestSerial = 0;
    this.activeCheckpointError = "";
    this.adjacentPrefetchKey = "";
    this.initialEnvironmentCatalog = null;
    this.initialCatalogConsumed = false;
    this.environmentCatalogCache = null;
    this.goalCatalogCache = new Map();
    this.goalActivityCache = new Map();
    this.checkpointCatalogCache = new Map();
    this.searchCatalogCache = new Map();
    this.checkpointTrainingPending = false;
    this.favoriteEnvironments = readEnvironmentFavorites();
    writeEnvironmentFavorites(this.favoriteEnvironments);
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
    let restoredCatalog = false;
    this.hideActiveCheckpointNavigation();
    this.app = snapshot?.app || { phase: "active" };
    if (
      !this.initialCatalogConsumed
      && this.app.catalog
      && typeof this.app.catalog === "object"
    ) {
      this.initialEnvironmentCatalog = this.app.catalog;
    }
    const appRoute = canonicalSourceRoute(this.selection.view.route || {});
    if (
      this.pendingLocationRoute
      && location.pathname !== "/"
      && this.app.phase === "selecting"
    ) {
      const pending = { ...this.pendingLocationRoute };
      this.pendingLocationRoute = null;
      this.lastAppRoute = routeSignature(appRoute);
      this.applyRoute(pending);
      this.selection.browse(pending, snapshot, { historyMode: null });
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
      this.runStatus = null;
      this.checkpointTrainingPending = false;
      this.checkpointTrainingController?.abort();
      this.checkpointTrainingController = null;
      this.checkpointTrainingSerial += 1;
      this.loadedKey = "";
      this.error = "";
      this.selectedCheckpoints.clear();
      this.resetGoalVariantDetail();
      this.autoSelectedRoute = "";
      restoredCatalog = this.restoreEnvironmentCatalog() || this.restoreGoalCatalog()
        || this.restoreGoalActivity() || this.restoreCheckpointCatalog();
      this.syncUrl("replace");
    }
    this.hydrateInitialEnvironments();
    this.renderView();
    if (this.app.phase === "selecting") {
      this.ensureLoaded({ quiet: restoredCatalog });
    }
  }

  stop({ preserveBreadcrumbs = false, preserveCheckpointNavigation = false } = {}) {
    clearTimeout(this.searchTimer);
    this.requestController?.abort();
    this.requestController = null;
    this.requestSerial += 1;
    this.goalEvidenceEpoch = (this.goalEvidenceEpoch || 0) + 1;
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
      this.view?.breadcrumbs([]);
    }
  }

  renderActiveBreadcrumbs(snapshot) {
    const app = snapshot?.app || {};
    const route = this.selection.view.route || {};
    const recording = snapshot?.mode === "trajectory" && Boolean(route.environment_id);
    const signature = routeSignature(route);
    if (
      (route.checkpoint_id || recording)
      && signature === this.activeBreadcrumbRoute
      && signature === routeSignature(this.route)
      && this.app?.phase === "active"
    ) {
      if (!recording) this.renderActiveCheckpointNavigation(route);
      return;
    }
    this.stop({ preserveBreadcrumbs: true, preserveCheckpointNavigation: !recording });
    this.app = app;
    if (!route.checkpoint_id && !recording) {
      this.activeBreadcrumbRoute = "";
      this.view?.breadcrumbs([]);
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
    this.activeCheckpointError = "";
    this.activeBreadcrumbRoute = signature;
    this.renderBreadcrumbs(this.breadcrumbsRoot);
    this.syncUrl("replace");
    if (recording) {
      this.hideActiveCheckpointNavigation();
      return;
    }
    this.renderActiveCheckpointNavigation(this.route);
    void this.loadActiveCheckpointNavigation(this.route, signature);
  }

  hideActiveCheckpointNavigation() {
    this.view?.navigation(null);
  }

  activeCheckpointItems(route = this.route) {
    return this.activeCheckpointCache.get(activeCheckpointCacheKey(route)) || null;
  }

  renderActiveCheckpointNavigation(route = this.route) {
    if (!route?.checkpoint_id || !route?.run_id) {
      this.hideActiveCheckpointNavigation();
      return;
    }
    const items = this.activeCheckpointItems(route);
    const p = checkpointNavigationPresentation(items, route.checkpoint_id);
    const loading = items === null && !this.activeCheckpointError;
    const canChange = this.hasControl() && !this.selection.view.navigationPending;
    this.view?.navigation({
      warning: Boolean(this.activeCheckpointError || (!loading && p.position === null)),
      position: loading ? "… / …" : `${p.position === null ? "—" : p.position.toLocaleString()} / ${p.count.toLocaleString()}`,
      title: this.activeCheckpointError || String(p.current?.checkpoint_id || route.checkpoint_id),
      previousDisabled: !canChange || !p.previous,
      nextDisabled: !canChange || !p.next,
      previousTitle: p.previous ? `Previous checkpoint · step ${Number(p.previous.step).toLocaleString()}` : loading ? "Loading checkpoints" : "This is the first checkpoint",
      nextTitle: p.next ? `Next checkpoint · step ${Number(p.next.step).toLocaleString()}` : loading ? "Loading checkpoints" : "This is the latest checkpoint",
    });
  }

  async loadActiveCheckpointNavigation(route, expectedSignature) {
    const requestRoute = { ...route };
    const cacheKey = activeCheckpointCacheKey(requestRoute);
    this.activeCheckpointController?.abort();
    this.activeCheckpointController = null;
    const serial = ++this.activeCheckpointRequestSerial;
    if (this.activeCheckpointCache.has(cacheKey)) {
      this.activeCheckpointError = "";
      this.prefetchAdjacentCheckpoints(requestRoute);
      return;
    }
    const controller = new AbortController();
    this.activeCheckpointController = controller;
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
    if (!item || this.selection.view.navigationPending) return false;
    const commandId = this.selectCheckpoint(item);
    if (!commandId) return false;
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

  resetGoalVariantDetail() {
    this.goalVariantDiffController?.abort();
    this.goalVariantDiffController = null;
    this.goalVariantDiffSerial += 1;
    this.selectedGoalVariantId = "";
    this.goalVariantDiff = null;
    this.goalVariantRunPages.clear();
    this.activityRevision = "";
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
    this.rememberEnvironmentCatalog();
    return true;
  }

  catalogSnapshot() {
    return {
      sourceItems: [...this.sourceItems],
      metricColumns: [...this.metricColumns],
      fallbackMetricColumns: [...this.fallbackMetricColumns],
      nextCursor: this.nextCursor,
      freshness: this.freshness,
      catalogWarnings: [...this.catalogWarnings],
      catalogSource: this.catalogSource ? { ...this.catalogSource } : null,
      generatedAt: this.generatedAt,
      selectionFence: this.selectionFence,
    };
  }

  restoreCatalogSnapshot(catalog) {
    this.sourceItems = [...catalog.sourceItems];
    this.items = [...this.sourceItems];
    this.metricColumns = [...catalog.metricColumns];
    this.fallbackMetricColumns = [...catalog.fallbackMetricColumns];
    this.nextCursor = catalog.nextCursor;
    this.freshness = catalog.freshness;
    this.catalogWarnings = [...catalog.catalogWarnings];
    this.catalogSource = catalog.catalogSource ? { ...catalog.catalogSource } : null;
    this.generatedAt = catalog.generatedAt;
    this.selectionFence = catalog.selectionFence;
  }

  rememberEnvironmentCatalog() {
    if (this.route.level !== "environments" || this.query.trim()) return false;
    this.environmentCatalogCache = {
      ...this.catalogSnapshot(),
      loaded: this.loadedKey === this.routeKey(),
    };
    return true;
  }

  restoreEnvironmentCatalog() {
    const catalog = this.environmentCatalogCache;
    if (!catalog || this.route.level !== "environments" || this.query.trim()) return false;
    this.restoreCatalogSnapshot(catalog);
    if (catalog.loaded) this.loadedKey = this.routeKey();
    return true;
  }

  rememberGoalCatalog() {
    if (this.route.level !== "goals" || this.query.trim() || !this.route.environment_id) {
      return false;
    }
    this.goalCatalogCache.set(this.route.environment_id, {
      ...this.catalogSnapshot(),
    });
    return true;
  }

  restoreGoalCatalog() {
    if (this.route.level !== "goals" || this.query.trim() || !this.route.environment_id) {
      return false;
    }
    const catalog = this.goalCatalogCache.get(this.route.environment_id);
    if (!catalog) return false;
    this.restoreCatalogSnapshot(catalog);
    this.sourceItems = catalog.sourceItems.map((item) => item.evidence_status === "pending"
      ? { ...item, evidence_status: "unavailable" }
      : item);
    this.items = [...this.sourceItems];
    this.loadedKey = this.routeKey();
    return true;
  }

  rememberGoalActivity() {
    if (this.route.level !== "goal_variants") return;
    this.goalActivityCache.set(this.routeKey(), {
      ...this.catalogSnapshot(),
      activityRevision: this.activityRevision,
      runPages: new Map(this.goalVariantRunPages),
    });
  }

  restoreGoalActivity() {
    if (this.route.level !== "goal_variants") return false;
    const cached = this.goalActivityCache.get(this.routeKey());
    if (!cached) return false;
    this.restoreCatalogSnapshot(cached);
    this.activityRevision = cached.activityRevision;
    this.goalVariantRunPages = new Map(cached.runPages);
    this.loadedKey = this.routeKey();
    return true;
  }

  rememberCheckpointCatalog() {
    if (this.route.level !== "runs" || !this.route.run_id
      || this.route.checkpoint_id || this.query.trim()) return false;
    const key = activeCheckpointCacheKey(this.route);
    this.checkpointCatalogCache.delete(key);
    this.activeCheckpointCache.set(key, [...this.sourceItems]);
    this.checkpointCatalogCache.set(key, {
      ...this.catalogSnapshot(),
      runStatus: this.runStatus ? { ...this.runStatus } : null,
      trainingPending: this.checkpointTrainingPending,
    });
    return true;
  }

  restoreCheckpointCatalog() {
    if (this.route.level !== "runs" || !this.route.run_id
      || this.route.checkpoint_id || this.query.trim()) return false;
    const cached = this.checkpointCatalogCache.get(activeCheckpointCacheKey(this.route));
    if (!cached) return false;
    this.restoreCatalogSnapshot(cached);
    this.runStatus = cached.runStatus ? { ...cached.runStatus } : null;
    this.checkpointTrainingPending = cached.trainingPending;
    this.loadedKey = this.routeKey();
    if (cached.trainingPending) {
      const key = this.loadedKey;
      queueMicrotask(() => {
        if (key === this.routeKey() && this.checkpointTrainingPending
          && !this.checkpointTrainingController) {
          void this.loadCheckpointTraining(key);
        }
      });
    }
    return true;
  }

  rememberSearchCatalog() {
    if (!this.query?.trim()) return false;
    this.searchCatalogCache.set(this.routeKey(), {
      ...this.catalogSnapshot(),
      runStatus: this.runStatus ? { ...this.runStatus } : null,
      activityRevision: this.activityRevision,
      trainingPending: this.checkpointTrainingPending,
      runPages: new Map(this.goalVariantRunPages),
    });
    return true;
  }

  restoreSearchCatalog() {
    if (!this.query?.trim()) return false;
    const cached = this.searchCatalogCache.get(this.routeKey());
    if (!cached) return false;
    this.restoreCatalogSnapshot(cached);
    this.runStatus = cached.runStatus ? { ...cached.runStatus } : null;
    this.activityRevision = cached.activityRevision;
    this.checkpointTrainingPending = cached.trainingPending;
    this.goalVariantRunPages = new Map(cached.runPages);
    this.loadedKey = this.routeKey();
    if (cached.trainingPending && this.route.level === "runs" && this.route.run_id) {
      const key = this.loadedKey;
      queueMicrotask(() => {
        if (key === this.routeKey() && this.checkpointTrainingPending
          && !this.checkpointTrainingController) {
          void this.loadCheckpointTraining(key);
        }
      });
    }
    return true;
  }

  endpoint(cursor = null, { force = false } = {}) {
    const query = new URLSearchParams();
    if (this.query.trim()) query.set("q", this.query.trim());
    if (cursor) query.set("cursor", cursor);
    if (force) query.set("refresh", "1");
    if (this.route.level === "goals") {
      if (!this.query.trim()) query.set("evidence", "0");
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
    return `/api/catalog/environments?${query}`;
  }

  async ensureLoaded({ quiet = false } = {}) {
    const key = this.routeKey();
    if ((this.loading && this.loadingKey === key) || this.loadedKey === key) return;
    await this.load({ quiet });
  }

  async load({ append = false, quiet = false, force = false } = {}) {
    const key = this.routeKey();
    if (this.loading && this.loadingKey === key) return;
    this.requestController?.abort();
    this.checkpointTrainingController?.abort();
    this.checkpointTrainingController = null;
    this.checkpointTrainingSerial += 1;
    const controller = new AbortController();
    this.requestController = controller;
    const cursor = append ? this.nextCursor : null;
    const serial = ++this.requestSerial;
    if (!append) this.goalEvidenceEpoch = (this.goalEvidenceEpoch || 0) + 1;
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
        this.runStatus = payload.run && typeof payload.run === "object"
          ? { ...payload.run }
          : null;
        this.checkpointTrainingPending = payload.training_enrichment === "pending";
        if (this.route.level === "goal_variants") {
          if (force) this.goalVariantRunPages.clear();
          this.activityRevision = String(payload.revision || "");
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
      this.rememberEnvironmentCatalog();
      this.rememberGoalCatalog();
      this.rememberGoalActivity();
      this.rememberCheckpointCatalog();
      this.rememberSearchCatalog();
      this.goalVariantDiff = null;
      if (this.route.level === "goals" && received.some((item) => item.evidence_status === "pending")) {
        void this.loadGoalEvidence(key, this.goalEvidenceEpoch, cursor, received);
      }
      if (
        !append
        && payload.training_enrichment === "pending"
        && this.route.level === "runs"
        && this.route.run_id
      ) {
        void this.loadCheckpointTraining(key);
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
      const message = timedOut
        ? "Catalog request timed out. Try Refresh."
        : String(error?.message || error);
      if (quiet && this.items.length) {
        this.error = "";
        this.showToast(message, true);
      } else {
        this.error = message;
        if (quiet) this.showToast(message, true);
      }
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

  async loadGoalEvidence(expectedKey, epoch, cursor, received = this.sourceItems) {
    const pendingIds = new Set(received.map((item) => item.goal_id));
    const endpoint = this.endpoint(cursor).replace("evidence=0", "evidence=1");
    try {
      const response = await fetch(endpoint, {
        headers: { Authorization: `Bearer ${this.token}` },
        cache: "no-store",
        signal: AbortSignal.timeout(this.catalogRequestTimeoutMs),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Success evidence unavailable");
      if (epoch !== this.goalEvidenceEpoch || expectedKey !== this.routeKey()) return;
      const evidence = new Map((payload.items || []).map((item) => [item.goal_id, item]));
      this.sourceItems = this.sourceItems.map((item) => evidence.has(item.goal_id)
        ? { ...item, ...evidence.get(item.goal_id), evidence_status: "ready" }
        : item);
    } catch {
      if (epoch !== this.goalEvidenceEpoch || expectedKey !== this.routeKey()) return;
      this.sourceItems = this.sourceItems.map((item) => item.evidence_status === "pending" && pendingIds.has(item.goal_id)
        ? { ...item, evidence_status: "unavailable" } : item);
    }
    this.items = [...this.sourceItems];
    this.rememberGoalCatalog();
    this.rememberSearchCatalog();
    this.renderView();
  }

  async loadCheckpointTraining(expectedKey) {
    const runId = String(this.route.run_id || "");
    if (!runId || expectedKey !== this.routeKey()) return;
    this.checkpointTrainingController?.abort();
    const controller = new AbortController();
    this.checkpointTrainingController = controller;
    const serial = ++this.checkpointTrainingSerial;
    this.sourceItems = this.sourceItems.map((item) => ({ ...item, training_pending: true, training_loaded_metrics: [] }));
    this.items = [...this.sourceItems];
    this.renderView();
    const query = new URLSearchParams({ stream: "1" });
    if (this.query.trim()) query.set("q", this.query.trim());
    if (this.route.goal_variant_id) {
      query.set("goal_variant_id", this.route.goal_variant_id);
    }
    let timedOut = false;
    const abortOnTimeout = () => { timedOut = true; controller.abort(); };
    let timeout = setTimeout(abortOnTimeout, this.catalogRequestTimeoutMs);
    try {
      const response = await fetch(
        `/api/catalog/runs/${encodeURIComponent(runId)}/checkpoint-training?${query}`,
        {
          headers: { Authorization: `Bearer ${this.token}` },
          cache: "no-store",
          signal: controller.signal,
        },
      );
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.error || `Training evidence request failed (${response.status})`);
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let payload = null;
      try {
        while (true) {
          const { value, done } = await reader.read();
          if (serial !== this.checkpointTrainingSerial || expectedKey !== this.routeKey()) {
            await reader.cancel();
            return;
          }
          clearTimeout(timeout);
          timeout = setTimeout(abortOnTimeout, this.catalogRequestTimeoutMs);
          buffer += decoder.decode(value, { stream: !done });
          const lines = buffer.split("\n");
          buffer = lines.pop();
          if (done && buffer.trim()) lines.push(buffer);
          let changed = false;
          for (const line of lines) {
            if (!line.trim()) continue;
            const event = JSON.parse(line);
            if (event.type === "error") throw new Error(event.error || "Training evidence unavailable");
            if (event.type === "complete") payload = event;
            if (event.type === "metrics") {
              const updates = new Map(event.items.map((item) => [item.checkpoint_id, item.metrics]));
              this.sourceItems = this.sourceItems.map((item) => {
                const metrics = updates.get(item.checkpoint_id);
                return metrics ? {
                  ...item,
                  metrics: { ...item.metrics, ...metrics },
                  training_loaded_metrics: [...new Set([...item.training_loaded_metrics, ...Object.keys(metrics)])],
                } : item;
              });
              changed = true;
            }
          }
          if (changed) {
            this.items = [...this.sourceItems];
            this.renderView();
          }
          if (done) break;
        }
      } finally {
        await reader.cancel().catch(() => {});
        reader.releaseLock();
      }
      if (!payload) throw new Error("Training evidence stream ended before completion. Try Refresh.");
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
      this.runStatus = payload.run && typeof payload.run === "object"
        ? { ...payload.run }
        : this.runStatus;
      this.catalogWarnings = Array.isArray(payload.warnings) ? payload.warnings : [];
      this.freshness = this.catalogWarnings.length ? "partial" : "fresh";
    } catch (error) {
      if (
        serial !== this.checkpointTrainingSerial
        || expectedKey !== this.routeKey()
      ) return;
      this.freshness = "partial";
      this.catalogWarnings = [
        ...this.catalogWarnings.filter((warning) => warning?.code !== "wandb_enrichment_pending"),
        {
          code: "wandb_enrichment_unavailable",
          message: timedOut
            ? "Training evidence request timed out. Try Refresh."
            : `Live W&B training evidence is unavailable: ${String(error?.message || error)}`,
          retryable: true,
          source: "wandb",
        },
      ];
    } finally {
      clearTimeout(timeout);
      if (serial === this.checkpointTrainingSerial) {
        this.checkpointTrainingController = null;
        this.checkpointTrainingPending = false;
        this.sourceItems = this.sourceItems.map((item) => ({ ...item, training_pending: false }));
        this.items = [...this.sourceItems];
        this.rememberCheckpointCatalog();
        this.rememberSearchCatalog();
        this.renderView();
      }
    }
  }

  applyRoute(route, { seedItems = null } = {}) {
    this.route = canonicalSourceRoute({ ...this.route, ...route });
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
    this.runStatus = null;
    this.checkpointTrainingPending = false;
    this.loadedKey = "";
    this.error = "";
    this.checkpointTrainingController?.abort();
    this.checkpointTrainingController = null;
    this.checkpointTrainingSerial += 1;
    this.selectedCheckpoints.clear();
    this.resetGoalVariantDetail();
    this.autoSelectedRoute = "";
    let restoredCatalog = false;
    if (Array.isArray(seedItems) && seedItems.length) {
      this.sourceItems = seedItems.map((item) => ({ ...item }));
      this.items = [...this.sourceItems];
      this.freshness = "partial";
    } else {
      restoredCatalog = this.restoreEnvironmentCatalog() || this.restoreGoalCatalog()
        || this.restoreGoalActivity() || this.restoreCheckpointCatalog();
    }
    this.hydrateInitialEnvironments();
    this.renderView();
    this.ensureLoaded({ quiet: restoredCatalog });
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
    const nextRoute = canonicalSourceRoute({ ...this.route, ...route });
    const current = this.getState()?.applicationSnapshot || { app: this.app };
    const decision = this.selection.browse(nextRoute, current, { historyMode });
    if (!decision) return false;
    this.applyRoute(decision.route, { seedItems });
    if (decision.historyMode) this.syncUrl(decision.historyMode);
    this.openSourceRoute?.({ ...this.route });
    return true;
  }

  goBack() {
    if (!this.selection.view.sourceMode) {
      this.browseCurrentSource();
      return;
    }
    const items = sourceBreadcrumbItems(this.selection.view.route || this.route);
    const current = items.findIndex((item) => item.current);
    const parent = items[current - 1];
    if (parent?.route) this.navigate(parent.route);
  }

  browseCurrentSource() {
    const route = this.selection.view.route || this.route;
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
    const decision = this.selection.select({
      kind: "public_run",
      value: item.manifest_url,
      run_id: item.run_id,
      checkpoint_id: item.checkpoint_id,
      seed: checkpointPlaybackSeed(item),
    }, route, { historyMode });
    if (!decision) return false;
    this.route = { ...decision.route };
    if (decision.historyMode) this.syncUrl(decision.historyMode);
    this.renderActiveCheckpointNavigation(this.route);
    return decision.commandId;
  }

  back() {
    if (this.route.level === "runs" && this.route.run_id) {
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
    this.requestController?.abort();
    this.requestController = null;
    this.requestSerial += 1;
    this.goalEvidenceEpoch = (this.goalEvidenceEpoch || 0) + 1;
    this.loading = false;
    this.loadingKey = "";
    this.checkpointTrainingController?.abort();
    this.checkpointTrainingController = null;
    this.checkpointTrainingSerial += 1;
    this.goalVariantRunPages.clear();
    this.items = this.sourceItems.filter((item) => catalogItemMatchesSearch(item, value));
    this.nextCursor = null;
    this.loadedKey = "";
    this.error = "";
    const restored = this.restoreSearchCatalog() || (!this.query.trim() && (
      this.restoreEnvironmentCatalog() || this.restoreGoalCatalog()
      || this.restoreGoalActivity() || this.restoreCheckpointCatalog()
    ));
    if (restored && this.loadedKey === this.routeKey()) {
      this.renderView();
      return;
    }
    this.renderView();
    this.searchTimer = window.setTimeout(() => {
      this.load();
    }, 80);
  }

  renderView() {
    if (this.route.level === "goal_variants") {
      const selected = this.items.find((v) => v.variant_id === this.selectedGoalVariantId)
        || this.items.find((v) => v.configuration_kind === "current_default") || this.items[0];
      if (selected) {
        this.selectedGoalVariantId = String(selected.variant_id || "");
        if (this.goalVariantDiff?.variantId !== this.selectedGoalVariantId) {
          this.goalVariantDiff = this.goalVariantDiffFromActivity(selected);
        }
      }
    }
    if (["error", "resolving", "verifying", "loading"].includes(this.app.phase)) {
      this.view?.breadcrumbs([]);
    } else this.renderBreadcrumbs();
    const presentation = { ...this,
      selectedCheckpoints: new Set(this.selectedCheckpoints),
      goalVariantRunPages: new Map(this.goalVariantRunPages),
      hasControl: this.hasControl(), heading: this.heading(),
      refreshing: this.loading || Boolean(this.checkpointTrainingController),
      searchPlaceholder: this.route.level === "environments" ? "Search environments"
        : this.route.level === "goals" ? "Search goals"
        : this.route.level === "goal_variants" ? "Search configuration, difference, status, date, or contract hash"
        : "Search checkpoint, step, hash, purpose, or evaluation",
    };
    presentation.table = sourceTablePresentation(presentation);
    this.view?.render(presentation);
  }

  heading() {
    if (this.app.phase === "error") return "Could not open checkpoint";
    if (["resolving", "verifying", "loading"].includes(this.app.phase)) return "Opening checkpoint";
    if (this.route.level === "runs" && this.route.run_id) {
      return "Runs · choose a checkpoint";
    }
    if (this.route.level === "goal_variants") {
      return "Choose a run";
    }
    if (this.route.level === "goals") return "Choose a goal";
    return "Choose an environment";
  }

  renderBreadcrumbs() {
    this.view?.breadcrumbs(sourceBreadcrumbItems(this.route));
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
      const withEvaluationStatus = (item) => {
        const status = statuses.get(String(item.checkpoint_id || ""));
        return status
          ? {
              ...item,
              evaluation: status.evaluation || item.evaluation,
              evaluation_queue: status,
            }
          : item;
      };
      this.items = this.items.map(withEvaluationStatus);
      this.sourceItems = this.sourceItems.map(withEvaluationStatus);
      this.rememberCheckpointCatalog();
      this.rememberSearchCatalog();
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

  embeddedGoalRunsHaveMore(variant) {
    const page = this.goalVariantRunPages.get(String(variant?.variant_id || ""));
    return page?.loaded ? Boolean(page.nextCursor) : Boolean(variant.has_more_runs);
  }

  async loadEmbeddedGoalRuns(variant, { append = false } = {}) {
    const variantId = String(variant?.variant_id || "");
    if (!variantId) return;
    const expectedKey = this.routeKey();
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
      if (expectedKey !== this.routeKey()) return;
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
        loaded: true,
        nextCursor: payload.next_cursor || null,
        loading: false,
        error: "",
      });
      this.rememberGoalActivity();
      this.rememberSearchCatalog();
    } catch (error) {
      if (expectedKey !== this.routeKey()) return;
      this.goalVariantRunPages.set(variantId, {
        ...current,
        loading: false,
        error: String(error?.message || error),
      });
    } finally {
      this.renderView();
    }
  }

}
