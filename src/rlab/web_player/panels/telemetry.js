const STATIC_DESCRIPTORS = Object.freeze({
  "reward/provider": {
    label: "Provider reward",
    shortLabel: "Provider r",
    type: "scalar",
    unit: "step-reward",
    phase: "post-action",
    color: "#76a9ff",
    digits: 3,
    current: (snapshot) => snapshot?.transition?.reward?.provider,
    history: (point) => point?.reward_provider,
  },
  "reward/shaped": {
    label: "Shaped reward",
    shortLabel: "Shaped r",
    type: "scalar",
    unit: "step-reward",
    phase: "post-action",
    color: "#d794ff",
    digits: 3,
    current: (snapshot) => snapshot?.transition?.reward?.shaped,
    history: (point) => point?.reward_shaped,
  },
  "reward/return": {
    label: "Episode return",
    shortLabel: "Return",
    type: "scalar",
    unit: "return",
    phase: "post-action",
    color: "#60d394",
    digits: 2,
    current: (snapshot) => snapshot?.transition?.reward?.return,
    history: (point) => point?.return,
  },
  "policy/value": {
    label: "Value estimate V(s)",
    shortLabel: "V(s)",
    type: "scalar",
    unit: "value",
    phase: "pre-action",
    color: "#76a9ff",
    digits: 4,
    current: (snapshot) => snapshot?.transition?.decision?.value,
    history: (point) => point?.value,
  },
  "policy/realized-return": {
    label: "Realized return-to-go G(s)",
    shortLabel: "Realized G(s)",
    type: "scalar",
    unit: "value",
    phase: "post-episode",
    color: "#f0c36a",
    digits: 3,
    history: (point) => point?.realized_return,
  },
  "policy/value-error": {
    label: "Value error V(s) − G(s)",
    shortLabel: "V − G",
    type: "scalar",
    unit: "value",
    phase: "post-episode",
    color: "#f28f6b",
    digits: 3,
    history: (point) => point?.value_error,
  },
  "policy/entropy": {
    label: "Policy entropy",
    shortLabel: "Entropy",
    type: "scalar",
    unit: "entropy",
    phase: "pre-action",
    color: "#53d4e8",
    digits: 4,
    current: (snapshot) => snapshot?.transition?.decision?.entropy,
    history: (point) => point?.entropy,
  },
  "policy/log-probability": {
    label: "Selected action log probability",
    shortLabel: "Log p",
    type: "scalar",
    unit: "log-probability",
    phase: "pre-action",
    color: "#f0c36a",
    digits: 4,
    current: (snapshot) => snapshot?.transition?.decision?.log_probability,
    history: (point) => point?.log_probability,
  },
  "policy/mode": {
    label: "Policy sampling mode",
    shortLabel: "Mode",
    type: "text",
    unit: "mode",
    phase: "pre-action",
    current: (snapshot) => {
      const decision = snapshot?.transition?.decision;
      if (!decision) return snapshot?.driver || null;
      return decision.sampled ? "Stochastic" : "Deterministic";
    },
  },
  "policy/distribution": {
    label: "Policy distribution",
    shortLabel: "Distribution",
    type: "distribution",
    unit: "probability",
    phase: "pre-action",
    current: (snapshot) => snapshot?.transition?.decision || null,
  },
  "action/policy": {
    label: "Policy-selected action",
    shortLabel: "Policy action",
    type: "categorical",
    unit: "action",
    phase: "pre-action",
    current: (snapshot) => snapshot?.transition?.decision?.selected_action,
    history: (point) => point?.policy_action,
  },
  "action/executed": {
    label: "Executed action",
    shortLabel: "Action",
    type: "categorical",
    unit: "action",
    phase: "post-action",
    current: (snapshot) => snapshot?.transition?.executed_action,
    history: (point) => point?.executed_action,
  },
  "transition/outcome": {
    label: "Transition outcome",
    shortLabel: "Outcome",
    type: "text",
    unit: "outcome",
    phase: "post-action",
    current: (snapshot) => snapshot?.transition?.outcome || "continuing",
    history: (point) => point?.outcome || "continuing",
  },
});

export function encodeDynamicSegment(value) {
  return encodeURIComponent(String(value));
}

export function decodeDynamicSegment(value) {
  try {
    return decodeURIComponent(String(value));
  } catch {
    return null;
  }
}

export function dynamicDescriptorKey(namespace, name) {
  return `${namespace}/${encodeDynamicSegment(name)}`;
}

function dynamicDescriptor(key) {
  for (const [prefix, namespace] of [
    ["signal/", "signal"],
    ["reward-component/", "reward-component"],
  ]) {
    if (!key.startsWith(prefix)) continue;
    const name = decodeDynamicSegment(key.slice(prefix.length));
    if (name === null || !name) return null;
    const signal = namespace === "signal";
    return {
      key,
      namespace,
      name,
      label: signal ? name : `${name} reward component`,
      shortLabel: name,
      type: "scalar",
      unit: signal ? `signal:${name}` : "step-reward",
      phase: "post-action",
      color: signal ? "#f0c36a" : "#f28f6b",
      digits: 4,
      current: (snapshot) => signal
        ? snapshot?.transition?.signals?.[name]
        : snapshot?.transition?.reward?.components?.[name],
      history: (point) => signal
        ? point?.signals?.[name]
        : point?.components?.[name],
    };
  }
  return null;
}

export function descriptorFor(key) {
  const descriptor = STATIC_DESCRIPTORS[key] || dynamicDescriptor(String(key || ""));
  return descriptor ? { key, ...descriptor } : null;
}

export function descriptorCatalog(snapshot, history = []) {
  const descriptors = new Map(
    Object.keys(STATIC_DESCRIPTORS).map((key) => [key, descriptorFor(key)]),
  );
  const transition = snapshot?.transition || {};
  const signals = new Set([
    ...Object.keys(transition.signals || {}),
    ...history.flatMap((point) => Object.keys(point.signals || {})),
  ]);
  const components = new Set([
    ...Object.keys(transition.reward?.components || {}),
    ...history.flatMap((point) => Object.keys(point.components || {})),
  ]);
  signals.forEach((name) => {
    const key = dynamicDescriptorKey("signal", name);
    descriptors.set(key, descriptorFor(key));
  });
  components.forEach((name) => {
    const key = dynamicDescriptorKey("reward-component", name);
    descriptors.set(key, descriptorFor(key));
  });
  return descriptors;
}

export function descriptorValue(descriptor, { snapshot = null, point = null } = {}) {
  if (!descriptor) return null;
  const historical = point && descriptor.history ? descriptor.history(point) : null;
  if (historical !== null && historical !== undefined) return historical;
  return descriptor.current ? descriptor.current(snapshot) : null;
}

export function formatTelemetryValue(value, descriptor = null) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "number") {
    return Number.isFinite(value)
      ? value.toFixed(descriptor?.digits ?? 3)
      : "—";
  }
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

export function seriesForMetric(key, history) {
  const descriptor = descriptorFor(key);
  if (!descriptor?.history) return [];
  return history.map((point) => {
    const value = descriptor.history(point);
    if (value === null || value === undefined || value === "") return Number.NaN;
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric : Number.NaN;
  });
}

export function compatibleMetricKeys(keys, catalog) {
  const descriptors = keys.map((key) => catalog.get(key) || descriptorFor(key)).filter(Boolean);
  if (!descriptors.length || descriptors.some((item) => item.type !== "scalar")) return false;
  return new Set(descriptors.map((item) => item.unit)).size === 1;
}

export function metricOptions(catalog, kind) {
  return [...catalog.values()]
    .filter((descriptor) => {
      if (kind === "line") return descriptor.type === "scalar" && descriptor.history;
      if (kind === "stats") return ["scalar", "text", "categorical"].includes(descriptor.type);
      if (kind === "histogram") return descriptor.type === "categorical" && descriptor.history;
      if (kind === "distribution") return descriptor.type === "distribution";
      return false;
    })
    .sort((left, right) => left.label.localeCompare(right.label));
}
