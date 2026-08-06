function sessionContract(snapshot) {
  const contract = snapshot?.session?.action_contract;
  return contract && typeof contract === "object" ? contract : null;
}

function semanticLabel(value) {
  if (typeof value !== "string") return null;
  const label = value.trim();
  return label && !/^<[^>]+>$/.test(label) ? label : null;
}

function fallbackActionLabel(snapshot, value) {
  const scalar = scalarActionIndex(value);
  if (scalar === null) return null;
  const names = snapshot?.session?.action_names;
  if (!Array.isArray(names)) return null;
  const start = Number(sessionContract(snapshot)?.policy?.space?.start || 0);
  const index = scalar - start;
  if (!Number.isInteger(index) || index < 0 || index >= names.length) return null;
  const label = semanticLabel(names[index]);
  return label ? label.replaceAll("_", " ") : null;
}

export function scalarActionIndex(value) {
  if (Number.isInteger(value)) return Number(value);
  if (Array.isArray(value) && value.length === 1 && Number.isInteger(value[0])) {
    return Number(value[0]);
  }
  return null;
}

function legalTupleEntry(policy, semantics, value) {
  if (
    policy?.space?.type !== "multi_discrete"
    || !Array.isArray(semantics?.legal_entries)
    || !Array.isArray(value)
  ) return null;
  const selected = value.flat(Infinity).map(Number);
  return semantics.legal_entries.find((entry) => {
    const candidate = Array.isArray(entry?.value) ? entry.value.flat(Infinity).map(Number) : null;
    return candidate
      && candidate.length === selected.length
      && candidate.every((item, index) => item === selected[index]);
  }) || null;
}

function mixedRadixEntry(semantics, value, start = 0) {
  let remaining = Number(value) - Number(start || 0);
  if (!Number.isInteger(remaining) || remaining < 0) return null;
  if (!Array.isArray(semantics?.axes)) return null;
  const atoms = [];
  const inputs = [];
  for (const axis of semantics.axes) {
    const radix = Number(axis.radix || axis.values?.length || 0);
    if (
      !Number.isInteger(radix)
      || radix <= 0
      || !Array.isArray(axis.values)
      || axis.values.length !== radix
    ) return null;
    const selected = axis.values?.[remaining % radix];
    if (!selected || typeof selected !== "object") return null;
    remaining = Math.floor(remaining / radix);
    if (
      (selected.atoms !== undefined && !Array.isArray(selected.atoms))
      || (selected.inputs !== undefined && !Array.isArray(selected.inputs))
    ) return null;
    (selected.atoms || []).forEach((atom) => {
      if (!atoms.includes(String(atom))) atoms.push(String(atom));
    });
    (selected.inputs || []).forEach((input) => {
      if (!inputs.includes(String(input))) inputs.push(String(input));
    });
  }
  if (remaining !== 0) return null;
  const semanticId = atoms.length ? atoms.join("_") : "noop";
  return {
    value: Number(value),
    semantic_id: semanticId,
    label: semanticId.replaceAll("_", " "),
    controls: [{ player: 1, atoms, inputs }],
  };
}

export function actionEntry(snapshot, value) {
  const contract = sessionContract(snapshot);
  const policy = contract?.policy;
  const semantics = policy?.semantics;
  if (semantics?.status !== "available") return null;
  const legalEntry = legalTupleEntry(policy, semantics, value);
  if (legalEntry) return legalEntry;
  const scalar = scalarActionIndex(value);
  if (
    scalar === null
    || policy?.space?.type !== "discrete"
  ) return null;
  if (semantics.encoding === "explicit") {
    if (!Array.isArray(semantics.entries)) return null;
    return semantics.entries.find(
      (entry) => (
        entry
        && typeof entry === "object"
        && Number.isInteger(entry.value)
        && entry.value === scalar
        && Boolean(semanticLabel(entry.label) || semanticLabel(entry.semantic_id))
      ),
    ) || null;
  }
  if (semantics.encoding === "mixed_radix") {
    return mixedRadixEntry(semantics, scalar, policy.space.start);
  }
  return null;
}

export function actionSemanticsReason(snapshot, value = undefined) {
  const contract = sessionContract(snapshot);
  if (!contract) {
    return "this source did not record an action contract";
  }
  const semantics = contract?.policy?.semantics;
  if (semantics?.status !== "available") {
    return semanticLabel(semantics?.reason)
      || "the policy action semantics are unavailable";
  }
  if (value === undefined) return null;
  const space = contract?.policy?.space;
  const resolved = actionEntry(snapshot, value)
    || (space?.type === "discrete"
      ? null
    : (
      semantics.encoding === "components"
      && ["multi_binary", "multi_discrete", "box"].includes(space?.type)
      && componentActionLabel(value, semantics, space)
    ));
  return resolved ? null : "the declared action semantics do not describe this value";
}

function componentLabel(component, value) {
  if (Array.isArray(component?.values)) {
    const selected = component.values[Number(value)];
    return semanticLabel(selected?.label) || semanticLabel(selected?.semantic_id);
  }
  return semanticLabel(component?.label) || semanticLabel(component?.semantic_id);
}

function componentActionLabel(value, semantics, space) {
  const flat = Array.isArray(value) ? value.flat(Infinity) : [value];
  const components = semantics?.components;
  if (!Array.isArray(components) || components.length !== flat.length) return null;
  if (space?.type === "multi_binary") {
    const active = [];
    for (const [index, component] of components.entries()) {
      if (!component || typeof component !== "object") return null;
      const componentIndex = Number(component.index);
      const label = componentLabel(component);
      if (!Number.isInteger(componentIndex) || componentIndex !== index || !label) return null;
      if (Number(flat[componentIndex]) !== 0) active.push(label);
    }
    return active.length ? active.join(" + ") : "noop";
  }
  if (space?.type === "multi_discrete") {
    const labels = [];
    for (const [index, component] of components.entries()) {
      if (!component || typeof component !== "object") return null;
      const selected = componentLabel(component, flat[index]);
      if (!selected) return null;
      const componentName = semanticLabel(component.label)
        || semanticLabel(component.semantic_id)
        || `component ${index}`;
      labels.push(`${componentName}: ${selected}`);
    }
    return labels.join(" · ");
  }
  const labels = [];
  for (const [index, component] of components.entries()) {
    if (!component || typeof component !== "object") return null;
    const componentName = componentLabel(component);
    if (!componentName) return null;
    labels.push(`${componentName}: ${flat[index]}`);
  }
  return labels.join(" · ");
}

export function formatActionValue(value, snapshot) {
  const entry = actionEntry(snapshot, value);
  if (entry) return semanticLabel(entry.label) || semanticLabel(entry.semantic_id);

  const contract = sessionContract(snapshot);
  const semantics = contract?.policy?.semantics;
  const space = contract?.policy?.space;
  if (
    semantics?.status === "available"
    && semantics?.encoding === "components"
    && ["multi_binary", "multi_discrete", "box"].includes(space?.type)
  ) {
    const label = componentActionLabel(value, semantics, space);
    if (label) return label;
  }

  const fallback = fallbackActionLabel(snapshot, value);
  if (fallback) return fallback;
  const scalar = scalarActionIndex(value);
  const encoded = JSON.stringify(value);
  const raw = scalar === null ? (encoded === undefined ? String(value) : encoded) : String(scalar);
  const reason = actionSemanticsReason(snapshot, value);
  return reason ? `raw action ${raw} · semantics unavailable: ${reason}` : `raw action ${raw}`;
}

export function discreteActionLabels(snapshot, count = null) {
  const contract = sessionContract(snapshot);
  const space = contract?.policy?.space;
  const legalEntries = contract?.policy?.semantics?.legal_entries;
  if (
    space?.type === "multi_discrete"
    && Array.isArray(legalEntries)
    && (count === null || Number(count) === legalEntries.length)
  ) {
    return legalEntries.map((entry) => (
      semanticLabel(entry?.label)
      || semanticLabel(entry?.semantic_id)?.replaceAll("_", " ")
      || "unknown action"
    ));
  }
  const size = count === null ? Number(space?.n || 0) : Number(count);
  const start = Number(space?.start || 0);
  if (Number.isInteger(size) && size > 0) {
    return Array.from(
      { length: size },
      (_, index) => formatActionValue(start + index, snapshot),
    );
  }
  return [...(snapshot?.session?.action_names || [])].map(
    (name) => String(name).replaceAll("_", " "),
  );
}
