function sessionContract(snapshot) {
  const contract = snapshot?.session?.action_contract;
  return contract && typeof contract === "object" ? contract : null;
}

export function scalarActionIndex(value) {
  if (Number.isInteger(value)) return Number(value);
  if (Array.isArray(value) && value.length === 1 && Number.isInteger(value[0])) {
    return Number(value[0]);
  }
  return null;
}

function mixedRadixEntry(semantics, value, start = 0) {
  let remaining = Number(value) - Number(start || 0);
  if (!Number.isInteger(remaining) || remaining < 0) return null;
  const atoms = [];
  const inputs = [];
  for (const axis of semantics.axes || []) {
    const radix = Number(axis.radix || axis.values?.length || 0);
    if (!Number.isInteger(radix) || radix <= 0) return null;
    const selected = axis.values?.[remaining % radix];
    if (!selected) return null;
    remaining = Math.floor(remaining / radix);
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
  const scalar = scalarActionIndex(value);
  if (
    scalar === null
    || semantics?.status !== "available"
    || policy?.space?.type !== "discrete"
  ) return null;
  if (semantics.encoding === "explicit") {
    return (semantics.entries || []).find(
      (entry) => Number(entry?.value) === scalar,
    ) || null;
  }
  if (semantics.encoding === "mixed_radix") {
    return mixedRadixEntry(semantics, scalar, policy.space.start);
  }
  return null;
}

export function actionSemanticsReason(snapshot) {
  const contract = sessionContract(snapshot);
  if (!contract) {
    return snapshot?.session?.action_names?.length
      ? "legacy session exposes labels without a structured action contract"
      : "this source did not record an action contract";
  }
  const semantics = contract?.policy?.semantics;
  if (semantics?.status === "available") return null;
  return semantics?.reason || "the policy action semantics are unavailable";
}

function componentLabel(component, value) {
  if (Array.isArray(component?.values)) {
    const selected = component.values[Number(value)];
    return selected?.label || selected?.semantic_id || String(value);
  }
  return component?.label || component?.semantic_id || String(value);
}

function componentActionLabel(value, semantics, space) {
  const flat = Array.isArray(value) ? value.flat(Infinity) : [value];
  const components = semantics?.components || [];
  if (space?.type === "multi_binary") {
    const active = components
      .filter((component) => Number(flat[Number(component.index)]) !== 0)
      .map((component) => component.label || component.semantic_id);
    return active.length ? active.join(" + ") : "noop";
  }
  if (space?.type === "multi_discrete") {
    return components.map((component, index) => (
      `${component.label || component.semantic_id || `component ${index}`}: ${
        componentLabel(component, flat[index])
      }`
    )).join(" · ");
  }
  return components.map((component, index) => (
    `${component.label || component.semantic_id || `component ${index}`}: ${flat[index]}`
  )).join(" · ");
}

export function formatActionValue(value, snapshot) {
  const entry = actionEntry(snapshot, value);
  if (entry) return entry.label || entry.semantic_id;

  const contract = sessionContract(snapshot);
  const semantics = contract?.policy?.semantics;
  const space = contract?.policy?.space;
  if (
    semantics?.status === "available"
    && semantics?.encoding === "components"
    && ["multi_binary", "multi_discrete", "box"].includes(space?.type)
  ) {
    return componentActionLabel(value, semantics, space);
  }

  const scalar = scalarActionIndex(value);
  const legacy = scalar === null ? null : snapshot?.session?.action_names?.[scalar];
  if (legacy) return String(legacy).replaceAll("_", " ");
  const raw = scalar === null ? JSON.stringify(value) : String(scalar);
  const reason = actionSemanticsReason(snapshot);
  return reason ? `raw action ${raw} · semantics unavailable: ${reason}` : `raw action ${raw}`;
}

export function discreteActionLabels(snapshot, count = null) {
  const contract = sessionContract(snapshot);
  const space = contract?.policy?.space;
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
