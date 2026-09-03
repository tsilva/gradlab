export function unavailableDiagnosticPresentation(statuses) {
  if (statuses.includes("protocol-error") || statuses.includes("error")) {
    return { label: "Protocol error", tone: "error" };
  }
  if (statuses.includes("contract-incomparable")) {
    return { label: "Incomparable", tone: "incomparable" };
  }
  if (statuses.includes("unsupported") || statuses.includes("disabled")) {
    return { label: "Unsupported", tone: "unsupported" };
  }
  return { label: "Waiting for data", tone: "waiting" };
}

export function unavailableDiagnosticRows(panels) {
  return panels.flatMap(({ panel, statuses }) => {
    if (
      !statuses.length
      || statuses.includes("available")
      || statuses.every((status) => status === "not-yet-observed")
    ) return [];
    return [{
      panel,
      ...unavailableDiagnosticPresentation(statuses),
    }];
  });
}
