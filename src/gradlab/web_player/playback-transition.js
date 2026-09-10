export function snapshotActivatesCheckpointSelection(checkpointLoad, snapshot) {
  const checkpointId = String(checkpointLoad?.checkpointId || "");
  return Boolean(
    checkpointId
    && snapshot?.app?.phase === "active"
    && String(snapshot?.app?.route?.checkpoint_id || "") === checkpointId
  );
}

// Loading is acknowledged before preparation finishes; failures arrive in snapshots.
export function snapshotFailsCheckpointSelection(checkpointLoad, snapshot) {
  return Boolean(checkpointLoad && (
    snapshot?.app?.phase === "error"
    || (snapshot?.app?.phase === "active" && snapshot?.app?.error)
  ));
}
