export function snapshotActivatesCheckpointSelection(checkpointLoad, snapshot) {
  const checkpointId = String(checkpointLoad?.checkpointId || "");
  return Boolean(
    checkpointId
    && snapshot?.app?.phase === "active"
    && String(snapshot?.app?.route?.checkpoint_id || "") === checkpointId
  );
}
