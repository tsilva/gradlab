"""Repair one terminal SQLite MLflow projection from a verified GradLab journal.

Run only while the MLflow service is stopped and the GradLab writer lease is
held. The evidence file contains the manifest, service_run_id and hash-verified
journal events. A verified online snapshot must be supplied. No receipts change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path


def repair(database: Path, evidence: dict, snapshot: Path) -> dict:
    receipt = json.loads(snapshot.with_suffix(snapshot.suffix + ".json").read_text())
    if hashlib.sha256(snapshot.read_bytes()).hexdigest() != receipt["sha256"]:
        raise ValueError("snapshot hash mismatch")
    from datetime import datetime
    manifest = evidence["manifest"]
    run_id = evidence["service_run_id"]
    created = int(datetime.fromisoformat(manifest["created_at"].replace("Z", "+00:00")).timestamp() * 1000)
    old, new = {}, {}
    events = evidence["events"]
    if [e["event_seq"] for e in events] != list(range(1, len(events) + 1)) or not events:
        raise ValueError("journal must be nonempty and contiguous")
    for event in events:
        if event["kind"] != "history":
            raise ValueError("this maintenance tool supports scalar history journals only")
        seq, step = event["event_seq"], int(event.get("step") or 0)
        timestamp = event["created_at"]
        import math
        if isinstance(timestamp, bool) or not isinstance(timestamp, int | float) or not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("invalid retained event time")
        payload = dict(event["payload"])
        source = str(event.get("source") or "")
        if source.startswith("eval"):
            payload["eval/step"] = float(step)
        elif not source.startswith("orchestration"):
            payload["train/step"] = float(step)
        payload["ops/sequence"] = float(seq)
        for key, value in payload.items():
            original = (key, created + seq, seq if key == "ops/sequence" else step, float(value))
            target = (key, int(timestamp * 1000), seq if key.startswith(("ops/", "system/")) else step, float(value))
            old[original] = target
            new[target] = target
    with sqlite3.connect(database) as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT status FROM runs WHERE run_uuid=?", (run_id,)).fetchone()
        identity = db.execute("SELECT value FROM tags WHERE run_uuid=? AND key='gradlab.run_id'", (run_id,)).fetchone()
        if row is None or row[0] not in ("FINISHED", "FAILED", "KILLED") or identity != (manifest["run_id"],):
            raise ValueError("target must be the same terminal GradLab Run")
        rows = db.execute("SELECT key,timestamp,step,value FROM metrics WHERE run_uuid=?", (run_id,)).fetchall()
        if len(rows) != len(old) or any(r not in old and r not in new for r in rows):
            raise ValueError("projection differs from supplied journal; no changes made")
        changed = 0
        for original in rows:
            target = old.get(original, new.get(original))
            if target == original:
                continue
            key, timestamp, step, value = original
            db.execute("UPDATE metrics SET timestamp=?,step=? WHERE run_uuid=? AND key=? AND timestamp=? AND step=? AND value=?",
                       (target[1], target[2], run_id, key, timestamp, step, value))
            changed += 1
        # Recompute MLflow's latest-value ordering after changing operational axes.
        db.execute("DELETE FROM latest_metrics WHERE run_uuid=?", (run_id,))
        db.execute("INSERT INTO latest_metrics (key,value,timestamp,step,is_nan,run_uuid) "
                   "SELECT key,value,timestamp,step,is_nan,run_uuid FROM ("
                   "SELECT *,ROW_NUMBER() OVER (PARTITION BY key ORDER BY step DESC,timestamp DESC,value DESC) AS rank "
                   "FROM metrics WHERE run_uuid=?) WHERE rank=1", (run_id,))
        actual = db.execute("SELECT key,timestamp,step,value FROM metrics WHERE run_uuid=?", (run_id,)).fetchall()
        if set(actual) != set(new) or db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("repaired projection verification failed")
    return {"run_id": manifest["run_id"], "service_run_id": run_id, "metrics": len(rows),
            "changed": changed, "snapshot_sha256": receipt["sha256"],
            "evidence_sha256": hashlib.sha256(json.dumps(evidence,sort_keys=True).encode()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("snapshot", type=Path)
    args = parser.parse_args()
    print(json.dumps(repair(args.database, json.loads(args.evidence.read_text()), args.snapshot),sort_keys=True))
