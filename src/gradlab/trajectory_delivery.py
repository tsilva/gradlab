"""Supervisor-owned immutable dataset delivery and recovery, independent of HF."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Callable

from gradlab.file_utils import atomic_write_json
from gradlab.json_utils import canonical_json_sha256
from gradlab.r2_store import R2Bucket
from gradlab.trajectory_config import FORMAT


def read_document(path):
    value = json.loads(Path(path).read_bytes())
    if not isinstance(value, dict):
        raise ValueError("dataset document must be an object")
    return value


def verify_bytes(payload, manifest):
    if (
        len(payload) != manifest["bytes"]
        or hashlib.sha256(payload).hexdigest() != manifest["sha256"]
    ):
        raise ValueError("dataset chunk size/checksum mismatch")


class DatasetDelivery:
    def __init__(
        self,
        root: Path,
        bucket: R2Bucket,
        run_id: str,
        attempt_id: str,
        contribution_bytes: int,
        *,
        heartbeat: Callable[[], None] = lambda: None,
    ):
        self.root = root
        self.bucket = bucket
        self.run_id = run_id
        self.attempt_id = attempt_id
        self.prefix = f"datasets/runs/{run_id}/attempts/{attempt_id}"
        self.limit = contribution_bytes
        self.heartbeat = heartbeat
        self._receipt = {"enabled": True, "complete": False}
        self.uploaded = 0
        self.failures = 0
        self.pending_bytes = 0
        self.oldest_pending_seconds = 0.0
        self.error = None
        self._thread = None
        self._final = False
        self._stopped = False
        self._verified = set()

    @staticmethod
    def reserved_bytes(bucket, run_id):
        total = 0
        for key in bucket.iter_keys(f"datasets/runs/{run_id}/attempts"):
            if key.endswith(".reservation.json"):
                total += int(bucket.get_json(key)["reserved_bytes"])
        return total

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._work, name="trajectory-delivery", daemon=True
            )
            self._thread.start()

    def finalize(self):
        self._final = True

    def stop(self):
        self._stopped = True

    def _work(self):
        while not self._stopped and not self._receipt["complete"]:
            try:
                self.advance(final=self._final)
                self.error = None
            except Exception as exc:
                self.failures += 1
                self.error = type(exc).__name__
            time.sleep(0.25)

    def _fence(self):
        if self._stopped:
            raise RuntimeError("dataset delivery owner stopped")
        self.heartbeat()

    def advance(self, *, final=False):
        if not (self.root / "producer.json").exists():
            return
        manifests = sorted(self.root.glob("*.manifest.json"))
        pending = []
        for path in manifests:
            manifest = read_document(path)
            if manifest.get("format") != FORMAT:
                raise ValueError("unsupported training dataset format")
            filename = manifest["file"]
            if Path(filename).name != filename or not filename.endswith(".zip"):
                raise ValueError("unsafe dataset chunk filename")
            local = self.root / filename
            if local.exists():
                pending.append(local)
        self.pending_bytes = sum(path.stat().st_size for path in pending)
        self.oldest_pending_seconds = max(
            (time.time() - p.stat().st_mtime for p in pending), default=0.0
        )
        # Keep each delivery bounded. The encoder never waits on this worker.
        for path in manifests:
            manifest = read_document(path)
            filename = manifest["file"]
            local = self.root / filename
            key = f"{self.prefix}/chunks/{filename}"
            manifest_key = f"{self.prefix}/chunks/{path.name}"
            if key in self._verified and not local.exists():
                continue
            self._fence()
            reservation_key = f"{self.prefix}/chunks/{filename}.reservation.json"
            reservation = {
                "reserved_bytes": manifest["reserved_bytes"],
                "sha256": manifest["sha256"],
                "episode_id": manifest["episode"]["episode_id"],
            }
            if self.bucket.get_json_optional(reservation_key) is None:
                if (
                    self.reserved_bytes(self.bucket, self.run_id) + reservation["reserved_bytes"]
                    > self.limit
                ):
                    raise ValueError("Run dataset contribution budget exhausted")
            self._fence()
            self.bucket.put_json(reservation_key, reservation)
            if local.exists():
                payload = local.read_bytes()
                verify_bytes(payload, manifest)
                self._fence()
                self.bucket.put_bytes(key, payload, metadata={"sha256": manifest["sha256"]})
            remote = self.bucket.get_bytes(key)
            verify_bytes(remote, manifest)
            self._fence()
            self.bucket.put_json(manifest_key, {**manifest, "key": key})
            committed = self.bucket.get_json(manifest_key)
            if committed != {**manifest, "key": key}:
                raise ValueError("dataset manifest identity conflict")
            if key not in self._verified:
                self.uploaded += manifest["bytes"]
            self._verified.add(key)
            # Failed deletion is retried and continues to count against disk capacity.
            local.unlink(missing_ok=True)
            break
        if not final or not (self.root / "closed.json").exists():
            return
        closure = read_document(self.root / "closed.json")
        if closure.get("fault") or (self.root / "fault.json").exists():
            raise ValueError("recording fault prevents complete dataset delivery")
        if list(self.root.glob("*.partial")) or list(self.root.glob("*.rows")):
            raise ValueError("unsealed trajectory evidence remains")
        if closure["chunks"] != len(manifests):
            raise ValueError("dataset closure does not match complete chunk inventory")
        if len(self._verified) != len(manifests):
            return
        producer = read_document(self.root / "producer.json")
        inventory = {
            "format": FORMAT,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "producer": producer,
            "closure": closure,
            "chunks": [
                {
                    **read_document(path),
                    "key": f"{self.prefix}/chunks/{read_document(path)['file']}",
                }
                for path in manifests
            ],
        }
        key = f"{self.prefix}/final.json"
        self._fence()
        self.bucket.put_json(key, inventory)
        if self.bucket.get_json(key) != inventory:
            raise ValueError("dataset inventory verification failed")
        self._receipt = {
            "enabled": True,
            "complete": True,
            "manifest_key": key,
            "manifest_uri": self.bucket.uri(key),
            "manifest_sha256": canonical_json_sha256(inventory),
            "chunk_count": len(manifests),
            "verified_bytes": self.uploaded,
        }
        atomic_write_json(self.root / "delivery.json", self._receipt)
        self.pending_bytes = 0
        self.oldest_pending_seconds = 0

    def receipt(self):
        return dict(self._receipt)

    def metrics(self, now):
        path = self.root / "status.json"
        status = read_document(path) if path.exists() else {}
        counters = (status.get("captured", 0), status.get("encoded_bytes", 0), self.uploaded)
        previous_time, previous = getattr(self, "_metric_sample", (now, counters))
        elapsed = max(now - previous_time, 1e-9)
        rates = [max(0, value - old) / elapsed for value, old in zip(counters, previous)]
        self._metric_sample = (now, counters)
        result = {
            "ops/dataset/capture/rate": rates[0],
            "ops/dataset/encode/rate": rates[1],
            "ops/dataset/upload/rate": rates[2],
            "ops/dataset/pending/bytes": self.pending_bytes,
            "ops/dataset/pending/seconds": self.oldest_pending_seconds,
            "ops/dataset/spool/bytes": sum(
                p.stat().st_size for p in self.root.rglob("*") if p.is_file()
            ),
            "ops/dataset/upload/failures": self.failures,
            "ops/dataset/pause/seconds": status.get("pause_seconds", 0),
            "ops/dataset/skipped/count": status.get("skipped", 0),
            "ops/dataset/incomplete/count": status.get("incomplete", 0),
        }
        reason = status.get("pause_reason", "")
        for name in ("budget", "stage_budget", "memory", "disk", "recording_fault"):
            result[f"ops/dataset/pause/{name}"] = float(reason == name)
        return result
