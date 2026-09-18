"""Supervisor-owned monitoring admission, recovery, and delivery accounting."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime

from gradlab.checkpoint_monitoring import episode_manifest, verify_monitoring_inventory
from gradlab.eval_backend import EvalHandle
from gradlab.json_utils import canonical_json_sha256


class MonitoringQueue:
    def __init__(self, supervisor, backend):
        self.owner = supervisor
        self.backend = backend
        self.settings = supervisor.train_config["checkpoint_monitoring"]
        self.contract_hash = canonical_json_sha256(
            {
                "recipe_sha256": supervisor.manifest.recipe_sha256,
                "settings": self.settings,
            }
        )
        self.prefix = f"runs/{supervisor.manifest.run_id}/monitoring/{self.contract_hash}"
        self.manifest = episode_manifest(self.settings["episodes"])
        original = supervisor.authority.control.get_json(
            f"runs/{supervisor.manifest.run_id}/manifest.json"
        )
        self.deadline = (
            datetime.fromisoformat(original["created_at"].replace("Z", "+00:00")).timestamp()
            + self.settings["whole_run_seconds"]
        )
        self.receipt = dict(enabled=True, complete=False, inventory=[], workers_quiescent=True)

    def _save(self, key, state):
        self.owner._lease_heartbeat()
        self.owner.authority.control.put_json(key, state, create_only=False)

    def _reclaim(self, state):
        forget = getattr(self.backend, "forget", None)
        if forget is not None and state.get("handle"):
            forget(EvalHandle(**state["handle"]))

    def _intent(self, checkpoint):
        owner = self.owner
        identity = canonical_json_sha256(
            {"checkpoint": checkpoint["checkpoint_id"], "contract": self.contract_hash}
        )
        return {
            "evaluation_id": identity,
            "checkpoint": checkpoint,
            "run_id": owner.manifest.run_id,
            "attempt_id": owner.manifest.attempt_id,
            "source_sha": owner.manifest.source_sha,
            "runtime": owner.manifest.image_digest,
            "recipe_sha256": owner.manifest.recipe_sha256,
            "goal_sha256": owner.manifest.goal_sha256,
            "environment_sha256": owner.manifest.environment_sha256,
            "goal_variant": owner.manifest.goal_variant,
            "training_seed": owner.manifest.seed,
            "contract_sha256": self.contract_hash,
            "manifest": self.manifest,
            "settings": self.settings,
            "deadline": self.deadline,
            "prefix": f"monitoring/{owner.manifest.run_id}/{identity}",
        }

    def _verify(self, result, intent):
        checkpoint = intent["checkpoint"]
        if (
            result["checkpoint_id"] != checkpoint["checkpoint_id"]
            or result["checkpoint_step"] != checkpoint["step"]
            or result["evaluation_id"] != intent["evaluation_id"]
            or result["contract_sha256"] != self.contract_hash
        ):
            raise ValueError("monitoring result does not bind its immutable checkpoint contract")
        verify_monitoring_inventory(
            self.owner.authority.models, result, self.manifest, prefix=intent["prefix"]
        )

    def advance(self, *, final=False, canceled=False):
        owner = self.owner
        rows = []
        for checkpoint in owner.store.checkpoint_publications():
            intent = self._intent(checkpoint)
            base = f"{self.prefix}/{intent['evaluation_id']}"
            owner._lease_heartbeat()
            existing = owner.authority.control.get_json_optional(base + "/intent.json")
            if existing is None:
                owner.authority.control.put_json(base + "/intent.json", intent)
            else:
                # A retry retains the producing Attempt and original immutable intent.
                intent = existing
            key = base + "/state.json"
            state = owner.authority.control.get_json_optional(key) or {
                "status": "pending",
                "attempts": 0,
            }
            rows.append((intent, key, state))
        expired = owner.clock.time() >= self.deadline
        for intent, key, state in rows:
            if canceled or expired:
                if state["status"] == "running":
                    self.backend.cancel(EvalHandle(**state["handle"]))
                if state["status"] not in {"complete", "failed", "canceled"}:
                    state.update(
                        status="canceled" if canceled else "failed",
                        error="canceled" if canceled else "whole-run monitoring deadline exhausted",
                    )
                    self._save(key, state)
                    self._reclaim(state)
                continue
            if state["status"] == "running":
                try:
                    poll = self.backend.poll(EvalHandle(**state["handle"]))
                    if poll.status == "running":
                        continue
                    if poll.status != "succeeded":
                        raise RuntimeError(poll.error or f"monitoring worker {poll.status}")
                    result = poll.provider_result
                    self._verify(result, intent)
                    reference = owner.authority.models.get_json(intent["prefix"] + "/result.json")
                    if reference != result:
                        raise ValueError("monitoring result is not durably committed")
                    state.update(
                        status="verified",
                        result_sha256=canonical_json_sha256(result),
                        delivery_started_at=owner.clock.time(),
                    )
                    self._save(key, state)
                    self._reclaim(state)
                except Exception as exc:
                    self.backend.cancel(EvalHandle(**state["handle"]))
                    state.update(
                        status="pending" if state["attempts"] < 2 else "failed",
                        error=str(exc)[:1000],
                    )
                    self._save(key, state)
                    self._reclaim(state)
            if state["status"] == "verified":
                result = owner.authority.models.get_json(intent["prefix"] + "/result.json")
                if canonical_json_sha256(result) != state["result_sha256"]:
                    raise ValueError("monitoring result changed after verification")
                owner.store.append_monitoring(
                    result,
                    bucket_uri=owner.authority.models.config.uri,
                    media_spool_bytes=self.settings.get("media_spool_bytes", 512 * 1024**2),
                    scratch_headroom_bytes=self.settings.get("scratch_headroom_bytes", 1024**3),
                )
                if owner.store.monitoring_delivered(intent["evaluation_id"]):
                    state.update(
                        status="complete",
                        wandb_media_delivery_seconds=owner.clock.time()
                        - state["delivery_started_at"],
                    )
                    self._save(key, state)
        limit = self.settings["task_cpus"] if final else self.settings["active_workers"]
        limit = min(
            limit,
            (self.settings["memory_bytes"] - self.settings.get("media_memory_bytes", 0))
            // self.settings["worker_memory_bytes"],
            (self.settings["spool_bytes"] - self.settings.get("media_spool_bytes", 0))
            // self.settings["worker_spool_bytes"],
        )
        available = limit - sum(s["status"] == "running" for _, _, s in rows)
        if not canceled and not expired:
            for intent, key, state in rows:
                if available <= 0:
                    break
                if state["status"] != "pending":
                    continue
                state["attempts"] += 1
                self._save(key, state)
                try:
                    handle = self.backend.submit({**intent, "execution_attempt": state["attempts"]})
                    state.update(status="running", handle=asdict(handle))
                    available -= 1
                except Exception as exc:
                    state.update(
                        status="pending" if state["attempts"] < 2 else "failed",
                        error=str(exc)[:1000],
                    )
                self._save(key, state)
        complete = bool(rows) and all(state["status"] == "complete" for _, _, state in rows)
        inventory = [
            dict(
                checkpoint_id=i["checkpoint"]["checkpoint_id"],
                checkpoint_step=i["checkpoint"]["step"],
                evaluation_id=i["evaluation_id"],
                status=s["status"],
                attempts=s["attempts"],
                result_key=i["prefix"] + "/result.json"
                if s["status"] in {"verified", "complete"}
                else None,
                result_sha256=s.get("result_sha256"),
                error=s.get("error"),
                wandb_media_delivery_seconds=s.get("wandb_media_delivery_seconds"),
            )
            for i, _, s in rows
        ]
        self.receipt = dict(
            enabled=True,
            complete=complete,
            inventory=inventory,
            workers_quiescent=not any(s["status"] == "running" for _, _, s in rows),
        )
        if final and not canceled and any(s["status"] == "failed" for _, _, s in rows):
            raise RuntimeError(
                "required checkpoint monitoring incomplete; recovery evidence retained"
            )
        return complete or canceled

    def close(self):
        for checkpoint in self.owner.store.checkpoint_publications():
            identity = self._intent(checkpoint)["evaluation_id"]
            key = f"{self.prefix}/{identity}/state.json"
            state = self.owner.authority.control.get_json_optional(key)
            if state and state["status"] == "running":
                self.backend.cancel(EvalHandle(**state["handle"]))
