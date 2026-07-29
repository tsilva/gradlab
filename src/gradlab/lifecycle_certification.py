from __future__ import annotations

import hashlib
import io
import json
import shutil
import socket
import tempfile
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from gradlab.checkpoint_acceptance import acceptance_aggregates
from gradlab.early_stop import (
    MetricEarlyStopStateMachine,
    MetricSample,
    validate_metric_early_stop_decision,
)
from gradlab.eval_backend import EvalHandle, EvalPoll
from gradlab.goal_variants import build_goal_variant_descriptor
from gradlab.modal_eval_protocol import execution_key
from gradlab.policy_bundle import (
    build_recipe_document,
    canonical_json_bytes as _canonical_bytes,
    canonical_json_sha256,
    evaluation_contract,
    model_document_path,
    recipe_document_path,
    write_canonical_json,
)
from gradlab.r2_store import (
    BucketConfig,
    ConditionalWriteConflict,
    RunStorageConfig,
)
from gradlab.recipe_documents import (
    compose_resolved_train_documents,
    load_goal_contract,
)
from gradlab.run_authority import LEASE_TTL_SECONDS, LeaseUnavailable, RunAuthority
from gradlab.run_contracts import (
    EarlyStopReceipt,
    PromotionReceipt,
    RunManifest,
    TerminalReceipt,
)
from gradlab.run_supervisor import (
    IncompleteEvaluationEvidence,
    METRIC_JOURNAL_RETENTION_DAYS,
    RunSupervisor,
    _terminal_outcome,
)
from gradlab.supervisor_runtime import LifecycleObserver, SupervisorRuntime
from gradlab.wandb_publisher import WandbProjector


CERTIFICATION_SCHEMA_VERSION = 1
SOURCE_SHA = "a" * 40
BUILD_SOURCE_SHA = "f" * 40
RUNTIME_INPUT_SHA256 = "e" * 64
IMAGE_REF = "docker:registry.example/gradlab-certification@sha256:" + "b" * 64
GOAL_PATH = Path("experiments/goals/SuperMarioBros-Nes-v0/Level1-1/_goal.yaml")
RECIPE_PATH = GOAL_PATH.parent / "recipes" / "ppo.yaml"
DEFAULT_SCENARIOS = (
    "full-lifecycle",
    "parallel-run-isolation",
    "same-run-lease-fencing",
    "wandb-retry-deduplication",
    "wandb-visibility-gating",
    "checkpoint-upload-retry",
    "eval-result-reconciliation",
    "modal-ambiguous-submit",
    "cancellation-terminalization",
    "scratch-preservation-stop",
    "drain-only-recovery",
    "terminal-receipt-gating",
    "verifier-tamper-detection",
    "early-stop-outcomes",
)


def _sha256(value: object) -> str:
    return canonical_json_sha256(value)


class DeterministicClock:
    """A manually advanced clock shared by every simulated boundary."""

    def __init__(self) -> None:
        self._wall = datetime(2026, 1, 1, tzinfo=UTC).timestamp()
        self._monotonic = 0.0

    def time(self) -> float:
        return self._wall

    def monotonic(self) -> float:
        return self._monotonic

    def sleep(self, seconds: float) -> None:
        self.advance(seconds)

    def advance(self, seconds: float) -> None:
        increment = float(seconds)
        if increment < 0:
            raise ValueError("deterministic clock cannot move backwards")
        self._wall += increment
        self._monotonic += increment

    def utc_datetime(self) -> datetime:
        return datetime.fromtimestamp(self._wall, UTC)

    def utc_now(self) -> str:
        return self.utc_datetime().isoformat().replace("+00:00", "Z")


class RecordingObserver(LifecycleObserver):
    def __init__(self, *, evidence_path: Path | None = None) -> None:
        self.events: list[dict[str, Any]] = []
        self.evidence_path = evidence_path

    def emit(self, kind: str, payload: Mapping[str, Any]) -> None:
        self.events.append(
            {
                "sequence": len(self.events) + 1,
                "kind": str(kind),
                **dict(payload),
            }
        )
        if self.evidence_path is not None:
            self.evidence_path.parent.mkdir(parents=True, exist_ok=True)
            self.evidence_path.write_bytes(_canonical_bytes({"events": self.events}))


class ScriptedEvalBackend:
    def __init__(self, *, ambiguous_submits: int = 0) -> None:
        self.ambiguous_submits = int(ambiguous_submits)
        self.submissions: list[dict[str, Any]] = []
        self.canceled: list[str] = []

    def submit(self, intent: dict[str, Any]) -> EvalHandle:
        self.submissions.append(dict(intent))
        if self.ambiguous_submits > 0:
            self.ambiguous_submits -= 1
            raise RuntimeError("simulated ambiguous Modal spawn")
        return EvalHandle(provider="modal", call_id=f"modal-call-{len(self.submissions):03d}")

    def poll(self, handle: EvalHandle) -> EvalPoll:
        return EvalPoll(status="running", provider_result={"call_id": handle.call_id})

    def cancel(self, handle: EvalHandle) -> None:
        self.canceled.append(handle.call_id)


class CertificationRuntime(SupervisorRuntime):
    """Scriptable stand-in for process, W&B, signals, and host state."""

    def __init__(
        self,
        *,
        clock: DeterministicClock,
        writer_id: str,
        publish_failures: int = 0,
        evidence_path: Path | None = None,
    ) -> None:
        super().__init__(clock=clock)
        self.writer_id = writer_id
        self.publish_failures = int(publish_failures)
        self.evidence_path = evidence_path
        self.wandb_events: list[dict[str, Any]] = []
        self.summary: dict[str, Any] = {}
        self.closed = False
        self.stop_requests = 0
        self.learner_starts = 0

    def runtime_contract(self, *, runtime_image_ref: str) -> dict[str, Any]:
        if runtime_image_ref != IMAGE_REF:
            raise ValueError("certification runtime received an unexpected image")
        return {
            "runtime_build_source_sha": BUILD_SOURCE_SHA,
            "runtime_input_sha256": RUNTIME_INPUT_SHA256,
        }

    def holder_id(self) -> str:
        return self.writer_id

    def disk_usage(self, path: Path) -> Any:
        del path
        return SimpleNamespace(total=100, used=10, free=90)

    def publish_frames(
        self,
        store,
        projector: WandbProjector,
        *,
        limit: int,
    ) -> int:
        del projector
        published = 0
        for row in store.pending_metric_frames(limit=limit):
            frame_id = int(row["id"])
            if not store.claim_metric_frame(frame_id):
                continue
            if self.publish_failures > 0:
                self.publish_failures -= 1
                store.mark_metric_frame_failed(frame_id, "simulated W&B outage")
                continue
            payload = json.loads(str(row["payload_json"]))
            event = {
                "event_seq": frame_id,
                "event_id": str(row["event_id"]),
                "kind": str(row["kind"]),
                "source": str(row["source"]),
                "step": int(row["step"] or 0),
                "writer_id": self.writer_id,
                "payload": payload,
            }
            if event["kind"] == "history":
                event["payload"]["orchestration/event_seq"] = frame_id
                event["payload"]["orchestration/event_id"] = event["event_id"]
                if event["source"].startswith("eval"):
                    event["payload"]["eval/checkpoint_step"] = event["step"]
                elif not event["source"].startswith("orchestration"):
                    event["payload"]["train/global_step"] = event["step"]
            self.wandb_events.append(event)
            if self.evidence_path is not None:
                self.evidence_path.parent.mkdir(parents=True, exist_ok=True)
                self.evidence_path.write_bytes(_canonical_bytes({"events": self.wandb_events}))
            self.summary["orchestration/event_seq"] = frame_id
            store.mark_metric_frame_published(frame_id, step=row["step"])
            published += 1
        return published

    def publish_promotion(
        self,
        projector: WandbProjector,
        *,
        checkpoint_step: int,
        checkpoint_url: str,
        metrics: Mapping[str, Any],
        updated_at: str,
    ) -> None:
        del projector, metrics, updated_at
        self.summary.update(
            {
                "gradlab/goal/outcome": "accepted",
                "leader/checkpoint/step": int(checkpoint_step),
                "leader/checkpoint/artifact_ref": str(checkpoint_url),
            }
        )

    def remote_summary(self, run_path: str) -> dict[str, Any]:
        if not run_path:
            raise ValueError("certification W&B path is empty")
        return dict(self.summary)

    def close_wandb(
        self,
        projector: WandbProjector,
        *,
        timeout_seconds: float,
    ) -> None:
        del projector, timeout_seconds
        self.closed = True

    def publish_terminal(
        self,
        train_config: Mapping[str, Any],
        receipt: TerminalReceipt,
        *,
        timeout_seconds: float,
    ) -> None:
        del train_config, timeout_seconds
        self.summary.update(
            {
                "gradlab/run/terminal_state": receipt.state,
                "gradlab/run/stop_reason": receipt.stop_reason,
                "gradlab/run/final_step": receipt.final_step,
                "gradlab/run/early_stop_trigger": str(
                    (receipt.early_stop or {}).get("trigger") or ""
                ),
                "gradlab/run/early_stop_condition": str(
                    (receipt.early_stop or {}).get("condition_id") or ""
                ),
            }
        )

    def request_learner_stop(self, learner) -> None:
        del learner
        self.stop_requests += 1

    def start_learner(self, command, *, log_path: Path, environment):
        del command, log_path, environment
        self.learner_starts += 1
        raise AssertionError("Tier 1 must not start a real learner process")

    def install_cancel_handlers(self, callback):
        del callback
        return None

    def restore_cancel_handlers(self, token) -> None:
        del token


@dataclass
class ScenarioRecorder:
    name: str
    invariants: list[dict[str, Any]]

    def require(
        self,
        invariant: str,
        condition: bool,
        *,
        evidence: Mapping[str, Any] | None = None,
    ) -> None:
        row = {
            "name": invariant,
            "status": "passed" if condition else "failed",
            "evidence": dict(evidence or {}),
        }
        self.invariants.append(row)
        if not condition:
            raise AssertionError(f"{self.name}: {invariant}: {row['evidence']}")


@dataclass
class PreparedSupervisor:
    supervisor: RunSupervisor
    runtime: CertificationRuntime
    backend: ScriptedEvalBackend
    observer: RecordingObserver


class CertificationFixture:
    def __init__(self, root: Path, *, clock: DeterministicClock | None = None) -> None:
        self.root = root
        self.clock = clock or DeterministicClock()
        self.storage = RunStorageConfig(
            control=BucketConfig(uri=f"file://{root / 'control'}"),
            evaluation=BucketConfig(uri=f"file://{root / 'eval'}"),
            models=BucketConfig(
                uri=f"file://{root / 'models'}",
                public_base_url=f"file://{root / 'models'}",
            ),
        )
        self.authority = RunAuthority(self.storage, clock=self.clock)
        self.asset = {
            "schema_version": 2,
            "game": "SuperMarioBros-Nes-v0",
            "filename": "certification.nes",
            "size_bytes": 1,
            "sha256": "c" * 64,
            "object_uri": self.authority.evaluation.uri("assets/certification.nes"),
            "provider_rom_identity": "d" * 40,
            "provider_rom_identity_algorithm": "sha1-provider-body-v1",
        }
        resolved_documents = compose_resolved_train_documents(
            GOAL_PATH,
            RECIPE_PATH,
            source_sha=SOURCE_SHA,
        )
        document = resolved_documents.effective
        contract_document = dict(document)
        config = dict(contract_document["train_config"])
        config["rom_asset_manifest"] = self.asset
        config["checkpoint_eval_backend"] = "modal"
        contract_document["train_config"] = config
        contract_document["goal_variant"] = build_goal_variant_descriptor(
            goal_slug="SuperMarioBros-Nes-v0/Level1-1",
            source_sha=SOURCE_SHA,
            authored_goal=load_goal_contract(GOAL_PATH, Path.cwd()),
            effective_goal=dict(document["goal"]),
        )
        self.composed = contract_document
        self.recipe_document = build_recipe_document(
            contract_document,
            repo_root=Path.cwd(),
            source_commit=SOURCE_SHA,
            run_description="deterministic Tier 1 lifecycle certification",
            seed=123,
            runtime_image_ref=IMAGE_REF,
            base_materialized_recipe={
                **resolved_documents.base,
                "train_config": {
                    **resolved_documents.base["train_config"],
                    "rom_asset_manifest": self.asset,
                    "checkpoint_eval_backend": "modal",
                },
            },
            canonical_goal=resolved_documents.canonical_goal,
        )
        self.authority.put_recipe_document(self.recipe_document)

    def manifest(
        self,
        *,
        run_number: int,
        attempt_number: int = 1,
        recovery_mode: str = "resume-training",
    ) -> RunManifest:
        run_id = f"gradlab-{run_number:032x}"
        attempt_id = f"attempt-{attempt_number:016x}"
        compute = {
            "request": {
                "kind": "local",
                "target": "simulated-b3",
                "max_price": None,
                "max_cost_usd": None,
                "allow_on_demand": False,
                "max_duration_seconds": 3600,
            },
            "selected": {
                "kind": "local",
                "target": "simulated-b3",
                "max_price": None,
                "max_cost_usd": None,
                "allow_on_demand": False,
                "max_duration_seconds": 3600,
            },
            "dstack_task": run_id,
            "runtime_workflow_run_id": "tier1",
            "runtime_input_sha256": RUNTIME_INPUT_SHA256,
            "runtime_build_source_sha": BUILD_SOURCE_SHA,
            "recovery_mode": recovery_mode,
        }
        return RunManifest(
            run_id=run_id,
            attempt_id=attempt_id,
            created_at=self.clock.utc_now(),
            source_sha=SOURCE_SHA,
            image_digest=IMAGE_REF,
            goal_slug="SuperMarioBros-Nes-v0/Level1-1",
            goal_sha256=str(self.composed["train_config"]["effective_goal_contract_sha256"]),
            recipe_slug="ppo",
            recipe_sha256=canonical_json_sha256(self.recipe_document),
            recipe_overrides=(),
            environment_sha256=str(self.composed["environment_hash"]).removeprefix("sha256:"),
            seed=123,
            run_description="deterministic Tier 1 lifecycle certification",
            compute=compute,
            wandb={
                "run_id": run_id,
                "entity": "certification",
                "project": "SuperMarioBros-Nes-v0",
                "url": (f"https://wandb.invalid/certification/SuperMarioBros-Nes-v0/runs/{run_id}"),
            },
            modal={
                "enabled": True,
                "environment_name": "certification",
                "app_name": f"gradlab-eval-v3-{SOURCE_SHA[:12]}",
                "function_name": "evaluate_checkpoint",
                "deployment_source_sha": SOURCE_SHA,
                "rom_asset_manifest": self.asset,
            },
            storage=self.storage.manifest_locations(),
            goal_variant=self.composed["goal_variant"],
        )

    def prepare(
        self,
        *,
        run_number: int,
        attempt_number: int = 1,
        recovery_mode: str = "resume-training",
        backend: ScriptedEvalBackend | None = None,
        publish_failures: int = 0,
    ) -> PreparedSupervisor:
        manifest = self.manifest(
            run_number=run_number,
            attempt_number=attempt_number,
            recovery_mode=recovery_mode,
        )
        if attempt_number == 1:
            self.authority.create_manifest(manifest)
        else:
            self.authority.create_attempt_manifest(manifest)
        evidence_root = self.root / "evidence" / f"{manifest.run_id}-{manifest.attempt_id}"
        runtime = CertificationRuntime(
            clock=self.clock,
            writer_id=f"writer-{run_number}-{attempt_number}",
            publish_failures=publish_failures,
            evidence_path=evidence_root / "wandb-events.json",
        )
        selected_backend = backend or ScriptedEvalBackend()
        observer = RecordingObserver(evidence_path=evidence_root / "transcript.json")
        supervisor = RunSupervisor(
            manifest_uri=self.authority.control.uri(
                (
                    f"runs/{manifest.run_id}/manifest.json"
                    if attempt_number == 1
                    else (f"runs/{manifest.run_id}/attempts/{manifest.attempt_id}/manifest.json")
                )
            ),
            storage=self.storage,
            eval_backend=selected_backend,
            repo_root=Path.cwd(),
            work_root=self.root / f"work-{run_number}-{attempt_number}",
            runtime=runtime,
            authority=self.authority,
            observer=observer,
        )
        supervisor.recipe_document = self.recipe_document
        supervisor.eval_contract = evaluation_contract(self.recipe_document)
        supervisor.store.init()
        supervisor.projector = WandbProjector(object())
        supervisor.wandb_run_path = f"certification/SuperMarioBros-Nes-v0/{manifest.run_id}"
        supervisor.lease = self.authority.acquire_lease(
            run_id=manifest.run_id,
            attempt_id=manifest.attempt_id,
            holder_id=runtime.writer_id,
        )
        supervisor.last_lease_renewal = self.clock.monotonic()
        supervisor._emit("writer_lease_acquired", holder_id=runtime.writer_id)
        return PreparedSupervisor(supervisor, runtime, selected_backend, observer)

    def record_checkpoint(
        self,
        prepared: PreparedSupervisor,
        *,
        step: int,
        kind: str,
    ) -> int:
        supervisor = prepared.supervisor
        directory = self.root / "checkpoint-source" / supervisor.manifest.run_id / str(step)
        directory.mkdir(parents=True, exist_ok=True)
        model_path = directory / "model.zip"
        model_path.write_bytes(f"checkpoint:{supervisor.manifest.run_id}:{step}".encode())
        write_canonical_json(
            model_document_path(model_path),
            {
                "schema_version": 1,
                "run_id": supervisor.manifest.run_id,
                "step": step,
            },
        )
        write_canonical_json(recipe_document_path(model_path), self.recipe_document)
        return supervisor.store.record_checkpoint(
            run_name=supervisor.manifest.run_id,
            kind=kind,
            step=step,
            path=model_path,
            eval_required=True,
        )


def _accepted_or_rejected_raw(
    row: Mapping[str, Any],
    *,
    accepted: bool,
) -> dict[str, Any]:
    intent = dict(row["intent"])
    contract = dict(intent["execution_contract"])
    entries = list(contract["manifest"]["episodes"])
    selected = entries if accepted else entries[:1]
    episodes = [
        {
            "episode_id": str(entry["episode_id"]),
            "seed_lane": int(entry["lane"]),
            "seed_episode_ordinal": int(entry["lane_episode_ordinal"]),
            "seed": int(entry["seed"]),
            "start_state": str(entry.get("start_state") or ""),
            "outcome": "success" if accepted else "failure",
            "level_complete": bool(accepted),
            "return": 1.0 if accepted else 0.0,
            "steps": 100,
        }
        for entry in selected
    ]
    aggregates = acceptance_aggregates(episodes, contract=contract)
    metrics: dict[str, float] = {}
    if accepted:
        metrics = {
            "eval/full/outcome/success/rate/min": 1.0,
            "eval/full/outcome/success/rate/mean": 1.0,
        }
    attempt = int(row["attempt"])
    asset = contract.get("asset")
    return {
        "schema_version": int(contract["schema_version"]),
        "attempt_id": f"{intent['idempotency_key'][:20]}-a{attempt}",
        "execution_key": execution_key(contract),
        "checkpoint_sha256": str(contract["checkpoint_sha256"]),
        "recipe_sha256": str(contract["recipe_sha256"]),
        "recipe_format_version": int(contract["recipe_format_version"]),
        "evaluation_contract_sha256": str(contract["evaluation_contract_sha256"]),
        "contract_schema_version": int(contract["schema_version"]),
        "runtime_image_ref": str(contract["runtime_image_ref"]),
        "rom_sha256": (str(asset.get("sha256") or "") if isinstance(asset, Mapping) else ""),
        "seed_protocol": str(contract["seed_protocol"]),
        "n_envs": int(contract["n_envs"]),
        "episodes": int(contract["episodes"]),
        "status": "succeeded",
        "verdict": "accepted" if accepted else "rejected",
        "episode_results": episodes,
        "claimed_aggregates": aggregates,
        "metrics": metrics,
        "duration_seconds": 1.0,
    }


def _write_raw_result(
    prepared: PreparedSupervisor,
    *,
    accepted: bool,
    position: int = 0,
) -> dict[str, Any]:
    rows = prepared.supervisor.store.evals(statuses=("submitted",))
    row = rows[position]
    raw = _accepted_or_rejected_raw(row, accepted=accepted)
    prepared.supervisor.authority.evaluation.put_json(
        str(row["intent"]["result_key"]),
        raw,
        create_only=True,
    )
    return row


def _trace_positions(
    events: Sequence[Mapping[str, Any]],
    *,
    checkpoint_id: str,
) -> dict[str, int]:
    selected: dict[str, int] = {}
    for event in events:
        if str(event.get("checkpoint_id") or "") != checkpoint_id:
            continue
        kind = str(event["kind"])
        if kind in {
            "checkpoint_published",
            "eval_intent_persisted",
            "eval_submitted",
            "eval_terminal",
        }:
            selected[kind] = int(event["sequence"])
    return selected


class LifecycleVerifier:
    """Independent reader of raw storage, metric, and transcript evidence."""

    def verify_success(
        self,
        *,
        prepared: PreparedSupervisor,
        receipt: TerminalReceipt,
    ) -> list[dict[str, Any]]:
        supervisor = prepared.supervisor
        authority = supervisor.authority
        run_id = supervisor.manifest.run_id
        checks: list[dict[str, Any]] = []

        def check(name: str, condition: bool, evidence: Mapping[str, Any]) -> None:
            if not condition:
                raise AssertionError(f"independent verifier: {name}: {dict(evidence)}")
            checks.append({"name": name, "status": "passed", "evidence": dict(evidence)})

        manifest = RunManifest(**authority.control.get_json(f"runs/{run_id}/manifest.json"))
        manifest.validate()
        check("manifest-valid", True, {"run_id": run_id})

        index = authority.models.get_json(f"runs/{run_id}/index.json")
        checkpoint_rows = [dict(row) for row in index["checkpoints"]]
        for checkpoint in checkpoint_rows:
            prefix = f"runs/{run_id}/checkpoints/{int(checkpoint['step'])}-{checkpoint['sha256']}"
            payload = authority.models.get_bytes(f"{prefix}/model.zip")
            check(
                f"checkpoint-hash-{checkpoint['step']}",
                hashlib.sha256(payload).hexdigest() == str(checkpoint["sha256"]),
                {"checkpoint_id": checkpoint["checkpoint_id"]},
            )
            stored = authority.models.get_json(f"{prefix}/manifest.json")
            check(
                f"checkpoint-manifest-{checkpoint['step']}",
                stored == checkpoint,
                {"checkpoint_id": checkpoint["checkpoint_id"]},
            )

        eval_rows = supervisor.store.evals()
        checkpoint_ids = {str(row["checkpoint_id"]) for row in checkpoint_rows}
        eval_checkpoint_ids = {str(row["checkpoint_id"]) for row in eval_rows}
        unevaluated = [
            row for row in checkpoint_rows if str(row["checkpoint_id"]) not in eval_checkpoint_ids
        ]
        check(
            "post-acceptance-checkpoints-remain-unevaluated",
            eval_checkpoint_ids < checkpoint_ids
            and all(
                int(row["step"]) > max(int(eval_row["checkpoint_step"]) for eval_row in eval_rows)
                for row in unevaluated
            )
            and any(str(row["purpose"]) == "periodic" for row in unevaluated)
            and any(str(row["purpose"]) == "final" for row in unevaluated),
            {
                "checkpoints": len(checkpoint_rows),
                "evals": len(eval_rows),
                "unevaluated": [str(row["checkpoint_id"]) for row in unevaluated],
            },
        )
        check(
            "all-evals-terminal",
            all(
                str(row["status"]) in {"accepted", "rejected", "failed", "expired", "canceled"}
                for row in eval_rows
            ),
            {"statuses": [row["status"] for row in eval_rows]},
        )
        for row in eval_rows:
            key = str(row["idempotency_key"])
            check(
                f"eval-intent-{row['checkpoint_step']}",
                authority.evaluation.get_json_optional(f"runs/{run_id}/evals/{key}/intent.json")
                is not None,
                {"idempotency_key": key},
            )
            check(
                f"eval-result-{row['checkpoint_step']}",
                authority.evaluation.get_json_optional(
                    f"runs/{run_id}/evals/{key}/verified-result.json"
                )
                is not None,
                {"idempotency_key": key},
            )

        frames = prepared.runtime.wandb_events
        event_ids = [str(row["event_id"]) for row in frames]
        writers = {str(row["writer_id"]) for row in frames}
        check(
            "single-wandb-writer",
            writers == {prepared.runtime.writer_id},
            {"writers": sorted(writers)},
        )
        check(
            "wandb-event-id-deduplication",
            len(event_ids) == len(set(event_ids)),
            {"events": len(event_ids), "unique": len(set(event_ids))},
        )
        check(
            "training-and-eval-metrics-share-run",
            any(str(row["source"]).startswith("learner") for row in frames)
            and any(str(row["source"]).startswith("eval") for row in frames),
            {"sources": sorted({str(row["source"]) for row in frames})},
        )
        max_event_seq = max((int(row["event_seq"]) for row in frames), default=0)
        check(
            "delivery-high-water",
            max_event_seq
            == int(receipt.wandb_high_water_mark)
            == int(receipt.drain["metric_segment_high_water"])
            and int(receipt.drain["wandb_remote_high_water_mark"]) >= max_event_seq,
            {"high_water": max_event_seq},
        )

        promotion = authority.control.get_json(f"runs/{run_id}/promotion.json")
        accepted = [row for row in eval_rows if str(row["status"]) == "accepted"]
        lowest = min(accepted, key=lambda row: int(row["checkpoint_step"]))
        check(
            "lowest-step-accepted-promotion",
            str(promotion["checkpoint_id"]) == str(lowest["checkpoint_id"]),
            {
                "promoted_step": int(promotion["checkpoint_step"]),
                "lowest_accepted_step": int(lowest["checkpoint_step"]),
            },
        )
        terminal = authority.control.get_json(f"runs/{run_id}/terminal.json")
        check(
            "scientific-terminal-receipt",
            terminal == receipt.to_dict()
            and terminal["state"] == "succeeded"
            and terminal["stop_reason"] == "eval_acceptance",
            {"state": terminal["state"], "stop_reason": terminal["stop_reason"]},
        )

        periodic_id = str(min(checkpoint_rows, key=lambda row: int(row["step"]))["checkpoint_id"])
        positions = _trace_positions(
            prepared.observer.events,
            checkpoint_id=periodic_id,
        )
        ordered = [
            positions.get(name, -1)
            for name in (
                "checkpoint_published",
                "eval_intent_persisted",
                "eval_submitted",
                "eval_terminal",
            )
        ]
        check(
            "checkpoint-to-eval-causality",
            ordered == sorted(ordered) and all(position > 0 for position in ordered),
            {"event_sequences": ordered},
        )
        stop_events = [
            row for row in prepared.observer.events if row["kind"] == "learner_stop_requested"
        ]
        check(
            "eval-driven-stop",
            len(stop_events) == 1
            and stop_events[0]["reason"] == "eval_acceptance"
            and receipt.final_step < 50_000_000,
            {
                "stop_events": len(stop_events),
                "final_step": receipt.final_step,
            },
        )
        return checks


def _finalize_success(prepared: PreparedSupervisor) -> TerminalReceipt:
    supervisor = prepared.supervisor
    supervisor._seal_metrics(supervisor.clock.monotonic(), force=True)
    while supervisor._publish_wandb():
        pass
    promotion = supervisor._create_promotion()
    if promotion is None:
        raise AssertionError("successful certification has no accepted checkpoint")
    supervisor._publish_promotion(promotion)
    wandb_high_water = supervisor._finish_wandb()
    supervisor._wait_for_remote_delivery(wandb_high_water)
    supervisor._wait_for_remote_promotion(promotion)
    journal = supervisor.authority.archive_metric_journals(run_id=supervisor.manifest.run_id)
    checkpoints, evals = supervisor._terminal_inventory()
    receipt = TerminalReceipt(
        run_id=supervisor.manifest.run_id,
        attempt_id=supervisor.manifest.attempt_id,
        state="succeeded",
        acceptance_required=True,
        stop_reason=supervisor.stop_reason,
        final_step=max(int(row["step"]) for row in checkpoints),
        checkpoint_inventory=checkpoints,
        eval_inventory=evals,
        wandb_high_water_mark=wandb_high_water,
        drain={
            "complete": True,
            "metric_segment_high_water": supervisor.store.metric_segment_high_water(),
            "eval_terminal_count": supervisor.store.terminal_eval_count(),
            "journal_archive": journal,
            "journal_expires_at": (
                supervisor.clock.utc_datetime() + timedelta(days=METRIC_JOURNAL_RETENTION_DAYS)
            )
            .isoformat()
            .replace("+00:00", "Z"),
            "wandb_remote_high_water_mark": int(
                prepared.runtime.summary.get("orchestration/event_seq") or 0
            ),
            "publication_capacity_ratio": None,
            "failure": None,
        },
        completed_at=supervisor.clock.utc_now(),
    )
    supervisor.authority.create_attempt_terminal(receipt)
    supervisor.authority.create_terminal(receipt)
    supervisor._emit(
        "attempt_terminal_created",
        state=receipt.state,
        stop_reason=receipt.stop_reason,
        final_step=receipt.final_step,
    )
    return receipt


def _scenario_full_lifecycle(root: Path) -> dict[str, Any]:
    recorder = ScenarioRecorder("full-lifecycle", [])
    fixture = CertificationFixture(root)
    prepared = fixture.prepare(run_number=1)
    supervisor = prepared.supervisor
    supervisor.store.append_metrics(
        {"train/episode/return/shaped/mean": 1.0},
        step=250_000,
        source="learner",
    )
    fixture.record_checkpoint(prepared, step=250_000, kind="checkpoint")
    fixture.clock.advance(5)
    supervisor.active_iteration()
    first_eval = supervisor.store.evals()[0]
    fixture.record_checkpoint(prepared, step=255_000, kind="checkpoint")
    fixture.clock.advance(5)
    supervisor.active_iteration()
    recorder.require(
        "second-eval-launched-before-acceptance",
        len(supervisor.store.evals(statuses=("submitted",))) == 2,
        evidence={"submitted": len(supervisor.store.evals(statuses=("submitted",)))},
    )
    _write_raw_result(prepared, accepted=True)
    fixture.clock.advance(2)
    supervisor.active_iteration()
    recorder.require(
        "accepted-eval-requested-stop",
        supervisor.stop_reason == "eval_acceptance",
        evidence={"stop_reason": supervisor.stop_reason},
    )

    fixture.record_checkpoint(prepared, step=258_000, kind="checkpoint")
    fixture.record_checkpoint(prepared, step=260_000, kind="final")
    fixture.clock.advance(5)
    supervisor.active_iteration()
    recorder.require(
        "post-acceptance-checkpoints-published-without-automatic-eval",
        len(supervisor.store.checkpoint_publications()) == 4
        and len(supervisor.store.evals()) == 2
        and all(
            int(row["checkpoint_step"]) not in {258_000, 260_000}
            for row in supervisor.store.evals()
        ),
        evidence={
            "checkpoints": len(supervisor.store.checkpoint_publications()),
            "evals": len(supervisor.store.evals()),
        },
    )
    _write_raw_result(prepared, accepted=False)
    fixture.clock.advance(2)
    supervisor.active_iteration()
    receipt = _finalize_success(prepared)
    verifier_checks = LifecycleVerifier().verify_success(
        prepared=prepared,
        receipt=receipt,
    )
    recorder.require(
        "all-one-hundred-episodes-accepted",
        len(first_eval["intent"]["execution_contract"]["manifest"]["episodes"]) == 100,
        evidence={"episodes": 100},
    )
    recorder.require(
        "already-submitted-rejection-does-not-displace-earlier-acceptance",
        prepared.supervisor.store.evals()[-1]["status"] == "rejected"
        and receipt.state == "succeeded",
        evidence={
            "final_status": prepared.supervisor.store.evals()[-1]["status"],
            "terminal_state": receipt.state,
        },
    )
    _write_evidence(root, prepared)
    return {
        "invariants": recorder.invariants + verifier_checks,
        "evidence": {
            "run_id": supervisor.manifest.run_id,
            "checkpoint_count": len(supervisor.store.checkpoint_publications()),
            "eval_statuses": [row["status"] for row in supervisor.store.evals()],
            "wandb_event_count": len(prepared.runtime.wandb_events),
            "terminal_state": receipt.state,
            "stop_reason": receipt.stop_reason,
            "final_step": receipt.final_step,
        },
    }


def _scenario_parallel_run_isolation(root: Path) -> dict[str, Any]:
    recorder = ScenarioRecorder("parallel-run-isolation", [])
    fixture = CertificationFixture(root)
    first = fixture.prepare(run_number=11)
    second = fixture.prepare(run_number=12)
    for index, prepared in enumerate((first, second), start=1):
        prepared.supervisor.store.append_metrics(
            {"train/episode/return/shaped/mean": float(index)},
            step=100 * index,
            source="learner",
        )
        fixture.record_checkpoint(prepared, step=100 * index, kind="checkpoint")
    fixture.clock.advance(5)
    first.supervisor.active_iteration()
    second.supervisor.active_iteration()
    recorder.require(
        "independent-run-leases-coexist",
        first.supervisor.lease is not None
        and second.supervisor.lease is not None
        and first.supervisor.lease.run_id != second.supervisor.lease.run_id,
        evidence={
            "run_ids": [
                first.supervisor.manifest.run_id,
                second.supervisor.manifest.run_id,
            ]
        },
    )
    first_keys = list(
        fixture.authority.control.iter_keys(f"runs/{first.supervisor.manifest.run_id}")
    )
    second_keys = list(
        fixture.authority.control.iter_keys(f"runs/{second.supervisor.manifest.run_id}")
    )
    recorder.require(
        "parallel-storage-prefixes-isolated",
        first_keys and second_keys and not set(first_keys).intersection(second_keys),
        evidence={"first_keys": len(first_keys), "second_keys": len(second_keys)},
    )
    recorder.require(
        "parallel-wandb-writers-isolated",
        {row["writer_id"] for row in first.runtime.wandb_events} == {first.runtime.writer_id}
        and {row["writer_id"] for row in second.runtime.wandb_events} == {second.runtime.writer_id},
        evidence={"writers": [first.runtime.writer_id, second.runtime.writer_id]},
    )
    recorder.require(
        "parallel-modal-dispatches-isolated",
        len(first.backend.submissions) == 1
        and len(second.backend.submissions) == 1
        and first.backend.submissions[0]["result_uri"]
        != second.backend.submissions[0]["result_uri"],
        evidence={
            "submission_counts": [
                len(first.backend.submissions),
                len(second.backend.submissions),
            ]
        },
    )
    return {
        "invariants": recorder.invariants,
        "evidence": {
            "interleaving": [
                first.supervisor.manifest.run_id,
                second.supervisor.manifest.run_id,
            ]
        },
    }


def _scenario_same_run_lease_fencing(root: Path) -> dict[str, Any]:
    recorder = ScenarioRecorder("same-run-lease-fencing", [])
    fixture = CertificationFixture(root)
    manifest = fixture.manifest(run_number=21)
    fixture.authority.create_manifest(manifest)
    first = fixture.authority.acquire_lease(
        run_id=manifest.run_id,
        attempt_id=manifest.attempt_id,
        holder_id="writer-a",
    )
    blocked = False
    try:
        fixture.authority.acquire_lease(
            run_id=manifest.run_id,
            attempt_id=manifest.attempt_id,
            holder_id="writer-b",
        )
    except LeaseUnavailable:
        blocked = True
    recorder.require(
        "competing-supervisor-blocked",
        blocked,
        evidence={"active_holder": first.holder_id},
    )
    fixture.clock.advance(LEASE_TTL_SECONDS + 1)
    takeover = fixture.authority.acquire_lease(
        run_id=manifest.run_id,
        attempt_id="attempt-0000000000000002",
        holder_id="writer-b",
    )
    recorder.require(
        "takeover-only-after-expiry",
        takeover.holder_id == "writer-b" and takeover.generation == first.generation + 1,
        evidence={"generation": takeover.generation},
    )
    stale_renewal_blocked = False
    try:
        fixture.authority.renew_lease(first)
    except LeaseUnavailable:
        stale_renewal_blocked = True
    recorder.require(
        "stale-writer-cannot-renew",
        stale_renewal_blocked,
        evidence={"stale_holder": first.holder_id},
    )
    return {"invariants": recorder.invariants, "evidence": {"run_id": manifest.run_id}}


def _scenario_wandb_retry_deduplication(root: Path) -> dict[str, Any]:
    recorder = ScenarioRecorder("wandb-retry-deduplication", [])
    fixture = CertificationFixture(root)
    prepared = fixture.prepare(run_number=31, publish_failures=1)
    supervisor = prepared.supervisor
    supervisor.store.append_metrics(
        {"train/episode/return/shaped/mean": 3.0},
        step=300,
        source="learner",
    )
    fixture.clock.advance(5)
    supervisor.active_iteration()
    pending = supervisor.store.pending_metric_frames()
    recorder.require(
        "wandb-failure-remains-retryable",
        len(pending) == 1 and int(pending[0]["attempts"]) == 1,
        evidence={"pending": len(pending)},
    )
    supervisor.active_iteration()
    with supervisor.store.connection() as connection:
        attempts = int(
            connection.execute("SELECT attempts FROM metric_frames ORDER BY id LIMIT 1").fetchone()[
                0
            ]
        )
    recorder.require(
        "wandb-retry-publishes-once",
        attempts == 2
        and len(prepared.runtime.wandb_events) == 1
        and supervisor.store.metric_outbox_stats()["frames"] == 0,
        evidence={
            "attempts": attempts,
            "remote_events": len(prepared.runtime.wandb_events),
        },
    )
    return {
        "invariants": recorder.invariants,
        "evidence": {"event_id": prepared.runtime.wandb_events[0]["event_id"]},
    }


def _scenario_wandb_visibility_gating(root: Path) -> dict[str, Any]:
    recorder = ScenarioRecorder("wandb-visibility-gating", [])
    fixture = CertificationFixture(root)
    prepared = fixture.prepare(run_number=36)
    supervisor = prepared.supervisor
    supervisor.store.append_metrics(
        {"train/episode/return/shaped/mean": 3.6},
        step=360,
        source="learner",
    )
    fixture.clock.advance(5)
    supervisor.active_iteration()
    high_water = supervisor._finish_wandb()
    prepared.runtime.summary["orchestration/event_seq"] = 0
    timed_out = False
    try:
        supervisor._wait_for_remote_delivery(high_water)
    except TimeoutError:
        timed_out = True
    recorder.require(
        "local-sdk-delivery-is-not-remote-visibility",
        high_water == 1 and timed_out,
        evidence={
            "local_high_water": high_water,
            "remote_high_water": supervisor.wandb_remote_high_water,
        },
    )
    return {
        "invariants": recorder.invariants,
        "evidence": {"virtual_timeout_seconds": 300},
    }


def _scenario_checkpoint_upload_retry(root: Path) -> dict[str, Any]:
    recorder = ScenarioRecorder("checkpoint-upload-retry", [])
    fixture = CertificationFixture(root)
    prepared = fixture.prepare(run_number=37)
    supervisor = prepared.supervisor
    checkpoint_ledger_id = fixture.record_checkpoint(
        prepared,
        step=370,
        kind="checkpoint",
    )
    fixture.clock.advance(5)
    with patch.object(
        supervisor.authority,
        "publish_checkpoint",
        side_effect=RuntimeError("simulated interrupted checkpoint upload"),
    ):
        supervisor.active_iteration()
    failed = supervisor.store.checkpoints()[0]
    recorder.require(
        "interrupted-upload-does-not-create-eval-intent",
        failed["upload_status"] == "failed_retryable"
        and not supervisor.store.evals()
        and supervisor.store.checkpoint_publication(checkpoint_ledger_id) is None,
        evidence={"upload_status": failed["upload_status"]},
    )
    supervisor.active_iteration()
    published = supervisor.store.checkpoint_publication(checkpoint_ledger_id)
    recorder.require(
        "verified-retry-unlocks-eval-dispatch",
        published is not None
        and supervisor.store.checkpoints()[0]["upload_status"] == "uploaded"
        and len(supervisor.store.evals()) == 1
        and len(prepared.backend.submissions) == 1,
        evidence={
            "checkpoint_id": (published or {}).get("checkpoint_id"),
            "submissions": len(prepared.backend.submissions),
        },
    )
    return {
        "invariants": recorder.invariants,
        "evidence": {"checkpoint_ledger_id": checkpoint_ledger_id},
    }


def _scenario_eval_result_reconciliation(root: Path) -> dict[str, Any]:
    recorder = ScenarioRecorder("eval-result-reconciliation", [])
    fixture = CertificationFixture(root)
    prepared = fixture.prepare(run_number=39)
    supervisor = prepared.supervisor
    fixture.record_checkpoint(prepared, step=390, kind="checkpoint")
    fixture.clock.advance(5)
    supervisor.active_iteration()
    recorder.require(
        "first-eval-submitted",
        len(prepared.backend.submissions) == 1,
        evidence={"submissions": len(prepared.backend.submissions)},
    )

    fixture.record_checkpoint(prepared, step=395, kind="final")
    _write_raw_result(prepared, accepted=True)
    fixture.clock.advance(5)
    supervisor.active_iteration()
    statuses = [str(row["status"]) for row in supervisor.store.evals()]
    recorder.require(
        "durable-acceptance-reconciled-before-next-submit",
        len(prepared.backend.submissions) == 1
        and statuses == ["accepted", "deferred"]
        and supervisor.eval_admission_closed
        and supervisor.store.all_evals_settled(),
        evidence={
            "submissions": len(prepared.backend.submissions),
            "statuses": statuses,
            "admission_closed": supervisor.eval_admission_closed,
        },
    )
    checkpoints, evals = supervisor._terminal_inventory()
    recorder.require(
        "deferred-intent-is-durable-terminal-inventory-evidence",
        len(checkpoints) == 2
        and [str(row["status"]) for row in evals] == ["accepted", "deferred"]
        and evals[1]["result_sha256"] is None,
        evidence={
            "checkpoint_count": len(checkpoints),
            "eval_inventory": evals,
        },
    )
    return {
        "invariants": recorder.invariants,
        "evidence": {
            "submissions": len(prepared.backend.submissions),
            "statuses": statuses,
        },
    }


def _scenario_modal_ambiguous_submit(root: Path) -> dict[str, Any]:
    recorder = ScenarioRecorder("modal-ambiguous-submit", [])
    fixture = CertificationFixture(root)
    backend = ScriptedEvalBackend(ambiguous_submits=1)
    prepared = fixture.prepare(run_number=41, backend=backend)
    fixture.record_checkpoint(prepared, step=400, kind="checkpoint")
    fixture.clock.advance(5)
    prepared.supervisor.active_iteration()
    row = prepared.supervisor.store.evals()[0]
    prepared.supervisor.active_iteration()
    recorder.require(
        "ambiguous-submit-not-immediately-repeated",
        len(backend.submissions) == 1
        and row["status"] == "submitted"
        and row["modal_call_id"] == "",
        evidence={"submissions": len(backend.submissions), "attempt": row["attempt"]},
    )
    with prepared.supervisor.store.connection() as connection:
        connection.execute(
            "UPDATE eval_dispatches SET attempt_expires_at = ?",
            (fixture.clock.time() + 1,),
        )
    fixture.clock.advance(2)
    prepared.supervisor.active_iteration()
    prepared.supervisor.active_iteration()
    retried = prepared.supervisor.store.evals()[0]
    recorder.require(
        "ambiguous-submit-retries-after-expiry",
        len(backend.submissions) == 2
        and int(retried["attempt"]) == 2
        and bool(retried["modal_call_id"]),
        evidence={
            "submissions": len(backend.submissions),
            "attempt": retried["attempt"],
        },
    )
    return {"invariants": recorder.invariants, "evidence": {"call_id": retried["modal_call_id"]}}


def _scenario_cancellation_terminalization(root: Path) -> dict[str, Any]:
    recorder = ScenarioRecorder("cancellation-terminalization", [])
    fixture = CertificationFixture(root)
    prepared = fixture.prepare(run_number=51)
    fixture.record_checkpoint(prepared, step=500, kind="checkpoint")
    fixture.clock.advance(5)
    prepared.supervisor.active_iteration()
    prepared.supervisor.cancel_requested = True
    prepared.supervisor.drain_iteration()
    row = prepared.supervisor.store.evals()[0]
    recorder.require(
        "cancel-terminalizes-eval",
        row["status"] == "canceled"
        and len(prepared.backend.canceled) == 1
        and prepared.supervisor.store.all_evals_terminal(),
        evidence={
            "status": row["status"],
            "canceled_calls": prepared.backend.canceled,
        },
    )
    return {"invariants": recorder.invariants, "evidence": {"eval_status": row["status"]}}


def _scenario_scratch_preservation_stop(root: Path) -> dict[str, Any]:
    recorder = ScenarioRecorder("scratch-preservation-stop", [])
    fixture = CertificationFixture(root)
    prepared = fixture.prepare(run_number=56)
    supervisor = prepared.supervisor
    supervisor.store.append_metrics(
        {"train/episode/return/shaped/mean": 5.6},
        step=560,
        source="learner",
    )
    prepared.runtime.disk_usage = lambda _path: SimpleNamespace(
        total=100,
        used=81,
        free=19,
    )
    fixture.clock.advance(5)
    stopped = False
    try:
        supervisor.active_iteration()
    except RuntimeError as exc:
        stopped = "evidence loss" in str(exc)
    recorder.require(
        "scratch-pressure-stops-before-evidence-loss",
        stopped
        and supervisor.stop_reason == "scratch_storage_above_80_percent"
        and supervisor.store.metric_segment_high_water() == 1,
        evidence={
            "stop_reason": supervisor.stop_reason,
            "r2_high_water": supervisor.store.metric_segment_high_water(),
        },
    )
    return {
        "invariants": recorder.invariants,
        "evidence": {"scratch_used_fraction": 0.81},
    }


def _scenario_drain_only_recovery(root: Path) -> dict[str, Any]:
    recorder = ScenarioRecorder("drain-only-recovery", [])
    fixture = CertificationFixture(root)
    first = fixture.prepare(run_number=61)
    first.supervisor.store.append_metrics(
        {"train/episode/return/shaped/mean": 6.0},
        step=600,
        source="learner",
    )
    fixture.record_checkpoint(first, step=600, kind="checkpoint")
    fixture.clock.advance(5)
    first.supervisor.active_iteration()
    _write_raw_result(first, accepted=True)
    first.supervisor._seal_metrics(first.supervisor.clock.monotonic(), force=True)
    first.supervisor._publish_wandb()
    first.supervisor.lease = None
    fixture.clock.advance(LEASE_TTL_SECONDS + 1)

    recovered = fixture.prepare(
        run_number=61,
        attempt_number=2,
        recovery_mode="drain-only",
    )
    recovered.supervisor._recover_durable_state()
    recovered_rows = recovered.supervisor.store.evals()
    recorder.require(
        "drain-only-recovers-checkpoint-and-eval",
        len(recovered.supervisor.store.checkpoint_publications()) == 1
        and len(recovered_rows) == 1
        and recovered_rows[0]["status"] == "accepted",
        evidence={
            "checkpoints": len(recovered.supervisor.store.checkpoint_publications()),
            "eval_status": recovered_rows[0]["status"],
        },
    )
    recorder.require(
        "drain-only-never-starts-learner",
        recovered.runtime.learner_starts == 0,
        evidence={"learner_starts": recovered.runtime.learner_starts},
    )
    return {
        "invariants": recorder.invariants,
        "evidence": {"recovered_attempt": recovered.supervisor.manifest.attempt_id},
    }


def _scenario_terminal_receipt_gating(root: Path) -> dict[str, Any]:
    recorder = ScenarioRecorder("terminal-receipt-gating", [])
    fixture = CertificationFixture(root)
    manifest = fixture.manifest(run_number=66)
    fixture.authority.create_manifest(manifest)
    receipt = TerminalReceipt(
        run_id=manifest.run_id,
        attempt_id=manifest.attempt_id,
        state="succeeded",
        acceptance_required=True,
        stop_reason="dstack_process_exited_zero",
        final_step=0,
        checkpoint_inventory=(),
        eval_inventory=(),
        wandb_high_water_mark=1,
        drain={
            "complete": True,
            "metric_segment_high_water": 1,
            "wandb_remote_high_water_mark": 1,
            "journal_archive": {
                "prefix": f"expiring-metric-journals/{manifest.run_id}/",
                "segment_count": 1,
                "keys": ["simulated.jsonl"],
            },
            "journal_expires_at": fixture.clock.utc_now(),
            "publication_capacity_ratio": None,
        },
        completed_at=fixture.clock.utc_now(),
    )
    fixture.authority.create_attempt_terminal(receipt)
    rejected = False
    try:
        fixture.authority.create_terminal(receipt)
    except ValueError:
        rejected = True
    recorder.require(
        "process-success-cannot-create-scientific-success",
        rejected
        and fixture.authority.control.get_json_optional(f"runs/{manifest.run_id}/terminal.json")
        is None
        and fixture.authority.control.get_json_optional(
            f"runs/{manifest.run_id}/attempts/{manifest.attempt_id}/terminal.json"
        )
        is not None,
        evidence={"canonical_terminal_created": False},
    )
    return {
        "invariants": recorder.invariants,
        "evidence": {"attempt_terminal_preserved": True},
    }


def _scenario_verifier_tamper_detection(root: Path) -> dict[str, Any]:
    recorder = ScenarioRecorder("verifier-tamper-detection", [])
    fixture = CertificationFixture(root)
    prepared = fixture.prepare(run_number=71)
    fixture.record_checkpoint(prepared, step=700, kind="checkpoint")
    fixture.clock.advance(5)
    prepared.supervisor.active_iteration()
    checkpoint = prepared.supervisor.store.checkpoint_publications()[0]
    key = (
        f"runs/{prepared.supervisor.manifest.run_id}/checkpoints/"
        f"{checkpoint['step']}-{checkpoint['sha256']}/model.zip"
    )
    original = prepared.supervisor.authority.models.get_bytes(key)
    object_path = Path(prepared.supervisor.authority.models.uri(key).removeprefix("file://"))
    object_path.write_bytes(b"tampered")
    detected = (
        hashlib.sha256(prepared.supervisor.authority.models.get_bytes(key)).hexdigest()
        != checkpoint["sha256"]
    )
    recorder.require(
        "checkpoint-corruption-detected",
        detected,
        evidence={"checkpoint_id": checkpoint["checkpoint_id"]},
    )
    object_path.write_bytes(original)

    etag = prepared.supervisor.authority.control.put_json(
        "cas/state.json",
        {"value": 1},
        create_only=True,
    )
    prepared.supervisor.authority.control.put_json(
        "cas/state.json",
        {"value": 2},
        create_only=False,
        if_match=etag,
    )
    conflict = False
    try:
        prepared.supervisor.authority.control.put_json(
            "cas/state.json",
            {"value": 3},
            create_only=False,
            if_match=etag,
        )
    except ConditionalWriteConflict:
        conflict = True
    recorder.require(
        "stale-etag-write-rejected",
        conflict,
        evidence={"initial_etag": etag},
    )
    return {"invariants": recorder.invariants, "evidence": {"tamper_repaired": True}}


def _scenario_early_stop_outcomes(root: Path) -> dict[str, Any]:
    recorder = ScenarioRecorder("early-stop-outcomes", [])
    fixture = CertificationFixture(root)
    prepared = fixture.prepare(run_number=76)
    manifest = prepared.supervisor.manifest

    def success_receipt_for(run_manifest: RunManifest) -> EarlyStopReceipt:
        receipt = EarlyStopReceipt(
            run_id=run_manifest.run_id,
            attempt_id=run_manifest.attempt_id,
            condition_id="target_reached",
            matched_condition_ids=("target_reached",),
            outcome="success",
            trigger="threshold",
            metric="train/outcome/success/window_100/rate/min",
            metric_step=10,
            value=0.95,
            best_value=0.95,
            elapsed_steps=0,
            patience_progress=1.0,
            condition={
                "metric": "train/outcome/success/window_100/rate/min",
                "trigger": "threshold",
                "operator": ">=",
                "threshold": 0.95,
                "start_after_steps": 0,
                "patience_steps": 0,
                "outcome": "success",
                "action": "stop",
            },
            early_stop_config_sha256="1" * 64,
            decision_sha256="2" * 64,
            recorded_at=fixture.clock.utc_now(),
        )
        receipt.validate()
        return receipt

    success_receipt = success_receipt_for(manifest)
    config = {
        "conditions": {
            "return_plateau": {
                "metric": "train/episode/return/shaped/from/target/mean",
                "trigger": "no_improvement",
                "direction": "maximize",
                "min_delta": 0.01,
                "delta_mode": "relative",
                "start_after_steps": 0,
                "patience_steps": 10,
                "outcome": "failure",
                "action": "stop",
            }
        }
    }
    machine = MetricEarlyStopStateMachine(config)
    metric = "train/episode/return/shaped/from/target/mean"
    machine.update({metric: MetricSample(value=100.0, step=0)})
    update = machine.update({metric: MetricSample(value=100.0, step=10)})
    decision = validate_metric_early_stop_decision(update.stop_decision, config)
    failure_receipt = EarlyStopReceipt(
        run_id=manifest.run_id,
        attempt_id=manifest.attempt_id,
        condition_id=str(decision["condition_id"]),
        matched_condition_ids=tuple(decision["matched_condition_ids"]),
        outcome="failure",
        trigger="no_improvement",
        metric=metric,
        metric_step=int(decision["metric_step"]),
        value=float(decision["value"]),
        best_value=float(decision["best_value"]),
        elapsed_steps=int(decision["elapsed_steps"]),
        patience_progress=float(decision["patience_progress"]),
        condition=dict(decision["condition"]),
        early_stop_config_sha256=str(decision["early_stop_config_sha256"]),
        decision_sha256=canonical_json_sha256(decision),
        recorded_at=fixture.clock.utc_now(),
    )
    fixture.authority.create_early_stop(failure_receipt)
    fixture.record_checkpoint(prepared, step=10, kind="final")
    fixture.clock.advance(5)
    prepared.supervisor.active_iteration()
    _write_raw_result(prepared, accepted=False)
    fixture.clock.advance(2)
    prepared.supervisor.active_iteration()
    prepared.supervisor._validate_no_acceptance_evidence()
    state, reason = _terminal_outcome(
        cancel_requested=False,
        failure=None,
        evaluation_required=True,
        promotion=None,
        early_stop=failure_receipt,
    )
    recorder.require(
        "plateau-is-scientific-failure-after-valid-rejection",
        state == "failed" and reason == "early_stop_failure:return_plateau",
        evidence={
            "state": state,
            "reason": reason,
            "eval_statuses": [row["status"] for row in prepared.supervisor.store.evals()],
        },
    )
    rejected_state, rejected_reason = _terminal_outcome(
        cancel_requested=False,
        failure=None,
        evaluation_required=True,
        promotion=None,
        early_stop=success_receipt,
    )
    recorder.require(
        "training-target-with-complete-rejection-is-scientific-failure",
        rejected_state == "failed"
        and rejected_reason == "early_stop_success_without_acceptance:target_reached",
        evidence={"state": rejected_state, "reason": rejected_reason},
    )

    incomplete = fixture.prepare(run_number=77)
    incomplete_success_receipt = success_receipt_for(incomplete.supervisor.manifest)
    fixture.record_checkpoint(incomplete, step=10, kind="final")
    fixture.clock.advance(5)
    incomplete.supervisor.active_iteration()
    incomplete.supervisor._mark_expired(
        incomplete.supervisor.store.evals(statuses=("submitted",))[0],
        error="simulated evaluation infrastructure failure",
    )
    incomplete_failure: BaseException | None = None
    try:
        incomplete.supervisor._validate_no_acceptance_evidence()
    except IncompleteEvaluationEvidence as exc:
        incomplete_failure = exc
    incomplete_state, incomplete_reason = _terminal_outcome(
        cancel_requested=False,
        failure=incomplete_failure,
        evaluation_required=True,
        promotion=None,
        early_stop=incomplete_success_receipt,
    )
    recorder.require(
        "training-target-with-incomplete-eval-evidence-is-resumable",
        isinstance(incomplete_failure, IncompleteEvaluationEvidence)
        and incomplete_state == "resumable_failure"
        and incomplete_reason == "evaluation_evidence_incomplete",
        evidence={
            "state": incomplete_state,
            "reason": incomplete_reason,
            "eval_statuses": [row["status"] for row in incomplete.supervisor.store.evals()],
        },
    )

    success_state, success_reason = _terminal_outcome(
        cancel_requested=False,
        failure=None,
        evaluation_required=False,
        promotion=None,
        early_stop=success_receipt,
    )
    recorder.require(
        "training-only-target-stop-succeeds-attempt",
        success_state == "succeeded" and success_reason == "early_stop_success:target_reached",
        evidence={"state": success_state, "reason": success_reason},
    )

    promotion = PromotionReceipt(
        run_id=manifest.run_id,
        checkpoint_id="checkpoint-accepted",
        checkpoint_step=5,
        eval_idempotency_key="3" * 64,
        eval_result_sha256="4" * 64,
        accepted_episode_count=100,
        promoted_at=fixture.clock.utc_now(),
    )
    promoted_state, promoted_reason = _terminal_outcome(
        cancel_requested=False,
        failure=None,
        evaluation_required=True,
        promotion=promotion,
        early_stop=success_receipt,
    )
    recorder.require(
        "evaluation-promotion-overrides-training-target-stop",
        promoted_state == "succeeded" and promoted_reason == "completed_after_eval_acceptance",
        evidence={"state": promoted_state, "reason": promoted_reason},
    )

    tampered = failure_receipt.to_dict()
    tampered["decision_sha256"] = "corrupted"
    corruption_rejected = False
    try:
        EarlyStopReceipt(**tampered).validate()
    except ValueError:
        corruption_rejected = True
    recorder.require(
        "early-stop-receipt-corruption-rejected",
        corruption_rejected,
        evidence={"decision_sha256": tampered["decision_sha256"]},
    )
    return {
        "invariants": recorder.invariants,
        "evidence": {
            "failure_condition_id": failure_receipt.condition_id,
            "success_condition_id": success_receipt.condition_id,
            "metric_step": failure_receipt.metric_step,
        },
    }


SCENARIOS: dict[str, Callable[[Path], dict[str, Any]]] = {
    "full-lifecycle": _scenario_full_lifecycle,
    "parallel-run-isolation": _scenario_parallel_run_isolation,
    "same-run-lease-fencing": _scenario_same_run_lease_fencing,
    "wandb-retry-deduplication": _scenario_wandb_retry_deduplication,
    "wandb-visibility-gating": _scenario_wandb_visibility_gating,
    "checkpoint-upload-retry": _scenario_checkpoint_upload_retry,
    "eval-result-reconciliation": _scenario_eval_result_reconciliation,
    "modal-ambiguous-submit": _scenario_modal_ambiguous_submit,
    "cancellation-terminalization": _scenario_cancellation_terminalization,
    "scratch-preservation-stop": _scenario_scratch_preservation_stop,
    "drain-only-recovery": _scenario_drain_only_recovery,
    "terminal-receipt-gating": _scenario_terminal_receipt_gating,
    "verifier-tamper-detection": _scenario_verifier_tamper_detection,
    "early-stop-outcomes": _scenario_early_stop_outcomes,
}


def _write_evidence(root: Path, prepared: PreparedSupervisor) -> None:
    (root / "evidence").mkdir(parents=True, exist_ok=True)
    (root / "evidence" / "transcript.json").write_bytes(
        _canonical_bytes({"events": prepared.observer.events})
    )
    (root / "evidence" / "wandb-events.json").write_bytes(
        _canonical_bytes({"events": prepared.runtime.wandb_events})
    )


@contextmanager
def _network_disabled() -> Iterator[None]:
    def denied(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("Tier 1 certification attempted network access")

    with (
        patch.object(socket.socket, "connect", denied),
        patch.object(socket, "create_connection", denied),
    ):
        yield


def run_simulated_certification(
    *,
    scenarios: Sequence[str] | None = None,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    selected = tuple(scenarios or DEFAULT_SCENARIOS)
    unknown = sorted(set(selected) - set(SCENARIOS))
    if unknown:
        raise ValueError(f"unknown Tier 1 scenario(s): {', '.join(unknown)}")
    owned_temporary: tempfile.TemporaryDirectory[str] | None = None
    if artifact_root is None:
        owned_temporary = tempfile.TemporaryDirectory(prefix="gradlab-tier1-")
        root = Path(owned_temporary.name)
    else:
        root = artifact_root.resolve()
        root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    try:
        with _network_disabled():
            for name in selected:
                scenario_root = root / name
                scenario_root.mkdir(parents=True, exist_ok=True)
                scenario_output = io.StringIO()
                try:
                    with redirect_stdout(scenario_output):
                        result = SCENARIOS[name](scenario_root)
                except BaseException as exc:
                    (scenario_root / "failure.txt").write_text(
                        traceback.format_exc(),
                        encoding="utf-8",
                    )
                    result = {
                        "invariants": [],
                        "evidence": {},
                        "failure": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        },
                    }
                (scenario_root / "scenario.log").write_text(
                    scenario_output.getvalue(),
                    encoding="utf-8",
                )
                results.append(
                    {
                        "name": name,
                        "status": "failed" if result.get("failure") else "passed",
                        **result,
                    }
                )
        report: dict[str, Any] = {
            "schema_version": CERTIFICATION_SCHEMA_VERSION,
            "tier": "simulated",
            "network_access": "denied",
            "credential_requirement": "none",
            "status": ("passed" if all(row["status"] == "passed" for row in results) else "failed"),
            "scenarios": results,
        }
        report["report_sha256"] = _sha256(report)
        (root / "report.json").write_bytes(_canonical_bytes(report))
        (root / "replay.json").write_bytes(
            _canonical_bytes(
                {
                    "schema_version": CERTIFICATION_SCHEMA_VERSION,
                    "tier": "simulated",
                    "scenarios": list(selected),
                    "expected_report_sha256": report["report_sha256"],
                }
            )
        )
        return report
    finally:
        if owned_temporary is not None:
            owned_temporary.cleanup()


def replay_simulated_certification(
    replay_path: Path,
    *,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    document = json.loads(replay_path.read_text(encoding="utf-8"))
    if int(document.get("schema_version") or 0) != CERTIFICATION_SCHEMA_VERSION:
        raise ValueError("unsupported certification replay schema")
    if str(document.get("tier") or "") != "simulated":
        raise ValueError("replay is not a simulated Tier 1 certification")
    scenarios = document.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("replay must contain at least one scenario")
    return run_simulated_certification(
        scenarios=[str(name) for name in scenarios],
        artifact_root=artifact_root,
    )


def preserve_failure_bundle(
    source: Path,
    destination: Path,
) -> Path:
    if destination.exists():
        raise FileExistsError(f"failure bundle destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    return destination
