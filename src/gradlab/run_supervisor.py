from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any

from gradlab.checkpoint_acceptance import manifest_index
from gradlab.dstack_backend import DSTACK_VERSION
from gradlab.early_stop import validate_metric_early_stop_decision
from gradlab.env import resolve_env_config
from gradlab.env_config import env_config_from_mapping
from gradlab.eval_metrics import eval_by_start_rows
from gradlab.eval_backend import EvalBackend, EvalHandle
from gradlab.file_utils import file_sha256
from gradlab.goal_variants import (
    build_goal_variant_descriptor,
    validate_goal_variant_descriptor,
)
from gradlab.metric_names import (
    EVAL_ACCEPTANCE_DURATION_SECONDS,
    EVAL_ACCEPTANCE_EPISODES_COMPLETED,
    EVAL_ACCEPTANCE_EPISODES_PLANNED,
    EVAL_ACCEPTANCE_FAILURE_COUNT,
    EVAL_ACCEPTANCE_PASS,
    METRICS_SCHEMA_VERSION,
    ORCHESTRATION_CHECKPOINT_BACKLOG,
    ORCHESTRATION_IDLE_GPU_TAIL_SECONDS,
    ORCHESTRATION_INGRESS_RATE,
    ORCHESTRATION_LOCAL_HIGH_WATER,
    ORCHESTRATION_OLDEST_UNPUBLISHED_SECONDS,
    ORCHESTRATION_PENDING_EVALS,
    ORCHESTRATION_PUBLICATION_CAPACITY_RATIO,
    ORCHESTRATION_PUBLISH_RATE,
    ORCHESTRATION_QUEUE_DEPTH,
    ORCHESTRATION_R2_HIGH_WATER,
    ORCHESTRATION_RESULT_TO_STOP_SECONDS,
    ORCHESTRATION_SCRATCH_USED_FRACTION,
    ORCHESTRATION_WANDB_HIGH_WATER,
    ORCHESTRATION_WANDB_REMOTE_HIGH_WATER,
    ORCHESTRATION_WANDB_REMOTE_VISIBLE_LAG_SECONDS,
    metric_definition,
)
from gradlab.metric_store import metric_store_path
from gradlab.model_sources import download_public_checkpoint_manifest_source
from gradlab.modal_eval_backend import ModalEvalBackend
from gradlab.modal_eval_config import load_modal_eval_config
from gradlab.modal_eval_protocol import (
    PROTOCOL_SCHEMA_VERSION,
    execution_key,
    validate_attempt_result,
)
from gradlab.policy_bundle import (
    build_recipe_document,
    canonical_json_sha256,
    evaluation_contract,
    evaluation_contract_sha256,
    write_canonical_json,
)
from gradlab.r2_store import ConditionalWriteConflict, RunStorageConfig
from gradlab.recipe_documents import (
    compose_resolved_train_documents,
    load_goal_contract,
    prepare_checkpoint_eval_mode,
    recipe_tags,
)
from gradlab.recipe_variants import recipe_variant_id
from gradlab.rom_assets import (
    CONTAINER_ROM_CACHE,
    cache_path,
    install_rom_file,
    validate_rom_asset_manifest,
    verify_rom_file,
)
from gradlab.run_authority import (
    LEASE_MISSES_BEFORE_STOP,
    LEASE_RENEW_SECONDS,
    Lease,
    LeaseUnavailable,
    RunAuthority,
)
from gradlab.run_contracts import (
    CheckpointManifest,
    EarlyStopReceipt,
    EvalIntent,
    EvalResult,
    PromotionReceipt,
    RunManifest,
    TerminalReceipt,
    document_sha256,
    eval_idempotency_key,
)
from gradlab.supervisor_ledger import SupervisorLedger
from gradlab.supervisor_runtime import (
    LearnerProcess,
    LifecycleObserver,
    NullLifecycleObserver,
    SupervisorRuntime,
)
from gradlab.train_config import (
    load_materialized_train_config,
    validate_and_normalize_train_config,
)
from gradlab.training_lifecycle import (
    LEARNER_READY_FILENAME,
    LEARNER_STATE_FORMAT_VERSION,
    TRAINING_RESULT_FILENAME,
    TerminalReason,
    TrainingExecutionMode,
)
from gradlab.trusted_inputs import stage_model_input
from gradlab.wandb_publisher import WandbProjector


METRIC_SEGMENT_SECONDS = 5.0
METRIC_SEGMENT_EVENTS = 1_000
EVAL_POLL_SECONDS = 2.0
WANDB_WARNING_SECONDS = 45.0
WANDB_UNHEALTHY_SECONDS = 60.0
WANDB_DRAIN_TIMEOUT_SECONDS = 300.0
SCRATCH_STOP_FRACTION = 0.80
METRIC_JOURNAL_RETENTION_DAYS = 7
HEALTH_SAMPLE_SECONDS = 15.0
WANDB_REMOTE_PROBE_SECONDS = 30.0
WANDB_DRAIN_REMOTE_PROBE_SECONDS = 2.0


class IncompleteEvaluationEvidence(RuntimeError):
    """Evaluation did not produce enough valid evidence for scientific rejection."""


class LearnerOperationalFailure(RuntimeError):
    stop_reason = "learner_failure"


class LearnerFailure(LearnerOperationalFailure):
    """The learner emitted an authoritative failed terminal result."""


class LearnerStartupTimeout(LearnerOperationalFailure):
    """The learner did not emit readiness or a terminal result before its deadline."""

    stop_reason = "startup_timeout"


class LearnerStateContractError(LearnerOperationalFailure):
    """The learner emitted malformed, stale, or identity-mismatched state."""

    stop_reason = "invalid_result"


class LearnerExitContractMismatch(LearnerOperationalFailure):
    """The learner process exit disagreed with its terminal document."""

    stop_reason = "exit_contract_mismatch"


class LearnerTeardownTimeout(LearnerOperationalFailure):
    """The learner process group remained alive after bounded escalation."""

    stop_reason = "teardown_timeout"


@dataclass(frozen=True)
class LearnerState:
    kind: str
    status: str
    terminal_reason: str | None
    final_step: int | None
    document: Mapping[str, Any]


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _timestamp(unix_seconds: float) -> str:
    return datetime.fromtimestamp(unix_seconds, UTC).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _bounded_exception_document(failure: BaseException) -> dict[str, str]:
    message = str(failure).replace("\x00", "\N{REPLACEMENT CHARACTER}")
    message = re.sub(
        r"(?i)(api[_-]?key|access[_-]?key|secret|token|password)(\\s*[:=]\\s*)\\S+",
        r"\1\2<redacted>",
        message,
    )
    return {
        "type": type(failure).__name__[:200],
        "message": message[:4_000],
    }


def _manifest_from_document(value: Mapping[str, Any]) -> RunManifest:
    manifest = RunManifest(**dict(value))
    manifest.validate()
    return manifest


def _bind_evaluation_contract(
    config: dict[str, Any],
    *,
    recipe_document: Mapping[str, Any],
    evaluation_required: bool,
) -> dict[str, Any]:
    if evaluation_required:
        contract = evaluation_contract(recipe_document)
        config["checkpoint_eval_contract"] = contract
        return contract
    config.pop("checkpoint_eval_contract", None)
    return {}


def _summary_scalar(value: Any) -> Any:
    getter = getattr(value, "get", None)
    if callable(getter):
        for key in ("max", "last"):
            nested = getter(key)
            if nested is not None:
                return nested
    return value


def _terminal_outcome(
    *,
    cancel_requested: bool,
    failure: BaseException | None,
    evaluation_required: bool,
    promotion: PromotionReceipt | None,
    early_stop: EarlyStopReceipt | None,
) -> tuple[str, str]:
    if isinstance(failure, IncompleteEvaluationEvidence):
        return "resumable_failure", "evaluation_evidence_incomplete"
    if isinstance(failure, LearnerOperationalFailure):
        return "resumable_failure", failure.stop_reason
    if failure is not None:
        return "resumable_failure", "supervisor_failure"
    if cancel_requested:
        return "canceled", "canceled"
    if evaluation_required and promotion is not None:
        return "succeeded", "completed_after_eval_acceptance"
    if early_stop is not None:
        if early_stop.outcome == "success":
            if evaluation_required:
                return (
                    "failed",
                    f"early_stop_success_without_acceptance:{early_stop.condition_id}",
                )
            return "succeeded", f"early_stop_success:{early_stop.condition_id}"
        return "failed", f"early_stop_failure:{early_stop.condition_id}"
    if evaluation_required:
        return "failed", "training_cap_without_acceptance"
    return "succeeded", "training_cap_complete"


class RunSupervisor:
    """Own all network-side effects for one learner container."""

    def __init__(
        self,
        *,
        manifest_uri: str,
        storage: RunStorageConfig | None = None,
        eval_backend: EvalBackend | None = None,
        repo_root: Path | None = None,
        work_root: Path | None = None,
        runtime: SupervisorRuntime | None = None,
        authority: RunAuthority | None = None,
        observer: LifecycleObserver | None = None,
    ):
        self.runtime = runtime or SupervisorRuntime()
        self.clock = self.runtime.clock
        self.observer = observer or NullLifecycleObserver()
        self.storage = storage or RunStorageConfig.from_env()
        self.authority = authority or RunAuthority(self.storage, clock=self.clock)
        manifest_key = self.authority.control.key_from_uri(manifest_uri)
        self.manifest = _manifest_from_document(self.authority.control.get_json(manifest_key))
        accepted_manifest_keys = {
            f"runs/{self.manifest.run_id}/manifest.json",
            f"runs/{self.manifest.run_id}/attempts/{self.manifest.attempt_id}/manifest.json",
        }
        if manifest_key not in accepted_manifest_keys:
            raise ValueError(
                f"run manifest is not at its canonical run or attempt key: {manifest_key}"
            )
        self.repo_root = (
            repo_root
            or Path(os.environ.get("GRADLAB_PROJECT_ROOT") or Path(__file__).resolve().parents[2])
        ).resolve()
        self.work_root = (
            work_root or Path(os.environ.get("GRADLAB_RUN_WORK_ROOT") or "/workspace")
        ).resolve()
        self.output_root = self.work_root / "gradlab" / self.manifest.run_id
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.run_dir = self.output_root / "runs" / self.manifest.run_id
        self.config_path = self.output_root / "train-config.json"
        self.recipe_path = self.output_root / "recipe.json"
        self.learner_log_path = self.output_root / "learner.log"
        self.cancel_requested = False
        self.lease_lost = False
        self.stop_reason = ""
        self.learner: LearnerProcess | None = None
        self.projector: WandbProjector | None = None
        self.wandb_run_path = ""
        self.lease: Lease | None = None
        self.last_lease_renewal = 0.0
        self.lease_misses = 0
        self.last_segment = 0.0
        self.last_eval_poll = 0.0
        self.last_health_sample = 0.0
        self.last_health_local_high_water = 0
        self.last_health_wandb_high_water = 0
        self.peak_ingress_rate = 0.0
        self.peak_publish_rate = 0.0
        self.peak_publish_capacity = 0.0
        self.last_remote_probe = 0.0
        self.wandb_remote_high_water = 0
        self.wandb_remote_visible_lag_seconds = 0.0
        self.learner_started_at: float | None = None
        self.expected_learner_pid: int | None = None
        self.learner_result_observed_at: float | None = None
        self.learner_final_step: int | None = None
        self.learner_terminal_document: dict[str, Any] | None = None
        self.learner_teardown_evidence: dict[str, Any] = {}
        self.accepted_observed_at: float | None = None
        self.eval_admission_closed = False
        self.recovered_early_stop: EarlyStopReceipt | None = None
        self.state_archive_publication: dict[str, Any] | None = None
        self.state_archive_closure_sha256 = ""
        self.recovery_mode = str(self.manifest.compute.get("recovery_mode") or "resume-training")
        if self.recovery_mode not in {"resume-training", "drain-only"}:
            raise ValueError(f"unsupported recovery mode: {self.recovery_mode}")
        self.evaluation_required = bool(self.manifest.modal["enabled"])
        self.eval_backend = eval_backend
        if self.eval_backend is None and self.evaluation_required:
            self.eval_backend = ModalEvalBackend(
                app_name=str(self.manifest.modal["app_name"]),
                function_name=str(
                    self.manifest.modal.get("function_name") or "evaluate_checkpoint"
                ),
                environment_name=str(self.manifest.modal.get("environment_name") or "gradlab-eval"),
            )
        self.store = SupervisorLedger(metric_store_path(self.run_dir), clock=self.clock)
        self.train_config: dict[str, Any] = {}
        self.recipe_document: dict[str, Any] = {}
        self.eval_contract: dict[str, Any] = {}
        self.modal_config = load_modal_eval_config(
            self.repo_root / "experiments" / "modal_eval.yaml"
        )

    def _emit(self, kind: str, **payload: Any) -> None:
        self.observer.emit(
            kind,
            {
                "run_id": self.manifest.run_id,
                "attempt_id": self.manifest.attempt_id,
                "supervision_liveness": dict(self.manifest.liveness or {}),
                "at": self.clock.utc_now(),
                **payload,
            },
        )

    def _authoritative_early_stop_receipt(
        self,
        *,
        attempt_id: str,
    ) -> EarlyStopReceipt | None:
        existing_document = self.authority.early_stop_receipt(
            run_id=self.manifest.run_id,
            attempt_id=attempt_id,
        )
        if existing_document is None:
            return None
        existing = EarlyStopReceipt(**existing_document)
        existing.validate()
        early_stop_config = self.train_config.get("early_stop")
        if not isinstance(early_stop_config, Mapping):
            raise ValueError(
                "authoritative early-stop receipt exists without configured conditions"
            )
        authoritative_decision = {
            "schema_version": 1,
            "kind": "metric_early_stop",
            "condition_id": existing.condition_id,
            "matched_condition_ids": list(existing.matched_condition_ids),
            "outcome": existing.outcome,
            "action": "stop",
            "trigger": existing.trigger,
            "metric": existing.metric,
            "metric_step": existing.metric_step,
            "value": existing.value,
            "best_value": existing.best_value,
            "elapsed_steps": existing.elapsed_steps,
            "patience_progress": existing.patience_progress,
            "condition": dict(existing.condition),
            "early_stop_config_sha256": existing.early_stop_config_sha256,
        }
        validated_authoritative_decision = validate_metric_early_stop_decision(
            authoritative_decision,
            early_stop_config,
            label="authoritative early-stop receipt",
        )
        if existing.decision_sha256 != canonical_json_sha256(validated_authoritative_decision):
            raise ValueError("authoritative early-stop receipt decision hash does not match")
        return existing

    def _prior_early_stop_receipt(self) -> EarlyStopReceipt | None:
        prefix = f"runs/{self.manifest.run_id}/attempts"
        manifests = [
            self.authority.control.get_json(key)
            for key in self.authority.control.iter_keys(prefix)
            if key.endswith("/manifest.json")
        ]
        manifests.sort(
            key=lambda row: (
                str(row.get("created_at") or ""),
                str(row.get("attempt_id") or ""),
            )
        )
        current_index = next(
            (
                index
                for index, row in enumerate(manifests)
                if str(row.get("attempt_id") or "") == self.manifest.attempt_id
            ),
            None,
        )
        if current_index is None:
            raise ValueError(f"current attempt manifest is missing: {self.manifest.attempt_id}")
        for row in reversed(manifests[:current_index]):
            receipt = self._authoritative_early_stop_receipt(attempt_id=str(row["attempt_id"]))
            if receipt is not None:
                return receipt
        return None

    def _resolve_early_stop_receipt(self) -> EarlyStopReceipt | None:
        existing = self._authoritative_early_stop_receipt(attempt_id=self.manifest.attempt_id)

        decision_path = self.run_dir / f"early_stop_decision-{self.manifest.attempt_id}.json"
        if not decision_path.is_file():
            return existing or self.recovered_early_stop or self._prior_early_stop_receipt()
        raw = json.loads(decision_path.read_text(encoding="utf-8"))
        early_stop_config = self.train_config.get("early_stop")
        if early_stop_config is None:
            raise ValueError("learner wrote an early-stop decision without configured conditions")
        decision = validate_metric_early_stop_decision(
            raw,
            early_stop_config,
            label=f"learner early-stop decision {decision_path}",
        )
        decision_sha256 = canonical_json_sha256(decision)
        if existing is not None:
            if existing.decision_sha256 != decision_sha256:
                raise ValueError(
                    "local early-stop decision conflicts with the authoritative receipt"
                )
            return existing

        receipt = EarlyStopReceipt(
            run_id=self.manifest.run_id,
            attempt_id=self.manifest.attempt_id,
            condition_id=str(decision["condition_id"]),
            matched_condition_ids=tuple(str(item) for item in decision["matched_condition_ids"]),
            outcome=str(decision["outcome"]),  # type: ignore[arg-type]
            trigger=str(decision["trigger"]),  # type: ignore[arg-type]
            metric=str(decision["metric"]),
            metric_step=int(decision["metric_step"]),
            value=float(decision["value"]),
            best_value=float(decision["best_value"]),
            elapsed_steps=int(decision["elapsed_steps"]),
            patience_progress=float(decision["patience_progress"]),
            condition=dict(decision["condition"]),
            early_stop_config_sha256=str(decision["early_stop_config_sha256"]),
            decision_sha256=decision_sha256,
            recorded_at=self.clock.utc_now(),
        )
        self.authority.create_early_stop(receipt)
        self._emit(
            "early_stop_receipt_created",
            condition_id=receipt.condition_id,
            outcome=receipt.outcome,
            metric_step=receipt.metric_step,
            decision_sha256=receipt.decision_sha256,
        )
        return receipt

    def _liveness_seconds(self, key: str) -> float:
        value = (self.manifest.liveness or {}).get(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise RuntimeError(f"run manifest has no valid liveness.{key}")
        return float(value)

    def _parse_learner_state(
        self,
        document: Mapping[str, Any],
        *,
        kind: str,
        live: bool,
    ) -> LearnerState:
        expected_type = (
            "gradlab.learner-ready" if kind == "ready" else "gradlab.training-result"
        )
        if document.get("document_type") != expected_type:
            raise LearnerStateContractError(
                f"learner {kind} document_type is not {expected_type!r}"
            )
        version = document.get("format_version")
        if version == 2 and kind == "result" and not live:
            if document.get("execution_mode") != TrainingExecutionMode.SUPERVISED.value:
                raise LearnerStateContractError("legacy learner result is not supervised")
            try:
                reason = TerminalReason(str(document.get("terminal_reason") or ""))
            except ValueError as exc:
                raise LearnerStateContractError(
                    "legacy learner result has an invalid terminal_reason"
                ) from exc
            final_step = document.get("final_step")
            if (
                isinstance(final_step, bool)
                or not isinstance(final_step, int)
                or final_step < 0
            ):
                raise LearnerStateContractError(
                    "legacy learner result has an invalid final_step"
                )
            return LearnerState(
                kind="result",
                status=str(document.get("status") or ""),
                terminal_reason=reason.value,
                final_step=final_step,
                document=dict(document),
            )
        if version != LEARNER_STATE_FORMAT_VERSION:
            raise LearnerStateContractError(
                f"learner {kind} document has unsupported format_version {version!r}"
            )
        if document.get("run_id") != self.manifest.run_id:
            raise LearnerStateContractError(f"learner {kind} run_id does not match manifest")
        if document.get("attempt_id") != self.manifest.attempt_id:
            raise LearnerStateContractError(
                f"learner {kind} attempt_id does not match manifest"
            )
        learner_pid = document.get("learner_pid")
        if (
            isinstance(learner_pid, bool)
            or not isinstance(learner_pid, int)
            or learner_pid <= 0
        ):
            raise LearnerStateContractError(f"learner {kind} has an invalid learner_pid")
        if live and learner_pid != self.expected_learner_pid:
            raise LearnerStateContractError(
                f"learner {kind} pid {learner_pid} does not match "
                f"spawned pid {self.expected_learner_pid}"
            )
        if document.get("execution_mode") != TrainingExecutionMode.SUPERVISED.value:
            raise LearnerStateContractError(f"learner {kind} is not supervised")
        expected_backend = str(
            dict(self.train_config.get("training_backend") or {}).get("id") or ""
        )
        if not expected_backend or document.get("training_backend_id") != expected_backend:
            raise LearnerStateContractError(
                f"learner {kind} training_backend_id does not match materialized config"
            )
        timestamp_field = "ready_at" if kind == "ready" else "terminal_at"
        timestamp = document.get(timestamp_field)
        if not isinstance(timestamp, str):
            raise LearnerStateContractError(
                f"learner {kind} has no valid {timestamp_field}"
            )
        try:
            _parse_timestamp(timestamp)
        except (TypeError, ValueError) as exc:
            raise LearnerStateContractError(
                f"learner {kind} has an invalid {timestamp_field}"
            ) from exc
        if kind == "ready":
            if document.get("status") != "ready":
                raise LearnerStateContractError("learner readiness status is not 'ready'")
            return LearnerState(
                kind="ready",
                status="ready",
                terminal_reason=None,
                final_step=None,
                document=dict(document),
            )

        status = str(document.get("status") or "")
        if status not in {"completed", "interrupted", "failed"}:
            raise LearnerStateContractError("learner result has an invalid status")
        try:
            reason = TerminalReason(str(document.get("terminal_reason") or ""))
        except ValueError as exc:
            raise LearnerStateContractError(
                "learner result has an invalid terminal_reason"
            ) from exc
        if (status == "failed") != (reason == TerminalReason.FAILED):
            raise LearnerStateContractError(
                "learner result status and terminal_reason disagree"
            )
        interruption_reasons = {
            TerminalReason.LOCAL_INTERRUPTION,
            TerminalReason.EXTERNAL_SIGNAL,
        }
        if (status == "interrupted") != (reason in interruption_reasons):
            raise LearnerStateContractError(
                "learner result interrupted status and terminal_reason disagree"
            )
        final_step = document.get("final_step")
        if (
            isinstance(final_step, bool)
            or not isinstance(final_step, int)
            or final_step < 0
        ):
            raise LearnerStateContractError("learner result has an invalid final_step")
        if not isinstance(document.get("execution_policy"), Mapping):
            raise LearnerStateContractError("learner result has no execution_policy")
        if status == "failed":
            error_type = document.get("error_type")
            error_message = document.get("error_message")
            if (
                not isinstance(error_type, str)
                or not error_type
                or len(error_type) > 200
                or not isinstance(error_message, str)
                or len(error_message) > 2_000
            ):
                raise LearnerStateContractError(
                    "failed learner result has invalid bounded error evidence"
                )
        return LearnerState(
            kind="result",
            status=status,
            terminal_reason=reason.value,
            final_step=final_step,
            document=dict(document),
        )

    def _learner_state_file(
        self,
        *,
        kind: str,
        live: bool,
    ) -> LearnerState | None:
        filename = LEARNER_READY_FILENAME if kind == "ready" else TRAINING_RESULT_FILENAME
        path = self.run_dir / filename
        if not path.is_file():
            return None
        try:
            document = _read_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise LearnerStateContractError(
                f"learner {kind} document is unreadable"
            ) from exc
        return self._parse_learner_state(document, kind=kind, live=live)

    def _training_terminal_reason(self) -> str:
        state = self._learner_state_file(kind="result", live=False)
        return "" if state is None else str(state.terminal_reason or "")

    def _observe_live_learner_state(self, now: float) -> LearnerState | None:
        result = self._learner_state_file(kind="result", live=True)
        if result is not None:
            self.learner_final_step = result.final_step
            self.learner_terminal_document = dict(result.document)
            if self.learner_result_observed_at is None:
                self.learner_result_observed_at = now
                self._emit(
                    "learner_terminal_result_observed",
                    status=result.status,
                    terminal_reason=result.terminal_reason,
                    final_step=result.final_step,
                )
            if result.status == "failed":
                error_type = str(result.document.get("error_type") or "LearnerError")
                error_message = str(result.document.get("error_message") or "")
                raise LearnerFailure(f"{error_type}: {error_message}")
            elapsed = max(0.0, now - self.learner_result_observed_at)
            grace = self._liveness_seconds("result_exit_grace_seconds")
            if elapsed >= grace:
                raise LearnerTeardownTimeout(
                    "learner wrote terminal result "
                    f"{result.terminal_reason!r} but remained alive for {elapsed:.1f}s"
                )
            return result

        ready = self._learner_state_file(kind="ready", live=True)
        if ready is not None:
            return ready
        if self.learner_started_at is None:
            raise LearnerStateContractError("learner startup time was not recorded")
        elapsed = max(0.0, now - self.learner_started_at)
        timeout = self._liveness_seconds("startup_timeout_seconds")
        if elapsed >= timeout:
            raise LearnerStartupTimeout(
                f"learner emitted no readiness or terminal result within {timeout:.1f}s"
            )
        return None

    def _validate_learner_exit(self, returncode: int) -> LearnerState:
        terminal_state = self._learner_state_file(kind="result", live=True)
        if terminal_state is None:
            raise LearnerStateContractError(
                f"learner exited with code {returncode} without a terminal result"
            )
        self.learner_final_step = terminal_state.final_step
        self.learner_terminal_document = dict(terminal_state.document)
        if terminal_state.status == "failed":
            raise LearnerFailure(
                f"{terminal_state.document.get('error_type')}: "
                f"{terminal_state.document.get('error_message')}"
            )
        if returncode != 0:
            raise LearnerExitContractMismatch(
                f"learner emitted {terminal_state.status!r} but exited with code {returncode}"
            )
        return terminal_state

    @staticmethod
    def _checkpoint_manifest_url(checkpoint: Mapping[str, Any]) -> str:
        model_url = str(checkpoint.get("public_url") or "")
        if not model_url.endswith("/model.zip"):
            raise ValueError("public checkpoint URL is malformed")
        return f"{model_url.removesuffix('/model.zip')}/manifest.json"

    def _configure_resume(self, config: dict[str, Any]) -> None:
        backend = config.get("training_backend")
        if isinstance(backend, Mapping) and backend.get("id") == "gradlab.go-explore":
            return
        index = self.authority.models.get_json_optional(f"runs/{self.manifest.run_id}/index.json")
        checkpoints = [
            dict(row) for row in (index or {}).get("checkpoints") or [] if isinstance(row, Mapping)
        ]
        if not checkpoints:
            return
        if self.recovery_mode == "drain-only" and any(
            str(row.get("purpose") or "") == "final" for row in checkpoints
        ):
            return
        checkpoint = max(
            checkpoints,
            key=lambda row: (int(row["step"]), str(row["sha256"])),
        )
        manifest_url = self._checkpoint_manifest_url(checkpoint)
        resolved = download_public_checkpoint_manifest_source(
            manifest_url,
            root=self.output_root / ".resume-source",
        )
        staged = stage_model_input(
            resolved.model_path,
            source_identity=manifest_url,
        )
        try:
            backend = copy.deepcopy(dict(config["training_backend"]))
            backend_config = copy.deepcopy(dict(backend["config"]))
            backend_config["resume"] = manifest_url
            backend_config["resume_approval_hash"] = staged.manifest_hash
            backend_config["resume_manifest"] = [entry.as_dict() for entry in staged.manifest]
            backend["config"] = backend_config
            config["training_backend"] = backend
        finally:
            staged.cleanup()

    def validate_runtime(self) -> None:
        runtime = self.runtime.runtime_contract(runtime_image_ref=self.manifest.image_digest)
        observed_source = str(runtime.get("runtime_build_source_sha") or "")
        expected_source = str(self.manifest.compute.get("runtime_build_source_sha") or "")
        if observed_source != expected_source:
            raise RuntimeError(
                "runtime build source SHA does not match the immutable run manifest: "
                f"{observed_source or 'missing'} != {expected_source or 'missing'}"
            )
        observed_input = str(runtime.get("runtime_input_sha256") or "")
        expected_input = str(self.manifest.compute.get("runtime_input_sha256") or "")
        if observed_input != expected_input:
            raise RuntimeError(
                "runtime input SHA-256 does not match the immutable run manifest: "
                f"{observed_input or 'missing'} != {expected_input or 'missing'}"
            )
        if str(os.environ.get("GRADLAB_ORCHESTRATOR") or "") != "dstack":
            raise RuntimeError("run supervisor may execute only inside a dstack task")

    def materialize(self) -> None:
        goal_path = (
            self.repo_root / "experiments" / "goals" / self.manifest.goal_slug / "_goal.yaml"
        )
        recipe_path = goal_path.parent / "recipes" / f"{self.manifest.recipe_slug}.yaml"
        resolved_documents = compose_resolved_train_documents(
            goal_path,
            recipe_path,
            recipe_overrides=self.manifest.recipe_overrides,
            prepare_materialized=partial(
                prepare_checkpoint_eval_mode,
                checkpoint_eval_backend=("modal" if self.evaluation_required else "none"),
            ),
            source_sha=self.manifest.source_sha,
        )
        materialized = resolved_documents.effective
        base_materialized = resolved_documents.base
        materialized_goal_hash = str(materialized["train_config"]["effective_goal_contract_sha256"])
        if materialized_goal_hash != self.manifest.goal_sha256:
            raise RuntimeError("effective goal hash does not match the run manifest")
        materialized_environment_hash = str(
            materialized.get("environment_hash") or ""
        ).removeprefix("sha256:")
        if materialized_environment_hash != self.manifest.environment_sha256:
            raise RuntimeError("materialized environment hash does not match the run manifest")
        goal_variant = None
        if self.manifest.goal_variant is not None:
            goal_variant = build_goal_variant_descriptor(
                goal_slug=self.manifest.goal_slug,
                source_sha=self.manifest.source_sha,
                authored_goal=load_goal_contract(goal_path, self.repo_root),
                effective_goal=dict(materialized["goal"]),
            )
            if goal_variant != validate_goal_variant_descriptor(self.manifest.goal_variant):
                raise RuntimeError("materialized goal variant does not match the run manifest")
            materialized["goal_variant"] = goal_variant
            self.authority.register_goal_variant_best_effort(self.manifest)

        config = dict(materialized["train_config"])
        asset = self.manifest.modal.get("rom_asset_manifest")
        if isinstance(asset, Mapping):
            normalized_asset = validate_rom_asset_manifest(
                asset,
                expected_game=str(config["game"]),
            )
            cached_rom = cache_path(CONTAINER_ROM_CACHE, normalized_asset)
            try:
                verify_rom_file(cached_rom, normalized_asset)
            except FileNotFoundError, ValueError:
                if str(os.environ.get("GRADLAB_ROM_CACHE_READ_ONLY") or "") == "1":
                    raise
                object_key = self.authority.evaluation.key_from_uri(
                    str(normalized_asset["object_uri"])
                )
                with tempfile.TemporaryDirectory(
                    prefix="gradlab-rom-",
                    dir=self.output_root,
                ) as temporary:
                    staged = Path(temporary) / str(normalized_asset["filename"])
                    staged.write_bytes(self.authority.evaluation.get_bytes(object_key))
                    install_rom_file(staged, normalized_asset, CONTAINER_ROM_CACHE)
            config["rom_asset_manifest"] = normalized_asset
            base_config = dict(base_materialized["train_config"])
            base_config["rom_asset_manifest"] = normalized_asset
            base_materialized["train_config"] = base_config
        materialized["train_config"] = config
        self.recipe_document = build_recipe_document(
            materialized,
            repo_root=self.repo_root,
            source_commit=self.manifest.source_sha,
            run_description=self.manifest.run_description,
            seed=self.manifest.seed,
            runtime_image_ref=self.manifest.image_digest,
            base_materialized_recipe=base_materialized,
            canonical_goal=resolved_documents.canonical_goal,
        )
        if canonical_json_sha256(self.recipe_document) != self.manifest.recipe_sha256:
            raise RuntimeError("portable recipe hash does not match the run manifest")
        variant_id = recipe_variant_id(
            recipe_slug=self.manifest.recipe_slug,
            source_sha=self.manifest.source_sha,
            recipe_overrides=self.manifest.recipe_overrides,
        )
        config.update(
            {
                "seed": int(self.manifest.seed),
                "run_name": self.manifest.run_id,
                "run_description": self.manifest.run_description,
                "runs_dir": str(self.output_root / "runs"),
                "goal_slug": self.manifest.goal_slug,
                "goal_path": str(goal_path),
                "goal_sha256": self.manifest.goal_sha256,
                "recipe_slug": self.manifest.recipe_slug,
                "recipe_path": str(recipe_path),
                "recipe_sha256": self.manifest.recipe_sha256,
                "recipe_overrides": list(self.manifest.recipe_overrides),
                "recipe_variant_id": variant_id,
                "source_sha": self.manifest.source_sha,
                "runtime_build_source_sha": str(self.manifest.compute["runtime_build_source_sha"]),
                "runtime_input_sha256": str(os.environ.get("GRADLAB_RUNTIME_INPUT_SHA256") or ""),
                "runtime_image_ref": self.manifest.image_digest,
                "compute_target": str(
                    dict(
                        self.manifest.compute.get("selected")
                        or self.manifest.compute.get("request")
                        or {}
                    ).get("target")
                    or ""
                ),
                "attempt_id": self.manifest.attempt_id,
                "dstack_task": str(self.manifest.compute.get("dstack_task") or ""),
                "wandb": True,
                "wandb_mode": "online",
                "wandb_run_id": str(self.manifest.wandb.get("run_id") or self.manifest.run_id),
                "wandb_entity": str(self.manifest.wandb.get("entity") or ""),
                "wandb_project": str(self.manifest.wandb.get("project") or ""),
                "wandb_display_name": str(
                    self.manifest.wandb.get("display_name") or self.manifest.run_id
                ),
                "wandb_group": str(self.manifest.wandb.get("group") or ""),
                "wandb_tags": ",".join(
                    [
                        *recipe_tags(materialized),
                        f"gradlab_run_id:{self.manifest.run_id}",
                        f"attempt_id:{self.manifest.attempt_id}",
                        f"recipe_variant:{variant_id}",
                        *(
                            [f"goal_variant:{goal_variant['variant_id']}"]
                            if goal_variant is not None
                            else []
                        ),
                        "orchestrator:dstack",
                    ]
                ),
                "checkpoint_eval_backend": ("modal" if self.evaluation_required else "none"),
                "metrics_schema_version": METRICS_SCHEMA_VERSION,
            }
        )
        materialized["train_config"] = config
        write_canonical_json(self.recipe_path, self.recipe_document)
        self.eval_contract = _bind_evaluation_contract(
            config,
            recipe_document=self.recipe_document,
            evaluation_required=self.evaluation_required,
        )
        config["recipe_json_path"] = str(self.recipe_path)
        config["recipe_composition"] = dict(materialized.get("_composition") or {})
        self._configure_resume(config)
        self.train_config = validate_and_normalize_train_config(
            config,
            label="dstack run train_config",
            required_keys=("training_backend",),
        )
        write_canonical_json(self.config_path, self.train_config)

    def _start_wandb(self) -> None:
        train_config = load_materialized_train_config(self.config_path)
        config = resolve_env_config(env_config_from_mapping(train_config))
        receipt_key = f"runs/{self.manifest.run_id}/wandb.json"
        existing = self.authority.control.get_json_optional(receipt_key)
        if existing is None:
            self.projector = self.runtime.start_wandb(
                train_config,
                run_dir=str(self.run_dir),
                config=config,
                goal_variant=self.manifest.goal_variant,
            )
            run = self.projector.run
            receipt = {
                "schema_version": 1,
                "run_id": self.manifest.run_id,
                "wandb_run_id": str(getattr(run, "id", "") or ""),
                "url": str(getattr(run, "url", "") or ""),
                "created_at": self.clock.utc_now(),
            }
            self.authority.control.put_json(receipt_key, receipt, create_only=True)
        else:
            self.projector = self.runtime.resume_wandb(
                self.train_config,
                allow_create=False,
            )
        path = getattr(self.projector.run, "path", "")
        self.wandb_run_path = (
            "/".join(str(part) for part in path) if isinstance(path, list | tuple) else str(path)
        )
        if not self.wandb_run_path:
            self.wandb_run_path = "/".join(
                (
                    str(self.manifest.wandb["entity"]),
                    str(self.manifest.wandb["project"]),
                    str(self.manifest.wandb["run_id"]),
                )
            )

    def _start_learner(self) -> None:
        self._archive_pre_spawn_learner_state()
        environment = os.environ.copy()
        for name in tuple(environment):
            if (
                name in {"WANDB_API_KEY", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"}
                or name.startswith("GRADLAB_CONTROL_R2_")
                or name.startswith("GRADLAB_EVAL_R2_")
                or name.startswith("GRADLAB_MODELS_R2_")
                or name.startswith("AWS_")
            ):
                environment.pop(name, None)
        environment["GRADLAB_INTERNAL_LEARNER"] = "1"
        environment["GRADLAB_ROM_CACHE_DIR"] = str(CONTAINER_ROM_CACHE)
        fault_fixture = str(
            environment.get("GRADLAB_SUPERVISION_FAULT_FIXTURE") or ""
        ).strip()
        command = (
            [
                sys.executable,
                "-m",
                "gradlab.supervision_fault_learner",
                "--train-config-json",
                str(self.config_path),
                "--mode",
                fault_fixture,
            ]
            if fault_fixture
            else [
                sys.executable,
                "-m",
                "gradlab.train",
                "--train-config-json",
                str(self.config_path),
                "--execution-mode",
                "supervised",
            ]
        )
        self.learner = self.runtime.start_learner(
            command,
            log_path=self.learner_log_path,
            environment=environment,
        )
        learner_pid = getattr(self.learner, "pid", None)
        if isinstance(learner_pid, bool) or not isinstance(learner_pid, int) or learner_pid <= 0:
            raise RuntimeError("spawned learner has no valid pid")
        self.expected_learner_pid = learner_pid
        self.learner_started_at = self.clock.monotonic()
        print(
            f"learner started pid={learner_pid} log={self.learner_log_path}",
            flush=True,
        )

    def _archive_pre_spawn_learner_state(self) -> None:
        for filename in (
            LEARNER_READY_FILENAME,
            "learner_ready.json",
            TRAINING_RESULT_FILENAME,
        ):
            path = self.run_dir / filename
            if not path.exists():
                continue
            suffix = 0
            while True:
                marker = (
                    self.run_dir
                    / f"{filename}.pre-spawn-{self.manifest.attempt_id}-{suffix}.json"
                )
                if not marker.exists():
                    break
                suffix += 1
            path.replace(marker)
            self._emit(
                "pre_spawn_learner_state_archived",
                filename=filename,
                archived_path=str(marker),
            )

    def _learner_log_tail(self, *, max_bytes: int = 12_000) -> str:
        if max_bytes <= 0 or not self.learner_log_path.is_file():
            return ""
        with self.learner_log_path.open("rb") as source:
            source.seek(0, os.SEEK_END)
            size = source.tell()
            source.seek(max(size - max_bytes, 0))
            encoded = source.read(max_bytes)
        return encoded.decode("utf-8", errors="replace").strip()

    def _learner_log_evidence(self) -> dict[str, Any] | None:
        if not self.learner_log_path.is_file():
            return None
        payload = self.learner_log_path.read_bytes()
        return {
            "path": self.learner_log_path.name,
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    def _close_learner_log(self) -> None:
        if self.learner is None:
            return
        log = getattr(self.learner, "_gradlab_log", None)
        if log is not None and not getattr(log, "closed", False):
            log.close()

    def _recover_durable_state(self) -> None:
        if self._durable_state_archive_enabled():
            restored = self.authority.restore_state_archive(
                run_id=self.manifest.run_id,
                destination=self.run_dir / "state-archive",
            )
            if restored is not None:
                self.state_archive_publication = restored
                print(
                    "restored durable state archive "
                    f"step={int(restored['step'])} files={int(restored['file_count'])}",
                    flush=True,
                )
        prefix = f"runs/{self.manifest.run_id}"
        control_keys = list(self.authority.control.iter_keys(f"{prefix}/attempts"))
        expiring_journal_keys = list(
            self.authority.control.iter_keys(f"expiring-metric-journals/{self.manifest.run_id}")
        )
        attempts = []
        for key in control_keys:
            if not key.endswith("/manifest.json"):
                continue
            document = self.authority.control.get_json(key)
            attempts.append(
                (
                    str(document.get("created_at") or ""),
                    str(document.get("attempt_id") or ""),
                )
            )
        attempt_order = {
            attempt_id: index for index, (_created_at, attempt_id) in enumerate(sorted(attempts))
        }
        segment_keys = [
            key for key in control_keys if "/metric-segments/" in key and key.endswith(".jsonl")
        ]
        segment_keys.extend(key for key in expiring_journal_keys if key.endswith(".jsonl"))

        def segment_attempt_id(key: str) -> str:
            if "/attempts/" in key:
                return key.split("/attempts/", 1)[1].split("/", 1)[0]
            return key.split(
                f"expiring-metric-journals/{self.manifest.run_id}/",
                1,
            )[1].split("/", 1)[0]

        segment_keys.sort(
            key=lambda key: (
                attempt_order.get(segment_attempt_id(key), 1_000_000),
                key,
            )
        )
        recovered_events = 0
        for key in segment_keys:
            for encoded in self.authority.control.get_bytes(key).splitlines():
                if not encoded:
                    continue
                event = json.loads(encoded)
                if not isinstance(event, Mapping):
                    raise ValueError(f"metric journal event is not a mapping: {key}")
                self.store.enqueue_event(
                    kind=str(event["kind"]),
                    payload=dict(event["payload"]),
                    step=(None if event.get("step") is None else int(event["step"])),
                    source=str(event["source"]),
                    event_id=str(event["event_id"]),
                    created_at=float(event["created_at"]),
                )
                recovered_events += 1

        index = self.authority.models.get_json_optional(f"{prefix}/index.json")
        checkpoints = [
            dict(row) for row in (index or {}).get("checkpoints") or [] if isinstance(row, Mapping)
        ]
        checkpoints.sort(key=lambda row: (int(row["step"]), str(row["sha256"])))
        if self.evaluation_required and self.authority.has_accepted_eval(self.manifest.run_id):
            self.eval_admission_closed = True
            self.stop_reason = "eval_acceptance"
            self.store.set_state(
                "automatic_eval_admission",
                {
                    "closed": True,
                    "reason": "eval_acceptance",
                    "recovered": True,
                    "closed_at": self.clock.utc_now(),
                },
            )
        for position, document in enumerate(checkpoints, start=1):
            checkpoint = CheckpointManifest(**document)
            checkpoint.validate()
            ledger_id = -position
            self.store.record_checkpoint_publication(
                checkpoint_ledger_id=ledger_id,
                manifest=checkpoint.to_dict(),
            )
            if self.evaluation_required:
                if self.eval_admission_closed:
                    self._ensure_eval(
                        ledger_id,
                        checkpoint,
                        create_if_missing=False,
                    )
                else:
                    self._ensure_eval(ledger_id, checkpoint)

        for initial in self.store.evals(statuses=("pending",)):
            key = str(initial["idempotency_key"])
            selected_attempt = 0
            selected_prepared: Mapping[str, Any] | None = None
            selected_dispatch: Mapping[str, Any] | None = None
            for attempt in range(1, int(self.modal_config.protocol.max_attempts) + 1):
                prepared = self.authority.eval_attempt(
                    run_id=self.manifest.run_id,
                    idempotency_key=key,
                    attempt=attempt,
                )
                if prepared is None:
                    continue
                selected_attempt = attempt
                selected_prepared = prepared
                selected_dispatch = self.authority.eval_dispatch(
                    run_id=self.manifest.run_id,
                    idempotency_key=key,
                    attempt=attempt,
                )
            if selected_prepared is not None:
                self.store.mark_eval_submitted(
                    idempotency_key=key,
                    attempt=selected_attempt,
                    modal_call_id=str((selected_dispatch or {}).get("modal_call_id") or ""),
                    attempt_expires_at=float(selected_prepared["expires_at"]),
                )

            verified = self.authority.evaluation.get_json_optional(
                f"{prefix}/evals/{key}/verified-result.json"
            )
            raw = self.authority.eval_result(
                run_id=self.manifest.run_id,
                idempotency_key=key,
            )
            if verified is not None:
                result = EvalResult(**verified)
                result.validate()
                self.store.mark_eval_terminal(
                    idempotency_key=key,
                    status=result.status,
                    result=result.to_dict(),
                )
                if raw is not None:
                    self._record_eval_metrics(
                        self.store.eval(key) or initial,
                        result,
                        raw,
                    )
                if result.status == "accepted":
                    self.stop_reason = "eval_acceptance"
                continue
            if raw is not None:
                row = self.store.eval(key)
                if row is None:
                    raise RuntimeError(f"recovered eval disappeared from the ledger: {key}")
                if int(row["attempt"] or 0) == 0:
                    raw_attempt_id = str(raw.get("attempt_id") or "")
                    try:
                        attempt = int(raw_attempt_id.rsplit("-a", 1)[1])
                    except (IndexError, ValueError) as exc:
                        raise ValueError("raw eval result has no recoverable attempt") from exc
                    self.store.mark_eval_submitted(
                        idempotency_key=key,
                        attempt=attempt,
                        modal_call_id="",
                        attempt_expires_at=0.0,
                    )
                    row = self.store.eval(key)
                    assert row is not None
                self._observe_result(row)
        if recovered_events or checkpoints:
            print(
                f"recovered durable state events={recovered_events} "
                f"checkpoints={len(checkpoints)} evals={len(self.store.evals())}",
                flush=True,
            )

    def _request_learner_stop(self, reason: str) -> None:
        if not self.stop_reason:
            self.stop_reason = reason
            self._emit("learner_stop_requested", reason=reason)
        if self.learner is not None and self.learner.poll() is None:
            self.runtime.request_learner_stop(self.learner)
            print(f"learner stop requested: reason={reason}", flush=True)

    def _request_finalize_only_stop(self, reason: str) -> None:
        if self.learner is None or self.learner.poll() is not None:
            return
        self.runtime.request_learner_stop(self.learner)
        self._emit("learner_finalize_only_requested", reason=reason)
        print(f"learner finalize-only stop requested: reason={reason}", flush=True)

    def _close_eval_admission(self, result: EvalResult) -> None:
        if self.eval_admission_closed:
            return
        self.eval_admission_closed = True
        closed_at = self.clock.utc_now()
        self.store.set_state(
            "automatic_eval_admission",
            {
                "closed": True,
                "reason": "eval_acceptance",
                "checkpoint_id": result.checkpoint_id,
                "idempotency_key": result.idempotency_key,
                "closed_at": closed_at,
            },
        )
        self._emit(
            "automatic_eval_admission_closed",
            checkpoint_id=result.checkpoint_id,
            idempotency_key=result.idempotency_key,
            reason="eval_acceptance",
        )

    def _reconcile_verified_eval_result(
        self,
        row: Mapping[str, Any],
    ) -> bool:
        document = self.authority.evaluation.get_json_optional(
            f"runs/{self.manifest.run_id}/evals/{row['idempotency_key']}/verified-result.json"
        )
        if document is None:
            return False
        result = EvalResult(**document)
        result.validate()
        self.store.mark_eval_terminal(
            idempotency_key=result.idempotency_key,
            status=result.status,
            result=result.to_dict(),
        )
        return True

    def _cancel_outstanding_evals(self) -> None:
        for row in self.store.evals(statuses=("pending", "submitted")):
            if self._observe_result(row) or self._reconcile_verified_eval_result(row):
                continue
            call_id = str(row.get("modal_call_id") or "")
            if call_id:
                try:
                    assert self.eval_backend is not None
                    self.eval_backend.cancel(EvalHandle(provider="modal", call_id=call_id))
                except Exception as exc:
                    print(f"Modal cancel failed call={call_id}: {exc}", flush=True)
            result = EvalResult(
                run_id=self.manifest.run_id,
                checkpoint_id=str(row["checkpoint_id"]),
                idempotency_key=str(row["idempotency_key"]),
                modal_call_id=call_id or "not-submitted",
                status="canceled",
                episode_results=[],
                aggregates={},
                timings={"canceled_at": self.clock.utc_now()},
                evidence_sha256=[],
                completed_at=self.clock.utc_now(),
                error="run canceled",
            )
            try:
                self.authority.put_verified_eval_result(result)
            except ConditionalWriteConflict:
                if not self._reconcile_verified_eval_result(row):
                    raise
                continue
            self.store.mark_eval_terminal(
                idempotency_key=result.idempotency_key,
                status=result.status,
                result=result.to_dict(),
            )

    def _renew_lease(self, now: float) -> None:
        if now - self.last_lease_renewal < LEASE_RENEW_SECONDS:
            return
        assert self.lease is not None
        try:
            self.lease = self.authority.renew_lease(self.lease)
        except Exception as exc:
            self.lease_misses += 1
            print(
                f"writer lease renewal failed ({self.lease_misses}/"
                f"{LEASE_MISSES_BEFORE_STOP}): {exc}",
                flush=True,
            )
            if self.lease_misses >= LEASE_MISSES_BEFORE_STOP:
                self.lease_lost = True
                self._emit("writer_lease_lost", misses=self.lease_misses)
                self._request_learner_stop("writer_lease_lost")
        else:
            self.lease_misses = 0
            self.last_lease_renewal = now
            self._emit("writer_lease_renewed", holder_id=self.lease.holder_id)

    def _seal_metrics(self, now: float, *, force: bool = False) -> int:
        events = self.store.next_metric_events(limit=METRIC_SEGMENT_EVENTS)
        if not events:
            self.last_segment = now
            return 0
        if (
            not force
            and len(events) < METRIC_SEGMENT_EVENTS
            and now - self.last_segment < METRIC_SEGMENT_SECONDS
        ):
            return 0
        key, digest = self.authority.seal_metric_segment(
            run_id=self.manifest.run_id,
            attempt_id=self.manifest.attempt_id,
            events=events,
        )
        self.store.record_metric_segment(
            events=events,
            object_key=key,
            sha256=digest,
        )
        self._emit(
            "metric_segment_sealed",
            first_event_seq=int(events[0]["event_seq"]),
            last_event_seq=int(events[-1]["event_seq"]),
            object_key=key,
            sha256=digest,
        )
        self.last_segment = now
        return len(events)

    def _publish_wandb(self) -> int:
        if self.projector is None:
            return 0
        started = self.clock.monotonic()
        published = self.runtime.publish_frames(
            self.store,
            self.projector,
            limit=250,
        )
        elapsed = max(self.clock.monotonic() - started, 1e-6)
        if published:
            self.peak_publish_capacity = max(
                self.peak_publish_capacity,
                published / elapsed,
            )
        return published

    def _publish_checkpoints(self) -> int:
        published = 0
        contract_hashes = {
            "goal_sha256": self.manifest.goal_sha256,
            "recipe_sha256": self.manifest.recipe_sha256,
            "environment_sha256": self.manifest.environment_sha256,
            "evaluation_contract_sha256": (
                evaluation_contract_sha256(self.recipe_document)
                if self.evaluation_required
                else canonical_json_sha256(
                    {
                        "training_only": True,
                        "playback": self.recipe_document["recipe"]["playback"],
                    }
                )
            ),
        }
        for checkpoint in self.store.checkpoints():
            ledger_id = int(checkpoint["id"])
            if self.store.checkpoint_publication(ledger_id) is not None:
                continue
            path = Path(str(checkpoint["path"]))
            if not path.is_file():
                continue
            kind = str(checkpoint["kind"])
            purpose = "final" if kind in {"final", "interrupted"} else "periodic"
            try:
                manifest = self.authority.publish_checkpoint(
                    run_id=self.manifest.run_id,
                    model_path=path,
                    step=int(checkpoint["step"] or 0),
                    purpose=purpose,
                    contract_hashes=contract_hashes,
                    recovery_sidecar={
                        "schema_version": 1,
                        "run_id": self.manifest.run_id,
                        "attempt_id": self.manifest.attempt_id,
                        "checkpoint_ledger_id": ledger_id,
                        "kind": kind,
                        "local_path": str(path),
                    },
                    created_at=_timestamp(float(checkpoint["created_at"])),
                )
            except Exception as exc:
                self.store.mark_checkpoint_upload_failed(ledger_id, repr(exc))
                print(f"checkpoint publication failed id={ledger_id}: {exc}", flush=True)
                continue
            existing = self.store.checkpoint_publication_by_id(manifest.checkpoint_id)
            if existing is not None:
                self.store.mark_checkpoint_uploaded(
                    ledger_id,
                    manifest.public_url,
                )
                continue
            self.store.record_checkpoint_publication(
                checkpoint_ledger_id=ledger_id,
                manifest=manifest.to_dict(),
            )
            self.store.mark_checkpoint_uploaded(
                ledger_id,
                manifest.public_url,
            )
            self._emit(
                "checkpoint_published",
                checkpoint_id=manifest.checkpoint_id,
                checkpoint_step=manifest.step,
                public_url=manifest.public_url,
            )
            if bool(checkpoint["eval_required"]) and not self.eval_admission_closed:
                self._ensure_eval(ledger_id, manifest)
            print(
                f"checkpoint published id={manifest.checkpoint_id} "
                f"step={manifest.step} url={manifest.public_url}",
                flush=True,
            )
            published += 1
        return published

    def _execution_contract(self, checkpoint: CheckpointManifest) -> dict[str, Any]:
        contract = dict(self.eval_contract)
        contract.update(
            {
                "schema_version": PROTOCOL_SCHEMA_VERSION,
                "checkpoint_sha256": checkpoint.sha256,
                "runtime_image_ref": self.manifest.image_digest,
                "recipe_sha256": checkpoint.recipe_document_sha256,
                "recipe_format_version": int(self.recipe_document["format_version"]),
                "evaluation_contract_sha256": checkpoint.evaluation_contract_sha256,
            }
        )
        asset = contract.get("asset")
        if isinstance(asset, Mapping):
            contract["asset"] = {
                str(key): value for key, value in asset.items() if str(key) != "object_uri"
            }
        manifest_index(contract)
        return contract

    def _ensure_eval(
        self,
        checkpoint_ledger_id: int,
        checkpoint: CheckpointManifest,
        *,
        create_if_missing: bool = True,
    ) -> bool:
        contract = self._execution_contract(checkpoint)
        episode_manifest_sha = document_sha256(contract["manifest"])
        key = eval_idempotency_key(
            run_id=self.manifest.run_id,
            checkpoint_sha256=checkpoint.sha256,
            evaluation_contract_sha256=checkpoint.evaluation_contract_sha256,
            episode_manifest_sha256=episode_manifest_sha,
            protocol="modal-acceptance-v3",
        )
        timeout = int(self.modal_config.timeouts.acceptance_seconds)
        created = _parse_timestamp(checkpoint.created_at)
        result_key = f"runs/{self.manifest.run_id}/evals/{key}/result.json"
        intent = EvalIntent(
            run_id=self.manifest.run_id,
            checkpoint_id=checkpoint.checkpoint_id,
            idempotency_key=key,
            checkpoint_sha256=checkpoint.sha256,
            goal_sha256=self.manifest.goal_sha256,
            recipe_sha256=self.manifest.recipe_sha256,
            environment_sha256=self.manifest.environment_sha256,
            evaluation_contract_sha256=checkpoint.evaluation_contract_sha256,
            episode_manifest_sha256=episode_manifest_sha,
            protocol="modal-acceptance-v3",
            execution_contract=contract,
            result_key=result_key,
            timeout_seconds=timeout,
            created_at=created.isoformat().replace("+00:00", "Z"),
            expires_at=(created + timedelta(seconds=timeout)).isoformat().replace("+00:00", "Z"),
        )
        existing = self.authority.eval_intent(
            run_id=self.manifest.run_id,
            idempotency_key=key,
        )
        if existing is None:
            if not create_if_missing:
                return False
            self.authority.put_eval_intent(intent)
        elif existing != intent.to_dict():
            raise ValueError(
                f"evaluation intent conflicts with checkpoint {checkpoint.checkpoint_id}"
            )
        self.store.ensure_eval(
            checkpoint_ledger_id=checkpoint_ledger_id,
            intent={
                **intent.to_dict(),
                "checkpoint_step": checkpoint.step,
                "checkpoint": checkpoint.to_dict(),
            },
        )
        if existing is None:
            self._emit(
                "eval_intent_persisted",
                checkpoint_id=checkpoint.checkpoint_id,
                checkpoint_step=checkpoint.step,
                idempotency_key=key,
            )
        return True

    def _eval_payload(
        self,
        row: Mapping[str, Any],
        *,
        attempt: int,
        expires_at: float,
    ) -> dict[str, Any]:
        intent = dict(row["intent"])
        checkpoint = dict(intent["checkpoint"])
        contract = dict(intent["execution_contract"])
        timeout = int(intent["timeout_seconds"])
        result_key = str(intent["result_key"])
        attempt_id = f"{intent['idempotency_key'][:20]}-a{attempt}"
        payload: dict[str, Any] = {
            "attempt_id": attempt_id,
            "contract": contract,
            "expires_at": expires_at,
            "child_timeout_seconds": max(
                1,
                timeout - int(self.modal_config.timeouts.child_margin_seconds),
            ),
            "model_get_url": str(checkpoint["public_url"]),
            "model_document_get_url": str(checkpoint["model_document_url"]),
            "model_document_sha256": str(checkpoint["model_document_sha256"]),
            "recipe_get_url": str(checkpoint["recipe_document_url"]),
            "result_uri": self.authority.evaluation.uri(result_key),
            "result_put_url": self.authority.evaluation.presign_put(
                result_key,
                expires_seconds=timeout + int(self.modal_config.timeouts.expiry_margin_seconds),
            ),
        }
        asset = self.manifest.modal.get("rom_asset_manifest")
        if isinstance(asset, Mapping):
            rom_key = self.authority.evaluation.key_from_uri(str(asset["object_uri"]))
            payload["rom_get_url"] = self.authority.evaluation.presign_get(
                rom_key,
                expires_seconds=timeout + int(self.modal_config.timeouts.expiry_margin_seconds),
            )
        return payload

    def _submit_pending_evals(self) -> int:
        if self.eval_admission_closed:
            return 0
        submitted = 0
        for row in self.store.evals(statuses=("pending",)):
            attempt = int(row["attempt"] or 0) + 1
            if attempt > int(self.modal_config.protocol.max_attempts):
                self._mark_expired(row, error="eval exhausted two attempts")
                continue
            prepared = self.authority.eval_attempt(
                run_id=self.manifest.run_id,
                idempotency_key=str(row["idempotency_key"]),
                attempt=attempt,
            )
            if prepared is not None:
                dispatch = self.authority.eval_dispatch(
                    run_id=self.manifest.run_id,
                    idempotency_key=str(row["idempotency_key"]),
                    attempt=attempt,
                )
                self.store.mark_eval_submitted(
                    idempotency_key=str(row["idempotency_key"]),
                    attempt=attempt,
                    modal_call_id=str((dispatch or {}).get("modal_call_id") or ""),
                    attempt_expires_at=float(prepared["expires_at"]),
                )
                continue
            expires_at = self.clock.time() + int(row["intent"]["timeout_seconds"])
            self.authority.prepare_eval_attempt(
                run_id=self.manifest.run_id,
                idempotency_key=str(row["idempotency_key"]),
                attempt=attempt,
                expires_at=expires_at,
            )
            payload = self._eval_payload(
                row,
                attempt=attempt,
                expires_at=expires_at,
            )
            try:
                assert self.eval_backend is not None
                handle = self.eval_backend.submit(payload)
            except Exception as exc:
                self.store.mark_eval_submitted(
                    idempotency_key=str(row["idempotency_key"]),
                    attempt=attempt,
                    modal_call_id="",
                    attempt_expires_at=expires_at,
                )
                self.store.record_eval_error(
                    idempotency_key=str(row["idempotency_key"]),
                    error=f"ambiguous submit: {exc!r}",
                )
                print(
                    f"Modal spawn ambiguous key={row['idempotency_key']}: {exc}",
                    flush=True,
                )
                continue
            self.authority.put_eval_dispatch(
                run_id=self.manifest.run_id,
                idempotency_key=str(row["idempotency_key"]),
                attempt=attempt,
                modal_call_id=handle.call_id,
            )
            self.store.mark_eval_submitted(
                idempotency_key=str(row["idempotency_key"]),
                attempt=attempt,
                modal_call_id=handle.call_id,
                attempt_expires_at=expires_at,
            )
            self._emit(
                "eval_submitted",
                checkpoint_id=str(row["checkpoint_id"]),
                idempotency_key=str(row["idempotency_key"]),
                attempt=attempt,
                call_id=handle.call_id,
            )
            print(
                f"Modal eval submitted checkpoint={row['checkpoint_id']} "
                f"call={handle.call_id} attempt={attempt}",
                flush=True,
            )
            submitted += 1
        return submitted

    def _settle_closed_eval_admission(self) -> int:
        if not self.eval_admission_closed:
            return 0
        admission = self.store.state("automatic_eval_admission") or {}
        admission_reason = str(admission.get("reason") or "closed")
        settled = 0
        for row in self.store.evals(statuses=("pending",)):
            attempt = int(row.get("attempt") or 0)
            if attempt > 0:
                if self._observe_result(row):
                    settled += 1
                    continue
                self._mark_expired(
                    row,
                    error=(
                        "prior eval attempt expired without a valid result; "
                        f"automatic retry was suppressed after {admission_reason}"
                    ),
                )
            else:
                self.store.mark_eval_deferred(
                    idempotency_key=str(row["idempotency_key"]),
                    reason=f"automatic evaluation admission closed: {admission_reason}",
                )
                self._emit(
                    "eval_deferred",
                    checkpoint_id=str(row["checkpoint_id"]),
                    idempotency_key=str(row["idempotency_key"]),
                    reason="automatic_eval_admission_closed",
                )
            settled += 1
        return settled

    def _reconcile_evals_before_submission(self) -> int:
        reconciled = 0
        candidates = [
            *(
                row
                for row in self.store.evals(statuses=("pending",))
                if int(row.get("attempt") or 0) > 0
            ),
            *self.store.evals(statuses=("submitted",)),
        ]
        for row in candidates:
            current = self.store.eval(str(row["idempotency_key"]))
            if current is None or str(current["status"]) not in {"pending", "submitted"}:
                continue
            if self._observe_result(current):
                reconciled += 1
        reconciled += self._settle_closed_eval_admission()
        return reconciled

    def _verified_result(
        self,
        row: Mapping[str, Any],
        raw: Mapping[str, Any],
    ) -> EvalResult:
        intent = dict(row["intent"])
        attempt = int(row["attempt"])
        attempt_id = f"{intent['idempotency_key'][:20]}-a{attempt}"
        contract = dict(intent["execution_contract"])
        raw_status = str(raw.get("status") or "")
        if raw_status == "succeeded":
            validated = validate_attempt_result(
                raw,
                contract=contract,
                attempt_id=attempt_id,
            )
            verdict = str(validated.get("verdict") or "")
            status = "accepted" if verdict == "accepted" else "rejected"
            episodes = list(validated.get("episode_results") or [])
            aggregates = dict(validated.get("claimed_aggregates") or {})
            error = None
        else:
            if str(raw.get("attempt_id") or "") != attempt_id:
                raise ValueError("failed eval result attempt id mismatch")
            if str(raw.get("execution_key") or "") != execution_key(contract):
                raise ValueError("failed eval result execution key mismatch")
            status = "expired" if raw_status == "expired" else "failed"
            episodes = list(raw.get("episode_results") or [])
            aggregates = dict(raw.get("claimed_aggregates") or {})
            error = str(raw.get("error") or f"Modal eval status={raw_status or 'unknown'}")
        evidence_values = [
            episodes,
            raw.get("evaluation_evidence") or {},
            raw.get("preview") or {},
        ]
        evidence_hashes = [
            document_sha256({"evidence": value})
            for value in evidence_values
            if value not in (None, {}, [])
        ]
        return EvalResult(
            run_id=self.manifest.run_id,
            checkpoint_id=str(row["checkpoint_id"]),
            idempotency_key=str(row["idempotency_key"]),
            modal_call_id=str(row["modal_call_id"] or "not-recorded"),
            status=status,  # type: ignore[arg-type]
            episode_results=episodes,
            aggregates=aggregates,
            timings={
                "duration_seconds": float(raw.get("duration_seconds") or 0.0),
                "result_observed_at": self.clock.utc_now(),
            },
            evidence_sha256=evidence_hashes,
            completed_at=self.clock.utc_now(),
            error=error,
        )

    def _record_eval_metrics(
        self,
        row: Mapping[str, Any],
        result: EvalResult,
        raw: Mapping[str, Any],
    ) -> None:
        metrics = {
            str(name): value
            for name, value in dict(raw.get("metrics") or {}).items()
            if metric_definition(str(name)) is not None
        }
        metrics.update(
            {
                EVAL_ACCEPTANCE_PASS: 1.0 if result.status == "accepted" else 0.0,
                EVAL_ACCEPTANCE_EPISODES_PLANNED: float(
                    row["intent"]["execution_contract"]["episodes"]
                ),
                EVAL_ACCEPTANCE_EPISODES_COMPLETED: float(len(result.episode_results)),
                EVAL_ACCEPTANCE_FAILURE_COUNT: float(result.aggregates.get("failure_count") or 0),
                EVAL_ACCEPTANCE_DURATION_SECONDS: float(raw.get("duration_seconds") or 0.0),
            }
        )
        self.store.append_metrics(
            metrics,
            step=int(row["checkpoint_step"]),
            source=f"eval:{row['idempotency_key']}",
        )
        if result.status == "accepted":
            self.store.enqueue_event(
                kind="eval_by_start",
                payload={
                    "rows": eval_by_start_rows(
                        [dict(episode) for episode in result.episode_results]
                    )
                },
                step=int(row["checkpoint_step"]),
                source=f"eval:{row['idempotency_key']}:by-start",
            )

    def _observe_result(self, row: Mapping[str, Any]) -> bool:
        raw = self.authority.eval_result(
            run_id=self.manifest.run_id,
            idempotency_key=str(row["idempotency_key"]),
        )
        if raw is None:
            return False
        try:
            result = self._verified_result(row, raw)
        except Exception as exc:
            self.store.record_eval_error(
                idempotency_key=str(row["idempotency_key"]),
                error=f"invalid result: {exc!r}",
            )
            print(f"invalid Modal result ignored key={row['idempotency_key']}: {exc}", flush=True)
            return False
        self.authority.put_verified_eval_result(result)
        self.store.mark_eval_terminal(
            idempotency_key=result.idempotency_key,
            status=result.status,
            result=result.to_dict(),
        )
        self._emit(
            "eval_terminal",
            checkpoint_id=result.checkpoint_id,
            idempotency_key=result.idempotency_key,
            status=result.status,
        )
        if result.status == "accepted":
            observed = self.clock.time()
            self.accepted_observed_at = self.accepted_observed_at or observed
            self._close_eval_admission(result)
            self._request_learner_stop("eval_acceptance")
            signal_sent = self.clock.time()
            requested = self.store.mark_stop_requested(idempotency_key=result.idempotency_key)
            result_to_stop = signal_sent - observed
            self.store.append_metrics(
                {ORCHESTRATION_RESULT_TO_STOP_SECONDS: result_to_stop},
                step=int(row["checkpoint_step"]),
                source=f"orchestration:stop:{result.idempotency_key}",
            )
            if result_to_stop > 10.0 or requested - observed > 10.0:
                raise RuntimeError("accepted eval did not issue stop within ten seconds")
        self._record_eval_metrics(row, result, raw)
        print(
            f"Modal eval terminal checkpoint={result.checkpoint_id} status={result.status}",
            flush=True,
        )
        return True

    def _mark_expired(self, row: Mapping[str, Any], *, error: str) -> None:
        result = EvalResult(
            run_id=self.manifest.run_id,
            checkpoint_id=str(row["checkpoint_id"]),
            idempotency_key=str(row["idempotency_key"]),
            modal_call_id=str(row.get("modal_call_id") or "not-submitted"),
            status="expired",
            episode_results=[],
            aggregates={},
            timings={"result_observed_at": self.clock.utc_now()},
            evidence_sha256=[],
            completed_at=self.clock.utc_now(),
            error=error,
        )
        self.authority.put_verified_eval_result(result)
        self.store.mark_eval_terminal(
            idempotency_key=result.idempotency_key,
            status=result.status,
            result=result.to_dict(),
        )

    def _poll_evals(self, now: float, *, force: bool = False) -> int:
        if not force and now - self.last_eval_poll < EVAL_POLL_SECONDS:
            return 0
        self.last_eval_poll = now
        wall_now = self.clock.time()
        completed = 0
        for row in self.store.evals(statuses=("submitted",)):
            if self._observe_result(row):
                completed += 1
                continue
            call_id = str(row["modal_call_id"] or "")
            if call_id:
                handle = EvalHandle(provider="modal", call_id=call_id)
                assert self.eval_backend is not None
                poll = self.eval_backend.poll(handle)
                if poll.status == "failed" and poll.error:
                    self.store.record_eval_error(
                        idempotency_key=str(row["idempotency_key"]),
                        error=poll.error,
                    )
            expires_at = float(row.get("attempt_expires_at") or 0)
            if expires_at > wall_now:
                continue
            if not self.eval_admission_closed and int(row["attempt"] or 0) < int(
                self.modal_config.protocol.max_attempts
            ):
                self.store.reset_expired_eval(
                    idempotency_key=str(row["idempotency_key"]),
                    error="attempt expired without a valid result",
                )
            else:
                self._mark_expired(
                    row,
                    error=(
                        "eval expired after automatic admission closed"
                        if self.eval_admission_closed
                        else "eval expired twice without a valid result"
                    ),
                )
                completed += 1
        completed += self._settle_closed_eval_admission()
        return completed

    def _scratch_guard(self) -> None:
        usage = self.runtime.disk_usage(self.output_root)
        fraction = usage.used / max(usage.total, 1)
        if fraction >= SCRATCH_STOP_FRACTION:
            self._request_learner_stop("scratch_storage_above_80_percent")
            raise RuntimeError(
                f"scratch storage is {fraction:.1%} full; stopped before evidence loss"
            )

    def _frame_high_waters(self) -> tuple[int, int]:
        with self.store.connection() as connection:
            row = connection.execute(
                """
                SELECT COALESCE(MAX(id), 0) AS local_high_water,
                       COALESCE(MAX(id) FILTER (WHERE status = 'published'), 0)
                         AS wandb_high_water
                FROM metric_frames
                """
            ).fetchone()
        return int(row["local_high_water"]), int(row["wandb_high_water"])

    def _probe_wandb_remote(
        self,
        now: float,
        *,
        local_high_water: int,
        force: bool = False,
    ) -> None:
        minimum_interval = WANDB_DRAIN_REMOTE_PROBE_SECONDS if force else WANDB_REMOTE_PROBE_SECONDS
        if not self.wandb_run_path or now - self.last_remote_probe < minimum_interval:
            return
        self.last_remote_probe = now
        try:
            summary_value = self.runtime.remote_summary(self.wandb_run_path).get(
                "orchestration/event_seq"
            )
            summary_value = _summary_scalar(summary_value)
            remote_high_water = int(summary_value or 0)
            self.wandb_remote_high_water = max(
                self.wandb_remote_high_water,
                remote_high_water,
            )
            with self.store.connection() as connection:
                unseen = connection.execute(
                    "SELECT MIN(created_at) FROM metric_frames WHERE id > ?",
                    (self.wandb_remote_high_water,),
                ).fetchone()
            oldest_unseen = unseen[0] if unseen is not None else None
            self.wandb_remote_visible_lag_seconds = (
                0.0
                if oldest_unseen is None or self.wandb_remote_high_water >= local_high_water
                else max(0.0, self.clock.time() - float(oldest_unseen))
            )
        except Exception as exc:
            self.store.set_state(
                "wandb_remote_probe_error",
                {"error": repr(exc)[:1000], "at": self.clock.utc_now()},
            )

    def _emit_health(self, now: float) -> None:
        if now - self.last_health_sample < HEALTH_SAMPLE_SECONDS:
            return
        interval = (
            HEALTH_SAMPLE_SECONDS
            if self.last_health_sample == 0.0
            else max(now - self.last_health_sample, 0.001)
        )
        local_high_water, wandb_high_water = self._frame_high_waters()
        ingress_rate = max(
            0.0,
            (local_high_water - self.last_health_local_high_water) / interval,
        )
        publish_rate = max(
            0.0,
            (wandb_high_water - self.last_health_wandb_high_water) / interval,
        )
        self.peak_ingress_rate = max(self.peak_ingress_rate, ingress_rate)
        self.peak_publish_rate = max(self.peak_publish_rate, publish_rate)
        capacity_ratio = (
            self.peak_publish_capacity / self.peak_ingress_rate
            if self.peak_ingress_rate > 0.0
            else 0.0
        )
        self._probe_wandb_remote(now, local_high_water=local_high_water)
        usage = self.runtime.disk_usage(self.output_root)
        metrics = {
            ORCHESTRATION_QUEUE_DEPTH: float(self.store.metric_outbox_stats()["frames"]),
            ORCHESTRATION_OLDEST_UNPUBLISHED_SECONDS: self._oldest_unpublished_age(),
            ORCHESTRATION_INGRESS_RATE: ingress_rate,
            ORCHESTRATION_PUBLISH_RATE: publish_rate,
            ORCHESTRATION_PUBLICATION_CAPACITY_RATIO: capacity_ratio,
            ORCHESTRATION_LOCAL_HIGH_WATER: float(local_high_water),
            ORCHESTRATION_R2_HIGH_WATER: float(self.store.metric_segment_high_water()),
            ORCHESTRATION_WANDB_HIGH_WATER: float(wandb_high_water),
            ORCHESTRATION_WANDB_REMOTE_HIGH_WATER: float(self.wandb_remote_high_water),
            ORCHESTRATION_WANDB_REMOTE_VISIBLE_LAG_SECONDS: (self.wandb_remote_visible_lag_seconds),
            ORCHESTRATION_CHECKPOINT_BACKLOG: float(
                len(self.store.checkpoints()) - len(self.store.checkpoint_publications())
            ),
            ORCHESTRATION_PENDING_EVALS: float(
                len(
                    self.store.evals(
                        statuses=("pending", "submitted"),
                    )
                )
            ),
            ORCHESTRATION_SCRATCH_USED_FRACTION: (usage.used / max(usage.total, 1)),
        }
        step = int(self.store.outbox_health().get("local_latest_step") or 0)
        self.store.append_metrics(
            metrics,
            step=step,
            source="orchestration:health",
        )
        self.store.set_state(
            "backpressure",
            {
                **metrics,
                "publication_capacity_sufficient": capacity_ratio >= 2.0,
                "sampled_at": self.clock.utc_now(),
            },
        )
        self.last_health_sample = now
        self.last_health_local_high_water = local_high_water
        self.last_health_wandb_high_water = wandb_high_water

    def _oldest_unpublished_age(self) -> float:
        health = self.store.outbox_health()
        oldest = health.get("oldest_created_at")
        return 0.0 if oldest is None else max(0.0, self.clock.time() - float(oldest))

    def _all_ready_checkpoints_published(self) -> bool:
        for checkpoint in self.store.checkpoints():
            ledger_id = int(checkpoint["id"])
            if self.store.checkpoint_publication(ledger_id) is not None:
                continue
            digest = str(checkpoint.get("sha256") or "")
            if not digest:
                path = Path(str(checkpoint["path"]))
                if not path.is_file():
                    return False
                digest = file_sha256(path)
            checkpoint_id = f"checkpoint-{int(checkpoint['step'] or 0)}-{digest[:16]}"
            if self.store.checkpoint_publication_by_id(checkpoint_id) is None:
                return False
        return True

    def _has_public_final_checkpoint(self) -> bool:
        return any(
            str(row.get("purpose") or "") == "final" for row in self.store.checkpoint_publications()
        )

    def _durable_state_archive_enabled(self) -> bool:
        archive = self.train_config.get("state_archive")
        return isinstance(archive, Mapping) and archive.get("persistence") == "durable"

    def _publish_state_archive(self, *, require_closed: bool = False) -> int:
        if not self._durable_state_archive_enabled():
            return 0
        archive_root = self.run_dir / "state-archive"
        closure_path = archive_root / "closure.json"
        if not closure_path.is_file():
            if require_closed:
                raise RuntimeError("state archive is enabled but has no local closure")
            return 0
        closure_sha256 = file_sha256(closure_path)
        if closure_sha256 == self.state_archive_closure_sha256:
            publication = self.state_archive_publication
            if require_closed and (publication is None or publication.get("status") != "closed"):
                raise RuntimeError("state archive has no closed publication")
            return 0
        publication = self.authority.publish_state_archive(
            run_id=self.manifest.run_id,
            attempt_id=self.manifest.attempt_id,
            archive_root=archive_root,
        )
        if require_closed and publication.get("status") != "closed":
            raise RuntimeError("state archive final closure is not closed")
        self.state_archive_publication = publication
        self.state_archive_closure_sha256 = closure_sha256
        self._emit(
            "state_archive_published",
            step=int(publication["step"]),
            status=str(publication["status"]),
            generation_sha256=str(publication["generation_sha256"]),
            file_count=int(publication["file_count"]),
        )
        return 1

    def active_iteration(self, *, now: float | None = None) -> int:
        """Advance active supervision once without sleeping."""

        instant = self.clock.monotonic() if now is None else float(now)
        activity = 0
        self._renew_lease(instant)
        if self.lease_lost:
            return 0
        activity += self._seal_metrics(instant)
        activity += self._publish_checkpoints()
        activity += self._publish_state_archive()
        activity += self._reconcile_evals_before_submission()
        activity += self._submit_pending_evals()
        activity += self._poll_evals(instant)
        activity += self._publish_wandb()
        self._emit_health(instant)
        self._scratch_guard()
        unpublished_age = self._oldest_unpublished_age()
        if unpublished_age >= WANDB_WARNING_SECONDS:
            print(
                f"warning: oldest unpublished W&B event is {unpublished_age:.1f}s old",
                flush=True,
            )
        if unpublished_age >= WANDB_UNHEALTHY_SECONDS:
            self.store.set_state(
                "wandb_unhealthy",
                {
                    "oldest_unpublished_seconds": unpublished_age,
                    "at": self.clock.utc_now(),
                },
            )
        return activity

    def drain_iteration(self, *, now: float | None = None) -> tuple[int, bool]:
        """Advance terminal drain once and report whether it converged."""

        instant = self.clock.monotonic() if now is None else float(now)
        activity = 0
        self._renew_lease(instant)
        if self.lease_lost:
            raise LeaseUnavailable("writer lease was lost while draining")
        activity += self._seal_metrics(instant, force=True)
        activity += self._publish_checkpoints()
        activity += self._publish_state_archive()
        if self.cancel_requested:
            self._cancel_outstanding_evals()
        else:
            activity += self._reconcile_evals_before_submission()
            activity += self._submit_pending_evals()
        activity += self._poll_evals(instant, force=True)
        activity += self._publish_wandb()
        pending_frames = self.store.metric_outbox_stats()["frames"]
        converged = (
            self._all_ready_checkpoints_published()
            and self.store.all_evals_settled()
            and pending_frames == 0
        )
        return activity, converged

    def _drain(self) -> None:
        delivery_deadline: float | None = None
        while True:
            now = self.clock.monotonic()
            activity, converged = self.drain_iteration(now=now)
            if self.store.all_evals_settled():
                if delivery_deadline is None:
                    delivery_deadline = now + WANDB_DRAIN_TIMEOUT_SECONDS
            else:
                delivery_deadline = None
            if converged:
                if (
                    self.peak_ingress_rate > 0.0
                    and self.peak_publish_capacity < 2.0 * self.peak_ingress_rate
                ):
                    raise RuntimeError(
                        "measured W&B publication capacity is below twice peak metric ingress"
                    )
                return
            if delivery_deadline is not None and now >= delivery_deadline:
                raise TimeoutError(
                    "post-evaluation delivery drain exceeded 300 seconds before "
                    "checkpoints and local W&B delivery converged"
                )
            if activity == 0:
                self.clock.sleep(0.5)

    def _close_eval_admission_for_failure(self, failure: BaseException) -> None:
        if self.eval_admission_closed:
            return
        reason = (
            failure.stop_reason
            if isinstance(failure, LearnerOperationalFailure)
            else "supervisor_failure"
        )
        self.eval_admission_closed = True
        self.store.set_state(
            "automatic_eval_admission",
            {
                "closed": True,
                "reason": reason,
                "closed_at": self.clock.utc_now(),
            },
        )
        self._emit("automatic_eval_admission_closed", reason=reason)

    def _defer_unsettled_evals_after_failure(self) -> int:
        deferred = 0
        for row in self.store.evals(statuses=("pending", "submitted")):
            if self._observe_result(row) or self._reconcile_verified_eval_result(row):
                continue
            self.store.mark_eval_deferred(
                idempotency_key=str(row["idempotency_key"]),
                reason="failure drain deadline reached; explicit reconciliation required",
            )
            self._emit(
                "eval_deferred",
                checkpoint_id=str(row["checkpoint_id"]),
                idempotency_key=str(row["idempotency_key"]),
                reason="failure_drain_deadline",
            )
            deferred += 1
        return deferred

    def _failure_drain(self, failure: BaseException) -> None:
        self._close_eval_admission_for_failure(failure)
        deadline = (
            self.clock.monotonic()
            + self._liveness_seconds("failure_drain_timeout_seconds")
        )
        while True:
            now = self.clock.monotonic()
            activity, converged = self.drain_iteration(now=now)
            if converged:
                return
            if now >= deadline:
                activity += self._defer_unsettled_evals_after_failure()
                activity += self._publish_wandb()
                if (
                    self._all_ready_checkpoints_published()
                    and self.store.all_evals_settled()
                    and self.store.metric_outbox_stats()["frames"] == 0
                ):
                    return
                raise TimeoutError(
                    "failure drain deadline reached before durable publication converged"
                )
            if activity == 0:
                self.clock.sleep(self._liveness_seconds("poll_interval_seconds"))

    def _create_promotion(self) -> PromotionReceipt | None:
        existing = self.authority.control.get_json_optional(
            f"runs/{self.manifest.run_id}/promotion.json"
        )
        if existing is not None:
            receipt = PromotionReceipt(**existing)
            receipt.validate()
            self.authority.create_promotion(receipt)
            return receipt
        accepted = self.store.evals(statuses=("accepted",))
        if not accepted:
            return None
        selected = min(
            accepted,
            key=lambda row: (
                int(row["checkpoint_step"]),
                str(row["checkpoint_id"]),
            ),
        )
        result = dict(selected["result"])
        receipt = PromotionReceipt(
            run_id=self.manifest.run_id,
            checkpoint_id=str(selected["checkpoint_id"]),
            checkpoint_step=int(selected["checkpoint_step"]),
            eval_idempotency_key=str(selected["idempotency_key"]),
            eval_result_sha256=document_sha256(result),
            accepted_episode_count=len(result.get("episode_results") or []),
            promoted_at=self.clock.utc_now(),
        )
        self.authority.create_promotion(receipt)
        self._emit(
            "checkpoint_promoted",
            checkpoint_id=receipt.checkpoint_id,
            checkpoint_step=receipt.checkpoint_step,
        )
        return receipt

    def _validate_no_acceptance_evidence(self) -> None:
        if not self.evaluation_required or self.store.evals(statuses=("accepted",)):
            return
        checkpoints = {str(row["checkpoint_id"]) for row in self.store.checkpoint_publications()}
        evals = {str(row["checkpoint_id"]): row for row in self.store.evals()}
        missing = sorted(checkpoints - set(evals))
        invalid = sorted(
            (
                str(row["checkpoint_id"]),
                str(row["status"]),
            )
            for row in evals.values()
            if str(row["status"]) != "rejected"
        )
        if not checkpoints or missing or invalid:
            facts = {
                "checkpoint_count": len(checkpoints),
                "missing_checkpoint_ids": missing,
                "non_rejection_statuses": invalid,
            }
            raise IncompleteEvaluationEvidence(
                "scientific non-acceptance requires one valid rejected evaluation "
                f"for every published checkpoint: {facts}"
            )

    def _publish_promotion(self, receipt: PromotionReceipt) -> None:
        selected = self.store.eval(receipt.eval_idempotency_key)
        if selected is None:
            raise RuntimeError("promoted eval is absent from the supervisor ledger")
        raw = self.authority.eval_result(
            run_id=self.manifest.run_id,
            idempotency_key=receipt.eval_idempotency_key,
        )
        if raw is None:
            raise RuntimeError("promoted eval raw result is absent from private R2")
        checkpoint = self.store.checkpoint_publication_by_id(receipt.checkpoint_id)
        if checkpoint is None:
            raise RuntimeError("promoted checkpoint is absent from the public inventory")
        metrics = dict(raw.get("metrics") or {})
        metrics.update(dict(selected["result"].get("aggregates") or {}))
        assert self.projector is not None
        self.runtime.publish_promotion(
            self.projector,
            checkpoint_step=receipt.checkpoint_step,
            checkpoint_url=str(checkpoint["public_url"]),
            metrics=metrics,
            updated_at=receipt.promoted_at,
        )

    def _wait_for_remote_promotion(self, receipt: PromotionReceipt) -> None:
        if not self.wandb_run_path:
            raise RuntimeError("W&B run path is unavailable")
        deadline = self.clock.monotonic() + WANDB_DRAIN_TIMEOUT_SECONDS
        while True:
            try:
                summary = self.runtime.remote_summary(self.wandb_run_path)
                if (
                    str(summary.get("gradlab/goal/outcome") or "") == "accepted"
                    and int(summary.get("leader/checkpoint/step") or -1) == receipt.checkpoint_step
                ):
                    return
            except Exception as exc:
                self.store.set_state(
                    "wandb_promotion_probe_error",
                    {"error": repr(exc)[:1000], "at": self.clock.utc_now()},
                )
            if self.clock.monotonic() >= deadline:
                raise TimeoutError(
                    "W&B promotion summary did not become remotely visible within 300 seconds"
                )
            self.clock.sleep(WANDB_DRAIN_REMOTE_PROBE_SECONDS)

    def _wait_for_remote_delivery(self, local_high_water: int) -> None:
        deadline = self.clock.monotonic() + WANDB_DRAIN_TIMEOUT_SECONDS
        while True:
            self._probe_wandb_remote(
                self.clock.monotonic(),
                local_high_water=local_high_water,
                force=True,
            )
            if self.wandb_remote_high_water >= local_high_water:
                return
            if self.clock.monotonic() >= deadline:
                raise TimeoutError(
                    "W&B event high-water mark did not become remotely visible "
                    "within 300 seconds after the SDK run finished"
                )
            self.clock.sleep(WANDB_DRAIN_REMOTE_PROBE_SECONDS)

    def _wandb_high_water(self) -> int:
        with self.store.connection() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(id), 0) FROM metric_frames WHERE status = 'published'"
            ).fetchone()
        return int(row[0] if row else 0)

    def _finish_wandb(self) -> int:
        high_water = self._wandb_high_water()
        projector = self.projector
        self.projector = None
        if projector is not None:
            self.runtime.close_wandb(
                projector,
                timeout_seconds=WANDB_DRAIN_TIMEOUT_SECONDS,
            )
        return high_water

    def _terminal_inventory(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        checkpoints = self.store.checkpoint_publications()
        evals = []
        for row in self.store.evals(
            statuses=("accepted", "rejected", "failed", "expired", "canceled", "deferred")
        ):
            result = dict(row.get("result") or {})
            evals.append(
                {
                    "checkpoint_id": str(row["checkpoint_id"]),
                    "checkpoint_step": int(row["checkpoint_step"]),
                    "idempotency_key": str(row["idempotency_key"]),
                    "status": str(row["status"]),
                    "modal_call_id": str(row.get("modal_call_id") or ""),
                    "attempt": int(row.get("attempt") or 0),
                    "episode_count": len(result.get("episode_results") or []),
                    "result_sha256": document_sha256(result) if result else None,
                    "reason": (
                        str(row.get("last_error") or "")
                        if str(row["status"]) == "deferred"
                        else None
                    ),
                }
            )
        return checkpoints, evals

    def _wait_for_learner_exit_with_lease(self, timeout_seconds: float) -> bool:
        learner = self.learner
        if learner is None:
            return True
        deadline = self.clock.monotonic() + timeout_seconds
        while learner.poll() is None:
            now = self.clock.monotonic()
            if now >= deadline or self.lease_lost:
                return False
            self._renew_lease(now)
            self.clock.sleep(0.25)
        return True

    def _wait_for_learner_group_exit_with_lease(self, timeout_seconds: float) -> bool:
        learner = self.learner
        if learner is None:
            return True
        deadline = self.clock.monotonic() + timeout_seconds
        while self.runtime.learner_group_alive(learner):
            learner.poll()
            now = self.clock.monotonic()
            if now >= deadline:
                return False
            if not self.lease_lost:
                self._renew_lease(now)
            self.clock.sleep(self._liveness_seconds("poll_interval_seconds"))
        learner.poll()
        return True

    def _teardown_learner_group(
        self,
        *,
        primary_failure: BaseException,
    ) -> LearnerTeardownTimeout | None:
        learner = self.learner
        if learner is None:
            return None
        evidence: dict[str, Any] = {
            "primary_failure_type": type(primary_failure).__name__,
            "graceful_stop_requested": True,
            "term_sent": False,
            "kill_sent": False,
            "group_gone": False,
        }
        result_grace = self._liveness_seconds("result_exit_grace_seconds")
        if self._wait_for_learner_group_exit_with_lease(result_grace):
            evidence["group_gone"] = True
            evidence["completed_phase"] = "graceful"
            self.learner_teardown_evidence = evidence
            return None
        self.runtime.terminate_learner_group(learner)
        evidence["term_sent"] = True
        if self._wait_for_learner_group_exit_with_lease(
            self._liveness_seconds("terminate_grace_seconds")
        ):
            evidence["group_gone"] = True
            evidence["completed_phase"] = "term"
            self.learner_teardown_evidence = evidence
            return None
        self.runtime.kill_learner_group(learner)
        evidence["kill_sent"] = True
        if self._wait_for_learner_group_exit_with_lease(
            self._liveness_seconds("kill_grace_seconds")
        ):
            evidence["group_gone"] = True
            evidence["completed_phase"] = "kill"
            self.learner_teardown_evidence = evidence
            return None
        evidence["completed_phase"] = "timeout"
        self.learner_teardown_evidence = evidence
        return LearnerTeardownTimeout(
            "learner process group survived SIGUSR1, SIGTERM, and SIGKILL deadlines"
        )

    def _record_startup_failure(
        self,
        failure: BaseException,
        *,
        phase: str = "startup",
    ) -> int:
        receipt = TerminalReceipt(
            run_id=self.manifest.run_id,
            attempt_id=self.manifest.attempt_id,
            state="resumable_failure",
            acceptance_required=self.evaluation_required,
            stop_reason="supervisor_startup_failure",
            final_step=0,
            checkpoint_inventory=(),
            eval_inventory=(),
            wandb_high_water_mark=0,
            drain={
                "complete": False,
                "phase": phase,
                "metric_segment_high_water": 0,
                "eval_terminal_count": 0,
                "journal_archive": None,
                "journal_expires_at": None,
                "wandb_remote_high_water_mark": 0,
                "publication_capacity_ratio": None,
                "failure": _bounded_exception_document(failure),
            },
            completed_at=self.clock.utc_now(),
        )
        try:
            self.authority.create_attempt_terminal(receipt)
        except ConditionalWriteConflict:
            pass
        except Exception as receipt_failure:
            print(
                "startup failure receipt incomplete: "
                f"{receipt_failure!r}; original failure={failure!r}",
                flush=True,
            )
        print(f"run failed during supervisor startup: {failure!r}", flush=True)
        return 1

    def run(self) -> int:
        try:
            self.validate_runtime()
            self.materialize()
        except BaseException as failure:
            return self._record_startup_failure(failure)
        try:
            holder = self.runtime.holder_id()
            self.lease = self.authority.acquire_lease(
                run_id=self.manifest.run_id,
                attempt_id=self.manifest.attempt_id,
                holder_id=holder,
            )
            self._emit("writer_lease_acquired", holder_id=holder)
            self.last_lease_renewal = self.clock.monotonic()
            self.store.init()
            self.store.reset_interrupted_metric_frames()
            self._recover_durable_state()
            self.recovered_early_stop = (
                self._authoritative_early_stop_receipt(attempt_id=self.manifest.attempt_id)
                or self._prior_early_stop_receipt()
            )
            self._start_wandb()
            provisional_stop = self.eval_admission_closed or self.recovered_early_stop is not None
            final_checkpoint_published = self._has_public_final_checkpoint()
            if self.recovery_mode == "drain-only" and not (
                final_checkpoint_published or provisional_stop
            ):
                raise RuntimeError(
                    "drain-only recovery requires a published final checkpoint "
                    "or a durable provisional stop"
                )
            if final_checkpoint_published and (
                self.recovery_mode == "drain-only" or provisional_stop
            ):
                reason = (
                    "drain-only recovery"
                    if self.recovery_mode == "drain-only"
                    else "provisional-stop recovery"
                )
                print(f"{reason}: learner will not restart", flush=True)
            else:
                self._start_learner()
                if provisional_stop:
                    self._request_finalize_only_stop(
                        "eval_acceptance"
                        if self.eval_admission_closed
                        else "provisional_training_early_stop"
                    )
        except BaseException as failure:
            projector = self.projector
            self.projector = None
            if projector is not None:
                try:
                    self.runtime.close_wandb(
                        projector,
                        timeout_seconds=WANDB_DRAIN_TIMEOUT_SECONDS,
                    )
                except Exception as close_failure:
                    print(
                        "startup W&B cleanup incomplete: "
                        f"{close_failure!r}; original failure={failure!r}",
                        flush=True,
                    )
            return self._record_startup_failure(
                failure,
                phase="startup/recovery",
            )
        learner_exited_at: float | None = self.clock.time() if self.learner is None else None

        def cancel(_signum, _frame) -> None:
            self.cancel_requested = True
            self._request_learner_stop("canceled")

        handler_token = self.runtime.install_cancel_handlers(cancel)
        failure: BaseException | None = None
        promotion: PromotionReceipt | None = None
        early_stop: EarlyStopReceipt | None = None
        training_terminal_reason = ""
        try:
            while self.learner is not None and self.learner.poll() is None:
                self.active_iteration()
                self._observe_live_learner_state(self.clock.monotonic())
                if self.lease_lost:
                    break
                self.clock.sleep(self._liveness_seconds("poll_interval_seconds"))
            if self.learner is not None:
                learner_returncode = self.learner.wait()
                learner_exited_at = self.clock.time()
                self._close_learner_log()
                print(f"learner exited returncode={learner_returncode}", flush=True)
                early_stop = self._resolve_early_stop_receipt()
                terminal_state = self._validate_learner_exit(learner_returncode)
                training_terminal_reason = str(terminal_state.terminal_reason or "")
            else:
                early_stop = self._resolve_early_stop_receipt()
            if self.lease_lost:
                raise LeaseUnavailable("writer lease was lost")
            if self.cancel_requested:
                self._cancel_outstanding_evals()
                raise RuntimeError("run canceled")
            self._publish_state_archive(require_closed=True)
            self._publish_checkpoints()
            self.store.set_state(
                "checkpoint_set_frozen",
                {
                    "checkpoint_ledger_ids": [int(row["id"]) for row in self.store.checkpoints()],
                    "frozen_at": self.clock.utc_now(),
                },
            )
            self._drain()
            assert learner_exited_at is not None
            self.store.append_metrics(
                {
                    ORCHESTRATION_IDLE_GPU_TAIL_SECONDS: max(
                        0.0,
                        self.clock.time() - learner_exited_at,
                    )
                },
                step=max(
                    (int(row.get("step") or 0) for row in self.store.checkpoints()),
                    default=0,
                ),
                source="orchestration:drain",
            )
            self._drain()
            if self.evaluation_required:
                promotion = self._create_promotion()
                if promotion is not None:
                    self._publish_promotion(promotion)
                else:
                    self._validate_no_acceptance_evidence()
        except BaseException as exc:
            failure = exc
            if isinstance(exc, IncompleteEvaluationEvidence):
                self.stop_reason = "evaluation_evidence_incomplete"
            elif isinstance(exc, LearnerOperationalFailure):
                self.stop_reason = exc.stop_reason
            self._request_learner_stop("supervisor_failure")
            if self.learner is not None:
                teardown_failure = self._teardown_learner_group(primary_failure=exc)
                self._close_learner_log()
                if teardown_failure is not None and not isinstance(
                    failure, LearnerOperationalFailure
                ):
                    failure = teardown_failure
                    self.stop_reason = teardown_failure.stop_reason
            if not self.lease_lost:
                try:
                    self._publish_state_archive()
                    self._failure_drain(failure)
                except Exception as drain_exc:
                    print(f"failure drain incomplete: {drain_exc}", flush=True)
        finally:
            self.runtime.restore_cancel_handlers(handler_token)

        if self.lease_lost:
            self.projector = None
            print("run stopped after writer lease loss; no further state was mutated", flush=True)
            return 1
        try:
            wandb_high_water = self._finish_wandb()
        except Exception as exc:
            failure = failure or exc
            wandb_high_water = self._wandb_high_water()
        if failure is None:
            try:
                self._wait_for_remote_delivery(wandb_high_water)
                if promotion is not None:
                    self._wait_for_remote_promotion(promotion)
            except Exception as exc:
                failure = exc
        journal_archive: dict[str, Any] | None = None
        journal_expires_at: str | None = None
        if failure is None:
            try:
                journal_archive = self.authority.archive_metric_journals(
                    run_id=self.manifest.run_id
                )
                journal_expires_at = (
                    (self.clock.utc_datetime() + timedelta(days=METRIC_JOURNAL_RETENTION_DAYS))
                    .isoformat()
                    .replace("+00:00", "Z")
                )
            except Exception as exc:
                failure = exc
        checkpoints, evals = self._terminal_inventory()
        durable_checkpoint_step = max((int(row["step"]) for row in checkpoints), default=0)
        final_step = (
            int(self.learner_final_step)
            if self.learner_final_step is not None
            else durable_checkpoint_step
        )
        state, default_stop_reason = _terminal_outcome(
            cancel_requested=self.cancel_requested,
            failure=failure,
            evaluation_required=self.evaluation_required,
            promotion=promotion,
            early_stop=early_stop,
        )
        stop_reason = self.stop_reason or default_stop_reason
        if stop_reason == "training_cap_complete" and training_terminal_reason in {
            TerminalReason.FIRST_COMPLETION.value,
            TerminalReason.TRAINING_ACCEPTANCE.value,
        }:
            stop_reason = training_terminal_reason
        receipt = TerminalReceipt(
            run_id=self.manifest.run_id,
            attempt_id=self.manifest.attempt_id,
            state=state,  # type: ignore[arg-type]
            acceptance_required=self.evaluation_required,
            stop_reason=stop_reason,
            final_step=final_step,
            checkpoint_inventory=checkpoints,
            eval_inventory=evals,
            wandb_high_water_mark=wandb_high_water,
            drain={
                "complete": failure is None,
                "metric_segment_high_water": self.store.metric_segment_high_water(),
                "eval_terminal_count": self.store.terminal_eval_count(),
                "eval_deferred_count": self.store.deferred_eval_count(),
                "journal_archive": journal_archive,
                "journal_expires_at": journal_expires_at,
                "wandb_remote_high_water_mark": self.wandb_remote_high_water,
                "publication_capacity_ratio": (
                    self.peak_publish_capacity / self.peak_ingress_rate
                    if self.peak_ingress_rate > 0.0
                    else None
                ),
                "failure": (
                    _bounded_exception_document(failure) if failure is not None else None
                ),
                "learner_terminal": self.learner_terminal_document,
                "learner_teardown": self.learner_teardown_evidence or None,
                "learner_log": self._learner_log_evidence(),
            },
            completed_at=self.clock.utc_now(),
            early_stop=(early_stop.to_dict() if early_stop is not None else None),
            state_archive=self.state_archive_publication,
        )
        self.authority.create_attempt_terminal(
            receipt,
            metrics=self.store.latest_metrics(),
        )
        self._emit(
            "attempt_terminal_created",
            state=receipt.state,
            stop_reason=receipt.stop_reason,
            final_step=receipt.final_step,
        )
        try:
            self.runtime.publish_terminal(
                self.train_config,
                receipt,
                timeout_seconds=WANDB_DRAIN_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            print(f"W&B terminal summary projection failed: {exc!r}", flush=True)
        if failure is not None:
            print(f"run failed: {failure!r}", flush=True)
            return 1
        if self.evaluation_required and state == "succeeded":
            self.authority.create_terminal(receipt)
        if state == "failed":
            print(
                f"run completed without acceptance: run_id={self.manifest.run_id} "
                f"final_step={final_step} dstack={DSTACK_VERSION}",
                flush=True,
            )
            return 0
        print(
            f"{'run accepted' if self.evaluation_required else 'training-only run completed'}: "
            f"run_id={self.manifest.run_id} "
            f"final_step={final_step} dstack={DSTACK_VERSION}",
            flush=True,
        )
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Supervise one immutable dstack gradlab training run."
    )
    parser.add_argument("--manifest-uri", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return RunSupervisor(manifest_uri=args.manifest_uri).run()


if __name__ == "__main__":
    raise SystemExit(main())
