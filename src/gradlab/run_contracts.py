from __future__ import annotations

import math
import re
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from gradlab.json_utils import (
    canonical_json_bytes,
    canonical_json_sha256 as document_sha256,
)


SCHEMA_VERSION = 2
LEGACY_RUN_MANIFEST_SCHEMA_VERSION = 3
RUN_MANIFEST_SCHEMA_VERSION = 4
RUN_ID_PATTERN = re.compile(r"^gradlab-[0-9a-f]{32}$")
ATTEMPT_ID_PATTERN = re.compile(r"^attempt-[0-9a-f]{16}$")
RUN_TERMINAL_STATE_SUMMARY = "gradlab/run/terminal_state"
RUN_STOP_REASON_SUMMARY = "gradlab/run/stop_reason"
RUN_FINAL_STEP_SUMMARY = "gradlab/run/final_step"
RUN_EARLY_STOP_TRIGGER_SUMMARY = "gradlab/run/early_stop_trigger"
RUN_EARLY_STOP_CONDITION_SUMMARY = "gradlab/run/early_stop_condition"
canonical_json = canonical_json_bytes
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
EVAL_RESULT_TERMINAL_STATUSES = frozenset(
    {"accepted", "rejected", "failed", "expired", "canceled"}
)
EVAL_INVENTORY_SETTLED_STATUSES = frozenset(
    {*EVAL_RESULT_TERMINAL_STATUSES, "deferred"}
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def new_run_id() -> str:
    return f"gradlab-{secrets.token_hex(16)}"


def new_attempt_id() -> str:
    return f"attempt-{secrets.token_hex(8)}"


def _require_text(value: object, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} must not be empty")
    return text


def _require_sha256(value: object, label: str) -> str:
    digest = _require_text(value, label).lower()
    if SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return digest


def _require_run_id(value: object) -> str:
    run_id = _require_text(value, "run_id")
    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError("run_id must match gradlab-<32 lowercase hex>")
    return run_id


def _require_attempt_id(value: object) -> str:
    attempt_id = _require_text(value, "attempt_id")
    if ATTEMPT_ID_PATTERN.fullmatch(attempt_id) is None:
        raise ValueError("attempt_id must match attempt-<16 lowercase hex>")
    return attempt_id


def checkpoint_id(*, step: int, sha256: str) -> str:
    if int(step) < 0:
        raise ValueError("checkpoint step must be non-negative")
    digest = _require_sha256(sha256, "checkpoint sha256")
    return f"checkpoint-{int(step)}-{digest[:16]}"


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    attempt_id: str
    created_at: str
    source_sha: str
    image_digest: str
    goal_slug: str
    goal_sha256: str
    recipe_slug: str
    recipe_sha256: str
    recipe_overrides: Sequence[str]
    environment_sha256: str
    seed: int
    run_description: str
    compute: Mapping[str, Any]
    wandb: Mapping[str, Any]
    modal: Mapping[str, Any]
    storage: Mapping[str, Any]
    goal_variant: Mapping[str, Any] | None = None
    schema_version: int = RUN_MANIFEST_SCHEMA_VERSION

    def validate(self) -> None:
        _require_run_id(self.run_id)
        _require_attempt_id(self.attempt_id)
        _require_text(self.created_at, "created_at")
        source_sha = _require_text(self.source_sha, "source_sha")
        if re.fullmatch(r"[0-9a-f]{40}", source_sha) is None:
            raise ValueError("source_sha must be a full lowercase Git SHA")
        image = _require_text(self.image_digest, "image_digest")
        if re.fullmatch(r"docker:[^\s]+@sha256:[0-9a-f]{64}", image) is None:
            raise ValueError("image_digest must be a docker: immutable image reference")
        _require_text(self.goal_slug, "goal_slug")
        _require_sha256(self.goal_sha256, "goal_sha256")
        _require_text(self.recipe_slug, "recipe_slug")
        _require_sha256(self.recipe_sha256, "recipe_sha256")
        if isinstance(self.recipe_overrides, str | bytes) or not isinstance(
            self.recipe_overrides, Sequence
        ):
            raise ValueError("recipe_overrides must be a sequence")
        if any(not str(value).strip() for value in self.recipe_overrides):
            raise ValueError("recipe_overrides must contain non-empty strings")
        _require_sha256(self.environment_sha256, "environment_sha256")
        if isinstance(self.seed, bool) or int(self.seed) < 0:
            raise ValueError("seed must be a non-negative integer")
        _require_text(self.run_description, "run_description")
        if int(self.schema_version) not in {
            SCHEMA_VERSION,
            LEGACY_RUN_MANIFEST_SCHEMA_VERSION,
            RUN_MANIFEST_SCHEMA_VERSION,
        }:
            raise ValueError(f"unsupported run manifest schema: {self.schema_version}")
        if (
            int(self.schema_version) == RUN_MANIFEST_SCHEMA_VERSION
            and self.goal_variant is None
        ):
            raise ValueError("run manifest v4 requires goal_variant")
        if self.goal_variant is not None:
            from gradlab.goal_variants import validate_goal_variant_descriptor

            descriptor = validate_goal_variant_descriptor(self.goal_variant)
            if descriptor["goal_slug"] != self.goal_slug:
                raise ValueError("goal_variant.goal_slug must equal goal_slug")
            if descriptor["effective_goal_contract_sha256"] != self.goal_sha256:
                raise ValueError("goal_variant effective contract hash must equal goal_sha256")
        for label, value in (
            ("compute", self.compute),
            ("wandb", self.wandb),
            ("modal", self.modal),
            ("storage", self.storage),
        ):
            if not isinstance(value, Mapping):
                raise ValueError(f"{label} must be a mapping")
        request = self.compute.get("request")
        selected = self.compute.get("selected")
        if not isinstance(request, Mapping) or not isinstance(selected, Mapping):
            raise ValueError("compute must contain request and selected mappings")
        for label, value in (("request", request), ("selected", selected)):
            if str(value.get("kind") or "") not in {
                "auto",
                "local",
                "spot",
                "on-demand",
            }:
                raise ValueError(f"compute.{label}.kind is invalid")
            duration = value.get("max_duration_seconds")
            if isinstance(duration, bool) or not isinstance(duration, int) or duration <= 0:
                raise ValueError(f"compute.{label}.max_duration_seconds must be positive")
        _require_text(self.compute.get("dstack_task"), "compute.dstack_task")
        _require_text(
            self.compute.get("runtime_workflow_run_id"),
            "compute.runtime_workflow_run_id",
        )
        _require_sha256(
            self.compute.get("runtime_input_sha256"),
            "compute.runtime_input_sha256",
        )
        runtime_build_source_sha = _require_text(
            self.compute.get("runtime_build_source_sha"),
            "compute.runtime_build_source_sha",
        )
        if re.fullmatch(r"[0-9a-f]{40,64}", runtime_build_source_sha) is None:
            raise ValueError("compute.runtime_build_source_sha must be a full lowercase Git SHA")
        if str(self.wandb.get("run_id") or "") != self.run_id:
            raise ValueError("wandb.run_id must equal run_id")
        _require_text(self.wandb.get("entity"), "wandb.entity")
        _require_text(self.wandb.get("project"), "wandb.project")
        _require_text(self.wandb.get("url"), "wandb.url")
        for field in ("display_name", "group"):
            if field in self.wandb:
                _require_text(self.wandb.get(field), f"wandb.{field}")
        modal_enabled = self.modal.get("enabled")
        if not isinstance(modal_enabled, bool):
            raise ValueError("modal.enabled must be a boolean")
        if modal_enabled:
            _require_text(self.modal.get("environment_name"), "modal.environment_name")
            _require_text(self.modal.get("app_name"), "modal.app_name")
            _require_text(self.modal.get("function_name"), "modal.function_name")
            if str(self.modal.get("deployment_source_sha") or "") != source_sha:
                raise ValueError("modal.deployment_source_sha must equal source_sha")
        rom_asset = self.modal.get("rom_asset_manifest")
        if rom_asset is not None and not isinstance(rom_asset, Mapping):
            raise ValueError("modal.rom_asset_manifest must be a mapping or null")
        locations = [
            _require_text(self.storage.get(name), f"storage.{name}")
            for name in ("control", "evaluation", "models")
        ]
        if len(set(locations)) != len(locations):
            raise ValueError("control, evaluation, and model storage must be distinct")
        _require_text(
            self.storage.get("public_models_base_url"),
            "storage.public_models_base_url",
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class CheckpointManifest:
    run_id: str
    checkpoint_id: str
    step: int
    purpose: Literal["periodic", "final"]
    sha256: str
    size_bytes: int
    public_url: str
    model_document_url: str
    model_document_sha256: str
    recipe_document_url: str
    recipe_document_sha256: str
    goal_sha256: str
    recipe_sha256: str
    environment_sha256: str
    evaluation_contract_sha256: str
    recovery_sidecar_key: str
    created_at: str
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> None:
        _require_run_id(self.run_id)
        digest = _require_sha256(self.sha256, "sha256")
        if self.checkpoint_id != checkpoint_id(step=self.step, sha256=digest):
            raise ValueError("checkpoint_id does not match step and hash")
        if int(self.step) < 0:
            raise ValueError("step must be non-negative")
        if self.purpose not in {"periodic", "final"}:
            raise ValueError("purpose must be periodic or final")
        if int(self.size_bytes) <= 0:
            raise ValueError("size_bytes must be positive")
        _require_text(self.public_url, "public_url")
        _require_text(self.model_document_url, "model_document_url")
        _require_sha256(self.model_document_sha256, "model_document_sha256")
        _require_text(self.recipe_document_url, "recipe_document_url")
        _require_sha256(self.recipe_document_sha256, "recipe_document_sha256")
        _require_sha256(self.goal_sha256, "goal_sha256")
        _require_sha256(self.recipe_sha256, "recipe_sha256")
        _require_sha256(self.environment_sha256, "environment_sha256")
        _require_sha256(
            self.evaluation_contract_sha256,
            "evaluation_contract_sha256",
        )
        _require_text(self.recovery_sidecar_key, "recovery_sidecar_key")
        _require_text(self.created_at, "created_at")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class EvalIntent:
    run_id: str
    checkpoint_id: str
    idempotency_key: str
    checkpoint_sha256: str
    goal_sha256: str
    recipe_sha256: str
    environment_sha256: str
    evaluation_contract_sha256: str
    episode_manifest_sha256: str
    protocol: str
    execution_contract: Mapping[str, Any]
    result_key: str
    timeout_seconds: int
    created_at: str
    expires_at: str
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> None:
        _require_run_id(self.run_id)
        _require_text(self.checkpoint_id, "checkpoint_id")
        _require_sha256(self.idempotency_key, "idempotency_key")
        for label, value in (
            ("checkpoint_sha256", self.checkpoint_sha256),
            ("goal_sha256", self.goal_sha256),
            ("recipe_sha256", self.recipe_sha256),
            ("environment_sha256", self.environment_sha256),
            ("evaluation_contract_sha256", self.evaluation_contract_sha256),
            ("episode_manifest_sha256", self.episode_manifest_sha256),
        ):
            _require_sha256(value, label)
        _require_text(self.protocol, "protocol")
        if not isinstance(self.execution_contract, Mapping):
            raise ValueError("execution_contract must be a mapping")
        _require_text(self.result_key, "result_key")
        if int(self.timeout_seconds) <= 0:
            raise ValueError("timeout_seconds must be positive")
        _require_text(self.created_at, "created_at")
        _require_text(self.expires_at, "expires_at")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def eval_idempotency_key(
    *,
    run_id: str,
    checkpoint_sha256: str,
    evaluation_contract_sha256: str,
    episode_manifest_sha256: str,
    protocol: str,
) -> str:
    return document_sha256(
        {
            "run_id": _require_run_id(run_id),
            "checkpoint_sha256": _require_sha256(
                checkpoint_sha256,
                "checkpoint_sha256",
            ),
            "evaluation_contract_sha256": _require_sha256(
                evaluation_contract_sha256,
                "evaluation_contract_sha256",
            ),
            "episode_manifest_sha256": _require_sha256(
                episode_manifest_sha256,
                "episode_manifest_sha256",
            ),
            "protocol": _require_text(protocol, "protocol"),
        }
    )


@dataclass(frozen=True)
class EvalResult:
    run_id: str
    checkpoint_id: str
    idempotency_key: str
    modal_call_id: str
    status: Literal["accepted", "rejected", "failed", "expired", "canceled"]
    episode_results: Sequence[Mapping[str, Any]]
    aggregates: Mapping[str, Any]
    timings: Mapping[str, Any]
    evidence_sha256: Sequence[str]
    completed_at: str
    error: str | None = None
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> None:
        _require_run_id(self.run_id)
        _require_text(self.checkpoint_id, "checkpoint_id")
        _require_sha256(self.idempotency_key, "idempotency_key")
        _require_text(self.modal_call_id, "modal_call_id")
        if self.status not in {"accepted", "rejected", "failed", "expired", "canceled"}:
            raise ValueError(f"invalid terminal eval status: {self.status}")
        if self.status == "accepted" and not self.episode_results:
            raise ValueError("accepted evaluation must contain episode results")
        for index, digest in enumerate(self.evidence_sha256):
            _require_sha256(digest, f"evidence_sha256[{index}]")
        _require_text(self.completed_at, "completed_at")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class PromotionReceipt:
    run_id: str
    checkpoint_id: str
    checkpoint_step: int
    eval_idempotency_key: str
    eval_result_sha256: str
    accepted_episode_count: int
    promoted_at: str
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> None:
        _require_run_id(self.run_id)
        _require_text(self.checkpoint_id, "checkpoint_id")
        if int(self.checkpoint_step) < 0:
            raise ValueError("checkpoint_step must be non-negative")
        _require_sha256(self.eval_idempotency_key, "eval_idempotency_key")
        _require_sha256(self.eval_result_sha256, "eval_result_sha256")
        if int(self.accepted_episode_count) <= 0:
            raise ValueError("promotion requires accepted episodes")
        _require_text(self.promoted_at, "promoted_at")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class EarlyStopReceipt:
    run_id: str
    attempt_id: str
    condition_id: str
    matched_condition_ids: Sequence[str]
    outcome: Literal["success", "failure"]
    trigger: Literal["threshold", "no_improvement"]
    metric: str
    metric_step: int
    value: float
    best_value: float
    elapsed_steps: int
    patience_progress: float
    condition: Mapping[str, Any]
    early_stop_config_sha256: str
    decision_sha256: str
    recorded_at: str
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> None:
        _require_run_id(self.run_id)
        _require_attempt_id(self.attempt_id)
        condition_id = _require_text(self.condition_id, "condition_id")
        matched = [str(item) for item in self.matched_condition_ids]
        if not matched or matched != sorted(set(matched)) or condition_id not in matched:
            raise ValueError(
                "matched_condition_ids must be sorted, unique, and contain condition_id"
            )
        if self.outcome not in {"success", "failure"}:
            raise ValueError(f"invalid early-stop outcome: {self.outcome}")
        if self.trigger not in {"threshold", "no_improvement"}:
            raise ValueError(f"invalid early-stop trigger: {self.trigger}")
        metric = _require_text(self.metric, "metric")
        if not metric.startswith("train/"):
            raise ValueError("early-stop metric must use the train/* namespace")
        if int(self.metric_step) < 0 or int(self.elapsed_steps) < 0:
            raise ValueError("early-stop step fields must be non-negative")
        for label, value in (
            ("value", self.value),
            ("best_value", self.best_value),
            ("patience_progress", self.patience_progress),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"{label} must be finite")
        if not math.isclose(float(self.patience_progress), 1.0):
            raise ValueError("patience_progress must be one for an early-stop receipt")
        if not isinstance(self.condition, Mapping):
            raise ValueError("condition must be an object")
        _require_sha256(self.early_stop_config_sha256, "early_stop_config_sha256")
        _require_sha256(self.decision_sha256, "decision_sha256")
        _require_text(self.recorded_at, "recorded_at")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class TerminalReceipt:
    run_id: str
    attempt_id: str
    state: Literal["succeeded", "failed", "canceled", "interrupted", "resumable_failure"]
    acceptance_required: bool
    stop_reason: str
    final_step: int
    checkpoint_inventory: Sequence[Mapping[str, Any]]
    eval_inventory: Sequence[Mapping[str, Any]]
    wandb_high_water_mark: int
    drain: Mapping[str, Any]
    completed_at: str
    early_stop: Mapping[str, Any] | None = None
    state_archive: Mapping[str, Any] | None = None
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> None:
        _require_run_id(self.run_id)
        _require_attempt_id(self.attempt_id)
        if self.state not in {
            "succeeded",
            "failed",
            "canceled",
            "interrupted",
            "resumable_failure",
        }:
            raise ValueError(f"invalid terminal state: {self.state}")
        if not isinstance(self.acceptance_required, bool):
            raise ValueError("acceptance_required must be a boolean")
        _require_text(self.stop_reason, "stop_reason")
        if int(self.final_step) < 0:
            raise ValueError("final_step must be non-negative")
        if int(self.wandb_high_water_mark) < 0:
            raise ValueError("wandb_high_water_mark must be non-negative")
        for row in self.eval_inventory:
            if not isinstance(row, Mapping):
                raise ValueError("eval inventory entries must be objects")
            status = str(row.get("status") or "")
            if status not in EVAL_INVENTORY_SETTLED_STATUSES:
                raise ValueError(f"eval inventory contains unsettled status: {status}")
        if self.early_stop is not None:
            EarlyStopReceipt(**self.early_stop).validate()
        if self.state_archive is not None:
            if not isinstance(self.state_archive, Mapping):
                raise ValueError("state_archive terminal evidence must be an object")
            archive = self.state_archive
            if archive.get("semantic_id") != "state-archive-publication-v1":
                raise ValueError("state_archive terminal evidence semantic_id is unsupported")
            if int(archive.get("schema_version", 0)) != 1:
                raise ValueError("state_archive terminal evidence schema_version is unsupported")
            if str(archive.get("run_id") or "") != self.run_id:
                raise ValueError("state_archive terminal evidence run_id mismatch")
            _require_attempt_id(archive.get("attempt_id"))
            _require_sha256(
                archive.get("generation_sha256"),
                "state_archive generation_sha256",
            )
            _require_sha256(
                archive.get("inventory_sha256"),
                "state_archive inventory_sha256",
            )
            _require_text(
                archive.get("generation_key"),
                "state_archive generation_key",
            )
            if int(archive.get("step", -1)) < 0:
                raise ValueError("state_archive terminal evidence step must be non-negative")
            if int(archive.get("file_count", 0)) < 1:
                raise ValueError("state_archive terminal evidence must contain files")
            if int(archive.get("size_bytes", 0)) < 1:
                raise ValueError("state_archive terminal evidence must contain bytes")
            if self.state == "succeeded":
                if archive.get("status") != "closed":
                    raise ValueError("successful terminal state_archive must be closed")
                if int(archive["step"]) != int(self.final_step):
                    raise ValueError("successful terminal state_archive step must match final_step")
        _require_text(self.completed_at, "completed_at")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)
