from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from gradlab.checkpoint_acceptance import manifest_index, requires_complete_evaluation
from gradlab.clock import Clock, SystemClock
from gradlab.early_stop import EARLY_STOP_OPERATORS
from gradlab.eval_backend import EvalBackend
from gradlab.eval_metrics import eval_by_start_rows
from gradlab.evaluation_projection import (
    evaluation_wandb_projection,
    metrics_schema_version_from_recipe_document,
)
from gradlab.job_queue import (
    HandlerResult,
    JobStore,
    JobSubject,
    SubjectUpdate,
    WorkerStart,
    ensure_flusher,
    register_handler,
)
from gradlab.metric_names import (
    EVAL_FULL_EPISODE_RETURN_SHAPED_MEAN,
    EVAL_FULL_SUCCESS_ACROSS_STARTS_RATE_MIN,
)
from gradlab.modal_eval_backend import ModalEvalBackend
from gradlab.modal_eval_config import ModalEvalConfig, load_modal_eval_config
from gradlab.modal_eval_protocol import (
    PROTOCOL_SCHEMA_VERSION,
    execution_key,
    validate_attempt_result,
)
from gradlab.operator_environment import load_repository_operator_environment
from gradlab.policy_bundle import (
    evaluation_contract,
    evaluation_contract_sha256,
)
from gradlab.r2_store import ConditionalWriteConflict, RunStorageConfig
from gradlab.run_authority import Lease, LeaseUnavailable, RunAuthority
from gradlab.run_contracts import (
    CheckpointManifest,
    EvalIntent,
    EvalResult,
    PromotionReceipt,
    RunManifest,
    TerminalReceipt,
    document_sha256,
    eval_idempotency_key,
)
from gradlab.runtime_refs import RuntimeImageInfo, modal_readiness_for_release
from gradlab.supervisor_ledger import SupervisorLedger
from gradlab.supervisor_runtime import SupervisorRuntime


MANUAL_EVAL_PROTOCOL = "modal-acceptance-v4"
MAX_MANUAL_EVAL_SELECTION = 100
MANUAL_EVAL_JOB_TYPE = "evaluate-checkpoints"
MANUAL_EVAL_JOB_VERSION = 3
MANUAL_EVAL_RETRY_SECONDS = 2.0
MANUAL_EVAL_WAIT_SECONDS = 15.0

_STRICT_COMPLETE_ACCEPTANCE_METRIC_BY_GAME = {
    "VizdoomBasic-v1": EVAL_FULL_SUCCESS_ACROSS_STARTS_RATE_MIN,
    "VizdoomBasic-Plus-v1": EVAL_FULL_SUCCESS_ACROSS_STARTS_RATE_MIN,
    "VizdoomDeadlyCorridor-v1": EVAL_FULL_EPISODE_RETURN_SHAPED_MEAN,
    "VizdoomDeathmatch-v1": EVAL_FULL_EPISODE_RETURN_SHAPED_MEAN,
    "VizdoomDefendLine-v1": EVAL_FULL_EPISODE_RETURN_SHAPED_MEAN,
    "VizdoomDefendLine-Plus-v1": EVAL_FULL_EPISODE_RETURN_SHAPED_MEAN,
}


class EvaluationContractIneligible(ValueError):
    pass


class EvaluationProjectionPending(RuntimeError):
    pass


@dataclass(frozen=True)
class _EvaluationContext:
    manifest: RunManifest
    checkpoint: CheckpointManifest
    recipe_document: Mapping[str, Any]
    intent: EvalIntent


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _checkpoint_object_prefix(checkpoint: CheckpointManifest) -> str:
    return (
        f"runs/{checkpoint.run_id}/checkpoints/"
        f"{checkpoint.step}-{checkpoint.sha256}"
    )


def _evaluation_subject_type(run_id: str) -> str:
    return f"checkpoint-evaluation:{run_id}"


def _verified_result(
    *,
    context: _EvaluationContext,
    raw: Mapping[str, Any],
    attempt: int,
    modal_call_id: str,
    clock: Clock,
) -> EvalResult:
    intent = context.intent
    contract = dict(intent.execution_contract)
    attempt_id = f"{intent.idempotency_key[:20]}-a{attempt}"
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
    return EvalResult(
        run_id=context.manifest.run_id,
        checkpoint_id=context.checkpoint.checkpoint_id,
        idempotency_key=intent.idempotency_key,
        modal_call_id=modal_call_id or "not-recorded",
        status=status,  # type: ignore[arg-type]
        episode_results=episodes,
        aggregates=aggregates,
        timings={
            "duration_seconds": float(raw.get("duration_seconds") or 0.0),
            "result_observed_at": clock.utc_now(),
        },
        evidence_sha256=[
            document_sha256({"evidence": value})
            for value in evidence_values
            if value not in (None, {}, [])
        ],
        completed_at=clock.utc_now(),
        error=error,
    )


def _evaluation_summary(
    context: _EvaluationContext,
    result: EvalResult,
) -> dict[str, Any]:
    criteria = []
    for rule in context.intent.execution_contract.get("acceptance") or ():
        metric = str(rule["metric"])
        value = result.aggregates.get(metric)
        numeric = (
            float(value)
            if not isinstance(value, bool) and isinstance(value, int | float)
            else None
        )
        criteria.append(
            {
                "metric": metric,
                "operator": str(rule["operator"]),
                "threshold": float(rule["threshold"]),
                "value": numeric,
                "passed": (
                    None
                    if numeric is None
                    else bool(
                        EARLY_STOP_OPERATORS[str(rule["operator"])](
                            numeric,
                            float(rule["threshold"]),
                        )
                    )
                ),
            }
        )
    return {
        "status": result.status,
        "pass": result.status == "accepted",
        "episodes_planned": int(context.intent.execution_contract["episodes"]),
        "episodes_completed": len(result.episode_results),
        "failure_count": int(result.aggregates.get("failure_count") or 0),
        "criteria": criteria,
        "manual": True,
    }


class ManualEvaluationSupervisor:
    """Advance a post-training evaluation batch under one run writer lease."""

    def __init__(
        self,
        *,
        authority: RunAuthority,
        repo_root: Path,
        clock: Clock | None = None,
        backend_factory: Callable[[RunManifest], EvalBackend] | None = None,
        project_results: bool = True,
        holder_id: str | None = None,
        work_root: Path | None = None,
        runtime: SupervisorRuntime | None = None,
    ) -> None:
        self.authority = authority
        self.repo_root = Path(repo_root).resolve()
        self.clock = clock or SystemClock()
        self.modal_config: ModalEvalConfig = load_modal_eval_config(
            self.repo_root / "experiments" / "modal_eval.yaml"
        )
        self.backend_factory = backend_factory or self._modal_backend
        self.project_results = bool(project_results)
        self.runtime = runtime or SupervisorRuntime(clock=self.clock)
        self._backends: dict[str, EvalBackend] = {}
        self._leases: dict[str, Lease] = {}
        self._holder_id = holder_id or f"manual-eval-{uuid.uuid4().hex}"
        self.work_root = (
            Path(work_root).resolve()
            if work_root is not None
            else self.repo_root / "runs" / "manual-evaluation"
        )
        self.work_root.mkdir(parents=True, exist_ok=True)

    def _modal_backend(self, manifest: RunManifest) -> EvalBackend:
        app_name = str(manifest.modal.get("app_name") or "").strip()
        if not bool(manifest.modal.get("enabled")):
            readiness = modal_readiness_for_release(
                RuntimeImageInfo(
                    runtime_image_ref=manifest.image_digest,
                    source_sha=manifest.source_sha,
                    commit_message="",
                    published_at="",
                    workflow_run_id=str(
                        manifest.compute.get("runtime_workflow_run_id") or ""
                    ),
                    runtime_input_sha256=str(
                        manifest.compute.get("runtime_input_sha256") or ""
                    ),
                    runtime_build_source_sha=str(
                        manifest.compute.get("runtime_build_source_sha") or ""
                    ),
                ),
                require_current_contract=False,
            )
            app_name = readiness.modal_app_name
        return ModalEvalBackend(
            app_name=app_name,
            function_name=str(manifest.modal.get("function_name") or "evaluate_checkpoint"),
            environment_name=str(manifest.modal.get("environment_name") or "gradlab-eval"),
        )

    def _backend(self, manifest: RunManifest) -> EvalBackend:
        key = manifest.run_id
        backend = self._backends.get(key)
        if backend is None:
            backend = self.backend_factory(manifest)
            self._backends[key] = backend
        return backend

    def _manifest(self, run_id: str) -> RunManifest:
        documents: list[Mapping[str, Any]] = []
        root_document = self.authority.manifest(run_id)
        if root_document is not None:
            documents.append(root_document)
        for key in self.authority.control.iter_keys(f"runs/{run_id}/attempts"):
            if key.endswith("/manifest.json"):
                documents.append(self.authority.control.get_json(key))
        if not documents:
            raise ValueError(f"run manifest was not found: {run_id}")
        manifests = [RunManifest(**document) for document in documents]
        for manifest in manifests:
            manifest.validate()
            if manifest.run_id != run_id:
                raise ValueError("attempt manifest belongs to another run")
        manifest = max(
            manifests,
            key=lambda item: (item.created_at, item.attempt_id),
        )
        return manifest

    def _checkpoint_map(self, run_id: str) -> dict[str, CheckpointManifest]:
        index = self.authority.models.get_json(f"runs/{run_id}/index.json")
        if int(index.get("schema_version") or 0) != 1 or index.get("run_id") != run_id:
            raise ValueError("public checkpoint index identity or schema mismatch")
        result: dict[str, CheckpointManifest] = {}
        for raw in index.get("checkpoints") or ():
            if not isinstance(raw, Mapping):
                raise ValueError("public checkpoint index contains an invalid checkpoint")
            checkpoint = CheckpointManifest(**dict(raw))
            checkpoint.validate()
            if checkpoint.run_id != run_id:
                raise ValueError("public checkpoint belongs to another run")
            result[checkpoint.checkpoint_id] = checkpoint
        return result

    def _recipe_document(self, checkpoint: CheckpointManifest) -> dict[str, Any]:
        key = f"{_checkpoint_object_prefix(checkpoint)}/recipe.json"
        encoded = self.authority.models.get_bytes(key)
        if hashlib.sha256(encoded).hexdigest() != checkpoint.recipe_document_sha256:
            raise ValueError(f"checkpoint recipe document hash mismatch: {checkpoint.checkpoint_id}")
        document = json.loads(encoded)
        if not isinstance(document, dict):
            raise ValueError("checkpoint recipe document must be a mapping")
        return document

    def _context(
        self,
        *,
        manifest: RunManifest,
        checkpoint: CheckpointManifest,
        enforce_current_protocol: bool = True,
    ) -> _EvaluationContext:
        if (
            checkpoint.goal_sha256 != manifest.goal_sha256
            or checkpoint.recipe_sha256 != manifest.recipe_sha256
            or checkpoint.environment_sha256 != manifest.environment_sha256
        ):
            raise ValueError(
                f"checkpoint provenance does not match its run: {checkpoint.checkpoint_id}"
            )
        recipe_document = self._recipe_document(checkpoint)
        contract = evaluation_contract(recipe_document)
        contract_sha256 = evaluation_contract_sha256(recipe_document)
        if contract_sha256 != checkpoint.evaluation_contract_sha256:
            raise ValueError(
                f"checkpoint evaluation contract hash mismatch: "
                f"{checkpoint.checkpoint_id}"
            )
        contract.update(
            {
                "schema_version": PROTOCOL_SCHEMA_VERSION,
                "checkpoint_sha256": checkpoint.sha256,
                "runtime_image_ref": manifest.image_digest,
                "recipe_sha256": checkpoint.recipe_document_sha256,
                "recipe_format_version": int(recipe_document["format_version"]),
                "evaluation_contract_sha256": contract_sha256,
            }
        )
        asset = contract.get("asset")
        if isinstance(asset, Mapping):
            contract["asset"] = {
                str(key): value for key, value in asset.items() if str(key) != "object_uri"
            }
        contract = self._current_protocol_contract(contract)
        manifest_index(contract)
        if enforce_current_protocol:
            self._validate_current_protocol(contract)
        episode_manifest_sha = document_sha256(contract["manifest"])
        key = eval_idempotency_key(
            run_id=manifest.run_id,
            checkpoint_sha256=checkpoint.sha256,
            evaluation_contract_sha256=contract_sha256,
            episode_manifest_sha256=episode_manifest_sha,
            protocol=MANUAL_EVAL_PROTOCOL,
        )
        now = self.clock.utc_datetime().astimezone(UTC)
        timeout = int(self.modal_config.timeouts.acceptance_seconds)
        intent = EvalIntent(
            run_id=manifest.run_id,
            checkpoint_id=checkpoint.checkpoint_id,
            idempotency_key=key,
            checkpoint_sha256=checkpoint.sha256,
            goal_sha256=manifest.goal_sha256,
            recipe_sha256=manifest.recipe_sha256,
            environment_sha256=manifest.environment_sha256,
            evaluation_contract_sha256=contract_sha256,
            episode_manifest_sha256=episode_manifest_sha,
            protocol=MANUAL_EVAL_PROTOCOL,
            execution_contract=contract,
            result_key=f"runs/{manifest.run_id}/evals/{key}/result.json",
            timeout_seconds=timeout,
            created_at=now.isoformat().replace("+00:00", "Z"),
            expires_at=(now + timedelta(seconds=timeout)).isoformat().replace("+00:00", "Z"),
        )
        existing = self.authority.eval_intent(
            run_id=manifest.run_id,
            idempotency_key=key,
        )
        if existing is not None:
            # Creation timestamps describe the first admission and do not affect
            # scientific identity. All hash-bound fields must remain identical.
            expected = intent.to_dict()
            for timestamp in ("created_at", "expires_at"):
                expected[timestamp] = existing.get(timestamp)
            if existing != expected:
                raise ValueError(
                    f"evaluation intent conflicts with checkpoint {checkpoint.checkpoint_id}"
                )
            intent = EvalIntent(**existing)
            intent.validate()
        return _EvaluationContext(
            manifest=manifest,
            checkpoint=checkpoint,
            recipe_document=recipe_document,
            intent=intent,
        )

    def _contexts(
        self,
        run_id: str,
        checkpoint_ids: Sequence[str],
        *,
        enforce_current_protocol: bool = True,
    ) -> list[_EvaluationContext]:
        identifiers = tuple(dict.fromkeys(str(value).strip() for value in checkpoint_ids))
        if not identifiers or any(not value for value in identifiers):
            raise ValueError("select at least one checkpoint")
        if len(identifiers) > MAX_MANUAL_EVAL_SELECTION:
            raise ValueError(
                f"select at most {MAX_MANUAL_EVAL_SELECTION} checkpoints per request"
            )
        manifest = self._manifest(run_id)
        checkpoints = self._checkpoint_map(run_id)
        missing = [identifier for identifier in identifiers if identifier not in checkpoints]
        if missing:
            raise ValueError(f"unknown checkpoint: {missing[0]}")
        return [
            self._context(
                manifest=manifest,
                checkpoint=checkpoints[identifier],
                enforce_current_protocol=enforce_current_protocol,
            )
            for identifier in identifiers
        ]

    @staticmethod
    def _current_protocol_contract(
        contract: Mapping[str, Any],
    ) -> dict[str, Any]:
        normalized = dict(contract)
        environment = normalized.get("environment")
        if not requires_complete_evaluation(environment):
            return normalized
        policy_value = normalized.get("evidence_policy")
        policy = dict(policy_value) if isinstance(policy_value, Mapping) else {}
        policy["fail_fast"] = "disabled"
        policy["partial_rejection_metrics"] = False
        normalized["evidence_policy"] = policy
        return normalized

    @staticmethod
    def _validate_current_protocol(contract: Mapping[str, Any]) -> None:
        environment = contract.get("environment")
        game = str(environment.get("game") or "") if isinstance(environment, Mapping) else ""
        expected_metric = _STRICT_COMPLETE_ACCEPTANCE_METRIC_BY_GAME.get(game)
        if expected_metric is None:
            return
        if int(contract.get("episodes") or 0) != 100:
            raise EvaluationContractIneligible(
                f"{game} manual evaluation requires exactly 100 episodes"
            )
        policy = contract.get("evidence_policy")
        fail_fast = str(policy.get("fail_fast") or "") if isinstance(policy, Mapping) else ""
        if fail_fast != "disabled":
            raise EvaluationContractIneligible(
                f"{game} manual evaluation requires fail-fast to be disabled"
            )
        acceptance = contract.get("acceptance")
        if (
            not isinstance(acceptance, list)
            or len(acceptance) != 1
            or not isinstance(acceptance[0], Mapping)
            or str(acceptance[0].get("metric") or "") != expected_metric
        ):
            raise EvaluationContractIneligible(
                f"{game} acceptance must use only {expected_metric} over all 100 episodes"
            )

    def _training_terminal(self, manifest: RunManifest) -> Mapping[str, Any] | None:
        document = self.authority.control.get_json_optional(
            (
                f"runs/{manifest.run_id}/attempts/"
                f"{manifest.attempt_id}/terminal.json"
            )
        )
        if document is None:
            return None
        receipt = TerminalReceipt(**document)
        receipt.validate()
        if receipt.run_id != manifest.run_id or receipt.attempt_id != manifest.attempt_id:
            raise ValueError("training terminal receipt identity does not match the run")
        if receipt.state in {"interrupted", "resumable_failure"}:
            return None
        return document

    def _latest_attempt(
        self,
        context: _EvaluationContext,
    ) -> tuple[int, Mapping[str, Any] | None, Mapping[str, Any] | None]:
        selected_attempt = 0
        selected_prepared = None
        selected_dispatch = None
        for attempt in range(1, int(self.modal_config.protocol.max_attempts) + 1):
            prepared = self.authority.eval_attempt(
                run_id=context.manifest.run_id,
                idempotency_key=context.intent.idempotency_key,
                attempt=attempt,
            )
            if prepared is None:
                continue
            selected_attempt = attempt
            selected_prepared = prepared
            selected_dispatch = self.authority.eval_dispatch(
                run_id=context.manifest.run_id,
                idempotency_key=context.intent.idempotency_key,
                attempt=attempt,
            )
        return selected_attempt, selected_prepared, selected_dispatch

    def _payload(
        self,
        context: _EvaluationContext,
        *,
        attempt: int,
        expires_at: float,
    ) -> dict[str, Any]:
        checkpoint = context.checkpoint
        timeout = int(context.intent.timeout_seconds)
        payload: dict[str, Any] = {
            "attempt_id": f"{context.intent.idempotency_key[:20]}-a{attempt}",
            "contract": dict(context.intent.execution_contract),
            "expires_at": expires_at,
            "child_timeout_seconds": max(
                1,
                timeout - int(self.modal_config.timeouts.child_margin_seconds),
            ),
            "model_get_url": checkpoint.public_url,
            "model_document_get_url": checkpoint.model_document_url,
            "model_document_sha256": checkpoint.model_document_sha256,
            "recipe_get_url": checkpoint.recipe_document_url,
            "result_uri": self.authority.evaluation.uri(context.intent.result_key),
            "result_put_url": self.authority.evaluation.presign_put(
                context.intent.result_key,
                expires_seconds=(
                    timeout + int(self.modal_config.timeouts.expiry_margin_seconds)
                ),
            ),
        }
        asset = context.manifest.modal.get("rom_asset_manifest")
        if isinstance(asset, Mapping):
            rom_key = self.authority.evaluation.key_from_uri(str(asset["object_uri"]))
            payload["rom_get_url"] = self.authority.evaluation.presign_get(
                rom_key,
                expires_seconds=(
                    timeout + int(self.modal_config.timeouts.expiry_margin_seconds)
                ),
            )
        return payload

    def _projection_key(self, context: _EvaluationContext) -> str:
        return (
            f"runs/{context.manifest.run_id}/manual-evals/"
            f"{context.intent.idempotency_key}/wandb-projection.json"
        )

    def _request_key(self, context: _EvaluationContext) -> str:
        return (
            f"runs/{context.manifest.run_id}/manual-evals/"
            f"{context.intent.idempotency_key}/request.json"
        )

    def _event_seq_offset(
        self,
        manifest: RunManifest,
        training_terminal: Mapping[str, Any],
    ) -> int:
        high_water = int(training_terminal.get("wandb_high_water_mark") or 0)
        for key in self.authority.control.iter_keys(
            f"runs/{manifest.run_id}/manual-evals"
        ):
            if not (
                key.endswith("/wandb-projection.json")
                or key.endswith("/terminal.json")
            ):
                continue
            document = self.authority.control.get_json(key)
            if str(document.get("run_id") or "") != manifest.run_id:
                raise ValueError("manual evaluation projection belongs to another run")
            high_water = max(
                high_water,
                int(document.get("wandb_high_water_mark") or 0),
            )
        return high_water

    @staticmethod
    def _wandb_run_path(manifest: RunManifest) -> str:
        return (
            f"{manifest.wandb['entity']}/"
            f"{manifest.wandb['project']}/"
            f"{manifest.run_id}"
        )

    def _lease(self, manifest: RunManifest) -> Lease:
        existing = self._leases.get(manifest.run_id)
        now = self.clock.utc_datetime().astimezone(UTC)
        if existing is not None and _parse_timestamp(existing.expires_at) > now:
            existing = self.authority.renew_lease(existing, now=now)
        else:
            existing = self.authority.acquire_lease(
                run_id=manifest.run_id,
                attempt_id=manifest.attempt_id,
                holder_id=self._holder_id,
                now=now,
            )
        self._leases[manifest.run_id] = existing
        return existing

    def _project_result(
        self,
        context: _EvaluationContext,
        result: EvalResult,
        raw: Mapping[str, Any],
        *,
        ledger: SupervisorLedger,
        event_seq_offset: int,
    ) -> bool:
        if not self.project_results:
            return True
        projection_key = self._projection_key(context)
        if self.authority.control.get_json_optional(projection_key) is not None:
            return True
        self._lease(context.manifest)
        environment = context.intent.execution_contract["environment"]
        metrics_schema_version = metrics_schema_version_from_recipe_document(
            context.recipe_document
        )
        metrics = evaluation_wandb_projection(
            dict(raw.get("metrics") or {}),
            schema_version=metrics_schema_version,
            checkpoint_step=context.checkpoint.step,
            accepted=result.status == "accepted",
            episodes_planned=int(context.intent.execution_contract["episodes"]),
            episodes_completed=len(result.episode_results),
            duration_seconds=float(raw.get("duration_seconds") or 0.0),
        )
        ledger.append_metrics(
            metrics,
            step=context.checkpoint.step,
            source=f"eval:manual:{context.intent.idempotency_key}",
            metrics_schema_version=metrics_schema_version,
        )
        if result.status == "accepted":
            ledger.enqueue_event(
                kind="eval_by_start",
                payload={
                    "rows": eval_by_start_rows(
                        [dict(episode) for episode in result.episode_results]
                    )
                },
                step=context.checkpoint.step,
                source=f"eval:manual:{context.intent.idempotency_key}:by_start",
            )
        ledger.reset_interrupted_metric_frames()
        projector = self.runtime.resume_wandb(
            {
                "wandb_run_id": context.manifest.run_id,
                "wandb_entity": context.manifest.wandb["entity"],
                "wandb_project": context.manifest.wandb["project"],
                "wandb_mode": "online",
                "game": environment["game"],
                "env_provider": environment["env_provider"],
                "run_name": context.manifest.wandb.get("display_name"),
                "wandb_group": context.manifest.wandb.get("group"),
                "metrics_schema_version": metrics_schema_version,
            },
            allow_create=False,
            update_finish_state=False,
        )
        try:
            while ledger.pending_metric_frames(limit=1):
                if (
                    self.runtime.publish_frames(
                        ledger,
                        projector,
                        limit=100,
                        event_seq_offset=event_seq_offset,
                    )
                    <= 0
                ):
                    return False
        finally:
            self.runtime.close_wandb(projector, timeout_seconds=300)
        with ledger.connection() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(id), 0) FROM metric_frames"
            ).fetchone()
        high_water = event_seq_offset + int(row[0] if row else 0)
        remote = self.runtime.remote_summary(self._wandb_run_path(context.manifest))
        if int(remote.get("orchestration/event_seq") or 0) < high_water:
            return False
        self.authority.control.put_json(
            projection_key,
            {
                "schema_version": 1,
                "run_id": context.manifest.run_id,
                "checkpoint_id": context.checkpoint.checkpoint_id,
                "idempotency_key": context.intent.idempotency_key,
                "status": result.status,
                "wandb_high_water_mark": high_water,
                "projected_at": self.clock.utc_now(),
            },
            create_only=True,
        )
        return True

    def _promote(
        self,
        context: _EvaluationContext,
        result: EvalResult,
    ) -> None:
        key = f"runs/{context.manifest.run_id}/promotion.json"
        existing = self.authority.control.get_json_optional(key)
        if existing is not None:
            if int(existing.get("checkpoint_step") or 0) > context.checkpoint.step:
                raise ValueError(
                    "manual evaluation would invalidate the immutable lowest-step promotion"
                )
            return
        receipt = PromotionReceipt(
            run_id=context.manifest.run_id,
            checkpoint_id=context.checkpoint.checkpoint_id,
            checkpoint_step=context.checkpoint.step,
            eval_idempotency_key=result.idempotency_key,
            eval_result_sha256=document_sha256(result.to_dict()),
            accepted_episode_count=len(result.episode_results),
            promoted_at=self.clock.utc_now(),
        )
        self.authority.create_promotion(receipt)

    def _ensure_promotion_projection(
        self,
        context: _EvaluationContext,
        result: EvalResult,
        raw: Mapping[str, Any],
        receipt: PromotionReceipt,
    ) -> None:
        if not self.project_results:
            return
        run_path = self._wandb_run_path(context.manifest)
        try:
            remote = self.runtime.remote_summary(run_path)
            if (
                str(remote.get("gradlab/goal/outcome") or "") == "accepted"
                and int(remote.get("leader/checkpoint/step") or -1)
                == receipt.checkpoint_step
            ):
                return
        except Exception:
            pass
        metrics = {
            **dict(raw.get("metrics") or {}),
            **dict(result.aggregates),
        }
        metrics_schema_version = metrics_schema_version_from_recipe_document(
            context.recipe_document
        )
        recipe = context.recipe_document.get("recipe")
        train_config = recipe.get("train_config") if isinstance(recipe, Mapping) else None
        if not isinstance(train_config, Mapping):
            raise EvaluationProjectionPending(
                "checkpoint recipe has no train_config for promotion ranking"
            )
        try:
            environment = context.intent.execution_contract["environment"]
            projector = self.runtime.resume_wandb(
                {
                    "wandb_run_id": context.manifest.run_id,
                    "wandb_entity": context.manifest.wandb["entity"],
                    "wandb_project": context.manifest.wandb["project"],
                    "wandb_mode": "online",
                    "game": environment["game"],
                    "env_provider": environment["env_provider"],
                    "run_name": context.manifest.wandb.get("display_name"),
                    "wandb_group": context.manifest.wandb.get("group"),
                    "metrics_schema_version": metrics_schema_version,
                },
                allow_create=False,
                update_finish_state=False,
            )
            try:
                self.runtime.publish_promotion(
                    projector,
                    checkpoint_step=receipt.checkpoint_step,
                    checkpoint_url=context.checkpoint.public_url,
                    metrics=metrics,
                    updated_at=receipt.promoted_at,
                    selection_rank=train_config.get("selection_rank") or (),
                    evaluation_source="modal:manual",
                    metrics_schema_version=metrics_schema_version,
                )
            finally:
                self.runtime.close_wandb(projector, timeout_seconds=300)
            remote = self.runtime.remote_summary(run_path)
        except Exception as exc:
            raise EvaluationProjectionPending(
                f"W&B promotion projection is not yet complete: {exc}"
            ) from exc
        if (
            str(remote.get("gradlab/goal/outcome") or "") != "accepted"
            or int(remote.get("leader/checkpoint/step") or -1)
            != receipt.checkpoint_step
        ):
            raise EvaluationProjectionPending(
                "W&B promotion summary is not yet remotely visible"
            )

    def _terminal_status(
        self,
        context: _EvaluationContext,
        result: EvalResult,
        raw: Mapping[str, Any] | None,
        *,
        ledger: SupervisorLedger,
        event_seq_offset: int,
    ) -> dict[str, Any]:
        manually_requested = self.authority.control.get_json_optional(
            self._request_key(context)
        ) is not None
        projected = (
            not manually_requested
            or self.authority.control.get_json_optional(self._projection_key(context))
            is not None
        )
        projection_error = None
        if raw is not None and not projected:
            try:
                projected = self._project_result(
                    context,
                    result,
                    raw,
                    ledger=ledger,
                    event_seq_offset=event_seq_offset,
                )
            except LeaseUnavailable:
                projection_error = "waiting for the run writer lease"
            except Exception as exc:
                projection_error = str(exc)
        return {
            "checkpoint_id": context.checkpoint.checkpoint_id,
            "state": result.status if projected else "awaiting_projection",
            "evaluation": _evaluation_summary(context, result),
            "message": projection_error,
        }

    def _status(
        self,
        context: _EvaluationContext,
        *,
        flush: bool,
        ledger: SupervisorLedger,
        event_seq_offset: int,
    ) -> dict[str, Any]:
        key = context.intent.idempotency_key
        verified_document = self.authority.evaluation.get_json_optional(
            f"runs/{context.manifest.run_id}/evals/{key}/verified-result.json"
        )
        raw = self.authority.eval_result(
            run_id=context.manifest.run_id,
            idempotency_key=key,
        )
        if verified_document is not None:
            result = EvalResult(**verified_document)
            result.validate()
            return self._terminal_status(
                context,
                result,
                raw,
                ledger=ledger,
                event_seq_offset=event_seq_offset,
            )

        attempt, prepared, dispatch = self._latest_attempt(context)
        if raw is not None:
            if attempt <= 0:
                raw_attempt = str(raw.get("attempt_id") or "")
                try:
                    attempt = int(raw_attempt.rsplit("-a", 1)[1])
                except (IndexError, ValueError) as exc:
                    raise ValueError("raw eval result has no recoverable attempt") from exc
            result = _verified_result(
                context=context,
                raw=raw,
                attempt=attempt,
                modal_call_id=str((dispatch or {}).get("modal_call_id") or ""),
                clock=self.clock,
            )
            try:
                self.authority.put_verified_eval_result(result)
            except ConditionalWriteConflict:
                existing = self.authority.evaluation.get_json(
                    f"runs/{context.manifest.run_id}/evals/{key}/verified-result.json"
                )
                if existing != result.to_dict():
                    raise
            return self._terminal_status(
                context,
                result,
                raw,
                ledger=ledger,
                event_seq_offset=event_seq_offset,
            )

        if not flush:
            if dispatch is not None:
                return {
                    "checkpoint_id": context.checkpoint.checkpoint_id,
                    "state": "submitted",
                    "evaluation": None,
                    "message": None,
                }
            return {
                "checkpoint_id": context.checkpoint.checkpoint_id,
                "state": "queued",
                "evaluation": None,
                "message": None,
            }

        now = self.clock.time()
        if prepared is not None and float(prepared["expires_at"]) > now:
            return {
                "checkpoint_id": context.checkpoint.checkpoint_id,
                "state": "submitted" if dispatch is not None else "submission_uncertain",
                "evaluation": None,
                "message": (
                    None
                    if dispatch is not None
                    else "submission is being reconciled before any retry"
                ),
            }
        next_attempt = attempt + 1
        if next_attempt > int(self.modal_config.protocol.max_attempts):
            return {
                "checkpoint_id": context.checkpoint.checkpoint_id,
                "state": "expired",
                "evaluation": None,
                "message": "evaluation exhausted its retry allowance",
            }
        expires_at = now + int(context.intent.timeout_seconds)
        self.authority.prepare_eval_attempt(
            run_id=context.manifest.run_id,
            idempotency_key=key,
            attempt=next_attempt,
            expires_at=expires_at,
        )
        payload = self._payload(
            context,
            attempt=next_attempt,
            expires_at=expires_at,
        )
        try:
            handle = self._backend(context.manifest).submit(payload)
        except Exception as exc:
            return {
                "checkpoint_id": context.checkpoint.checkpoint_id,
                "state": "submission_uncertain",
                "evaluation": None,
                "message": str(exc),
            }
        self.authority.put_eval_dispatch(
            run_id=context.manifest.run_id,
            idempotency_key=key,
            attempt=next_attempt,
            modal_call_id=handle.call_id,
        )
        return {
            "checkpoint_id": context.checkpoint.checkpoint_id,
            "state": "submitted",
            "evaluation": None,
            "message": None,
        }

    def _ensure_intents(self, contexts: Sequence[_EvaluationContext], *, job_id: str) -> None:
        for context in contexts:
            existing = self.authority.eval_intent(
                run_id=context.manifest.run_id,
                idempotency_key=context.intent.idempotency_key,
            )
            if existing is None:
                self.authority.put_eval_intent(context.intent)
            if self.authority.control.get_json_optional(self._request_key(context)) is None:
                self.authority.control.put_json(
                    self._request_key(context),
                    {
                        "schema_version": 1,
                        "job_id": job_id,
                        "run_id": context.manifest.run_id,
                        "checkpoint_id": context.checkpoint.checkpoint_id,
                        "idempotency_key": context.intent.idempotency_key,
                        "requested_at": self.clock.utc_now(),
                    },
                    create_only=True,
                )

    def _promotion_candidate(
        self,
        manifest: RunManifest,
    ) -> tuple[_EvaluationContext, EvalResult, Mapping[str, Any]] | None:
        checkpoints = self._checkpoint_map(manifest.run_id)
        candidates: list[
            tuple[int, str, _EvaluationContext, EvalResult, Mapping[str, Any]]
        ] = []
        for key in self.authority.evaluation.iter_keys(
            f"runs/{manifest.run_id}/evals"
        ):
            if not key.endswith("/verified-result.json"):
                continue
            document = self.authority.evaluation.get_json(key)
            result = EvalResult(**document)
            result.validate()
            if result.status != "accepted":
                continue
            checkpoint = checkpoints.get(result.checkpoint_id)
            if checkpoint is None:
                raise ValueError("accepted evaluation references an unknown checkpoint")
            context = self._context(manifest=manifest, checkpoint=checkpoint)
            raw = self.authority.eval_result(
                run_id=manifest.run_id,
                idempotency_key=result.idempotency_key,
            )
            if raw is None:
                raise ValueError("accepted evaluation is missing its raw result")
            candidates.append(
                (
                    checkpoint.step,
                    checkpoint.checkpoint_id,
                    context,
                    result,
                    raw,
                )
            )
        if not candidates:
            return None
        _step, _checkpoint_id, context, result, raw = min(candidates)
        return context, result, raw

    def _finalize_promotion(self, manifest: RunManifest) -> Mapping[str, Any] | None:
        candidate = self._promotion_candidate(manifest)
        existing = self.authority.control.get_json_optional(
            f"runs/{manifest.run_id}/promotion.json"
        )
        if candidate is None:
            if existing is not None:
                raise ValueError("promotion has no matching accepted evaluation evidence")
            return None
        context, result, raw = candidate
        if existing is not None:
            if (
                str(existing.get("checkpoint_id") or "")
                != context.checkpoint.checkpoint_id
                or int(existing.get("checkpoint_step") or -1) != context.checkpoint.step
            ):
                raise EvaluationContractIneligible(
                    "new accepted evidence conflicts with the immutable "
                    "lowest-step promotion"
                )
        else:
            self._promote(context, result)
            existing = self.authority.control.get_json(
                f"runs/{manifest.run_id}/promotion.json"
            )
        receipt = PromotionReceipt(**existing)
        receipt.validate()
        self._ensure_promotion_projection(context, result, raw, receipt)
        return existing

    def _promotion_retry_result(
        self,
        *,
        run_id: str,
        statuses: list[dict[str, Any]],
        error: EvaluationProjectionPending,
    ) -> HandlerResult:
        message = str(error)
        for status in statuses:
            if str(status["state"]) == "accepted":
                status["state"] = "awaiting_projection"
                status["message"] = message
        return HandlerResult(
            state="retry_wait",
            available_at=self.clock.time() + MANUAL_EVAL_RETRY_SECONDS,
            message=message,
            subjects=self._subject_updates(run_id, statuses),
        )

    def _manual_terminal(
        self,
        *,
        job_id: str,
        manifest: RunManifest,
        state: str,
        statuses: Sequence[Mapping[str, Any]],
        promotion: Mapping[str, Any] | None,
    ) -> None:
        key = f"runs/{manifest.run_id}/manual-evals/jobs/{job_id}/terminal.json"
        projection_high_waters = []
        for status in statuses:
            checkpoint_id = str(status["checkpoint_id"])
            context = next(
                item
                for item in self._contexts(manifest.run_id, [checkpoint_id])
                if item.checkpoint.checkpoint_id == checkpoint_id
            )
            projection = self.authority.control.get_json_optional(
                self._projection_key(context)
            )
            if projection is not None:
                projection_high_waters.append(
                    int(projection.get("wandb_high_water_mark") or 0)
                )
        document = {
            "schema_version": 1,
            "job_id": job_id,
            "run_id": manifest.run_id,
            "attempt_id": manifest.attempt_id,
            "state": state,
            "subjects": [dict(item) for item in statuses],
            "promotion": None if promotion is None else dict(promotion),
            "wandb_high_water_mark": max(projection_high_waters, default=0),
            "completed_at": self.clock.utc_now(),
        }
        existing = self.authority.control.get_json_optional(key)
        if existing is not None:
            comparable = dict(document)
            comparable["completed_at"] = existing.get("completed_at")
            if existing != comparable:
                raise ValueError("manual evaluation terminal receipt conflicts")
            return
        self.authority.control.put_json(key, document, create_only=True)

    def _subject_updates(
        self,
        run_id: str,
        statuses: Sequence[Mapping[str, Any]],
    ) -> tuple[SubjectUpdate, ...]:
        return tuple(
            SubjectUpdate(
                subject_type=_evaluation_subject_type(run_id),
                subject_id=str(status["checkpoint_id"]),
                state=str(status["state"]),
                detail={
                    "evaluation": status.get("evaluation"),
                    "message": status.get("message"),
                },
            )
            for status in statuses
        )

    def advance_batch(
        self,
        *,
        job_id: str,
        run_id: str,
        checkpoint_ids: Sequence[str],
        cancel_requested: bool,
    ) -> HandlerResult:
        contexts = self._contexts(run_id, checkpoint_ids)
        manifest = contexts[0].manifest
        terminal = self._training_terminal(manifest)
        if terminal is None:
            statuses = [
                {
                    "checkpoint_id": context.checkpoint.checkpoint_id,
                    "state": "waiting_for_training_terminal",
                    "evaluation": None,
                    "message": "waiting for authoritative training terminal evidence",
                }
                for context in contexts
            ]
            return HandlerResult(
                state="retry_wait",
                available_at=self.clock.time() + MANUAL_EVAL_WAIT_SECONDS,
                message="waiting for training to become terminal",
                subjects=self._subject_updates(run_id, statuses),
            )

        try:
            self._lease(manifest)
        except LeaseUnavailable as exc:
            statuses = [
                {
                    "checkpoint_id": context.checkpoint.checkpoint_id,
                    "state": "waiting_for_run_lease",
                    "evaluation": None,
                    "message": str(exc),
                }
                for context in contexts
            ]
            return HandlerResult(
                state="retry_wait",
                available_at=self.clock.time() + MANUAL_EVAL_WAIT_SECONDS,
                message="waiting for the run writer lease",
                subjects=self._subject_updates(run_id, statuses),
            )

        self._ensure_intents(contexts, job_id=job_id)
        ledger = SupervisorLedger(self.work_root / job_id / "supervisor.sqlite3", clock=self.clock)
        ledger.init()
        event_seq_offset = self._event_seq_offset(manifest, terminal)
        statuses = [
            self._status(
                context,
                flush=not cancel_requested,
                ledger=ledger,
                event_seq_offset=event_seq_offset,
            )
            for context in contexts
        ]

        if cancel_requested:
            for context, status in zip(contexts, statuses, strict=True):
                if str(status["state"]) in {
                    "accepted",
                    "rejected",
                    "failed",
                    "expired",
                }:
                    continue
                _attempt, _prepared, dispatch = self._latest_attempt(context)
                if dispatch is not None:
                    try:
                        from gradlab.eval_backend import EvalHandle

                        self._backend(manifest).cancel(
                            EvalHandle(
                                provider="modal",
                                call_id=str(dispatch["modal_call_id"]),
                            )
                        )
                    except Exception as exc:
                        status["message"] = f"cancel reconciliation failed: {exc}"
                status["state"] = "canceled"
            try:
                promotion = self._finalize_promotion(manifest)
            except EvaluationProjectionPending as exc:
                return self._promotion_retry_result(
                    run_id=run_id,
                    statuses=statuses,
                    error=exc,
                )
            self._manual_terminal(
                job_id=job_id,
                manifest=manifest,
                state="canceled",
                statuses=statuses,
                promotion=promotion,
            )
            return HandlerResult(
                state="canceled",
                message="canceled by operator",
                subjects=self._subject_updates(run_id, statuses),
            )

        unsettled = [
            status
            for status in statuses
            if str(status["state"])
            not in {"accepted", "rejected", "failed", "expired"}
        ]
        if unsettled:
            return HandlerResult(
                state="retry_wait",
                available_at=self.clock.time() + MANUAL_EVAL_RETRY_SECONDS,
                message="evaluation work is still settling",
                subjects=self._subject_updates(run_id, statuses),
            )

        try:
            promotion = self._finalize_promotion(manifest)
        except EvaluationProjectionPending as exc:
            return self._promotion_retry_result(
                run_id=run_id,
                statuses=statuses,
                error=exc,
            )
        failed = [
            status
            for status in statuses
            if str(status["state"]) in {"failed", "expired"}
        ]
        state = "failed" if failed else "succeeded"
        self._manual_terminal(
            job_id=job_id,
            manifest=manifest,
            state=state,
            statuses=statuses,
            promotion=promotion,
        )
        return HandlerResult(
            state=state,  # type: ignore[arg-type]
            message=(
                "one or more evaluations failed or expired"
                if failed
                else None
            ),
            subjects=self._subject_updates(run_id, statuses),
        )


class ManualEvaluationJobHandler:
    job_type = MANUAL_EVAL_JOB_TYPE
    version = MANUAL_EVAL_JOB_VERSION

    @classmethod
    def validate_payload(cls, payload: Mapping[str, Any]) -> dict[str, Any]:
        allowed = {"repo_root", "queue_root", "run_id", "checkpoint_ids"}
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"manual evaluation job has unknown field: {sorted(unknown)[0]}")
        repo_root = Path(str(payload.get("repo_root") or "")).expanduser()
        queue_root = Path(str(payload.get("queue_root") or "")).expanduser()
        if not repo_root.is_absolute() or not queue_root.is_absolute():
            raise ValueError("manual evaluation job paths must be absolute")
        run_id = str(payload.get("run_id") or "")
        if not run_id.startswith("gradlab-") or len(run_id) != 40:
            raise ValueError("manual evaluation job has an invalid run id")
        raw_ids = payload.get("checkpoint_ids")
        if isinstance(raw_ids, str | bytes) or not isinstance(raw_ids, Sequence):
            raise ValueError("manual evaluation checkpoint_ids must be an array")
        checkpoint_ids = tuple(dict.fromkeys(str(value).strip() for value in raw_ids))
        if not checkpoint_ids or any(not value for value in checkpoint_ids):
            raise ValueError("manual evaluation job requires checkpoints")
        if len(checkpoint_ids) > MAX_MANUAL_EVAL_SELECTION:
            raise ValueError(
                f"select at most {MAX_MANUAL_EVAL_SELECTION} checkpoints per request"
            )
        return {
            "repo_root": str(repo_root.resolve()),
            "queue_root": str(queue_root.resolve()),
            "run_id": run_id,
            "checkpoint_ids": list(checkpoint_ids),
        }

    def advance(self, job: Mapping[str, Any]) -> HandlerResult:
        payload = self.validate_payload(job["payload"])
        repo_root = Path(payload["repo_root"])
        load_repository_operator_environment(repo_root)
        queue_root = Path(payload["queue_root"])
        holder_fingerprint = hashlib.sha256(
            str(queue_root).encode("utf-8")
        ).hexdigest()[:16]
        supervisor = ManualEvaluationSupervisor(
            authority=RunAuthority(RunStorageConfig.from_env()),
            repo_root=repo_root,
            holder_id=f"manual-eval-local-{holder_fingerprint}",
            work_root=queue_root / "work",
        )
        try:
            return supervisor.advance_batch(
                job_id=str(job["job_id"]),
                run_id=str(payload["run_id"]),
                checkpoint_ids=list(payload["checkpoint_ids"]),
                cancel_requested=bool(job.get("cancel_requested")),
            )
        except EvaluationContractIneligible as exc:
            return HandlerResult(
                state="blocked",
                message=str(exc),
                subjects=tuple(
                    SubjectUpdate(
                        subject_type=_evaluation_subject_type(str(payload["run_id"])),
                        subject_id=checkpoint_id,
                        state="blocked",
                        detail={"evaluation": None, "message": str(exc)},
                    )
                    for checkpoint_id in payload["checkpoint_ids"]
                ),
            )


def register_job_handler() -> None:
    register_handler(
        MANUAL_EVAL_JOB_TYPE,
        MANUAL_EVAL_JOB_VERSION,
        ManualEvaluationJobHandler,
        replace=True,
    )


class ManualEvaluationQueue:
    """Player-facing facade over the durable per-user local job queue."""

    def __init__(
        self,
        *,
        repo_root: Path,
        store: JobStore | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.store = store or JobStore()
        self.store.init()
        self._last_ensure = 0.0

    def _planner(self) -> ManualEvaluationSupervisor:
        load_repository_operator_environment(self.repo_root)
        return ManualEvaluationSupervisor(
            authority=RunAuthority(RunStorageConfig.from_env()),
            repo_root=self.repo_root,
            project_results=False,
            work_root=self.store.work_root,
        )

    @staticmethod
    def _queue_item(subject: Mapping[str, Any]) -> dict[str, Any]:
        detail = subject.get("detail")
        details = detail if isinstance(detail, Mapping) else {}
        return {
            "checkpoint_id": str(subject["subject_id"]),
            "job_id": str(subject["job_id"]),
            "state": str(subject["state"]),
            "evaluation": details.get("evaluation"),
            "message": details.get("message") or subject.get("job_error"),
        }

    def enqueue(
        self,
        *,
        run_id: str,
        checkpoint_ids: Sequence[str],
    ) -> dict[str, Any]:
        planner = self._planner()
        contexts = planner._contexts(
            run_id,
            checkpoint_ids,
            enforce_current_protocol=False,
        )
        terminal_items: dict[str, dict[str, Any]] = {}
        pending_contexts: list[_EvaluationContext] = []
        for context in contexts:
            verified = planner.authority.evaluation.get_json_optional(
                (
                    f"runs/{run_id}/evals/{context.intent.idempotency_key}"
                    "/verified-result.json"
                )
            )
            if verified is None:
                pending_contexts.append(context)
                continue
            result = EvalResult(**verified)
            result.validate()
            terminal_items[context.checkpoint.checkpoint_id] = {
                "checkpoint_id": context.checkpoint.checkpoint_id,
                "job_id": None,
                "state": result.status,
                "evaluation": _evaluation_summary(context, result),
                "message": result.error,
            }
        subject_ids = [
            context.checkpoint.checkpoint_id for context in pending_contexts
        ]
        subject_type = _evaluation_subject_type(run_id)

        for _attempt in range(3):
            existing = self.store.subject_statuses(
                subject_type=subject_type,
                subject_ids=subject_ids,
            )
            pending = [
                context
                for context in pending_contexts
                if context.checkpoint.checkpoint_id not in existing
            ]
            if not pending:
                break
            identity_payload = {
                "type": MANUAL_EVAL_JOB_TYPE,
                "version": MANUAL_EVAL_JOB_VERSION,
                "run_id": run_id,
                "evaluations": sorted(
                    context.intent.idempotency_key for context in pending
                ),
            }
            idempotency_key = hashlib.sha256(
                json.dumps(
                    identity_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            try:
                self.store.enqueue(
                    job_type=MANUAL_EVAL_JOB_TYPE,
                    handler_version=MANUAL_EVAL_JOB_VERSION,
                    payload={
                        "repo_root": str(self.repo_root),
                        "queue_root": str(self.store.root),
                        "run_id": run_id,
                        "checkpoint_ids": [
                            context.checkpoint.checkpoint_id for context in pending
                        ],
                    },
                    idempotency_key=idempotency_key,
                    subjects=[
                        JobSubject(
                            subject_type=subject_type,
                            subject_id=context.checkpoint.checkpoint_id,
                            exclusive_key=(
                                f"{MANUAL_EVAL_JOB_TYPE}:"
                                f"{context.intent.idempotency_key}"
                            ),
                            detail={
                                "evaluation": None,
                                "message": None,
                                "eval_idempotency_key": (
                                    context.intent.idempotency_key
                                ),
                            },
                        )
                        for context in pending
                    ],
                )
            except sqlite3.IntegrityError:
                continue
            break
        else:
            raise RuntimeError("checkpoint evaluation admission did not converge")

        statuses = self.store.subject_statuses(
            subject_type=subject_type,
            subject_ids=subject_ids,
        )
        missing = [identifier for identifier in subject_ids if identifier not in statuses]
        if missing:
            raise RuntimeError(f"checkpoint evaluation was not durably queued: {missing[0]}")
        worker = (
            ensure_flusher(self.store)
            if subject_ids
            else WorkerStart("already_running", "all checkpoints are already terminal")
        )
        job_ids = {
            str(subject["job_id"])
            for subject in statuses.values()
        }
        jobs = [
            job
            for job_id in sorted(job_ids)
            if (job := self.store.job(job_id)) is not None
        ]
        items = [
            (
                terminal_items[context.checkpoint.checkpoint_id]
                if context.checkpoint.checkpoint_id in terminal_items
                else self._queue_item(statuses[context.checkpoint.checkpoint_id])
            )
            for context in contexts
        ]
        if worker.state == "start_failed":
            items = [
                {
                    **item,
                    "state": (
                        "flusher_unavailable"
                        if str(item["state"]) in {"queued", "retry_wait", "running"}
                        else item["state"]
                    ),
                    "message": item.get("message") or worker.message,
                }
                for item in items
            ]
        return {
            "items": items,
            "jobs": jobs,
            "worker": worker.to_dict(),
        }

    def statuses(
        self,
        *,
        run_id: str,
        checkpoint_ids: Sequence[str],
    ) -> dict[str, dict[str, Any]]:
        statuses = self.store.subject_statuses(
            subject_type=_evaluation_subject_type(run_id),
            subject_ids=checkpoint_ids,
        )
        now = self.store.clock.monotonic()
        if self.store.has_unfinished() and now - self._last_ensure >= 5.0:
            self._last_ensure = now
            worker = ensure_flusher(self.store)
            if worker.state == "start_failed":
                for subject in statuses.values():
                    if str(subject["state"]) in {
                        "queued",
                        "retry_wait",
                        "waiting_for_training_terminal",
                        "waiting_for_run_lease",
                    }:
                        detail = subject.get("detail")
                        values = dict(detail) if isinstance(detail, Mapping) else {}
                        values["message"] = worker.message
                        subject["detail"] = values
                        subject["state"] = "flusher_unavailable"
        return {
            checkpoint_id: self._queue_item(subject)
            for checkpoint_id, subject in statuses.items()
        }


def build_manual_evaluation_queue(repo_root: Path) -> ManualEvaluationQueue:
    # Queue inspection is local and credential-free. Credentials are resolved
    # only by explicit enqueue and worker advancement paths.
    return ManualEvaluationQueue(repo_root=Path(repo_root).resolve())


__all__ = [
    "MAX_MANUAL_EVAL_SELECTION",
    "ManualEvaluationQueue",
    "ManualEvaluationSupervisor",
    "build_manual_evaluation_queue",
    "register_job_handler",
]
