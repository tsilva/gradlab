from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timedelta
from typing import Any

from gradlab.checkpoint_acceptance import manifest_index
from gradlab.clock import Clock, format_utc_datetime
from gradlab.eval_metrics import eval_by_start_records
from gradlab.evaluation_projection import evaluation_wandb_projection
from gradlab.modal_eval_protocol import PROTOCOL_SCHEMA_VERSION, normalize_attempt_result
from gradlab.run_contracts import (
    CheckpointManifest,
    EvalIntent,
    EvalResult,
    PromotionReceipt,
    RunManifest,
    document_sha256,
    eval_idempotency_key,
)
from gradlab.vizdoom_assets import validate_vizdoom_iwad_binding


def compile_execution_contract(
    base_contract: Mapping[str, Any],
    *,
    manifest: RunManifest,
    checkpoint: CheckpointManifest,
    recipe_format_version: int,
    evaluation_contract_sha256: str,
    transform: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    contract = dict(base_contract)
    contract.update(
        {
            "schema_version": PROTOCOL_SCHEMA_VERSION,
            "checkpoint_sha256": checkpoint.sha256,
            "runtime_image_ref": manifest.image_digest,
            "recipe_sha256": checkpoint.recipe_document_sha256,
            "recipe_format_version": int(recipe_format_version),
            "evaluation_contract_sha256": evaluation_contract_sha256,
        }
    )
    asset = contract.get("asset")
    if isinstance(asset, Mapping):
        contract["asset"] = {
            str(key): value for key, value in asset.items() if str(key) != "object_uri"
        }
    if transform is not None:
        contract = dict(transform(contract))
    manifest_index(contract)
    return contract


def compile_eval_intent(
    *,
    manifest: RunManifest,
    checkpoint: CheckpointManifest,
    execution_contract: Mapping[str, Any],
    evaluation_contract_sha256: str,
    protocol: str,
    timeout_seconds: int,
    created_at: datetime,
) -> EvalIntent:
    episode_manifest_sha256 = document_sha256(execution_contract["manifest"])
    idempotency_key = eval_idempotency_key(
        run_id=manifest.run_id,
        checkpoint_sha256=checkpoint.sha256,
        evaluation_contract_sha256=evaluation_contract_sha256,
        episode_manifest_sha256=episode_manifest_sha256,
        protocol=protocol,
    )
    return EvalIntent(
        run_id=manifest.run_id,
        checkpoint_id=checkpoint.checkpoint_id,
        idempotency_key=idempotency_key,
        checkpoint_sha256=checkpoint.sha256,
        goal_sha256=manifest.goal_sha256,
        recipe_sha256=manifest.recipe_sha256,
        environment_sha256=manifest.environment_sha256,
        evaluation_contract_sha256=evaluation_contract_sha256,
        episode_manifest_sha256=episode_manifest_sha256,
        protocol=protocol,
        execution_contract=dict(execution_contract),
        result_key=f"runs/{manifest.run_id}/evals/{idempotency_key}/result.json",
        timeout_seconds=int(timeout_seconds),
        created_at=format_utc_datetime(created_at),
        expires_at=format_utc_datetime(
            created_at + timedelta(seconds=int(timeout_seconds))
        ),
    )


def build_modal_eval_payload(
    *,
    manifest: RunManifest,
    checkpoint: CheckpointManifest,
    intent: EvalIntent,
    attempt: int,
    expires_at: float,
    evaluation_store: Any,
    child_margin_seconds: int,
    expiry_margin_seconds: int,
) -> dict[str, Any]:
    timeout = int(intent.timeout_seconds)
    payload: dict[str, Any] = {
        "attempt_id": f"{intent.idempotency_key[:20]}-a{attempt}",
        "contract": dict(intent.execution_contract),
        "expires_at": expires_at,
        "child_timeout_seconds": max(1, timeout - int(child_margin_seconds)),
        "model_get_url": checkpoint.public_url,
        "model_document_get_url": checkpoint.model_document_url,
        "model_document_sha256": checkpoint.model_document_sha256,
        "recipe_get_url": checkpoint.recipe_document_url,
        "result_uri": evaluation_store.uri(intent.result_key),
        "result_put_url": evaluation_store.presign_put(
            intent.result_key,
            expires_seconds=timeout + int(expiry_margin_seconds),
        ),
    }
    asset = manifest.modal.get("rom_asset_manifest")
    if isinstance(asset, Mapping):
        rom_key = evaluation_store.key_from_uri(str(asset["object_uri"]))
        payload["rom_get_url"] = evaluation_store.presign_get(
            rom_key,
            expires_seconds=timeout + int(expiry_margin_seconds),
        )
    iwad = manifest.modal.get("vizdoom_iwad_binding")
    if isinstance(iwad, Mapping):
        normalized_iwad = validate_vizdoom_iwad_binding(iwad)
        iwad_key = evaluation_store.key_from_uri(str(normalized_iwad["object_uri"]))
        payload["vizdoom_iwad_binding"] = {
            key: value for key, value in normalized_iwad.items() if key != "object_uri"
        }
        payload["vizdoom_iwad_get_url"] = evaluation_store.presign_get(
            iwad_key,
            expires_seconds=timeout + int(expiry_margin_seconds),
        )
    return payload


def verify_eval_result(
    *,
    run_id: str,
    checkpoint_id: str,
    intent: EvalIntent,
    raw: Mapping[str, Any],
    attempt: int,
    modal_call_id: str,
    clock: Clock,
) -> EvalResult:
    normalized = normalize_attempt_result(
        raw,
        contract=dict(intent.execution_contract),
        attempt_id=f"{intent.idempotency_key[:20]}-a{attempt}",
    )
    observed_at = clock.utc_now()
    return EvalResult(
        run_id=run_id,
        checkpoint_id=checkpoint_id,
        idempotency_key=intent.idempotency_key,
        modal_call_id=modal_call_id or "not-recorded",
        status=normalized["status"],  # type: ignore[arg-type]
        episode_results=normalized["episode_results"],
        aggregates=normalized["aggregates"],
        timings={
            "duration_seconds": normalized["duration_seconds"],
            "result_observed_at": observed_at,
        },
        evidence_sha256=normalized["evidence_sha256"],
        completed_at=observed_at,
        error=normalized["error"],
    )


def evaluation_metric_records(
    result: EvalResult,
    *,
    schema_version: int,
    checkpoint_step: int,
    episodes_planned: int,
) -> tuple[dict[str, Any], list[dict[str, Any]] | None]:
    metrics = evaluation_wandb_projection(
        result.aggregates,
        schema_version=schema_version,
        checkpoint_step=checkpoint_step,
        accepted=result.status == "accepted",
        episodes_planned=episodes_planned,
        episodes_completed=len(result.episode_results),
    )
    by_start = None
    if result.status in {"accepted", "rejected"} and len(
        result.episode_results
    ) == episodes_planned:
        by_start = eval_by_start_records(
            [dict(episode) for episode in result.episode_results]
        )
    return metrics, by_start


def promotion_receipt(
    *,
    run_id: str,
    checkpoint_id: str,
    checkpoint_step: int,
    eval_idempotency_key: str,
    result: Mapping[str, Any],
    promoted_at: str,
) -> PromotionReceipt:
    return PromotionReceipt(
        run_id=run_id,
        checkpoint_id=checkpoint_id,
        checkpoint_step=checkpoint_step,
        eval_idempotency_key=eval_idempotency_key,
        eval_result_sha256=document_sha256(result),
        accepted_episode_count=len(result.get("episode_results") or []),
        promoted_at=promoted_at,
    )


__all__ = [
    "build_modal_eval_payload",
    "compile_eval_intent",
    "compile_execution_contract",
    "evaluation_metric_records",
    "promotion_receipt",
    "verify_eval_result",
]
