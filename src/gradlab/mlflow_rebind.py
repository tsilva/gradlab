"""Audited reconstruction after an irrecoverable private MLflow database loss."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import time
from uuid import uuid4

from gradlab.metric_journal import read_control_journal
from gradlab.mlflow_delivery import MlflowDelivery
from gradlab.operator_environment import load_repository_operator_environment
from gradlab.r2_store import RunStorageConfig
from gradlab.run_authority import RunAuthority
from gradlab.run_contracts import RunManifest


def replace_mlflow_binding(
    authority: RunAuthority,
    *,
    run_id: str,
    tracking_uri: str,
    operator: str,
    reason: str,
) -> dict:
    """Replay immutable journal evidence, then atomically replace the service pointer."""
    if not operator.strip() or not reason.strip():
        raise ValueError("MLflow replacement requires an operator and reason")
    raw_manifest = authority.manifest(run_id)
    if raw_manifest is None:
        raise ValueError("GradLab Run manifest is missing")
    manifest = RunManifest.from_dict(raw_manifest)
    if (manifest.tracking or {}).get("backend") != "mlflow":
        raise ValueError("only an MLflow Run can replace an MLflow binding")
    prefix = authority.run_prefix(run_id)
    binding_key = f"{prefix}/metrics-binding.json"
    original = authority.control.get_json_optional(binding_key)
    if original is None or original.get("backend") != "mlflow":
        raise ValueError("original MLflow binding is missing")
    lease = authority.acquire_lease(
        run_id=run_id,
        attempt_id=manifest.attempt_id,
        holder_id=f"mlflow-rebind-{uuid4().hex}",
    )
    try:
        events = read_control_journal(authority.control, run_id)
        if not events:
            raise ValueError("replacement requires a complete retained journal")
        terminals = [
            authority.control.get_json(key)
            for key in authority.control.iter_keys(f"{prefix}/attempts/")
            if key.endswith("/terminal.json")
        ]
        if not terminals:
            raise ValueError("replacement requires an immutable terminal receipt")
        terminal = max(terminals, key=lambda row: str(row.get("completed_at") or ""))
        if int((terminal.get("drain") or {}).get("metric_segment_high_water") or 0) != int(
            events[-1]["event_seq"]
        ):
            raise ValueError("retained journal does not match terminal high-water evidence")
        created_at_ms = int(
            datetime.fromisoformat(manifest.created_at.replace("Z", "+00:00")).timestamp()
            * 1000
        )
        recipe = authority.recipe_document(manifest.recipe_sha256)
        game = str(recipe["recipe"]["train_config"]["game_family"])
        service = MlflowDelivery.open(
            tracking_uri=tracking_uri,
            experiment_name=f"gradlab-{game}",
            gradlab_run_id=run_id,
            created_at_ms=created_at_ms,
        )
        if service.run_id == original.get("service_run_id"):
            raise ValueError("replacement service still has the original binding")
        for event in events:
            service.publish_frame(
                {
                    "id": int(event["event_seq"]),
                    "kind": event["kind"],
                    "step": event.get("step"),
                    "source": event.get("source"),
                    "payload_json": json.dumps(event["payload"], sort_keys=True),
                }
            )
        high_water = int(events[-1]["event_seq"])
        deadline = time.monotonic() + 120
        while service.remote_high_water() < high_water:
            if time.monotonic() >= deadline:
                raise TimeoutError("replacement MLflow projection is not remotely visible")
            time.sleep(1)
        promotion = authority.control.get_json_optional(f"{prefix}/promotion.json")
        if promotion is not None:
            evaluation = authority.eval_result(
                run_id=run_id,
                idempotency_key=str(promotion["eval_idempotency_key"]),
            )
            index = authority.models.get_json(f"{prefix}/index.json")
            checkpoint = next(
                row for row in index["checkpoints"]
                if row["checkpoint_id"] == promotion["checkpoint_id"]
            )
            if evaluation is None:
                raise ValueError("promoted evaluation result is missing")
            service.publish_promotion(
                checkpoint_step=int(promotion["checkpoint_step"]),
                checkpoint_url=str(checkpoint["public_url"]),
                metrics=evaluation["aggregates"],
                updated_at=str(promotion["promoted_at"]),
                selection_rank=recipe["recipe"]["train_config"]["selection_rank"],
                evaluation_source="reconstructed:journal",
            )
        service.publish_terminal(
            state=str(terminal["state"]), reason=str(terminal["stop_reason"])
        )
        inventory = authority.retain_metric_journals(run_id=run_id)
        replacement = {
            "schema_version": 2,
            "run_id": run_id,
            "backend": "mlflow",
            "service_run_id": service.run_id,
            "service_url": service.service_url,
            "operator_profile": (manifest.tracking or {}).get("operator_profile"),
            "created_at": authority.clock.utc_now(),
            "replaces_service_run_id": original["service_run_id"],
            "journal_inventory_sha256": inventory["inventory_sha256"],
            "journal_high_water": high_water,
            "operator": operator.strip(),
            "reason": reason.strip(),
        }
        audit_key = f"{prefix}/metrics-bindings/replacements/{uuid4().hex}.json"
        authority.control.put_json(audit_key, replacement, create_only=True)
        replacement["audit_key"] = audit_key
        authority.control.put_json(
            binding_key,
            replacement,
            create_only=False,
            if_match=str(authority.control.head(binding_key)["etag"]),
        )
        return replacement
    finally:
        authority.release_lease(lease)


def main(argv: list[str] | None = None) -> int:
    from gradlab.cli_parser import ExactArgumentParser

    parser = ExactArgumentParser(
        prog="gradlab rebind-mlflow",
        description="Replay a Run into a replacement private MLflow service after database loss.",
    )
    parser.add_argument("run_id")
    parser.add_argument("--operator", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--confirm-irrecoverable", action="store_true", required=True)
    args = parser.parse_args(argv)
    load_repository_operator_environment(Path.cwd())
    uri = str(os.environ.get("MLFLOW_TRACKING_URI") or "").strip()
    if not uri:
        raise RuntimeError("replacement MLflow service URI is missing")
    result = replace_mlflow_binding(
        RunAuthority(RunStorageConfig.from_env()),
        run_id=str(args.run_id),
        tracking_uri=uri,
        operator=str(args.operator),
        reason=str(args.reason),
    )
    print(f"Replacement MLflow binding: {result['audit_key']}")
    return 0
