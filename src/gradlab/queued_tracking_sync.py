"""Explicit later service projection for a queued complete_local Run."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from gradlab.local_wandb import sync_local_run
from gradlab.metric_journal import read_control_journal
from gradlab.metric_names import (
    LEADER_CHECKPOINT_ARTIFACT_REF,
    LEADER_CHECKPOINT_STEP,
    METRICS_SCHEMA_VERSION,
    ORCHESTRATION_RUN_TERMINAL_REASON,
    ORCHESTRATION_RUN_TERMINAL_STATE,
)
from gradlab.operator_environment import load_repository_operator_environment
from gradlab.policy_bundle import write_canonical_json
from gradlab.r2_store import RunStorageConfig
from gradlab.run_authority import LeaseUnavailable, RunAuthority
from gradlab.run_contracts import RunManifest, TerminalReceipt
from gradlab.supervisor_ledger import SupervisorLedger


def _terminal(authority: RunAuthority, manifest: RunManifest) -> TerminalReceipt:
    prefix = f"{authority.run_prefix(manifest.run_id)}/attempts/"
    receipts = [
        TerminalReceipt.from_dict(authority.control.get_json(key))
        for key in authority.control.iter_keys(prefix)
        if key.endswith("/terminal.json")
    ]
    completed = [row for row in receipts if row.state == "complete_local"]
    if not completed:
        raise ValueError("queued tracker sync requires an immutable terminal receipt")
    receipt = max(completed, key=lambda row: row.completed_at)
    if receipt.run_id != manifest.run_id:
        raise ValueError("queued tracker sync requires a complete_local Run")
    return receipt


def _prepare_replay(
    directory: Path,
    *,
    manifest: RunManifest,
    recipe: dict,
    receipt: TerminalReceipt,
    events: list[dict],
) -> None:
    config = dict(recipe["recipe"]["train_config"])
    config.update(
        {
            "tracking": dict(manifest.tracking or {}),
            "wandb_run_id": manifest.run_id,
            "wandb_group": manifest.run_id,
            "wandb_display_name": str(manifest.wandb.get("display_name") or manifest.run_id),
            "run_name": manifest.run_id,
            "run_description": manifest.run_description,
            "metrics_schema_version": METRICS_SCHEMA_VERSION,
        }
    )
    write_canonical_json(directory / "train-config.json", config)
    write_canonical_json(directory / "recipe.json", recipe)
    write_canonical_json(
        directory / "local-run.json",
        {"run_id": manifest.run_id, "status": "complete_local", "started_at": manifest.created_at},
    )
    write_canonical_json(
        directory / "training-result.json",
        {"status": "complete_local", "terminal_reason": receipt.stop_reason},
    )
    store = SupervisorLedger(directory / "gradlab.sqlite")
    store.init()
    with store.connection() as connection:
        now = time.time()
        for event in events:
            connection.execute(
                "INSERT INTO metric_frames "
                "(id, event_id, step, source, kind, payload_json, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 'local_only', ?, ?)",
                (
                    int(event["event_seq"]),
                    str(event["event_id"]),
                    event.get("step"),
                    str(event.get("source") or ""),
                    str(event["kind"]),
                    json.dumps(event["payload"], sort_keys=True, separators=(",", ":")),
                    now,
                    now,
                ),
            )


def _project_queued_outcome(
    authority: RunAuthority,
    *,
    manifest: RunManifest,
    recipe: dict,
    receipt: TerminalReceipt,
    config: dict,
    service_run_id: str,
) -> None:
    prefix = authority.run_prefix(manifest.run_id)
    promotion = authority.control.get_json_optional(f"{prefix}/promotion.json")
    projection = None
    if promotion is not None:
        evaluation = authority.eval_result(
            run_id=manifest.run_id,
            idempotency_key=str(promotion["eval_idempotency_key"]),
        )
        if evaluation is None:
            raise ValueError("queued tracker sync cannot find promoted evaluation evidence")
        index = authority.models.get_json(f"{prefix}/index.json")
        checkpoint = next(
            (
                row for row in index["checkpoints"]
                if row["checkpoint_id"] == promotion["checkpoint_id"]
            ),
            None,
        )
        if checkpoint is None:
            raise ValueError("queued tracker sync cannot find promoted checkpoint")
        projection = {
            "checkpoint_step": int(promotion["checkpoint_step"]),
            "checkpoint_url": str(checkpoint["public_url"]),
            "metrics": evaluation["aggregates"],
            "updated_at": str(promotion["promoted_at"]),
            "selection_rank": recipe["recipe"]["train_config"]["selection_rank"],
            "evaluation_source": "synced:journal",
        }
    if manifest.tracking["backend"] == "mlflow":
        from gradlab.mlflow_delivery import MlflowDelivery

        created_at_ms = int(
            datetime.fromisoformat(manifest.created_at.replace("Z", "+00:00")).timestamp()
            * 1000
        )
        service = MlflowDelivery.open(
            tracking_uri=str(os.environ["MLFLOW_TRACKING_URI"]),
            experiment_name=f"gradlab-{config.get('game_family') or config['game']}",
            gradlab_run_id=manifest.run_id,
            created_at_ms=created_at_ms,
        )
        if service.run_id != service_run_id:
            raise RuntimeError("queued MLflow outcome changed the service binding")
        if projection is not None:
            service.publish_promotion(**projection)
        service.publish_terminal(state=receipt.state, reason=receipt.stop_reason)
        deadline = time.monotonic() + 60
        while True:
            summary = service.remote_summary()
            if (
                summary.get(ORCHESTRATION_RUN_TERMINAL_STATE) == receipt.state
                and summary.get(ORCHESTRATION_RUN_TERMINAL_REASON) == receipt.stop_reason
                and (
                    projection is None
                    or (
                        int(summary.get(LEADER_CHECKPOINT_STEP) or -1)
                        == projection["checkpoint_step"]
                        and summary.get(LEADER_CHECKPOINT_ARTIFACT_REF)
                        == projection["checkpoint_url"]
                    )
                )
            ):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("queued MLflow outcome is not remotely visible")
            time.sleep(2)
        return
    from gradlab.wandb_publisher import (
        WandbProjector,
        publish_promotion_summary,
        publish_terminal_summary,
        promotion_summary_matches,
    )

    projector = WandbProjector.resume(
        {**config, "tracking": {"backend": "wandb", "delivery": "online"}},
        update_finish_state=True,
    )
    try:
        if str(projector.run.id) != service_run_id:
            raise RuntimeError("queued W&B outcome changed the service binding")
        run_path = str(projector.run.path)
        if projection is not None:
            publish_promotion_summary(projector.run, **projection)
        publish_terminal_summary(projector.run, receipt)
    finally:
        projector.close(timeout_seconds=60, exit_code=0)
    import wandb

    deadline = time.monotonic() + 60
    while True:
        summary = dict(wandb.Api(timeout=10).run(run_path).summary)
        visible = (
            summary.get(ORCHESTRATION_RUN_TERMINAL_STATE) == receipt.state
            and summary.get(ORCHESTRATION_RUN_TERMINAL_REASON) == receipt.stop_reason
            and (
                projection is None
                or promotion_summary_matches(
                    summary,
                    checkpoint_step=projection["checkpoint_step"],
                    checkpoint_url=projection["checkpoint_url"],
                    updated_at=projection["updated_at"],
                    selection_rank=projection["selection_rank"],
                )
            )
        )
        if visible:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("queued W&B outcome is not remotely visible")
        time.sleep(2)


def sync_queued_run(authority: RunAuthority, run_id: str) -> dict:
    """Replay R2 evidence to the frozen backend without changing the terminal receipt."""
    raw = authority.manifest(run_id)
    if raw is None:
        raise ValueError("queued tracker sync requires a Run manifest")
    manifest = RunManifest.from_dict(raw)
    if (manifest.tracking or {}).get("delivery") != "local_only":
        raise ValueError("queued tracker sync requires frozen local_only delivery")
    receipt = _terminal(authority, manifest)
    events = read_control_journal(authority.control, run_id)
    high_water = int(events[-1]["event_seq"]) if events else 0
    if high_water < int(receipt.drain.get("metric_segment_high_water") or 0):
        raise ValueError("retained journal is shorter than the terminal receipt")
    recipe = authority.recipe_document(manifest.recipe_sha256)
    lease = authority.acquire_lease(
        run_id=run_id,
        attempt_id=receipt.attempt_id,
        holder_id=f"queued-tracker-sync-{uuid4().hex}",
    )
    stop = threading.Event()
    errors: list[BaseException] = []

    def renew() -> None:
        nonlocal lease
        while not stop.wait(15):
            try:
                lease = authority.renew_lease(lease)
            except BaseException as exc:
                errors.append(exc)
                return

    worker = threading.Thread(target=renew, name="queued-tracker-sync-lease", daemon=True)
    worker.start()
    try:
        with tempfile.TemporaryDirectory(prefix="gradlab-queued-sync-") as temporary:
            directory = Path(temporary)
            _prepare_replay(
                directory, manifest=manifest, recipe=recipe, receipt=receipt, events=events
            )
            url = sync_local_run(directory)
            evidence = json.loads((directory / "tracker-sync.json").read_text())
            config = json.loads((directory / "train-config.json").read_text())
        if errors:
            raise LeaseUnavailable("queued tracker sync lost its writer lease") from errors[0]
        if (
            evidence.get("run_id") != run_id
            or evidence.get("backend") != manifest.tracking["backend"]
            or evidence.get("status") != "delivered"
            or int(evidence.get("high_water") or 0) != high_water
        ):
            raise RuntimeError("queued tracker sync evidence disagrees with retained journal")
        _project_queued_outcome(
            authority,
            manifest=manifest,
            recipe=recipe,
            receipt=receipt,
            config=config,
            service_run_id=str(evidence["service_run_id"]),
        )
        prefix = authority.run_prefix(run_id)
        binding_key = f"{prefix}/metrics-binding.json"
        binding = authority.control.get_json_optional(binding_key)
        service_run_id = str(evidence["service_run_id"])
        if binding is not None and (
            binding.get("backend") != manifest.tracking["backend"]
            or binding.get("service_run_id") != service_run_id
        ):
            raise RuntimeError("queued tracker sync would change the frozen service binding")
        if binding is None:
            authority.control.put_json(
                binding_key,
                {
                    "schema_version": 1,
                    "run_id": run_id,
                    "backend": manifest.tracking["backend"],
                    "service_run_id": service_run_id,
                    "service_url": url,
                    "created_at": authority.clock.utc_now(),
                },
                create_only=True,
            )
        result = {
            **evidence,
            "service_url": url,
            "terminal_receipt_key": (
                f"{prefix}/attempts/{receipt.attempt_id}/terminal.json"
            ),
            "completed_at": authority.clock.utc_now(),
        }
        authority.control.put_json(
            f"{prefix}/tracker-sync/evidence/{uuid4().hex}.json",
            result,
            create_only=True,
        )
        latest_key = f"{prefix}/tracker-sync/latest.json"
        existing_latest = authority.control.get_json_optional(latest_key)
        authority.control.put_json(
            latest_key,
            result,
            create_only=existing_latest is None,
            if_match=(
                str(authority.control.head(latest_key)["etag"])
                if existing_latest is not None
                else None
            ),
        )
        if authority.models.get_json_optional(f"{prefix}/index.json") is not None:
            authority.publish_run_telemetry(run_id, tracker_sync_status="delivered")
        return result
    finally:
        stop.set()
        worker.join()
        authority.release_lease(lease)


def main(argv: list[str] | None = None) -> int:
    from gradlab.cli_parser import ExactArgumentParser

    parser = ExactArgumentParser(
        prog="gradlab sync-run",
        description="Project a queued complete_local Run's retained journal to its frozen service.",
    )
    parser.add_argument("run_id")
    args = parser.parse_args(argv)
    load_repository_operator_environment(Path.cwd())
    authority = RunAuthority(RunStorageConfig.from_env())
    raw = authority.manifest(str(args.run_id))
    if raw is None:
        raise ValueError("queued tracker sync requires a Run manifest")
    manifest = RunManifest.from_dict(raw)
    if (manifest.tracking or {}).get("backend") == "mlflow":
        from gradlab.experiment_cli import _preflight_mlflow_service_uri

        _preflight_mlflow_service_uri(os.environ.get("MLFLOW_TRACKING_URI", ""))
        if not all(
            str(os.environ.get(name) or "").strip()
            for name in ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD")
        ):
            raise RuntimeError("queued MLflow sync requires private service credentials")
        profile = str(os.environ.get("MLFLOW_OPERATOR_PROFILE") or "mlflow-default").strip()
        if profile != (manifest.tracking or {}).get("operator_profile"):
            raise RuntimeError("queued MLflow sync operator profile differs from the frozen Run")
    result = sync_queued_run(authority, str(args.run_id))
    print(f"Synchronized queued run: {result['service_url']}")
    return 0
