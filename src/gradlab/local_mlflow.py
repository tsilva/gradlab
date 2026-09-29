"""Lease-owned MLflow delivery for direct local Runs."""

from __future__ import annotations

from contextlib import contextmanager, ExitStack
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import threading
import time
from collections.abc import Callable

from gradlab.env import resolve_env_config
from gradlab.env_config import env_config_from_mapping
from gradlab.local_publication import local_publication
from gradlab.mlflow_delivery import MlflowDelivery, publish_pending_frames
from gradlab.operator_environment import load_repository_operator_environment
from gradlab.policy_bundle import write_canonical_json
from gradlab.supervisor_ledger import SupervisorLedger


@contextmanager
def local_mlflow_writer(run_dir: Path, config: dict):
    """Publish the local outbox while the learner runs, then prove remote visibility."""
    load_repository_operator_environment(Path.cwd(), requested_names={"MLFLOW_TRACKING_URI"})
    uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if not uri:
        raise RuntimeError("selected MLflow service has no private tracking URI")
    with (run_dir / ".metrics-writer.lock").open("a") as lease, ExitStack() as stack:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"a metrics writer already owns {run_dir}") from exc
        store = SupervisorLedger(run_dir / "gradlab.sqlite")
        store.init()
        store.reset_interrupted_metric_frames()
        environment = resolve_env_config(env_config_from_mapping(config))
        publication = stack.enter_context(local_publication(run_dir, config, store, environment))
        started_at = json.loads((run_dir / "local-run.json").read_text())["started_at"]
        created_at_ms = int(datetime.fromisoformat(started_at.replace("Z", "+00:00")).timestamp() * 1000)
        delivery = MlflowDelivery.open(
            tracking_uri=uri,
            experiment_name=f"gradlab-{config.get('game_family') or environment.game}",
            gradlab_run_id=config["wandb_run_id"],
            created_at_ms=created_at_ms,
        )
        if publication is not None:
            key = f"runs/{config['wandb_run_id']}/metrics-binding.json"
            existing = publication.authority.control.get_json_optional(key)
            if existing is None:
                publication.authority.control.put_json(
                    key,
                    {"schema_version": 1, "run_id": config["wandb_run_id"],
                     "backend": "mlflow", "service_run_id": delivery.run_id,
                     "service_url": delivery.service_url,
                     "created_at": started_at},
                    create_only=True,
                )
            elif existing.get("service_run_id") != delivery.run_id:
                raise RuntimeError("MLflow binding changed during local Run recovery")
        finished = threading.Event()
        errors: list[BaseException] = []

        def publish() -> None:
            try:
                while not finished.is_set():
                    if publication is not None:
                        publication.publish()
                    publish_pending_frames(
                        store, delivery, limit=100,
                        heartbeat=publication.check_lease if publication else None,
                    )
                    finished.wait(0.5)
            except BaseException as exc:
                errors.append(exc)

        worker = threading.Thread(target=publish, name="local-mlflow-supervisor", daemon=True)
        worker.start()
        failed = False
        try:
            experiment_id = delivery.client.get_run(delivery.run_id).info.experiment_id
            url = (
                f"{uri.rstrip('/')}/#/experiments/{experiment_id}/runs/{delivery.run_id}"
                if uri.startswith(("http://", "https://"))
                else f"mlflow:{delivery.run_id}"
            )
            yield url
        except BaseException:
            failed = True
            raise
        finally:
            finished.set()
            worker.join()
            try:
                if errors:
                    raise RuntimeError("local MLflow publisher failed") from errors[0]
                deadline = time.monotonic() + 60
                if publication is not None:
                    publication.publish()
                while store.pending_metric_frames(limit=1):
                    count = publish_pending_frames(
                        store, delivery, limit=100,
                        heartbeat=publication.check_lease if publication else None,
                    )
                    if time.monotonic() >= deadline:
                        raise TimeoutError("local MLflow outbox did not drain within 60 seconds")
                    if count == 0:
                        time.sleep(1)
                with store.connection() as connection:
                    high_water = int(connection.execute(
                        "SELECT COALESCE(MAX(id), 0) FROM metric_frames WHERE status = 'published'"
                    ).fetchone()[0])
                while delivery.remote_high_water() < high_water:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("MLflow has not confirmed all local metric frames")
                    time.sleep(2)
                result_path = run_dir / "training-result.json"
                if result_path.is_file():
                    result = json.loads(result_path.read_text())
                    if publication is not None:
                        state, reason = publication.terminal_summary(result)
                    else:
                        state, reason = result["status"], result["terminal_reason"]
                    delivery.publish_terminal(state=state, reason=reason)
                else:
                    delivery.close(success=False)
                if publication is not None:
                    publication.finish(wandb_high_water=high_water)
                write_canonical_json(run_dir / "metrics-delivery.json", {
                    "backend": "mlflow", "service_run_id": delivery.run_id,
                    "high_water": high_water, "status": "delivered",
                })
            except BaseException:
                if not failed:
                    raise


def sync_local_mlflow(
    run_dir: Path, config: dict, *, heartbeat: Callable[[], None] | None = None
) -> str:
    """Append service-delivery evidence for a completed frozen local-only Run."""
    receipt = json.loads((run_dir / "local-run.json").read_text())
    if receipt.get("status") != "complete_local":
        raise ValueError("MLflow sync requires a complete_local training receipt")
    load_repository_operator_environment(Path.cwd(), requested_names={"MLFLOW_TRACKING_URI"})
    uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if not uri:
        raise RuntimeError("selected MLflow service has no private tracking URI")
    with (run_dir / ".metrics-writer.lock").open("a") as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"a metrics writer already owns {run_dir}") from exc
        store = SupervisorLedger(run_dir / "gradlab.sqlite")
        store.init()
        store.reset_interrupted_metric_frames()
        with store.connection() as connection:
            connection.execute(
                "UPDATE metric_frames SET status = 'pending' WHERE status = 'local_only'"
            )
        started_at = json.loads((run_dir / "local-run.json").read_text())["started_at"]
        created_at_ms = int(
            datetime.fromisoformat(started_at.replace("Z", "+00:00")).timestamp() * 1000
        )
        environment = resolve_env_config(env_config_from_mapping(config))
        if heartbeat is not None:
            heartbeat()
        delivery = MlflowDelivery.open(
            tracking_uri=uri,
            experiment_name=f"gradlab-{config.get('game_family') or environment.game}",
            gradlab_run_id=str(config["wandb_run_id"]),
            created_at_ms=created_at_ms,
        )
        deadline = time.monotonic() + 120
        while store.pending_metric_frames(limit=1):
            if heartbeat is not None:
                heartbeat()
            count = publish_pending_frames(store, delivery, limit=100, heartbeat=heartbeat)
            if time.monotonic() >= deadline:
                raise TimeoutError("local MLflow sync did not drain within 120 seconds")
            if count == 0:
                time.sleep(1)
        with store.connection() as connection:
            high_water = int(connection.execute(
                "SELECT COALESCE(MAX(id), 0) FROM metric_frames WHERE status = 'published'"
            ).fetchone()[0])
        while True:
            if heartbeat is not None:
                heartbeat()
            if delivery.remote_high_water() >= high_water:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("MLflow sync did not become remotely visible")
            time.sleep(2)
        if heartbeat is not None:
            heartbeat()
        write_canonical_json(run_dir / "tracker-sync.json", {
            "backend": "mlflow", "run_id": config["wandb_run_id"],
            "service_run_id": delivery.run_id, "high_water": high_water,
            "status": "delivered",
        })
        experiment_id = delivery.client.get_run(delivery.run_id).info.experiment_id
        return (
            f"{uri.rstrip('/')}/#/experiments/{experiment_id}/runs/{delivery.run_id}"
            if uri.startswith(("http://", "https://"))
            else f"mlflow:{delivery.run_id}"
        )
