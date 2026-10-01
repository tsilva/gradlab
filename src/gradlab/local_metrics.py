"""One lease-owned metrics writer for direct training and later journal sync."""

from __future__ import annotations

import fcntl
import json
import os
import threading
from collections.abc import Callable
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path

from gradlab.clock import parse_utc_datetime
from gradlab.env import resolve_env_config
from gradlab.env_config import env_config_from_mapping
from gradlab.local_publication import local_publication
from gradlab.mlflow_delivery import MlflowDelivery, mlflow_experiment_name
from gradlab.operator_environment import load_repository_operator_environment
from gradlab.policy_bundle import write_canonical_json
from gradlab.selected_delivery import MetricsWriter
from gradlab.supervisor_ledger import SupervisorLedger
from gradlab.train_config import wandb_publication_enabled
from gradlab.wandb_publisher import WandbProjector, wandb_delivery_high_water


def _open_delivery(run_dir, config, environment, publication, *, backfill):
    backend = (config.get("tracking") or {}).get("backend", "wandb")
    if backend == "mlflow":
        load_repository_operator_environment(Path.cwd(), requested_names={"MLFLOW_TRACKING_URI"})
        uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
        if not uri:
            raise RuntimeError("selected MLflow service has no private tracking URI")
        started_at = json.loads((run_dir / "local-run.json").read_text())["started_at"]
        delivery = MlflowDelivery.open(
            tracking_uri=uri,
            experiment_name=mlflow_experiment_name(config),
            gradlab_run_id=config["wandb_run_id"],
            created_at_ms=int(parse_utc_datetime(started_at).timestamp() * 1000),
        )
        if publication is not None:
            key = f"runs/{config['wandb_run_id']}/metrics-binding.json"
            existing = publication.authority.control.get_json_optional(key)
            if existing is None:
                publication.authority.control.put_json(
                    key,
                    {
                        "schema_version": 1,
                        "run_id": config["wandb_run_id"],
                        "backend": backend,
                        "service_run_id": delivery.run_id,
                        "service_url": delivery.service_url,
                        "created_at": started_at,
                    },
                    create_only=True,
                )
            elif existing.get("service_run_id") != delivery.run_id:
                raise RuntimeError("MLflow binding changed during local Run recovery")
        return (
            delivery,
            delivery.service_url or f"mlflow:{delivery.run_id}",
            delivery.remote_high_water(),
        )
    if backfill:
        config = {
            **config,
            "tracking_original": config.get("tracking"),
            "tracking": {"backend": "wandb", "delivery": "online"},
            "wandb_mode": "online",
            "wandb_group": config["wandb_run_id"],
            "wandb_display_name": config.get("wandb_display_name") or config["run_name"],
        }
    recipe = run_dir / "recipe.json"
    delivery = WandbProjector.start_live(
        config,
        run_dir=str(run_dir),
        config=environment,
        goal_variant=json.loads(recipe.read_text())["recipe"].get("goal_variant")
        if recipe.is_file()
        else None,
    )
    url = str(delivery.run.url or "")
    (run_dir / "wandb_url.txt").write_text(url + "\n")
    (run_dir / "wandb_run_id.txt").write_text(str(delivery.run.id) + "\n")
    print(f"W&B run: {url}", flush=True)
    return delivery, url, wandb_delivery_high_water(dict(delivery.run.summary))


def _finish(delivery, backend, result, *, failed):
    if backend == "wandb":
        if result is not None:
            delivery.run.summary.update({"ops/state": result[0], "ops/reason": result[1]})
        delivery.close(timeout_seconds=60, exit_code=int(failed))
    elif result is not None:
        delivery.publish_terminal(state=result[0], reason=result[1])
    else:
        delivery.close(success=False)


@contextmanager
def local_metrics_writer(
    run_dir: Path,
    config: dict,
    *,
    backfill: bool = False,
    heartbeat: Callable[[], None] | None = None,
):
    """Own recovery, publication, remote acknowledgement, and terminal evidence."""
    if not backfill and not wandb_publication_enabled(config):
        yield None
        return
    backend = (config.get("tracking") or {}).get("backend", "wandb")
    with (run_dir / ".metrics-writer.lock").open("a") as lease, ExitStack() as stack:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"a metrics writer already owns {run_dir}") from exc
        store = SupervisorLedger(run_dir / "gradlab.sqlite")
        store.init()
        environment = resolve_env_config(env_config_from_mapping(config))
        publication = stack.enter_context(
            nullcontext(None)
            if backfill
            else local_publication(run_dir, config, store, environment)
        )

        def check_writer():
            if publication is not None:
                publication.check_lease()
            if heartbeat is not None:
                heartbeat()

        if publication is not None:
            config = {
                **config,
                "attempt_id": publication.manifest.attempt_id,
                "public_run_index_url": publication.public_index_url,
            }
        check_writer()
        delivery, url, remote_high_water = _open_delivery(
            run_dir, config, environment, publication, backfill=backfill
        )
        writer = MetricsWriter(store, delivery, heartbeat=check_writer)
        writer.recover(remote_high_water, backfill=backfill)
        stopped = threading.Event()
        errors = []

        def publish():
            try:
                while not stopped.is_set():
                    if publication is not None:
                        publication.publish()
                    writer.publish()
                    stopped.wait(0.5)
            except BaseException as exc:
                errors.append(exc)

        worker = threading.Thread(target=publish, name="local-metrics-supervisor", daemon=True)
        worker.start()
        failed = closed = False
        try:
            yield url
        except BaseException:
            failed = True
            raise
        finally:
            stopped.set()
            worker.join()
            try:
                if errors:
                    raise RuntimeError("local metrics publisher failed") from errors[0]
                if publication is not None:
                    publication.publish()
                high_water = writer.drain(timeout_seconds=120 if backfill else 60)
                result_path = run_dir / "training-result.json"
                result = None
                if result_path.is_file():
                    document = json.loads(result_path.read_text())
                    result = (
                        publication.terminal_summary(document)
                        if publication
                        else (document["status"], document["terminal_reason"])
                    )
                check_writer()
                # W&B must finish its SDK queue before checking remote visibility;
                # MLflow writes synchronously and confirms before terminal mutation.
                if backend == "wandb":
                    closed = True
                    _finish(delivery, backend, result, failed=failed)
                if (
                    backfill
                    or config.get("wandb_mode", "online") == "online"
                    or backend == "mlflow"
                ):
                    writer.confirm(high_water)
                if backend == "mlflow" and not backfill:
                    closed = True
                    _finish(delivery, backend, result, failed=failed)
                if publication is not None:
                    publication.finish(wandb_high_water=high_water)
                evidence = {
                    "backend": backend,
                    "run_id": config.get("wandb_run_id", delivery.run_id),
                    "service_run_id": delivery.run_id,
                    "high_water": high_water,
                    "status": "delivered",
                }
                if backfill:
                    write_canonical_json(run_dir / "tracker-sync.json", evidence)
                elif backend == "mlflow":
                    write_canonical_json(run_dir / "metrics-delivery.json", evidence)
                else:
                    mode = config.get("wandb_mode", "online")
                    write_canonical_json(
                        run_dir / "wandb-delivery.json",
                        {
                            "run_id": delivery.run_id,
                            "url": url,
                            "mode": mode,
                            "high_water": high_water,
                            "status": "delivered" if mode == "online" else "offline",
                        },
                    )
            except BaseException:
                if not closed:
                    try:
                        check_writer()
                        _finish(delivery, backend, None, failed=True)
                    except Exception:
                        pass
                if not failed:
                    raise
                print(
                    "Run publication also failed; saved metrics and checkpoints require synchronization.",
                    flush=True,
                )


def sync_local_run(run_dir: Path, *, heartbeat: Callable[[], None] | None = None) -> str:
    """Append delivery evidence without rewriting the frozen Run or terminal receipt."""
    receipt = json.loads((run_dir / "local-run.json").read_text())
    if receipt.get("status") != "complete_local":
        raise ValueError("tracker sync requires a complete_local training receipt")
    config = json.loads((run_dir / "train-config.json").read_text())
    if (config.get("tracking") or {}).get("delivery") != "local_only":
        raise ValueError("explicit sync requires a frozen local-only Run")
    if not config.get("wandb_run_id"):
        raise ValueError("local Run has no frozen GradLab identity for tracker sync")
    with local_metrics_writer(run_dir, config, backfill=True, heartbeat=heartbeat) as url:
        return str(url)


def main(argv: list[str] | None = None) -> int:
    from gradlab.cli_parser import ExactArgumentParser

    parser = ExactArgumentParser(
        prog="gradlab sync",
        description="Project a finished local Run to its frozen metrics service without retraining.",
    )
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    print(f"Synchronized local run: {sync_local_run(args.run_dir.expanduser().resolve())}")
    return 0
