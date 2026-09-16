"""Single-writer W&B delivery for local training and retained local metrics."""

from __future__ import annotations

from contextlib import contextmanager, ExitStack
import fcntl
import json
from pathlib import Path
import threading
import time
from uuid import uuid4

from gradlab.env import resolve_env_config
from gradlab.env_config import env_config_from_mapping
from gradlab.supervisor_ledger import SupervisorLedger
from gradlab.local_publication import local_publication
from gradlab.policy_bundle import write_canonical_json
from gradlab.train_config import wandb_publication_enabled
from gradlab.wandb_publisher import (
    WandbProjector,
    publish_pending_frames,
    wandb_delivery_high_water,
)


@contextmanager
def local_wandb_writer(run_dir: Path, config: dict, *, backfill: bool = False):
    """Own the local writer lease until the SDK and remote metric drain finish."""
    if not backfill and not wandb_publication_enabled(config):
        yield None
        return
    with (run_dir / ".wandb-writer.lock").open("a") as lease, ExitStack() as stack:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"a W&B writer already owns {run_dir}") from exc
        if backfill:
            config = _backfill_config(run_dir, config)
        store = SupervisorLedger(run_dir / "gradlab.sqlite")
        store.init()
        store.reset_interrupted_metric_frames()
        if backfill:
            with store.connection() as connection:
                connection.execute(
                    "UPDATE metric_frames SET status = 'pending' WHERE status = 'local_only'"
                )
        recipe_path = run_dir / "recipe.json"
        goal_variant = (
            json.loads(recipe_path.read_text())["recipe"].get("goal_variant")
            if recipe_path.is_file()
            else None
        )
        environment = resolve_env_config(env_config_from_mapping(config))
        publication = stack.enter_context(local_publication(run_dir, config, store, environment))
        if publication is not None:
            config = {
                **config,
                "attempt_id": publication.manifest.attempt_id,
                "public_run_index_url": publication.public_index_url,
            }
        projector = WandbProjector.start_live(
            config,
            run_dir=str(run_dir),
            config=environment,
            goal_variant=goal_variant,
        )
        run = projector.run
        # Reconcile SDK-acknowledged rows against the resumed remote run after a crash.
        remote_high_water = wandb_delivery_high_water(dict(run.summary))
        with store.connection() as connection:
            connection.execute(
                "UPDATE metric_frames SET status = 'pending' WHERE status = 'published' AND id > ?",
                (remote_high_water,),
            )
        url = str(run.url or "")
        (run_dir / "wandb_url.txt").write_text(url + "\n")
        (run_dir / "wandb_run_id.txt").write_text(str(run.id) + "\n")
        print(f"W&B run: {url}", flush=True)
        finished = threading.Event()
        errors: list[BaseException] = []

        def publish() -> None:
            try:
                while not finished.is_set():
                    if publication is not None:
                        publication.publish()
                    publish_pending_frames(
                        store,
                        run,
                        limit=100,
                        heartbeat=publication.check_lease if publication else None,
                    )
                    finished.wait(0.5)
            except BaseException as exc:
                errors.append(exc)

        worker = threading.Thread(target=publish, name="local-wandb-supervisor", daemon=True)
        worker.start()
        learner_failed = False
        try:
            yield url
        except BaseException:
            learner_failed = True
            raise
        finally:
            finished.set()
            # SDK log() queues locally; no network request runs on the learner thread.
            worker.join()
            close_started = False
            try:
                if errors:
                    raise RuntimeError("local run publisher failed") from errors[0]
                deadline = time.monotonic() + 60
                if publication is not None:
                    publication.publish()
                while store.pending_metric_frames(limit=1):
                    if publication is not None:
                        publication.check_lease()
                    count = publish_pending_frames(
                        store,
                        run,
                        limit=100,
                        heartbeat=publication.check_lease if publication else None,
                    )
                    if time.monotonic() >= deadline:
                        raise TimeoutError("local W&B outbox did not drain within 60 seconds")
                    if not count:
                        time.sleep(1)
                result_file = run_dir / "training-result.json"
                if result_file.is_file():
                    result = json.loads(result_file.read_text())
                    if publication is not None:
                        publication.check_lease()
                        state, reason = publication.terminal_summary(result)
                    else:
                        state, reason = result["status"], result["terminal_reason"]
                    run.summary["ops/state"] = state
                    run.summary["ops/reason"] = reason
                with store.connection() as connection:
                    high_water = int(
                        connection.execute(
                            "SELECT COALESCE(MAX(id), 0) FROM metric_frames WHERE status = 'published'"
                        ).fetchone()[0]
                    )
                close_started = True
                projector.close(timeout_seconds=60, exit_code=1 if learner_failed else 0)
                if config.get("wandb_mode", "online") == "online":
                    _verify_remote_delivery(run.path, high_water)
                if publication is not None:
                    publication.finish(wandb_high_water=high_water)
                write_canonical_json(
                    run_dir / "wandb-delivery.json",
                    {
                        "run_id": str(run.id),
                        "url": url,
                        "mode": config.get("wandb_mode", "online"),
                        "high_water": high_water,
                        "status": "delivered"
                        if config.get("wandb_mode", "online") == "online"
                        else "offline",
                    },
                )
            except BaseException:
                if not close_started:
                    try:
                        projector.close(timeout_seconds=60, exit_code=1)
                    except Exception:
                        pass
                if not learner_failed:
                    raise
                # Preserve the learner exception; retained outbox rows remain recoverable.
                print(
                    "Run publication also failed; saved metrics and checkpoints require synchronization.",
                    flush=True,
                )


def _verify_remote_delivery(path: str, high_water: int) -> None:
    import wandb

    deadline = time.monotonic() + 60
    while True:
        if wandb_delivery_high_water(dict(wandb.Api(timeout=10).run(path).summary)) >= high_water:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("W&B has not confirmed all local metric frames")
        time.sleep(2)


def _backfill_config(run_dir: Path, original: dict) -> dict:
    config = dict(original)
    identity_path = run_dir / "wandb-backfill.json"
    # Called under the writer lease; retries retain one W&B identity.
    if not identity_path.exists():
        write_canonical_json(
            identity_path, {"run_id": config.get("wandb_run_id") or f"gradlab-{uuid4().hex}"}
        )
    run_id = json.loads(identity_path.read_text())["run_id"]
    config.update(
        {
            "original_wandb_mode": config.get("wandb_mode", "disabled"),
            "wandb_mode": "online",
            "wandb_run_id": run_id,
            "wandb_group": run_id,
            "wandb_display_name": config.get("wandb_display_name") or config["run_name"],
        }
    )
    return config


def sync_local_run(run_dir: Path) -> str:
    """Upload a completed local run without changing its checkpoint or training contract."""
    receipt = json.loads((run_dir / "local-run.json").read_text())
    if receipt.get("status") == "running":
        raise ValueError("cannot backfill a running local training session")
    config = json.loads((run_dir / "train-config.json").read_text())
    with local_wandb_writer(run_dir, config, backfill=True) as url:
        return str(url)


def main(argv: list[str] | None = None) -> int:
    from gradlab.cli_parser import ExactArgumentParser

    parser = ExactArgumentParser(
        prog="gradlab sync",
        description="Sync a finished local run to W&B, R2, and the playback catalog without retraining.",
    )
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    url = sync_local_run(args.run_dir.expanduser().resolve())
    print(f"Synchronized local run: {url}")
    return 0
