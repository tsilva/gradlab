"""R2 authority and publication for a lease-owning local training supervisor."""

from __future__ import annotations

from datetime import timedelta
import json
from pathlib import Path
import threading
from uuid import uuid4

from gradlab.checkpoint_contract import checkpoint_manifest_contract_sha256
from gradlab.clock import utc_timestamp
from gradlab.early_stop import validate_metric_early_stop_decision
from gradlab.json_utils import canonical_json_sha256
from gradlab.operator_environment import load_repository_operator_environment
from gradlab.policy_bundle import write_canonical_json, canonical_json_sha256 as recipe_sha256
from gradlab.r2_store import RunStorageConfig
from gradlab.run_authority import RunAuthority, LeaseUnavailable, parse_utc_datetime
from gradlab.run_contracts import RunManifest, TerminalReceipt, EarlyStopReceipt, new_attempt_id
from gradlab.supervisor_ledger import SupervisorLedger
from gradlab.wandb_utils import load_wandb_env, resolve_wandb_namespace


def local_publication(run_dir, config, store, environment):
    """Offline and explicitly disabled runs make no R2 requests."""
    if config.get("wandb_mode", "online") != "online":
        from contextlib import nullcontext

        return nullcontext(None)
    return LocalRunPublication(run_dir, config, store, environment)


class LocalRunPublication:
    def __init__(
        self, run_dir: Path, config: dict, store: SupervisorLedger, environment, *, authority=None
    ):
        self.original_attempt_id = config.get("attempt_id") or ""
        self.run_dir = run_dir
        self.store = store
        self.recipe = json.loads((run_dir / "recipe.json").read_text())
        self.receipt = json.loads((run_dir / "local-run.json").read_text())
        if authority is None:
            names = {
                f"GRADLAB_{scope}_R2_{field}"
                for scope in ("CONTROL", "EVAL", "MODELS")
                for field in (
                    "URI",
                    "ENDPOINT_URL",
                    "REGION",
                    "ACCESS_KEY_ID",
                    "SECRET_ACCESS_KEY",
                    "PUBLIC_BASE_URL",
                )
            }
            load_repository_operator_environment(Path.cwd(), requested_names=names)
            authority = RunAuthority(RunStorageConfig.from_env())
        self.authority = authority
        load_wandb_env()
        entity, project = resolve_wandb_namespace(
            config.get("wandb_entity"),
            config.get("wandb_project"),
            environment.game,
            env_provider=environment.env_provider,
        )
        identity = store.state("local_publication_identity")
        if identity is None:
            identity = {
                "run_id": config["wandb_run_id"],
                "attempt_id": config.get("attempt_id") or new_attempt_id(),
            }
            store.set_state("local_publication_identity", identity)
        if identity["run_id"] != config["wandb_run_id"]:
            raise ValueError("local publication and W&B run identities differ")
        recipe = self.recipe["recipe"]
        runtime = self.recipe["provenance"]["runtime"]
        target = {"kind": "local", "max_steps": int(config["timesteps"])}
        self.manifest = RunManifest(
            **identity,
            created_at=self.receipt["started_at"],
            source_sha=self.recipe["provenance"]["source_commit"],
            image_digest=f"local:sha256:{canonical_json_sha256(runtime)}",
            goal_slug=recipe["goal_variant"]["goal_slug"],
            goal_sha256=recipe["goal_variant"]["effective_goal_contract_sha256"],
            recipe_slug=config.get("recipe_slug") or recipe["recipe_id"],
            recipe_sha256=recipe_sha256(self.recipe),
            recipe_overrides=recipe.get("recipe_overrides") or [],
            environment_sha256=recipe["environment_hash"].removeprefix("sha256:"),
            seed=int(config["seed"]),
            run_description=config["run_description"],
            compute={
                "execution_backend": "local-process",
                "request": target,
                "selected": target,
                "runtime": runtime,
            },
            wandb={
                "run_id": identity["run_id"],
                "entity": entity,
                "project": project,
                "url": f"https://wandb.ai/{entity}/{project}/runs/{identity['run_id']}",
                "group": identity["run_id"],
                "display_name": config.get("wandb_display_name") or config["run_name"],
            },
            modal={"enabled": False},
            storage=authority.storage.manifest_locations(),
            goal_variant=recipe["goal_variant"],
        )
        self.manifest.validate()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._error = None
        self._lease = None

    def __enter__(self):
        manifest = self.manifest
        self._lease = self.authority.acquire_lease(
            run_id=manifest.run_id, attempt_id=manifest.attempt_id, holder_id=f"local-{uuid4().hex}"
        )
        self._worker = threading.Thread(target=self._renew, name="local-r2-lease", daemon=True)
        self._worker.start()
        try:
            self.authority.put_recipe_document(self.recipe, expected_sha256=manifest.recipe_sha256)
            self.check_lease()
            self.authority.create_manifest(manifest)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def _renew(self):
        while not self._stop.wait(15):
            try:
                with self._lock:
                    self._lease = self.authority.renew_lease(self._lease)
            except BaseException as exc:
                self._error = exc
                return

    def check_lease(self):
        with self._lock:
            if self._error is not None:
                raise LeaseUnavailable("local supervisor lost its R2 writer lease") from self._error
            if self.authority.clock.utc_datetime() >= parse_utc_datetime(self._lease.expires_at):
                raise LeaseUnavailable("local supervisor R2 writer lease expired")

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._worker.join()
        try:
            self.authority.release_lease(self._lease)
        except LeaseUnavailable:
            if exc_type is None:
                raise

    @property
    def public_index_url(self):
        return f"{self.manifest.storage['public_models_base_url']}/runs/{self.manifest.run_id}/index.json"

    def publish(self):
        self.check_lease()
        manifest = self.manifest
        for row in self.store.checkpoints():
            self.check_lease()
            published = self.store.checkpoint_publication(row["id"])
            if published is None:
                try:
                    checkpoint = self.authority.publish_checkpoint(
                        run_id=manifest.run_id,
                        model_path=Path(row["path"]),
                        step=int(row["step"]),
                        purpose="final" if row["kind"] in {"final", "interrupted"} else "periodic",
                        contract_hashes={
                            "goal_sha256": manifest.goal_sha256,
                            "recipe_sha256": manifest.recipe_sha256,
                            "environment_sha256": manifest.environment_sha256,
                            "evaluation_contract_sha256": checkpoint_manifest_contract_sha256(
                                self.recipe
                            ),
                        },
                        recovery_sidecar={
                            "schema_version": 1,
                            "run_id": manifest.run_id,
                            "attempt_id": manifest.attempt_id,
                            "checkpoint_ledger_id": row["id"],
                            "kind": row["kind"],
                            "local_path": row["path"],
                        },
                        created_at=utc_timestamp(row["created_at"]),
                        heartbeat=self.check_lease,
                    )
                    published = checkpoint.to_dict()
                    self.store.record_checkpoint_publication(
                        checkpoint_ledger_id=row["id"], manifest=published
                    )
                except Exception as exc:
                    self.store.mark_checkpoint_upload_failed(row["id"], str(exc))
                    raise
            self.store.mark_checkpoint_uploaded(row["id"], published["public_url"])
        while events := self.store.next_metric_events():
            self.check_lease()
            key, digest = self.authority.seal_metric_segment(
                run_id=manifest.run_id, attempt_id=manifest.attempt_id, events=events
            )
            self.store.record_metric_segment(events=events, object_key=key, sha256=digest)

    def _outcome(self, result, completed_at):
        state = {"completed": "succeeded", "interrupted": "interrupted", "failed": "failed"}.get(
            result["status"]
        )
        if state is None:
            raise ValueError("local training result has an invalid terminal status")
        reason = result["terminal_reason"]
        if not reason.startswith("early_stop_"):
            return state, reason, None
        path = self.run_dir / f"early_stop_decision-{self.original_attempt_id}.json"
        decision = validate_metric_early_stop_decision(
            json.loads(path.read_text()),
            self.recipe["recipe"]["train_config"]["early_stop"],
            label="local learner early-stop decision",
        )
        outcome = decision["outcome"]
        if reason != f"early_stop_{outcome}":
            raise ValueError("local terminal reason disagrees with its early-stop decision")
        keys = (
            "condition_id",
            "matched_condition_ids",
            "outcome",
            "trigger",
            "metric",
            "metric_step",
            "value",
            "best_value",
            "elapsed_steps",
            "patience_progress",
            "condition",
            "early_stop_config_sha256",
        )
        receipt = EarlyStopReceipt(
            run_id=self.manifest.run_id,
            attempt_id=self.manifest.attempt_id,
            **{key: decision[key] for key in keys},
            decision_sha256=canonical_json_sha256(decision),
            recorded_at=completed_at,
        )
        return (
            {"success": "succeeded", "neutral": "stopped", "failure": "failed"}[outcome],
            f"{reason}:{receipt.condition_id}",
            receipt.to_dict(),
        )

    def terminal_summary(self, result):
        """Project the same outcome to W&B and the catalog before SDK finish."""
        terminal = self.store.state("local_publication_terminal")
        if terminal is not None:
            return terminal["state"], terminal["stop_reason"]
        state, reason, _ = self._outcome(
            result, result.get("terminal_at") or self.authority.clock.utc_now()
        )
        return state, reason

    def finish(self, *, wandb_high_water: int):
        self.publish()
        result_path = self.run_dir / "training-result.json"
        if not result_path.is_file():
            raise RuntimeError(
                "local training has no terminal result; artifacts retained for recovery"
            )
        result = json.loads(result_path.read_text())
        if self.store.metric_segment_high_water() != wandb_high_water:
            raise RuntimeError("local R2 and W&B metric high-water marks differ")
        if int(result.get("final_step") or 0) > 0 and not any(
            row["purpose"] == "final" and row["step"] == int(result["final_step"])
            for row in self.store.checkpoint_publications()
        ):
            raise RuntimeError("local training final checkpoint is missing from R2")
        terminal = self.store.state("local_publication_terminal")
        if terminal is None:
            archive = self.authority.archive_metric_journals(
                run_id=self.manifest.run_id, heartbeat=self.check_lease
            )
            completed_at = (
                result.get("terminal_at")
                or self.receipt.get("completed_at")
                or self.authority.clock.utc_now()
            )
            state, reason, early_stop = self._outcome(result, completed_at)
            if early_stop is not None:
                self.check_lease()
                self.authority.create_early_stop(EarlyStopReceipt.from_dict(early_stop))
            receipt = TerminalReceipt(
                run_id=self.manifest.run_id,
                attempt_id=self.manifest.attempt_id,
                state=state,
                acceptance_required=False,
                stop_reason=reason,
                early_stop=early_stop,
                final_step=int(result["final_step"]),
                checkpoint_inventory=self.store.checkpoint_publications(),
                eval_inventory=[],
                wandb_high_water_mark=wandb_high_water,
                drain={
                    "complete": True,
                    "metric_segment_high_water": wandb_high_water,
                    "wandb_remote_high_water_mark": wandb_high_water,
                    "journal_archive": archive,
                    "journal_expires_at": (
                        self.authority.clock.utc_datetime() + timedelta(days=7)
                    ).isoformat(),
                },
                completed_at=completed_at,
            )
            terminal = receipt.to_dict()
            self.store.set_state("local_publication_terminal", terminal)
        self.check_lease()
        self.authority.create_attempt_terminal(
            TerminalReceipt.from_dict(terminal), metrics=self.store.latest_metrics()
        )
        write_canonical_json(
            self.run_dir / "publication-delivery.json",
            {
                "status": "delivered",
                "run_id": self.manifest.run_id,
                "attempt_id": self.manifest.attempt_id,
                "public_index_url": self.public_index_url,
                "checkpoint_count": len(self.store.checkpoint_publications()),
                "metric_high_water": wandb_high_water,
            },
        )
