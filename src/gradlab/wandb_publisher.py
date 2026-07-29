from __future__ import annotations

import json
import math
import os
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from gradlab.env_metadata import env_config_metadata, training_metadata
from gradlab.evaluation_projection import validate_evaluation_metric_payload
from gradlab.metric_names import (
    EVAL_FULL_BY_START,
    LEADER_CHECKPOINT_ARTIFACT_REF,
    LEADER_CHECKPOINT_EVALUATION_SOURCE,
    LEADER_CHECKPOINT_STEP,
    LEADER_CHECKPOINT_UPDATED_AT,
    LEGACY_METRICS_SCHEMA_VERSION,
    METRICS_SCHEMA_VERSION,
    V13_LEADER_CHECKPOINT_ARTIFACT_REF,
    V13_LEADER_CHECKPOINT_EVALUATION_SOURCE,
    evaluation_metric_schema,
    leader_metric_for_rank_metric,
    validate_metric_payload,
)
from gradlab.metric_store import MetricStore
from gradlab.ranking import require_objective_rank
from gradlab.run_contracts import (
    RUN_EARLY_STOP_CONDITION_SUMMARY,
    RUN_EARLY_STOP_TRIGGER_SUMMARY,
    RUN_FINAL_STEP_SUMMARY,
    RUN_STOP_REASON_SUMMARY,
    RUN_TERMINAL_STATE_SUMMARY,
    TerminalReceipt,
)
from gradlab.wandb_utils import (
    configure_wandb_metric_axes,
    configure_wandb_metrics,
    game_family_for_environment,
    load_wandb_env,
    resolve_wandb_namespace,
)


def _write_wandb_identity(run, run_dir: str) -> None:
    if run is None:
        return
    for attribute, filename in (
        ("url", "wandb_url.txt"),
        ("id", "wandb_run_id.txt"),
    ):
        value = getattr(run, attribute, None)
        if value:
            Path(run_dir, filename).write_text(f"{value}\n", encoding="utf-8")


def _wandb_tags(value: Any) -> list[str]:
    if isinstance(value, str):
        return [tag.strip() for tag in value.split(",") if tag.strip()]
    if isinstance(value, list | tuple):
        return [str(tag).strip() for tag in value if str(tag).strip()]
    return []


def _start_wandb(
    train_config: Mapping[str, Any],
    *,
    run_dir: str,
    config: Any,
    goal_variant: Mapping[str, Any] | None = None,
):
    if not train_config.get("wandb"):
        raise ValueError("supervised training requires W&B metric publication")
    load_wandb_env()
    wandb_dir = os.path.abspath(run_dir)
    for env_name, path in {
        "WANDB_DIR": wandb_dir,
        "WANDB_CACHE_DIR": os.path.join(wandb_dir, "wandb", "cache"),
        "WANDB_CONFIG_DIR": os.path.join(wandb_dir, "wandb", "config"),
        "WANDB_DATA_DIR": os.path.join(wandb_dir, "wandb", "data"),
    }.items():
        os.environ.setdefault(env_name, path)
        os.makedirs(os.environ[env_name], exist_ok=True)

    import wandb

    entity, project = resolve_wandb_namespace(
        train_config.get("wandb_entity"),
        train_config.get("wandb_project"),
        config.game,
        env_provider=config.env_provider,
    )
    game_family = game_family_for_environment(config.env_provider, config.game)
    tags = _wandb_tags(train_config.get("wandb_tags"))
    family_tag = f"game_family:{game_family}"
    if family_tag not in tags:
        tags.append(family_tag)
    wandb_config: dict[str, Any] = {
        **train_config,
        "wandb_entity": entity,
        "wandb_project": project,
        "game_family": game_family,
        "wandb_tags": tags,
        **env_config_metadata(config),
    }
    if goal_variant is not None:
        from gradlab.goal_variants import goal_variant_projection

        wandb_config.update(goal_variant_projection(goal_variant))
    wandb_config["metrics_schema_version"] = METRICS_SCHEMA_VERSION
    training = training_metadata(
        config,
        rom_asset_manifest=train_config.get("rom_asset_manifest"),
    )
    wandb_config["environment"] = training["environment"]
    wandb_config["environment_hash"] = training["environment_hash"]
    display_name = str(train_config.get("wandb_display_name") or "").strip() or str(
        train_config["run_name"]
    )
    return configure_wandb_metrics(
        wandb.init(
            project=project,
            entity=entity,
            group=train_config.get("wandb_group"),
            name=display_name,
            notes=train_config.get("run_description") or None,
            tags=tags,
            config=wandb_config,
            dir=wandb_dir,
            sync_tensorboard=False,
            save_code=False,
            mode=str(train_config["wandb_mode"]),
            id=str(train_config["wandb_run_id"]),
            resume="allow",
            settings=wandb.Settings(
                x_server_side_expand_glob_metrics=False,
            ),
        ),
        metrics_schema_version=METRICS_SCHEMA_VERSION,
    )


class WandbProjector:
    """The sole W&B SDK owner for one logical dstack run."""

    def __init__(
        self,
        run,
        *,
        run_dir: str | None = None,
        metrics_schema_version: int = METRICS_SCHEMA_VERSION,
    ) -> None:
        self.run = run
        self.run_dir = run_dir
        self.metrics_schema_version = evaluation_metric_schema(
            metrics_schema_version
        ).version

    @classmethod
    def start_live(
        cls,
        train_config: Mapping[str, Any],
        *,
        run_dir: str,
        config: Any,
        goal_variant: Mapping[str, Any] | None = None,
    ) -> WandbProjector:
        return cls(
            _start_wandb(
                train_config,
                run_dir=run_dir,
                config=config,
                goal_variant=goal_variant,
            ),
            run_dir=run_dir,
        )

    @classmethod
    def resume(
        cls,
        train_config: Mapping[str, Any],
        *,
        allow_create: bool = False,
        update_finish_state: bool = True,
    ) -> WandbProjector:
        run_id = str(train_config.get("wandb_run_id") or "")
        if not run_id:
            raise ValueError("W&B projection requires the producing run id")
        load_wandb_env()
        import wandb

        entity, project = resolve_wandb_namespace(
            train_config.get("wandb_entity"),
            train_config.get("wandb_project"),
            str(train_config.get("game") or ""),
            env_provider=train_config.get("env_provider"),
        )
        raw_tags = train_config.get("wandb_tags") or ()
        tags = (
            [part.strip() for part in str(raw_tags).split(",") if part.strip()]
            if isinstance(raw_tags, str)
            else [str(tag) for tag in raw_tags]
        )
        display_name = (
            str(train_config.get("wandb_display_name") or "").strip()
            or str(train_config.get("run_name") or "").strip()
            or None
        )
        run = configure_wandb_metrics(
            wandb.init(
                entity=entity,
                project=project,
                id=run_id,
                resume="allow" if allow_create else "must",
                mode=str(train_config.get("wandb_mode") or "online"),
                name=display_name,
                group=str(train_config.get("wandb_group") or "") or None,
                tags=tags,
                config=dict(train_config) if allow_create else None,
                settings=wandb.Settings(
                    x_update_finish_state=update_finish_state,
                    x_server_side_expand_glob_metrics=False,
                ),
            ),
            metrics_schema_version=int(
                train_config.get("metrics_schema_version") or METRICS_SCHEMA_VERSION
            ),
        )
        return cls(
            run,
            metrics_schema_version=int(
                train_config.get("metrics_schema_version") or METRICS_SCHEMA_VERSION
            ),
        )

    def close(
        self,
        *,
        timeout_seconds: float | None = None,
        exit_code: int | None = None,
    ) -> None:
        if self.run_dir is not None:
            _write_wandb_identity(self.run, self.run_dir)
        if self.run is None:
            return
        finish_kwargs = {} if exit_code is None else {"exit_code": int(exit_code)}
        if timeout_seconds is None:
            self.run.finish(**finish_kwargs)
            return
        if timeout_seconds <= 0:
            raise ValueError("W&B finish timeout must be positive")
        errors: list[BaseException] = []
        finished = threading.Event()

        def finish() -> None:
            try:
                self.run.finish(**finish_kwargs)
            except BaseException as exc:
                errors.append(exc)
            finally:
                finished.set()

        thread = threading.Thread(
            target=finish,
            name="gradlab-wandb-finish",
            daemon=True,
        )
        thread.start()
        if not finished.wait(timeout_seconds):
            raise TimeoutError(
                f"W&B did not finish uploading within {timeout_seconds:g} seconds"
            )
        if errors:
            raise errors[0]


def _publish_frame(
    run,
    row: Mapping[str, Any],
    *,
    event_seq_offset: int = 0,
    metrics_schema_version: int = METRICS_SCHEMA_VERSION,
) -> None:
    if run is None:
        raise RuntimeError("W&B run is unavailable")
    payload = json.loads(str(row["payload_json"]))
    kind = str(row["kind"])
    event_seq = int(row["id"]) + int(event_seq_offset)
    event_id = str(row["event_id"])
    step = int(row["step"] or 0)
    source = str(row.get("source") or "")

    if kind == "history":
        if source.startswith("eval"):
            validate_evaluation_metric_payload(
                payload,
                schema_version=metrics_schema_version,
            )
        else:
            validate_metric_payload(payload)
        payload["orchestration/event_seq"] = event_seq
        payload["orchestration/event_id"] = event_id
        if source.startswith("eval"):
            payload[evaluation_metric_schema(metrics_schema_version).checkpoint_step] = step
        elif not source.startswith("orchestration"):
            payload["train/global_step"] = step
        # Use the durable outbox sequence as W&B's internal step. If the SDK call
        # succeeded but the local acknowledgement was interrupted, replaying the
        # same sequence is rejected by W&B as an already-committed step instead
        # of appending a second scientific point.
        configure_wandb_metric_axes(
            run,
            payload,
            metrics_schema_version=metrics_schema_version,
        )
        run.log(payload, step=event_seq)
        return

    if kind == "histogram":
        import wandb

        converted: dict[str, object] = {
            "train/global_step": step,
            "orchestration/event_seq": event_seq,
            "orchestration/event_id": event_id,
        }
        for name, values in payload.get("histograms", {}).items():
            converted[str(name)] = wandb.Histogram(values)
        validate_metric_payload(converted)
        configure_wandb_metric_axes(
            run,
            converted,
            metrics_schema_version=metrics_schema_version,
        )
        run.log(converted, step=event_seq)
        return

    if kind == "eval_by_start":
        import wandb

        rows = payload.get("rows")
        if not isinstance(rows, list):
            raise ValueError("eval_by_start frame must contain rows")
        schema = evaluation_metric_schema(metrics_schema_version)
        converted = {
            schema.checkpoint_step: step,
            "orchestration/event_seq": event_seq,
            "orchestration/event_id": event_id,
            EVAL_FULL_BY_START: wandb.Table(
                columns=list(schema.table_columns),
                data=[[step, *list(result)] for result in rows],
            ),
        }
        configure_wandb_metric_axes(
            run,
            converted,
            metrics_schema_version=metrics_schema_version,
        )
        run.log(converted, step=event_seq)
        return

    raise ValueError(f"unsupported supervisor telemetry frame kind: {kind}")


def publish_pending_frames(
    store: MetricStore,
    run,
    *,
    limit: int,
    event_seq_offset: int = 0,
    metrics_schema_version: int = METRICS_SCHEMA_VERSION,
) -> int:
    published = 0
    for row in store.pending_metric_frames(limit=limit):
        frame_id = int(row["id"])
        if not store.claim_metric_frame(frame_id):
            continue
        try:
            _publish_frame(
                run,
                row,
                event_seq_offset=event_seq_offset,
                metrics_schema_version=metrics_schema_version,
            )
        except Exception as exc:
            store.mark_metric_frame_failed(frame_id, repr(exc))
            print(f"W&B frame publish failed id={frame_id}: {exc}", flush=True)
            break
        store.mark_metric_frame_published(
            frame_id,
            step=int(row["step"]) if row.get("step") is not None else None,
        )
        published += 1
    return published


def publish_promotion_summary(
    run,
    *,
    checkpoint_step: int,
    checkpoint_url: str,
    metrics: Mapping[str, Any],
    updated_at: str,
    selection_rank: Sequence[str],
    evaluation_source: str,
    metrics_schema_version: int = METRICS_SCHEMA_VERSION,
) -> None:
    if run is None:
        raise RuntimeError("W&B run is unavailable")
    criteria = require_objective_rank(
        selection_rank,
        metrics_schema_version=metrics_schema_version,
    )
    legacy = metrics_schema_version == LEGACY_METRICS_SCHEMA_VERSION
    artifact_metric = (
        V13_LEADER_CHECKPOINT_ARTIFACT_REF
        if legacy
        else LEADER_CHECKPOINT_ARTIFACT_REF
    )
    source_metric = (
        V13_LEADER_CHECKPOINT_EVALUATION_SOURCE
        if legacy
        else LEADER_CHECKPOINT_EVALUATION_SOURCE
    )
    projection: dict[str, Any] = {
        "gradlab/goal/outcome": "accepted",
        LEADER_CHECKPOINT_STEP: int(checkpoint_step),
        artifact_metric: checkpoint_url,
        source_metric: str(evaluation_source),
        LEADER_CHECKPOINT_UPDATED_AT: updated_at,
    }
    available: dict[str, float] = {}
    for name, value in metrics.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            continue
        try:
            leader_name = leader_metric_for_rank_metric(
                str(name),
                schema_version=metrics_schema_version,
            )
        except ValueError:
            continue
        available[str(name)] = numeric
        projection[leader_name] = numeric
    for criterion in criteria:
        if criterion.metric == LEADER_CHECKPOINT_STEP:
            continue
        if criterion.metric not in available:
            raise ValueError(
                f"promoted checkpoint is missing finite rank metric: {criterion.metric}"
            )
    run.summary.update(projection)


def publish_terminal_summary(run, receipt: TerminalReceipt) -> None:
    if run is None:
        raise RuntimeError("W&B run is unavailable")
    receipt.validate()
    early_stop = receipt.early_stop if isinstance(receipt.early_stop, Mapping) else {}
    run.summary.update(
        {
            RUN_TERMINAL_STATE_SUMMARY: receipt.state,
            RUN_STOP_REASON_SUMMARY: receipt.stop_reason,
            RUN_FINAL_STEP_SUMMARY: int(receipt.final_step),
            RUN_EARLY_STOP_TRIGGER_SUMMARY: str(early_stop.get("trigger") or ""),
            RUN_EARLY_STOP_CONDITION_SUMMARY: str(early_stop.get("condition_id") or ""),
        }
    )
