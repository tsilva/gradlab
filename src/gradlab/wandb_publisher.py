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
    EVAL_CHECKPOINT_STEP,
    EVAL_FULL_START_TABLE,
    EVAL_START_TABLE_COLUMNS,
    LEADER_CHECKPOINT_ARTIFACT_REF,
    LEADER_CHECKPOINT_EVALUATION_SOURCE,
    LEADER_CHECKPOINT_PROJECTION_TIMESTAMP,
    LEADER_CHECKPOINT_STEP,
    EPISODE_METRIC_WINDOW_SIZE,
    METRICS_SCHEMA_VERSION,
    METRICS_EPISODE_WINDOW_SIZE_CONFIG,
    ORCHESTRATION_EVENT_SEQUENCE,
    ORCHESTRATION_RUN_TERMINAL_REASON,
    ORCHESTRATION_RUN_TERMINAL_STATE,
    leader_metric_for_rank_metric,
    require_current_metrics_schema,
    summary_metric_value,
    summary_value,
    validate_metric_payload,
)
from gradlab.metric_store import MetricStore
from gradlab.ranking import require_objective_rank
from gradlab.run_contracts import TerminalReceipt
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


def wandb_delivery_high_water(summary: Mapping[str, Any]) -> int:
    reducer_high_water = int(
        summary_metric_value(summary, ORCHESTRATION_EVENT_SEQUENCE) or 0
    )
    history_high_water = int(summary_value(summary.get("_step")) or 0)
    return max(reducer_high_water, history_high_water)


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
    from gradlab.train_config import wandb_publication_enabled

    if not wandb_publication_enabled(train_config):
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
    wandb_config[METRICS_EPISODE_WINDOW_SIZE_CONFIG] = EPISODE_METRIC_WINDOW_SIZE
    training = training_metadata(
        config,
        rom_asset_manifest=train_config.get("rom_asset_manifest"),
    )
    wandb_config["environment"] = training["environment"]
    wandb_config["environment_hash"] = training["environment_hash"]
    display_name = str(train_config.get("wandb_display_name") or "").strip()
    if not display_name:
        raise ValueError("W&B projection requires wandb_display_name")
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
        self.metrics_schema_version = require_current_metrics_schema(metrics_schema_version)

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
        display_name = str(train_config.get("wandb_display_name") or "").strip()
        if not display_name:
            raise ValueError("W&B projection requires wandb_display_name")
        metrics_schema_version = require_current_metrics_schema(
            train_config.get("metrics_schema_version")
        )
        resume_config = dict(train_config)
        resume_config[METRICS_EPISODE_WINDOW_SIZE_CONFIG] = EPISODE_METRIC_WINDOW_SIZE
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
                config=resume_config if allow_create else None,
                settings=wandb.Settings(
                    x_update_finish_state=update_finish_state,
                    x_server_side_expand_glob_metrics=False,
                ),
            ),
            metrics_schema_version=metrics_schema_version,
        )
        return cls(
            run,
            metrics_schema_version=metrics_schema_version,
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
        payload[ORCHESTRATION_EVENT_SEQUENCE] = event_seq
        if source.startswith("eval"):
            require_current_metrics_schema(metrics_schema_version)
            payload[EVAL_CHECKPOINT_STEP] = step
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

    if kind == "eval_by_start":
        import wandb

        records = payload.get("records")
        if not isinstance(records, list) or not all(
            isinstance(record, Mapping) for record in records
        ):
            raise ValueError("eval_by_start frame must contain records")
        require_current_metrics_schema(metrics_schema_version)
        converted = {
            EVAL_CHECKPOINT_STEP: step,
            ORCHESTRATION_EVENT_SEQUENCE: event_seq,
            EVAL_FULL_START_TABLE: wandb.Table(
                columns=list(EVAL_START_TABLE_COLUMNS),
                data=[
                    [
                        str(record["start_id"]),
                        int(record["episode_count"]),
                        int(record["success_count"]),
                        float(record["success_rate"]),
                        float(record["shaped_return_mean"]),
                        dict(record.get("failure_reasons") or {}),
                    ]
                    for record in records
                ],
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
    projection: dict[str, Any] = {
        LEADER_CHECKPOINT_STEP: int(checkpoint_step),
        LEADER_CHECKPOINT_ARTIFACT_REF: checkpoint_url,
        LEADER_CHECKPOINT_EVALUATION_SOURCE: str(evaluation_source),
        LEADER_CHECKPOINT_PROJECTION_TIMESTAMP: updated_at,
    }
    for criterion in criteria:
        leader_name = leader_metric_for_rank_metric(
            criterion.metric,
            schema_version=metrics_schema_version,
        )
        if leader_name == LEADER_CHECKPOINT_STEP:
            continue
        value = metrics.get(criterion.metric)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(
                f"promoted checkpoint is missing finite rank metric: {criterion.metric}"
            )
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(
                f"promoted checkpoint is missing finite rank metric: {criterion.metric}"
            )
        projection[leader_name] = numeric
    validate_metric_payload(projection, placement="summary")
    run.summary.update(projection)


def promotion_summary_matches(
    summary: Mapping[str, Any],
    *,
    checkpoint_step: int,
    checkpoint_url: str,
    updated_at: str,
    selection_rank: Sequence[str],
    metrics_schema_version: int = METRICS_SCHEMA_VERSION,
) -> bool:
    """Return whether W&B exposes the complete projection for one promotion receipt."""

    criteria = require_objective_rank(
        selection_rank,
        metrics_schema_version=metrics_schema_version,
    )
    if str(summary_value(summary.get(LEADER_CHECKPOINT_ARTIFACT_REF)) or "") != str(
        checkpoint_url
    ):
        return False
    try:
        remote_step = int(summary_value(summary.get(LEADER_CHECKPOINT_STEP)))
    except (TypeError, ValueError):
        return False
    if remote_step != int(checkpoint_step):
        return False
    if str(summary_value(summary.get(LEADER_CHECKPOINT_PROJECTION_TIMESTAMP)) or "") != str(
        updated_at
    ):
        return False
    if not str(summary_value(summary.get(LEADER_CHECKPOINT_EVALUATION_SOURCE)) or "").strip():
        return False
    for criterion in criteria:
        leader_name = leader_metric_for_rank_metric(
            criterion.metric,
            schema_version=metrics_schema_version,
        )
        value = summary_value(summary.get(leader_name))
        if isinstance(value, bool) or not isinstance(value, int | float):
            return False
        if not math.isfinite(float(value)):
            return False
    return True


def publish_terminal_summary(run, receipt: TerminalReceipt) -> None:
    if run is None:
        raise RuntimeError("W&B run is unavailable")
    receipt.validate()
    projection = {
        ORCHESTRATION_RUN_TERMINAL_STATE: receipt.state,
        ORCHESTRATION_RUN_TERMINAL_REASON: receipt.stop_reason,
    }
    validate_metric_payload(projection, placement="summary")
    run.summary.update(projection)
