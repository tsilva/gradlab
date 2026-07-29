from __future__ import annotations

import os
from pathlib import Path
from weakref import WeakKeyDictionary

from gradlab.dotenv import load_env_file
from gradlab.env_registry import game_family_for_environment, wandb_project_for_environment
from gradlab.metric_names import (
    EVAL_ACCEPTANCE_PASS,
    METRICS_SCHEMA_VERSION,
    ORCHESTRATION_EVENT_SEQ,
    TRAIN_EPISODE_RETURN_SHAPED_ACROSS_ORIGINS_ROLLING_UP_TO_100_MEAN,
    TRAIN_GLOBAL_STEP,
    evaluation_metric_schema,
)
from gradlab.operator_credentials import (
    load_operator_environment,
    reject_protected_dotenv,
)

DEFAULT_WANDB_ENTITY = "tsilva"
DEFAULT_WANDB_PROJECT = "SuperMarioBros-Nes-v0"
DEFAULT_WANDB_PROJECT_PATH = f"{DEFAULT_WANDB_ENTITY}/{DEFAULT_WANDB_PROJECT}"

WANDB_ENV_PREFIXES = ("WANDB_",)
_WANDB_AXIS_METRICS: WeakKeyDictionary[object, set[str]] = WeakKeyDictionary()


def load_wandb_env(dotenv_path: str | Path = ".env") -> None:
    """Resolve only W&B configuration from safe local operator sources."""
    reject_protected_dotenv(dotenv_path)
    load_env_file(
        dotenv_path,
        key_filter=lambda key: key.startswith(WANDB_ENV_PREFIXES),
    )
    load_operator_environment(
        requested_names={"WANDB_API_KEY", "WANDB_ENTITY"},
    )


def wandb_entity_from_env(*, fallback: str = DEFAULT_WANDB_ENTITY) -> str:
    entity = str(os.environ.get("WANDB_ENTITY") or "").strip()
    return entity or fallback


def canonical_wandb_environment(
    env_provider: object,
    env_id: object,
    *,
    fallback: str = DEFAULT_WANDB_PROJECT,
) -> tuple[str, str]:
    """Return the canonical W&B project and provider-neutral game family."""

    project = wandb_project_for_environment(
        env_provider,
        env_id,
        fallback=fallback,
    )
    family = game_family_for_environment(
        env_provider,
        env_id,
        fallback=project,
    )
    return project, family


def wandb_project_for_env_id(
    env_id: str | None,
    *,
    env_provider: object = None,
    fallback: str = DEFAULT_WANDB_PROJECT,
) -> str:
    """Return the default W&B project name for a provider-local environment id."""

    return canonical_wandb_environment(env_provider, env_id, fallback=fallback)[0]


def resolve_wandb_project(
    explicit_project: object,
    env_id: str | None,
    *,
    env_provider: object = None,
) -> str:
    """Use explicit W&B project when supplied, otherwise default to the env id."""

    project = str(explicit_project or "").strip()
    return project or wandb_project_for_env_id(env_id, env_provider=env_provider)


def resolve_wandb_namespace(
    explicit_entity: object,
    explicit_project: object,
    env_id: str | None,
    *,
    env_provider: object = None,
) -> tuple[str, str]:
    entity = str(explicit_entity or "").strip() or wandb_entity_from_env()
    project = resolve_wandb_project(
        explicit_project,
        env_id,
        env_provider=env_provider,
    )
    return entity, project


def configure_wandb_metrics(
    run,
    *,
    metrics_schema_version: int = METRICS_SCHEMA_VERSION,
):
    eval_schema = evaluation_metric_schema(metrics_schema_version)
    if run is not None:
        run.define_metric(TRAIN_GLOBAL_STEP, summary="max")
        run.define_metric(eval_schema.checkpoint_step, summary="max")
        run.define_metric(ORCHESTRATION_EVENT_SEQ, summary="max")
        run.define_metric(
            EVAL_ACCEPTANCE_PASS,
            step_metric=eval_schema.checkpoint_step,
            summary="max",
        )
        run.define_metric(
            TRAIN_EPISODE_RETURN_SHAPED_ACROSS_ORIGINS_ROLLING_UP_TO_100_MEAN,
            step_metric=TRAIN_GLOBAL_STEP,
            summary="last",
        )
    return run


def configure_wandb_metric_axes(
    run,
    metric_names,
    *,
    metrics_schema_version: int = METRICS_SCHEMA_VERSION,
):
    """Bind concrete history metrics to their scientific W&B X-axis."""

    if run is None:
        return run
    eval_schema = evaluation_metric_schema(metrics_schema_version)
    axes_by_prefix = (
        ("train/", TRAIN_GLOBAL_STEP),
        ("eval/", eval_schema.checkpoint_step),
        ("orchestration/", ORCHESTRATION_EVENT_SEQ),
    )
    configured = _WANDB_AXIS_METRICS.setdefault(run, set())
    for name in sorted({str(metric_name) for metric_name in metric_names}):
        if name in configured:
            continue
        axis = next(
            (
                candidate_axis
                for prefix, candidate_axis in axes_by_prefix
                if name.startswith(prefix)
            ),
            None,
        )
        if axis is None or name == axis:
            continue
        run.define_metric(name, step_metric=axis)
        configured.add(name)
    return run
