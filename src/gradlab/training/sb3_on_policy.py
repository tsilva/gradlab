from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gymnasium import spaces

from gradlab.artifacts import install_model_bundle
from gradlab.state_archive import state_archive_artifact_summary
from gradlab.training_backend import BackendContext


ModelFactory = Callable[[BackendContext, Any, Any, str], Any]
ConfigNormalizer = Callable[..., dict[str, Any]]
ModelClassResolver = Callable[[Mapping[str, Any]], str]


@dataclass(frozen=True)
class OnPolicyBackend:
    """Shared module protocol for one available SB3 on-policy algorithm."""

    backend_id: str
    defaults: Mapping[str, Any]
    normalize_config: ConfigNormalizer
    model_factory: ModelFactory
    model_class: str | ModelClassResolver

    @property
    def algorithm_id(self) -> str:
        return self.backend_id.rsplit(".", 1)[-1]

    def validate(
        self,
        common_config: Mapping[str, Any],
        backend_config: Mapping[str, Any],
    ) -> None:
        del common_config, backend_config

    def run(self, context: BackendContext) -> None:
        run_sb3_on_policy(context, algorithm_id=self.algorithm_id, model_factory=self.model_factory)

    def acceptance_mode(self, backend_config: Mapping[str, Any]) -> str:
        del backend_config
        from gradlab.training_backend import CHECKPOINT_EVAL_ACCEPTANCE

        return CHECKPOINT_EVAL_ACCEPTANCE

    def state_archive_priority_metrics(self) -> tuple[str, ...]:
        return ("value_error",)

    def contract_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": "available",
            "defaults": self.defaults,
            "state_archive_priority_metrics": ["value_error"],
        }

    def runtime_metadata(
        self,
        backend_config: Mapping[str, Any],
    ) -> Mapping[str, str]:
        model_class = (
            self.model_class(backend_config) if callable(self.model_class) else self.model_class
        )
        return {
            "training_backend_id": self.backend_id,
            "algorithm_id": self.algorithm_id,
            "model_class": model_class,
        }


_INTEGER_FIELDS = (
    "learning_rate_schedule_timesteps",
    "n_steps",
    "ent_coef_schedule_timesteps",
)
_NON_NEGATIVE_INTEGER_FIELDS = (
    "learning_rate_schedule_timesteps",
    "ent_coef_schedule_timesteps",
)
_NUMBER_FIELDS = (
    "learning_rate",
    "learning_rate_final",
    "gamma",
    "gae_lambda",
    "ent_coef",
    "ent_coef_final",
    "vf_coef",
)


def normalize_on_policy_config(
    config: Mapping[str, Any],
    *,
    defaults: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    unexpected = sorted(set(config) - set(defaults))
    if unexpected:
        raise ValueError(f"{label} has unexpected fields: {unexpected}")
    normalized = {**defaults, **dict(config)}
    for key in _INTEGER_FIELDS:
        value = normalized[key]
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{label}.{key} must be an integer")
    if normalized["n_steps"] <= 0:
        raise ValueError(f"{label}.n_steps must be positive")
    for key in _NON_NEGATIVE_INTEGER_FIELDS:
        if normalized[key] < 0:
            raise ValueError(f"{label}.{key} must be non-negative")
    for key in _NUMBER_FIELDS:
        value = normalized[key]
        if value is None:
            continue
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise ValueError(f"{label}.{key} must be a number or null")
    if normalized["device"] not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"{label}.device must be one of auto, cpu, cuda, mps")
    for key in ("policy_net_arch", "value_net_arch"):
        if not isinstance(normalized[key], str):
            raise ValueError(f"{label}.{key} must be a string")
    if not isinstance(normalized["normalize_advantage"], bool):
        raise ValueError(f"{label}.normalize_advantage must be a boolean")
    resume = normalized["resume"]
    if resume is not None and not isinstance(resume, str):
        raise ValueError(f"{label}.resume must be a string or null")
    approval = normalized["resume_approval_hash"]
    manifest = normalized["resume_manifest"]
    if resume is None:
        if approval is not None or manifest is not None:
            raise ValueError(f"{label} resume approval fields require resume")
    elif (
        not isinstance(approval, str)
        or not re.fullmatch(r"[0-9a-f]{64}", approval)
        or not isinstance(manifest, list)
        or not manifest
    ):
        raise ValueError(f"{label}.resume requires a pinned approval hash and byte manifest")
    return normalized


def active_reward_components(task: Mapping[str, object]) -> tuple[str, ...]:
    reward = task.get("reward")
    if not isinstance(reward, Mapping):
        return ()
    components: list[str] = []
    reward_mode = str(reward.get("reward_mode") or "")
    if reward_mode == "native" or bool(reward.get("use_native_reward")):
        components.append("native")
    if (
        float(reward.get("progress_reward_scale") or 0.0) != 0.0
        or float(reward.get("progress_reward_boost_scale") or 0.0) != 0.0
    ):
        components.append("progress")
    if reward_mode == "score":
        components.append("score")
    if (
        float(reward.get("terminal_reward") or 0.0) != 0.0
        or float(reward.get("completion_reward") or 0.0) != 0.0
    ):
        components.append("completion")
    if float(reward.get("death_penalty") or 0.0) != 0.0:
        components.append("death")
    if float(reward.get("time_penalty") or 0.0) != 0.0:
        components.append("time")
    return tuple(components)


def active_reward_signals(task: Mapping[str, object]) -> tuple[str, ...]:
    components = set(active_reward_components(task))
    return tuple(name for name in ("progress", "score") if name in components)


def policy_name_for_observation_space(observation_space) -> str:
    if isinstance(observation_space, spaces.Dict):
        return "MultiInputPolicy"
    if isinstance(observation_space, spaces.Box) and len(observation_space.shape) == 3:
        return "CnnPolicy"
    return "MlpPolicy"


def validate_action_space(action_space, *, algorithm_id: str) -> None:
    supported = (spaces.Box, spaces.Discrete, spaces.MultiBinary, spaces.MultiDiscrete)
    if not isinstance(action_space, supported):
        raise ValueError(
            f"SB3 {algorithm_id.upper()} does not support action space "
            f"{type(action_space).__name__}; configure a task action codec or choose another backend"
        )


def checkpoint_prefix(game: str, *, algorithm_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", game).strip("_").lower()
    return f"{algorithm_id}_{slug or 'retro'}"


def parse_net_arch(value: str) -> list[int]:
    layers = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if any(size <= 0 for size in layers):
        raise ValueError("policy/value network layer sizes must be positive")
    return layers


def policy_kwargs_from_config(
    backend_config: Mapping[str, Any],
    *,
    optimizer_eps: float | None = None,
) -> dict[str, object]:
    policy_kwargs: dict[str, object] = {}
    if optimizer_eps is not None:
        policy_kwargs["optimizer_kwargs"] = {"eps": optimizer_eps}
    pi_arch = parse_net_arch(str(backend_config["policy_net_arch"]))
    vf_arch = parse_net_arch(str(backend_config["value_net_arch"]))
    if pi_arch or vf_arch:
        policy_kwargs["net_arch"] = {"pi": pi_arch, "vf": vf_arch}
    return policy_kwargs


def checkpoint_save_frequency(checkpoint_freq: int, n_envs: int) -> int | None:
    if checkpoint_freq <= 0:
        return None
    return max(checkpoint_freq // max(n_envs, 1), 1)


def save_model_bundle(
    *,
    model,
    context: BackendContext,
    model_path: Path,
    kind: str,
    step: int | None,
) -> Path:
    model_path = install_model_bundle(
        model_path,
        save_checkpoint=lambda path: model.save(str(path)),
        train_config=context.train_config,
        config=context.environment,
        kind=kind,
        checkpoint_step_value=step,
        state_archive_summary=state_archive_artifact_summary(getattr(model, "env", None)),
    )
    checkpoint_id = context.metric_store.record_checkpoint(
        run_name=str(context.train_config["run_name"]),
        kind=kind,
        step=step,
        path=model_path,
        sha256=None,
        eval_required=context.train_config["checkpoint_eval_backend"] != "none",
    )
    print(f"{kind} model ready: id={checkpoint_id} step={step} path={model_path}", flush=True)
    return model_path


def run_sb3_on_policy(
    context: BackendContext,
    *,
    algorithm_id: str,
    model_factory: ModelFactory,
) -> None:
    from stable_baselines3.common.utils import set_random_seed

    from gradlab.callbacks import (
        LedgerCheckpointHelper,
        MetricStoreLoggerHelper,
        MetricEarlyStopHelper,
        GradLabCallback,
        RolloutDiagnosticsHelper,
        RuntimeMetricsHelper,
        ArchiveCurriculumFeedbackHelper,
        ThroughputHelper,
    )
    from gradlab.device import resolve_sb3_device
    from gradlab.env import (
        make_training_vec_env,
        preflight_state_archive_provider,
        task_termination,
    )
    from gradlab.file_utils import file_sha256
    from gradlab.metric_store import metric_store_path
    from gradlab.policy_bundle import write_canonical_json
    from gradlab.schedules import EntropyCoefficientScheduleHelper
    from gradlab.training.sb3_helpers import (
        GracefulStopHelper,
        Sb3HumanOutputFormatHelper,
        install_on_policy_safe_boundary_stop,
    )

    common_config = context.train_config
    backend_config = context.backend_config
    config = context.environment
    n_envs = int(common_config["resolved_n_envs"])
    preflight = preflight_state_archive_provider(
        config=config,
        n_envs=n_envs,
        seed=int(common_config["seed"]),
        rom_binding=getattr(context, "rom_binding", None),
        state_archive=common_config.get("state_archive"),
    )
    if preflight is not None:
        preflight_path = context.run_dir / "state_archive_preflight.json"
        write_canonical_json(preflight_path, preflight)
        common_config["state_archive_preflight_sha256"] = file_sha256(preflight_path)
        print(
            "state archive provider preflight passed: "
            f"provider={preflight['provider_id']} codec={preflight['codec_id']} "
            f"lanes={preflight['preflight_lanes']}",
            flush=True,
        )
    env = make_training_vec_env(
        config=config,
        n_envs=n_envs,
        seed=int(common_config["seed"]),
        rom_binding=getattr(context, "rom_binding", None),
        state_archive=common_config.get("state_archive"),
        state_archive_root=context.run_dir / "state-archive",
    )
    try:
        store_path = metric_store_path(context.run_dir)
        set_random_seed(int(common_config["seed"]))
        validate_action_space(env.action_space, algorithm_id=algorithm_id)
        device = resolve_sb3_device(str(backend_config["device"]))
        print(f"Using torch device: {device}", flush=True)
        model = model_factory(context, env, config, device)

        graceful_stop = GracefulStopHelper(
            context.stop_flag,
            marker_path=context.run_dir / "learner_stop_observed.json",
        )
        install_on_policy_safe_boundary_stop(
            model,
            graceful_stop=graceful_stop,
        )
        components: list[Any] = [
            graceful_stop,
            Sb3HumanOutputFormatHelper(compact=context.compact_console),
            ThroughputHelper(
                metric_store_path=store_path,
                wandb_enabled=context.wandb_enabled,
            ),
            RuntimeMetricsHelper(
                event_names=tuple(task_termination(config).get("failure", ())),
                active_reward_components=active_reward_components(config.task),
                active_reward_signals=active_reward_signals(config.task),
                configured_starts=tuple(config.states or ((config.state,) if config.state else ())),
                track_success=bool(
                    isinstance(config.task.get("termination"), Mapping)
                    and config.task["termination"].get("success")
                ),
            ),
        ]
        if common_config.get("state_archive") is not None:
            archive_config = common_config.get("state_archive")
            if isinstance(archive_config, Mapping) and archive_config.get("curriculum") is not None:
                components.append(ArchiveCurriculumFeedbackHelper())
        if common_config["early_stop"]:
            components.append(
                MetricEarlyStopHelper(
                    decision_path=(
                        context.run_dir
                        / f"early_stop_decision-{str(common_config['attempt_id'])}.json"
                    ),
                    config=common_config["early_stop"],
                    stop_flag=context.stop_flag,
                    metric_store_path=store_path,
                )
            )
        components.extend(
            [
                RolloutDiagnosticsHelper(
                    algorithm_id=algorithm_id,
                    metric_store_path=store_path,
                    wandb_enabled=context.wandb_enabled,
                    histogram_interval=64,
                ),
                MetricStoreLoggerHelper(
                    store_path,
                    algorithm_id=algorithm_id,
                    wandb_enabled=context.wandb_enabled,
                ),
            ]
        )
        checkpoint_save_freq = checkpoint_save_frequency(
            int(common_config["checkpoint_freq"]),
            n_envs,
        )
        if checkpoint_save_freq is not None:
            components.append(
                LedgerCheckpointHelper(
                    train_config=common_config,
                    config=config,
                    save_freq=checkpoint_save_freq,
                    save_path=str(context.checkpoint_dir),
                    name_prefix=checkpoint_prefix(config.game, algorithm_id=algorithm_id),
                    metric_store_path=store_path,
                    eval_required=common_config["checkpoint_eval_backend"] != "none",
                )
            )
        if backend_config["ent_coef_final"] is not None:
            components.append(
                EntropyCoefficientScheduleHelper(
                    initial_value=backend_config["ent_coef"],
                    final_value=backend_config["ent_coef_final"],
                    schedule_timesteps=(
                        backend_config["ent_coef_schedule_timesteps"]
                        if backend_config["ent_coef_schedule_timesteps"] > 0
                        else common_config["timesteps"]
                    ),
                    algorithm_id=algorithm_id,
                )
            )
        if common_config["checkpoint_eval_backend"] == "none":
            print(
                "checkpoint evaluation disabled; this run cannot establish promotion or acceptance"
            )
        else:
            print("training-loop eval disabled; async checkpoint eval handles promotion metrics")
        callback = GradLabCallback(components)
        context.mark_ready()

        final_model_path = context.run_dir / "final_model.zip"
        if backend_config["resume"]:
            remaining_timesteps = max(
                0,
                int(common_config["timesteps"]) - int(model.num_timesteps),
            )
            print(
                f"resuming learner at step={model.num_timesteps} "
                f"remaining={remaining_timesteps} cap={common_config['timesteps']}",
                flush=True,
            )
            if remaining_timesteps:
                model.learn(
                    total_timesteps=remaining_timesteps,
                    callback=callback,
                    progress_bar=True,
                    reset_num_timesteps=False,
                )
        else:
            model.learn(
                total_timesteps=int(common_config["timesteps"]),
                callback=callback,
                progress_bar=True,
            )
        save_model_bundle(
            model=model,
            context=context,
            model_path=final_model_path,
            kind="final",
            step=model.num_timesteps,
        )
        print(f"saved {final_model_path}")
    finally:
        env.close()
