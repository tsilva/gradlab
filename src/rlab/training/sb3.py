from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rlab.training.sb3_on_policy import (
    OnPolicyBackend,
    normalize_on_policy_config,
    policy_kwargs_from_args,
    policy_name_for_observation_space,
)
from rlab.training_backend import BackendContext


PPO_DEFAULT_CONFIG: dict[str, Any] = {
    "learning_rate": 1e-4,
    "learning_rate_final": None,
    "learning_rate_schedule_timesteps": 0,
    "n_steps": 512,
    "batch_size": 256,
    "n_epochs": 10,
    "device": "auto",
    "gamma": 0.9,
    "gae_lambda": 1.0,
    "ent_coef": 0.01,
    "ent_coef_final": None,
    "ent_coef_schedule_timesteps": 0,
    "vf_coef": 1.0,
    "clip_range": 0.2,
    "clip_range_vf": None,
    "policy_net_arch": "",
    "value_net_arch": "",
    "normalize_advantage": False,
    "advantage_normalization": "auto",
    "adam_eps": 1e-8,
    "target_kl": None,
    "resume": None,
    "resume_approval_hash": None,
    "resume_manifest": None,
}
A2C_DEFAULT_CONFIG: dict[str, Any] = {
    "learning_rate": 7e-4,
    "learning_rate_final": None,
    "learning_rate_schedule_timesteps": 0,
    "n_steps": 5,
    "device": "auto",
    "gamma": 0.99,
    "gae_lambda": 1.0,
    "ent_coef": 0.0,
    "ent_coef_final": None,
    "ent_coef_schedule_timesteps": 0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "rms_prop_eps": 1e-5,
    "use_rms_prop": True,
    "policy_net_arch": "",
    "value_net_arch": "",
    "normalize_advantage": False,
    "resume": None,
    "resume_approval_hash": None,
    "resume_manifest": None,
}


def _normalize_ppo(config: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    normalized = normalize_on_policy_config(config, defaults=PPO_DEFAULT_CONFIG, label=label)
    for key in ("batch_size", "n_epochs"):
        value = normalized[key]
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{label}.{key} must be an integer")
        if value <= 0:
            raise ValueError(f"{label}.{key} must be positive")
    for key in ("clip_range", "clip_range_vf", "adam_eps", "target_kl"):
        value = normalized[key]
        if value is not None and (not isinstance(value, int | float) or isinstance(value, bool)):
            raise ValueError(f"{label}.{key} must be a number or null")
    if normalized["advantage_normalization"] not in {"auto", "none", "global", "per-task"}:
        raise ValueError(
            f"{label}.advantage_normalization must be one of auto, none, global, per-task"
        )
    return normalized


def _normalize_a2c(config: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    normalized = normalize_on_policy_config(config, defaults=A2C_DEFAULT_CONFIG, label=label)
    for key in ("max_grad_norm", "rms_prop_eps"):
        value = normalized[key]
        if value is not None and (not isinstance(value, int | float) or isinstance(value, bool)):
            raise ValueError(f"{label}.{key} must be a number or null")
    if not isinstance(normalized["use_rms_prop"], bool):
        raise ValueError(f"{label}.use_rms_prop must be a boolean")
    return normalized


def _ppo_model_factory(context: BackendContext, env: Any, config: Any, device: str):
    from rlab.env import task_conditioning
    from rlab.policy_models import load_pinned_remote_policy_model
    from rlab.schedules import apply_resume_hyperparameters, learning_rate_schedule
    from rlab.task_advantage import PerTaskAdvantagePPO, resolve_advantage_normalization_mode

    args = context.args
    advantage_normalization = resolve_advantage_normalization_mode(args)
    if advantage_normalization == "per-task" and not task_conditioning(config).get("enabled"):
        raise ValueError("per-task advantage normalization requires task conditioning")
    sb3_normalize_advantage = advantage_normalization == "global"
    if args.resume:
        model = load_pinned_remote_policy_model(
            args.resume,
            download_root=context.run_dir / ".resume-source",
            approval_hash=args.resume_approval_hash,
            manifest=args.resume_manifest,
            metadata={"algorithm_id": "ppo"},
            env=env,
            tensorboard_log=str(context.run_dir),
            device=device,
        )
        if advantage_normalization == "per-task":
            raise ValueError("per-task advantage normalization is not supported with resume")
        apply_resume_hyperparameters(model, args)
        model.normalize_advantage = sb3_normalize_advantage
        return model

    from stable_baselines3 import PPO

    model_cls = PerTaskAdvantagePPO if advantage_normalization == "per-task" else PPO
    return model_cls(
        policy_name_for_observation_space(env.observation_space),
        env,
        learning_rate=learning_rate_schedule(args),
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        clip_range=args.clip_range,
        clip_range_vf=args.clip_range_vf,
        normalize_advantage=sb3_normalize_advantage,
        target_kl=args.target_kl,
        policy_kwargs=policy_kwargs_from_args(args, optimizer_eps=args.adam_eps),
        tensorboard_log=str(context.run_dir),
        device=device,
        verbose=1,
    )


def _a2c_model_factory(context: BackendContext, env: Any, config: Any, device: str):
    del config
    from rlab.policy_models import load_pinned_remote_policy_model
    from rlab.schedules import apply_a2c_resume_hyperparameters, learning_rate_schedule

    args = context.args
    if args.resume:
        model = load_pinned_remote_policy_model(
            args.resume,
            download_root=context.run_dir / ".resume-source",
            approval_hash=args.resume_approval_hash,
            manifest=args.resume_manifest,
            metadata={"algorithm_id": "a2c"},
            env=env,
            tensorboard_log=str(context.run_dir),
            device=device,
        )
        apply_a2c_resume_hyperparameters(model, args)
        return model

    from stable_baselines3 import A2C

    return A2C(
        policy_name_for_observation_space(env.observation_space),
        env,
        learning_rate=learning_rate_schedule(args),
        n_steps=args.n_steps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        rms_prop_eps=args.rms_prop_eps,
        use_rms_prop=args.use_rms_prop,
        normalize_advantage=args.normalize_advantage,
        policy_kwargs=policy_kwargs_from_args(args),
        tensorboard_log=str(context.run_dir),
        device=device,
        verbose=1,
    )


def _ppo_model_class(config: Mapping[str, Any]) -> str:
    return (
        "rlab.task_advantage.PerTaskAdvantagePPO"
        if config.get("advantage_normalization") == "per-task"
        else "stable_baselines3.ppo.ppo.PPO"
    )


_BACKENDS = {
    "sb3.ppo": OnPolicyBackend(
        "sb3.ppo",
        PPO_DEFAULT_CONFIG,
        _normalize_ppo,
        _ppo_model_factory,
        _ppo_model_class,
    ),
    "sb3.a2c": OnPolicyBackend(
        "sb3.a2c",
        A2C_DEFAULT_CONFIG,
        _normalize_a2c,
        _a2c_model_factory,
        "stable_baselines3.a2c.a2c.A2C",
    ),
}


def _backend(backend_id: str) -> OnPolicyBackend:
    try:
        return _BACKENDS[backend_id]
    except KeyError as exc:
        raise ValueError(f"SB3 backend module does not define {backend_id!r}") from exc


def normalize_config(
    backend_id: str,
    config: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    return _backend(backend_id).normalize_config(config, label=label)


def backend_for_id(backend_id: str) -> OnPolicyBackend:
    return _backend(backend_id)


def contract_payload(backend_id: str) -> dict[str, Any]:
    return _backend(backend_id).contract_payload(backend_id)


def state_archive_priority_metrics(backend_id: str) -> tuple[str, ...]:
    return _backend(backend_id).state_archive_priority_metrics(backend_id)


def runtime_metadata(
    backend_id: str,
    backend_config: Mapping[str, Any],
) -> Mapping[str, str]:
    return _backend(backend_id).runtime_metadata(backend_id, backend_config)
