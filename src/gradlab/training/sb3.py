from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from gradlab.training.sb3_on_policy import (
    OnPolicyBackend,
    normalize_on_policy_config,
    policy_kwargs_from_config,
    policy_name_for_observation_space,
)
from gradlab.training_backend import BackendContext


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
    from gradlab.env import task_conditioning
    from gradlab.policy_models import load_pinned_remote_policy_model
    from gradlab.schedules import apply_resume_hyperparameters, learning_rate_schedule
    from gradlab.task_advantage import PerTaskAdvantagePPO, resolve_advantage_normalization_mode

    common_config = context.train_config
    backend_config = context.backend_config
    advantage_normalization = resolve_advantage_normalization_mode(backend_config)
    if advantage_normalization == "per-task" and not task_conditioning(config).get("enabled"):
        raise ValueError("per-task advantage normalization requires task conditioning")
    sb3_normalize_advantage = advantage_normalization == "global"
    if backend_config["resume"]:
        model = load_pinned_remote_policy_model(
            backend_config["resume"],
            download_root=context.run_dir / ".resume-source",
            approval_hash=backend_config["resume_approval_hash"],
            manifest=backend_config["resume_manifest"],
            metadata={"algorithm_id": "ppo"},
            env=env,
            tensorboard_log=str(context.run_dir),
            device=device,
        )
        if advantage_normalization == "per-task":
            raise ValueError("per-task advantage normalization is not supported with resume")
        apply_resume_hyperparameters(model, common_config, backend_config)
        model.normalize_advantage = sb3_normalize_advantage
        return model

    from stable_baselines3 import PPO

    model_cls = PerTaskAdvantagePPO if advantage_normalization == "per-task" else PPO
    return model_cls(
        policy_name_for_observation_space(env.observation_space),
        env,
        learning_rate=learning_rate_schedule(common_config, backend_config),
        n_steps=backend_config["n_steps"],
        batch_size=backend_config["batch_size"],
        n_epochs=backend_config["n_epochs"],
        gamma=backend_config["gamma"],
        gae_lambda=backend_config["gae_lambda"],
        ent_coef=backend_config["ent_coef"],
        vf_coef=backend_config["vf_coef"],
        clip_range=backend_config["clip_range"],
        clip_range_vf=backend_config["clip_range_vf"],
        normalize_advantage=sb3_normalize_advantage,
        target_kl=backend_config["target_kl"],
        policy_kwargs=policy_kwargs_from_config(
            backend_config,
            optimizer_eps=backend_config["adam_eps"],
        ),
        tensorboard_log=str(context.run_dir),
        device=device,
        verbose=0,
    )


def _a2c_model_factory(context: BackendContext, env: Any, config: Any, device: str):
    del config
    from gradlab.policy_models import load_pinned_remote_policy_model
    from gradlab.schedules import apply_a2c_resume_hyperparameters, learning_rate_schedule

    common_config = context.train_config
    backend_config = context.backend_config
    if backend_config["resume"]:
        model = load_pinned_remote_policy_model(
            backend_config["resume"],
            download_root=context.run_dir / ".resume-source",
            approval_hash=backend_config["resume_approval_hash"],
            manifest=backend_config["resume_manifest"],
            metadata={"algorithm_id": "a2c"},
            env=env,
            tensorboard_log=str(context.run_dir),
            device=device,
        )
        apply_a2c_resume_hyperparameters(model, common_config, backend_config)
        return model

    from stable_baselines3 import A2C

    return A2C(
        policy_name_for_observation_space(env.observation_space),
        env,
        learning_rate=learning_rate_schedule(common_config, backend_config),
        n_steps=backend_config["n_steps"],
        gamma=backend_config["gamma"],
        gae_lambda=backend_config["gae_lambda"],
        ent_coef=backend_config["ent_coef"],
        vf_coef=backend_config["vf_coef"],
        max_grad_norm=backend_config["max_grad_norm"],
        rms_prop_eps=backend_config["rms_prop_eps"],
        use_rms_prop=backend_config["use_rms_prop"],
        normalize_advantage=backend_config["normalize_advantage"],
        policy_kwargs=policy_kwargs_from_config(backend_config),
        tensorboard_log=str(context.run_dir),
        device=device,
        verbose=0,
    )


def _ppo_model_class(config: Mapping[str, Any]) -> str:
    return (
        "gradlab.task_advantage.PerTaskAdvantagePPO"
        if config.get("advantage_normalization") == "per-task"
        else "stable_baselines3.ppo.ppo.PPO"
    )


BACKENDS = {
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
