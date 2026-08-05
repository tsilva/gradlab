from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from gradlab.training.sb3_on_policy import (
    OnPolicyBackend,
    normalize_on_policy_config,
    policy_kwargs_from_config,
    policy_type_for_config,
)
from gradlab.training_backend import BackendContext
from gradlab.training_lifecycle import ProgressField


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
    "normalize_advantage": False,
    "resume": None,
    "resume_approval_hash": None,
    "resume_manifest": None,
}
PPO_PROGRESS_FIELDS = (
    ProgressField("train/approx_kl", "KL"),
    ProgressField("train/explained_variance", "explained var"),
    ProgressField("train/entropy_loss", "entropy loss"),
)
A2C_PROGRESS_FIELDS = (
    ProgressField("train/explained_variance", "explained var"),
    ProgressField("train/entropy_loss", "entropy loss"),
)


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
    from gradlab.task_advantage import normalize_advantage_normalization

    normalized["advantage_normalization"] = normalize_advantage_normalization(
        normalized["advantage_normalization"],
        label=f"{label}.advantage_normalization",
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
    from gradlab.model_inputs import model_input_fields
    from gradlab.policy_models import load_pinned_remote_policy_model
    from gradlab.schedules import apply_resume_hyperparameters, learning_rate_schedule
    from gradlab.task_advantage import GroupedAdvantagePPO, resolve_advantage_normalization_mode
    from gradlab.training.sb3_on_policy import validate_resumed_policy_model

    common_config = context.train_config
    backend_config = context.backend_config
    advantage_normalization, advantage_context = resolve_advantage_normalization_mode(
        backend_config
    )
    if advantage_normalization == "grouped":
        fields = model_input_fields(config.task)
        field = fields.get(str(advantage_context))
        if field is None:
            raise ValueError(
                "grouped advantage normalization references undeclared context "
                f"{advantage_context!r}"
            )
        if field["encoding"]["kind"] != "categorical":
            raise ValueError(
                "grouped advantage normalization requires categorical context, got "
                f"{advantage_context!r}"
            )
    sb3_normalize_advantage = advantage_normalization == "global"
    from stable_baselines3 import PPO

    model_cls = GroupedAdvantagePPO if advantage_normalization == "grouped" else PPO
    if backend_config["resume"]:
        model = load_pinned_remote_policy_model(
            backend_config["resume"],
            download_root=context.run_dir / ".resume-source",
            approval_hash=backend_config["resume_approval_hash"],
            manifest=backend_config["resume_manifest"],
            expected_algorithm_id="ppo",
            env=env,
            tensorboard_log=str(context.run_dir),
            device=device,
            ppo_model_class=model_cls,
        )
        validate_resumed_policy_model(model, common_config)
        if advantage_normalization == "grouped":
            if not isinstance(model, GroupedAdvantagePPO):
                raise ValueError(
                    "resume artifact does not use grouped advantage normalization"
                )
            if model.advantage_context != advantage_context:
                raise ValueError(
                    "resume artifact grouped advantage context does not match the recipe"
                )
        elif isinstance(model, GroupedAdvantagePPO):
            raise ValueError(
                "resume artifact uses grouped advantage normalization but the recipe does not"
            )
        apply_resume_hyperparameters(model, common_config, backend_config)
        model.normalize_advantage = sb3_normalize_advantage
        return model

    model_kwargs: dict[str, Any] = {}
    if advantage_context is not None:
        model_kwargs["advantage_context"] = advantage_context
    return model_cls(
        policy_type_for_config(env.observation_space, common_config),
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
            common_config=common_config,
            optimizer_eps=backend_config["adam_eps"],
        ),
        tensorboard_log=str(context.run_dir),
        device=device,
        verbose=0,
        **model_kwargs,
    )


def _a2c_model_factory(context: BackendContext, env: Any, config: Any, device: str):
    del config
    from gradlab.policy_models import load_pinned_remote_policy_model
    from gradlab.schedules import apply_a2c_resume_hyperparameters, learning_rate_schedule
    from gradlab.training.sb3_on_policy import validate_resumed_policy_model

    common_config = context.train_config
    backend_config = context.backend_config
    if backend_config["resume"]:
        model = load_pinned_remote_policy_model(
            backend_config["resume"],
            download_root=context.run_dir / ".resume-source",
            approval_hash=backend_config["resume_approval_hash"],
            manifest=backend_config["resume_manifest"],
            expected_algorithm_id="a2c",
            env=env,
            tensorboard_log=str(context.run_dir),
            device=device,
        )
        validate_resumed_policy_model(model, common_config)
        apply_a2c_resume_hyperparameters(model, common_config, backend_config)
        return model

    from stable_baselines3 import A2C

    return A2C(
        policy_type_for_config(env.observation_space, common_config),
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
        policy_kwargs=policy_kwargs_from_config(
            backend_config,
            common_config=common_config,
        ),
        tensorboard_log=str(context.run_dir),
        device=device,
        verbose=0,
    )


def _ppo_model_class(config: Mapping[str, Any]) -> str:
    value = config.get("advantage_normalization")
    return (
        "gradlab.task_advantage.GroupedAdvantagePPO"
        if isinstance(value, Mapping) and value.get("mode") == "grouped"
        else "stable_baselines3.ppo.ppo.PPO"
    )


BACKENDS = {
    "sb3.ppo": OnPolicyBackend(
        "sb3.ppo",
        PPO_DEFAULT_CONFIG,
        _normalize_ppo,
        _ppo_model_factory,
        _ppo_model_class,
        PPO_PROGRESS_FIELDS,
    ),
    "sb3.a2c": OnPolicyBackend(
        "sb3.a2c",
        A2C_DEFAULT_CONFIG,
        _normalize_a2c,
        _a2c_model_factory,
        "stable_baselines3.a2c.a2c.A2C",
        A2C_PROGRESS_FIELDS,
    ),
}
