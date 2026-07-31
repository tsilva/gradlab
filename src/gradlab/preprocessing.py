from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from gradlab.validation import normalize_obs_crop, normalize_obs_resize


def _value(source: Mapping[str, Any] | Any, key: str, default: Any) -> Any:
    if isinstance(source, Mapping):
        value = source.get(key, default)
    else:
        value = getattr(source, key, default)
    return default if value is None else value


def _environment_argument(
    source: Mapping[str, Any] | Any,
    key: str,
    default: Any,
) -> Any:
    direct = _value(source, key, None)
    if direct is not None:
        return direct
    env_args = _value(source, "env_args", {})
    if isinstance(env_args, Mapping):
        value = env_args.get(key)
        if value is not None:
            return value
    return default


def preprocessing_contract(
    source: Mapping[str, Any] | Any,
    *,
    provider_id: str | None = None,
    task: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the canonical policy-facing preprocessing contract."""

    provider = str(provider_id or _value(source, "env_provider", "")).strip()
    if not provider:
        raise ValueError("preprocessing provider identity is required")
    pipeline = (
        "stable_retro_native_vec_env"
        if provider == "stable-retro-turbo"
        else f"{provider.replace('-', '_')}_native_vec_env"
    )
    obs_resize = normalize_obs_resize(_value(source, "obs_resize", (84, 84)))
    raw_crop = _value(source, "obs_crop", None)
    crop = normalize_obs_crop(raw_crop, label="environment.preprocessing.obs_crop")
    task_config = task or _value(source, "task", {})
    conditioning = task_config.get("conditioning", {}) if isinstance(task_config, Mapping) else {}
    model_inputs = (
        task_config.get("model_inputs", {}) if isinstance(task_config, Mapping) else {}
    )
    context = model_inputs.get("context", {}) if isinstance(model_inputs, Mapping) else {}
    max_pool_frames = _value(source, "max_pool_frames", True)
    return {
        "pipeline": pipeline,
        "obs_resize": list(obs_resize),
        "obs_crop": list(crop) if crop is not None else None,
        "obs_crop_mode": str(_value(source, "obs_crop_mode", "remove")),
        "obs_crop_fill": int(_value(source, "obs_crop_fill", 0)),
        "obs_grayscale": bool(
            _environment_argument(
                source,
                "obs_grayscale",
                True,
            )
        ),
        "obs_resize_algorithm": str(_value(source, "obs_resize_algorithm", "area")),
        "frame_skip": int(_value(source, "frame_skip", 4)),
        "frame_stack": int(
            _environment_argument(
                source,
                "frame_stack",
                4,
            )
        ),
        "max_pool_frames": bool(max_pool_frames),
        "sticky_action_prob": float(_value(source, "sticky_action_prob", 0.0)),
        "obs_copy": str(_value(source, "obs_copy", "safe_view")),
        "policy_observation_layout": (
            "dict_observation_context_v1"
            if bool(context)
            else "dict_image_task"
            if bool(conditioning.get("enabled"))
            else "channel_first"
        ),
    }
