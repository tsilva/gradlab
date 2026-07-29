#!/usr/bin/env python
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from collections.abc import Mapping

import numpy as np

from gradlab.env import EnvConfig, native_obs_crop, resolve_env_config
from gradlab.env_providers import make_provider_vec_env, provider_native_vec_kwargs


SMOKE_CONTRACT_VERSION = 1
PROVIDER_DISTRIBUTION = "vizdoom-turbo"


def _canonical_sha256(document: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(document),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    config = resolve_env_config(
        EnvConfig(
            env_provider="vizdoom-turbo",
            game="VizdoomBasic-v1",
            state="default",
            frame_skip=4,
            max_pool_frames=False,
            sticky_action_prob=0.0,
            obs_resize=(84, 84),
            obs_crop=(0, 0, 0, 0),
            env_args={
                "scenario": "scenario",
                "info": "data",
                "use_restricted_actions": "discrete",
                "record": False,
                "players": 1,
                "inttype": "stable",
                "obs_type": "image",
                "render_mode": "rgb_array",
                "num_threads": 1,
                "rom_path": None,
                "obs_copy": "safe_view",
                "obs_grayscale": True,
                "obs_layout": "chw",
                "frame_stack": 4,
                "noop_reset_max": 0,
                "info_filter": {"mode": "all", "keys": []},
                "use_fire_reset": False,
                "treat_episode_timeout_as_truncation": True,
            },
        )
    )
    kwargs = provider_native_vec_kwargs(
        config,
        n_envs=1,
        native_obs_crop=native_obs_crop,
        state_weight_mapping=lambda _config: {},
    )
    env = None
    closed = False
    try:
        env = make_provider_vec_env(config, native_kwargs=kwargs)
        observations, reset_info = env.reset(seed=17)
        step = env.step(np.zeros((1,), dtype=np.int64))
        if not isinstance(reset_info, Mapping):
            raise TypeError("reset info must be a columnar mapping")
        if not isinstance(step, tuple) or len(step) != 5:
            raise TypeError("step must return the Gymnasium five-tuple")
        next_observations, rewards, terminated, truncated, step_info = step
        if not isinstance(step_info, Mapping):
            raise TypeError("step info must be a columnar mapping")
        for label, value in (
            ("reset observations", observations),
            ("step observations", next_observations),
            ("rewards", rewards),
            ("terminated", terminated),
            ("truncated", truncated),
        ):
            if np.asarray(value).shape[0] != 1:
                raise ValueError(f"{label} must contain exactly one lane")
        native = getattr(env, "env", env)
        capabilities = getattr(native, "capabilities", {})
        evidence: dict[str, object] = {
            "smoke_contract_version": SMOKE_CONTRACT_VERSION,
            "provider_distribution": PROVIDER_DISTRIBUTION,
            "provider_version": importlib.metadata.version(PROVIDER_DISTRIBUTION),
            "game": config.game,
            "num_envs": int(env.num_envs),
            "observation_shape": list(np.asarray(observations).shape),
            "observation_dtype": str(np.asarray(observations).dtype),
            "reset_info_keys": sorted(str(key) for key in reset_info),
            "step_info_keys": sorted(str(key) for key in step_info),
            "capability_keys": sorted(str(key) for key in capabilities),
            "reset_succeeded": True,
            "step_succeeded": True,
        }
    finally:
        if env is not None:
            env.close()
            closed = True
    evidence["close_succeeded"] = closed
    evidence["evidence_sha256"] = _canonical_sha256(evidence)
    print(json.dumps(evidence, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
