from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from gradlab.artifacts import build_model_provenance
from gradlab.env import EnvConfig
from gradlab.policy_execution import (
    compile_policy_execution_contract,
    verify_policy_execution_contract,
)
from gradlab.routed_policy import RoutedActorCriticPolicy


def _model_and_env():
    policy_model = {
        "schema_version": 1,
        "topology": {
            "kind": "shared_encoder",
            "encoder": {"kind": "flatten"},
        },
        "fusion": "post_encoder_concat",
        "context_encoders": {"health": {"kind": "identity"}},
        "routes": {"health": ["state_value"]},
        "heads": {
            "action": {"hidden_sizes": [], "activation": "tanh"},
            "state_value": {"hidden_sizes": [], "activation": "tanh"},
        },
        "normalize_images": False,
        "orthogonal_init": True,
    }
    observation_space = gym.spaces.Dict(
        {
            "observation": gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
            "context/health": gym.spaces.Box(
                np.asarray([0.0], dtype=np.float32),
                np.asarray([1.0], dtype=np.float32),
                dtype=np.float32,
            ),
        }
    )
    policy = RoutedActorCriticPolicy(
        observation_space,
        gym.spaces.Discrete(2),
        lambda _: 1e-3,
        policy_model=policy_model,
    )
    model = SimpleNamespace(policy=policy)
    model_input_contract = {
        "schema_version": 1,
        "layout": "dict_observation_context_v1",
        "base_observation_space": {
            "kind": "box",
            "dtype": "<f4",
            "shape": [4],
        },
        "context": {
            "health": {
                "signal": "health",
                "update": "transition",
                "encoding": {
                    "kind": "continuous",
                    "scale": 0.01,
                    "offset": 0.0,
                    "low": 0.0,
                    "high": 1.0,
                },
                "source": [
                    {
                        "name": "health",
                        "dtype": "<f4",
                        "shape": [],
                        "available_on_reset": True,
                        "available_on_step": True,
                    }
                ],
                "output": {"kind": "box", "dtype": "<f4", "shape": [1]},
            }
        },
    }
    env = SimpleNamespace(
        runtime=SimpleNamespace(
            kernel=SimpleNamespace(model_input_contract=model_input_contract)
        )
    )
    return model, env


def test_policy_execution_contract_binds_provider_schema_model_and_routes() -> None:
    model, env = _model_and_env()

    contract = compile_policy_execution_contract(model, env)

    assert contract is not None
    assert contract["role_inputs"] == {
        "action": ["observation"],
        "state_value": ["observation", "context/health"],
    }
    assert contract["model_inputs"]["context"]["health"]["source"][0]["name"] == "health"
    assert len(contract["sha256"]) == 64
    assert verify_policy_execution_contract(model, env, contract) == contract


def test_policy_execution_contract_supports_configured_policy_without_context() -> None:
    policy = RoutedActorCriticPolicy(
        gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
        gym.spaces.Discrete(2),
        lambda _: 1e-3,
        policy_model={
            "schema_version": 1,
            "topology": {
                "kind": "shared_encoder",
                "encoder": {"kind": "flatten"},
            },
            "fusion": "post_encoder_concat",
            "context_encoders": {},
            "routes": {},
            "heads": {
                "action": {"hidden_sizes": [8], "activation": "tanh"},
                "state_value": {"hidden_sizes": [8], "activation": "tanh"},
            },
            "normalize_images": False,
            "orthogonal_init": True,
        },
    )
    model = SimpleNamespace(policy=policy)
    env = SimpleNamespace(runtime=SimpleNamespace(kernel=SimpleNamespace()))

    contract = compile_policy_execution_contract(model, env)

    assert contract is not None
    assert contract["model_inputs"] is None
    assert contract["role_inputs"] == {
        "action": ["observation"],
        "state_value": ["observation"],
    }
    assert verify_policy_execution_contract(model, env, contract) == contract


def test_policy_execution_contract_rejects_runtime_drift() -> None:
    model, env = _model_and_env()
    contract = compile_policy_execution_contract(model, env)
    assert contract is not None
    drifted = deepcopy(env.runtime.kernel.model_input_contract)
    drifted["context"]["health"]["source"][0]["dtype"] = "<f8"
    env.runtime.kernel.model_input_contract = drifted

    with pytest.raises(ValueError, match="does not match"):
        verify_policy_execution_contract(model, env, contract)


def test_policy_execution_contract_is_persisted_in_existing_training_metadata(
    tmp_path,
) -> None:
    model, env = _model_and_env()
    contract = compile_policy_execution_contract(model, env)
    assert contract is not None

    provenance = build_model_provenance(
        {
            "training_backend_id": "sb3.ppo",
            "training_backend_config_hash": "0" * 64,
            "algorithm_id": "ppo",
            "model_class": "stable_baselines3.ppo.ppo.PPO",
        },
        EnvConfig(),
        tmp_path / "model.zip",
        "final",
        policy_execution_contract=contract,
    )

    assert provenance["training_metadata"]["policy_execution_contract"] == contract
