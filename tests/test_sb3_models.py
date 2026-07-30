from __future__ import annotations

import pytest

from gradlab.policy_models import resolve_policy_algorithm
from gradlab.policy_registry import (
    ALGORITHM_MODEL_CLASSES,
    MODEL_CLASS_ALGORITHMS,
    POLICY_ALGORITHM_SPECS,
    RUNTIME_POLICY_ALGORITHMS,
    SB3_ALGORITHMS,
    default_action_selection_mode,
)
from gradlab.sb3_models import resolve_sb3_algorithm
from gradlab.training_backend import load_training_backend


def test_checkpoint_identity_is_required() -> None:
    with pytest.raises(ValueError, match="must identify"):
        resolve_sb3_algorithm({})


def test_a2c_checkpoint_identity_resolves_consistently() -> None:
    assert (
        resolve_sb3_algorithm(
            {
                "training_backend_id": "sb3.a2c",
                "algorithm_id": "a2c",
                "model_class": "stable_baselines3.a2c.a2c.A2C",
            }
        )
        == "a2c"
    )


def test_checkpoint_identity_rejects_conflicting_metadata() -> None:
    with pytest.raises(ValueError, match="metadata disagree"):
        resolve_sb3_algorithm(
            {
                "training_backend_id": "sb3.a2c",
                "algorithm_id": "ppo",
            }
        )


def test_checkpoint_identity_rejects_unknown_model_class() -> None:
    with pytest.raises(ValueError, match="unsupported checkpoint model class"):
        resolve_sb3_algorithm(
            {
                "algorithm_id": "ppo",
                "model_class": "example.Unknown",
            }
        )


def test_retired_dqn_backend_is_not_supported() -> None:
    with pytest.raises(ValueError, match="unsupported checkpoint algorithm"):
        resolve_policy_algorithm(
            {
                "training_backend_id": "sb3.dqn",
                "algorithm_id": "dqn",
                "model_class": "stable_baselines3.dqn.dqn.DQN",
            }
        )
    with pytest.raises(ValueError, match="unknown training backend"):
        load_training_backend("sb3.dqn")


def test_algorithm_registry_views_are_derived_from_authoritative_specs() -> None:
    for algorithm_id, spec in POLICY_ALGORITHM_SPECS.items():
        assert ALGORITHM_MODEL_CLASSES[algorithm_id] == frozenset(spec.model_classes)
        assert all(
            MODEL_CLASS_ALGORITHMS[model_class] == algorithm_id
            for model_class in spec.model_classes
        )
        assert (algorithm_id in RUNTIME_POLICY_ALGORITHMS) is (
            spec.runtime_family is not None
        )
        assert (algorithm_id in SB3_ALGORITHMS) is (spec.runtime_family == "sb3")
        assert default_action_selection_mode(algorithm_id) == (spec.default_action_selection_mode)
