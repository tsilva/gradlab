from __future__ import annotations

import pytest

from gradlab.policy_models import resolve_policy_algorithm
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
        resolve_sb3_algorithm({"model_class": "example.Unknown"})


def test_dqn_is_portable_runtime_provenance_but_not_a_launchable_backend() -> None:
    metadata = {
        "training_backend_id": "sb3.dqn",
        "algorithm_id": "dqn",
        "model_class": "stable_baselines3.dqn.dqn.DQN",
    }

    assert resolve_policy_algorithm(metadata) == "dqn"
    assert resolve_sb3_algorithm(metadata) == "dqn"
    with pytest.raises(ValueError, match="unknown training backend"):
        load_training_backend("sb3.dqn")
