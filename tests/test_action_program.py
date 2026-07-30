from __future__ import annotations

import json
import zipfile
from types import SimpleNamespace
from unittest import mock

import gymnasium as gym
import numpy as np
import pytest

from gradlab.action_program import (
    ACTION_PROGRAM_MEMBER,
    ACTION_PROGRAM_MODEL_CLASS,
    ACTION_PROGRAM_POLICY_TYPE,
    ACTION_PROGRAM_SCHEMA_VERSION,
    ActionProgramPolicy,
    ActionRun,
)
from gradlab.policy_models import load_policy_model, resolve_policy_algorithm


ACTIONS = ("noop", "right", "right_b", "right_a", "right_a_b", "a", "left")


def test_action_program_round_trip_and_lane_resets(tmp_path) -> None:
    path = tmp_path / "model.zip"
    policy = ActionProgramPolicy(
        action_names=ACTIONS,
        action_runs=(
            ActionRun(2, 2),
            ActionRun(2, 1),
            ActionRun(4, 2),
        ),
        fallback_action=0,
        initial_seed=127,
    )
    policy.save(path)
    loaded = ActionProgramPolicy.load(path)
    loaded.bind_action_space(gym.spaces.Discrete(len(ACTIONS)))

    assert loaded.action_runs == (ActionRun(2, 3), ActionRun(4, 2))
    assert loaded.run_count == 2
    assert loaded.step_count == 5
    with zipfile.ZipFile(path) as archive:
        payload = json.loads(archive.read(ACTION_PROGRAM_MEMBER))
    assert payload == {
        "schema_version": ACTION_PROGRAM_SCHEMA_VERSION,
        "policy_type": ACTION_PROGRAM_POLICY_TYPE,
        "model_class": ACTION_PROGRAM_MODEL_CLASS,
        "action_names": list(ACTIONS),
        "action_runs": [[2, 3], [4, 2]],
        "fallback_action": 0,
        "initial_seed": 127,
    }
    assert loaded.default_playback_seed == 127

    obs = np.zeros((2, 1), dtype=np.float32)
    assert loaded.predict(obs, deterministic=False)[0].tolist() == [2, 2]
    assert loaded.predict(obs, deterministic=False)[0].tolist() == [2, 2]
    assert loaded.predict(obs, deterministic=False)[0].tolist() == [2, 2]
    assert loaded.predict(obs, deterministic=False)[0].tolist() == [4, 4]
    loaded.reset_lanes([True, False])
    assert loaded.predict(obs, deterministic=False)[0].tolist() == [2, 4]


def test_action_program_artifact_discriminator_prevents_same_step_role_collision(
    tmp_path,
) -> None:
    policy = ActionProgramPolicy(
        action_names=ACTIONS,
        action_runs=(ActionRun(2, 3),),
        fallback_action=0,
    )
    checkpoint = tmp_path / "checkpoint.zip"
    final = tmp_path / "final.zip"
    policy.save(checkpoint, artifact_discriminator="checkpoint:100")
    policy.save(final, artifact_discriminator="final:100")

    assert checkpoint.read_bytes() != final.read_bytes()
    assert (
        ActionProgramPolicy.load(checkpoint).action_runs
        == ActionProgramPolicy.load(final).action_runs
    )


def test_action_program_rejects_noncurrent_jerk_artifact(tmp_path) -> None:
    path = tmp_path / "noncurrent-model.zip"
    payload = {
        "schema_version": 2,
        "algorithm_id": "jerk",
        "model_class": "gradlab.jerk.JerkPolicy",
        "action_names": list(ACTIONS),
        "action_runs": [[2, 2], [4, 1]],
        "fallback_action": 0,
    }
    with zipfile.ZipFile(path, mode="w") as archive:
        archive.writestr("jerk_policy.json", json.dumps(payload))

    with pytest.raises(ValueError, match="missing action_program.json"):
        ActionProgramPolicy.load(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 1, "unsupported action-program schema version"),
        ("schema_version", 3, "unsupported action-program schema version"),
        ("policy_type", "jerk", "wrong policy type"),
        ("model_class", "gradlab.jerk.JerkPolicy", "wrong model class"),
    ],
)
def test_action_program_rejects_invalid_identity(tmp_path, field, value, message) -> None:
    path = tmp_path / "model.zip"
    payload = ActionProgramPolicy(
        action_names=ACTIONS,
        action_runs=(ActionRun(2, 1),),
        fallback_action=0,
    ).payload()
    payload[field] = value
    with zipfile.ZipFile(path, mode="w") as archive:
        archive.writestr(ACTION_PROGRAM_MEMBER, json.dumps(payload))

    with pytest.raises(ValueError, match=message):
        ActionProgramPolicy.load(path)


def test_generic_policy_loader_dispatches_action_program(tmp_path) -> None:
    path = tmp_path / "model.zip"
    ActionProgramPolicy(
        action_names=ACTIONS,
        action_runs=(ActionRun(2, 1),),
        fallback_action=0,
    ).save(path)
    metadata = {
        "training_backend_id": "gradlab.jerk",
        "algorithm_id": "action-program",
        "search_algorithm_id": "jerk",
        "model_class": ACTION_PROGRAM_MODEL_CLASS,
    }

    assert resolve_policy_algorithm(metadata) == "action-program"
    from gradlab.trusted_inputs import ApprovedModelInput

    approved = ApprovedModelInput(
        staged=SimpleNamespace(model_path=path),
        approval_hash="unit-test",
    )
    with mock.patch.object(approved, "verify"):
        loaded = load_policy_model(
            approved,
            device="cpu",
            algorithm_id="action-program",
        )
    assert isinstance(loaded, ActionProgramPolicy)


def test_policy_registry_rejects_noncurrent_jerk_metadata() -> None:
    with pytest.raises(ValueError, match="unsupported checkpoint algorithm: jerk"):
        resolve_policy_algorithm(
            {
                "algorithm_id": "jerk",
                "model_class": "gradlab.jerk.JerkPolicy",
            }
        )
