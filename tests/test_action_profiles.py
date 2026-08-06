from __future__ import annotations

from pathlib import Path

import pytest

from gradlab.action_codecs import (
    VIZDOOM_SHARED_MULTIDISCRETE_BUTTONS,
    VIZDOOM_SHARED_MULTIDISCRETE_CODEC,
    VIZDOOM_SHARED_MULTIDISCRETE_NVEC,
)
from gradlab.action_profiles import VIZDOOM_SHARED_ACTION_PROFILE
from gradlab.recipe_documents import compose_train_document


GOALS_ROOT = Path("experiments/goals")


def _recipe(goal: Path) -> Path:
    recipes = sorted((goal.parent / "recipes").glob("*.yaml"))
    return next((path for path in recipes if path.name == "ppo.yaml"), recipes[0])


def test_shared_action_profile_materializes_all_current_vizdoom_goals_without_changing_defaults() -> (
    None
):
    goals = sorted(GOALS_ROOT.glob("Vizdoom*/_goal.yaml"))
    assert len(goals) == 13
    for goal in goals:
        recipe = _recipe(goal)
        baseline = compose_train_document(goal, recipe)
        repeated = compose_train_document(goal, recipe)
        selected = compose_train_document(
            goal,
            recipe,
            recipe_overrides=(f"action_profile={VIZDOOM_SHARED_ACTION_PROFILE}",),
        )

        assert baseline["environment_hash"] == repeated["environment_hash"]
        assert "action_profile" not in baseline["train_config"]
        assert selected["train_config"]["action_profile"] == VIZDOOM_SHARED_ACTION_PROFILE
        assert selected["environment_hash"] != baseline["environment_hash"]
        assert selected["goal_variant"]["source_relation"] == "changed"
        assert (
            selected["train_config"]["goal_contract_sha256"]
            != selected["train_config"]["effective_goal_contract_sha256"]
        )

        train = selected["goal"]["train"]["environment"]
        evaluation = selected["goal"]["eval"]["environment"]
        for environment in (train, evaluation):
            assert environment["env_config"]["env_args"]["use_restricted_actions"] == "filtered"
            assert (
                tuple(environment["env_config"]["env_args"]["vizdoom_config"]["available_buttons"])
                == VIZDOOM_SHARED_MULTIDISCRETE_BUTTONS
            )
            codec = environment["task"]["action"]["codec"]
            assert codec["type"] == VIZDOOM_SHARED_MULTIDISCRETE_CODEC
            assert len(codec["legal_tuples"]) == len(codec["source_table"])
            assert codec["legal_tuples"][0] == [0] * len(VIZDOOM_SHARED_MULTIDISCRETE_NVEC)
        assert train["task"]["action"] == evaluation["task"]["action"]


def test_shared_action_profile_rejects_unknown_profiles_and_raw_action_overrides() -> None:
    goal = GOALS_ROOT / "VizdoomBasic-v1" / "_goal.yaml"
    recipe = _recipe(goal)
    with pytest.raises(ValueError, match="unknown action_profile"):
        compose_train_document(
            goal,
            recipe,
            recipe_overrides=("action_profile=missing-v1",),
        )
    with pytest.raises(ValueError, match="cannot be combined with raw action overrides"):
        compose_train_document(
            goal,
            recipe,
            recipe_overrides=(
                f"action_profile={VIZDOOM_SHARED_ACTION_PROFILE}",
                "train.environment.task.action.set=native",
            ),
        )
