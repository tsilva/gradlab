from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from gradlab.recipe_documents import compose_train_document, load_goal_contract


ROOT = Path(__file__).resolve().parents[1]


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    goals = tmp_path / "experiments/goals/gradlab__bandit"
    recipes = goals / "recipes"
    presets = tmp_path / "experiments/recipes/_presets"
    recipes.mkdir(parents=True)
    presets.mkdir(parents=True)
    goal = goals / "_goal.yaml"
    recipe = recipes / "ppo.yaml"
    shutil.copyfile(ROOT / "experiments/goals/gradlab__bandit/_goal.yaml", goal)
    shutil.copyfile(ROOT / "experiments/goals/gradlab__bandit/recipes/ppo.yaml", recipe)
    recipe.write_text(
        recipe.read_text()
        .replace("logging:\n  wandb_mode: online\n", "")
        .replace("tracking:\n  delivery: local_only\n", "")
    )
    return goal, recipe


def test_tracking_precedence_and_scientific_identity(tmp_path: Path) -> None:
    goal, recipe = _fixture(tmp_path)
    (tmp_path / "experiments/tracking.yaml").write_text(
        "backend: mlflow\ndelivery: online\n"
    )
    baseline = compose_train_document(goal, recipe)
    assert baseline["train_config"]["tracking"]["backend"] == "mlflow"
    assert baseline["train_config"]["tracking"]["sources"]["backend"].endswith(
        "experiments/tracking.yaml"
    )

    goal.write_text(goal.read_text() + "\ntracking:\n  backend: wandb\n")
    preset = tmp_path / "experiments/recipes/_presets/base.yaml"
    preset.write_text("tracking:\n  backend: mlflow\n  delivery: local_only\n")
    recipe.write_text(
        "defaults:\n- ../../../recipes/_presets/base@_global_\n- _self_\n"
        + recipe.read_text()
        + "\ntracking:\n  backend: wandb\n"
    )
    composed = compose_train_document(goal, recipe)
    assert composed["train_config"]["tracking"]["backend"] == "wandb"
    assert composed["train_config"]["tracking"]["delivery"] == "local_only"
    assert composed["train_config"]["tracking"]["sources"]["backend"].endswith(
        "experiments/goals/gradlab__bandit/recipes/ppo.yaml"
    )
    assert composed["train_config"]["tracking"]["sources"]["delivery"].endswith(
        "experiments/recipes/_presets/base.yaml"
    )
    overridden = compose_train_document(
        goal, recipe, recipe_overrides=("tracking.backend=mlflow",)
    )
    assert overridden["train_config"]["tracking"]["backend"] == "mlflow"
    assert overridden["train_config"]["tracking"]["sources"]["backend"] == "launch override"
    assert overridden["train_config"]["goal_contract_sha256"] == composed["train_config"][
        "goal_contract_sha256"
    ]
    assert overridden["goal_variant"]["variant_id"] == composed["goal_variant"][
        "variant_id"
    ]
    assert "tracking" not in load_goal_contract(goal)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("unknown", "tracking.backend"),
        ("", "tracking.backend"),
    ],
)
def test_invalid_backend_is_rejected_before_launch(
    tmp_path: Path, value: str, message: str
) -> None:
    goal, recipe = _fixture(tmp_path)
    with pytest.raises(ValueError, match=message):
        compose_train_document(goal, recipe, recipe_overrides=(f"tracking.backend={value}",))


def test_legacy_wandb_mode_is_rejected(tmp_path: Path) -> None:
    goal, recipe = _fixture(tmp_path)
    recipe.write_text(recipe.read_text() + "\nlogging:\n  wandb_mode: online\n")
    with pytest.raises(ValueError, match="tracking.delivery"):
        compose_train_document(goal, recipe)
