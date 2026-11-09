from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path

import pytest

from gradlab.checkpoint_acceptance import (
    acceptance_aggregates,
    evaluate_acceptance,
    validate_episode_rows,
)
from gradlab.policy_bundle import (
    build_recipe_document,
    canonical_json_sha256,
    evaluation_contract,
    load_recipe_document,
    write_canonical_json,
)
from gradlab.recipe_documents import compose_resolved_train_documents


ROOT = Path(__file__).resolve().parents[1]
GOAL = ROOT / "experiments/goals/Breakout-Atari2600-v0/TwoWalls/_goal.yaml"
SOURCE_SHA = "a" * 40
RUNTIME = "docker:example.invalid/gradlab@sha256:" + "b" * 64
OVERRIDE = "train.backend.config.learning_rate=0.00015"


@pytest.fixture(scope="module")
def resolved_recipes():
    return tuple(
        compose_resolved_train_documents(
            GOAL,
            GOAL.parent / "recipes" / f"{name}.yaml",
            recipe_overrides=(OVERRIDE,),
            source_sha=SOURCE_SHA,
        )
        for name in ("ppo-two-walls", "ppo-screening")
    )


@pytest.fixture(scope="module")
def recipe_documents(resolved_recipes) -> tuple[dict, dict]:
    documents = []
    for name, resolved in zip(("ppo-two-walls", "ppo-screening"), resolved_recipes, strict=True):
        documents.append(
            build_recipe_document(
                resolved.effective,
                repo_root=ROOT,
                source_commit=SOURCE_SHA,
                run_description=f"{name} screening regression",
                seed=123,
                runtime_image_ref=RUNTIME,
                base_materialized_recipe=resolved.base,
                canonical_goal=resolved.canonical_goal,
            )
        )
    return documents[0], documents[1]


def test_screening_preserves_training_and_checkpoint_contracts(resolved_recipes) -> None:
    evaluated, screening = resolved_recipes
    original = deepcopy(evaluated.effective["train_config"])
    screened = deepcopy(screening.effective["train_config"])

    assert original.pop("checkpoint_eval_backend") == "modal"
    assert screened.pop("checkpoint_eval_backend") == "none"
    original_monitoring = original.pop("checkpoint_monitoring", {})
    screened_monitoring = screened.pop("checkpoint_monitoring")
    assert not original_monitoring.get("enabled", False)
    assert not screened_monitoring["enabled"]
    assert screened == original
    assert screened["checkpoint_freq"] == 10_000_000
    assert screening.effective["goal"] == evaluated.effective["goal"]
    assert screening.effective["goal_variant"] == evaluated.effective["goal_variant"]


def test_screening_bundle_keeps_provenance_and_separate_evaluation(
    recipe_documents, tmp_path: Path,
) -> None:
    evaluated, screening = recipe_documents
    path = tmp_path / "recipe.json"
    write_canonical_json(path, screening)
    restored = load_recipe_document(path)

    assert canonical_json_sha256(restored) == canonical_json_sha256(screening)
    assert restored["recipe"]["goal"]["evaluation_mode"] == "evaluated"
    assert "eval" not in restored["recipe"]
    assert "playback" in restored["recipe"]
    assert restored["recipe"]["train_config"]["seed"] == 123
    assert restored["recipe"]["recipe_overrides"] == [OVERRIDE]
    assert restored["provenance"]["source_commit"] == SOURCE_SHA
    assert restored["provenance"]["runtime"]["image_ref"] == RUNTIME
    sources = {
        item["path"]: item["sha256"] for item in restored["provenance"]["source_files"]
    }
    assert GOAL.relative_to(ROOT).as_posix() in sources
    assert "experiments/recipes/_presets/breakout-two-walls-ppo.yaml" in sources
    assert "experiments/recipes/_presets/screening.yaml" in sources
    assert GOAL.parent.joinpath("recipes/ppo-screening.yaml").relative_to(ROOT).as_posix() in sources
    for name, digest in sources.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    contract = evaluation_contract(restored)
    assert contract == evaluation_contract(evaluated)
    assert contract["seed"] == 10_000
    assert contract["episodes"] == 100


def test_screening_acceptance_still_requires_complete_independent_evidence(
    recipe_documents,
) -> None:
    _, screening = recipe_documents
    contract = evaluation_contract(screening)
    rows = [
        {
            "episode_id": entry["episode_id"],
            "seed_lane": entry["lane"],
            "seed_episode_ordinal": entry["lane_episode_ordinal"],
            "seed": entry["seed"],
            "start_state": entry["start_state"],
            "seed_protocol": contract["seed_protocol"],
            "outcome": "success",
            "return": 1.0,
            "steps": 100,
            "bricks_destroyed_normalized": 1.0,
        }
        for entry in contract["manifest"]["episodes"]
    ]
    assert all(row["seed"] != screening["recipe"]["train_config"]["seed"] for row in rows)
    assert not evaluate_acceptance(
        {"train/progress/bricks_destroyed_normalized/mean": 1.0}, contract=contract,
    )[0]
    with pytest.raises(ValueError, match="every planned episode"):
        validate_episode_rows(rows[:-1], contract=contract)
    assert not evaluate_acceptance(
        acceptance_aggregates(rows[:-1], contract=contract), contract=contract,
    )[0]
    validate_episode_rows(rows, contract=contract)
    rows[-1]["bricks_destroyed_normalized"] = 0.5
    assert not evaluate_acceptance(
        acceptance_aggregates(rows, contract=contract), contract=contract,
    )[0]
    rows[-1]["bricks_destroyed_normalized"] = 1.0
    assert evaluate_acceptance(
        acceptance_aggregates(rows, contract=contract), contract=contract,
    )[0]
