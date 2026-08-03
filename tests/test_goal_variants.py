from __future__ import annotations

from copy import deepcopy

import pytest

from gradlab.goal_variants import (
    build_goal_variant_descriptor,
    goal_contract_diff,
    goal_contract_diff_labels,
    goal_variant_id,
    goal_variant_projection,
    validate_goal_variant_descriptor,
)


def goal_document() -> dict[str, object]:
    return {
        "goal_id": "Level1-1",
        "title": "Mario Level1-1",
        "train": {
            "environment": {
                "task": {
                    "action": {
                        "sticky_probability": 0,
                    }
                }
            }
        },
        "eval": {
            "environment": {
                "task": {
                    "action": {
                        "sticky_probability": 0,
                    }
                }
            }
        },
    }


def test_goal_variant_identity_uses_authored_and_effective_contracts() -> None:
    authored = goal_document()
    effective = deepcopy(authored)
    effective["train"]["environment"]["task"]["action"][  # type: ignore[index]
        "sticky_probability"
    ] = 0.25
    effective["eval"]["environment"]["task"]["action"][  # type: ignore[index]
        "sticky_probability"
    ] = 0.25
    descriptor = build_goal_variant_descriptor(
        goal_slug="SuperMarioBros-Nes-v0/Level1-1",
        source_sha="a" * 40,
        authored_goal=authored,
        effective_goal=effective,
    )

    assert descriptor["source_relation"] == "changed"
    assert descriptor["variant_id"] == goal_variant_id(
        goal_slug=descriptor["goal_slug"],
        goal_contract_sha256_value=descriptor["goal_contract_sha256"],
        effective_goal_contract_sha256=descriptor["effective_goal_contract_sha256"],
    )
    assert "sticky probability 0 → 0.25" in descriptor["label"]
    assert descriptor["diff"] == [
        {
            "path": "train+eval.environment.task.action.sticky_probability",
            "before": 0,
            "after": 0.25,
            "kind": "changed",
        }
    ]
    projection = goal_variant_projection(descriptor)
    assert projection["goal_variant_id"] == descriptor["variant_id"]
    assert "sticky_probability" in projection["goal_variant_diff_json"]


def test_goal_variant_identity_preserves_canonical_json_hash_bytes() -> None:
    assert (
        goal_variant_id(
            goal_slug="SuperMarioBros-Nes-v0/Level1-1",
            goal_contract_sha256_value="1" * 64,
            effective_goal_contract_sha256="2" * 64,
        )
        == "goal-variant-a4af1a829454a35e02f38c72"
    )


def test_presentation_only_changes_do_not_fabricate_a_goal_diff() -> None:
    authored = goal_document()
    effective = deepcopy(authored)
    effective["title"] = "Renamed for display"
    descriptor = build_goal_variant_descriptor(
        goal_slug="SuperMarioBros-Nes-v0/Level1-1",
        source_sha="b" * 40,
        authored_goal=authored,
        effective_goal=effective,
    )

    assert descriptor["source_relation"] == "changed"
    assert descriptor["diff"] == []
    assert descriptor["label"] == "Mario Level1-1"


def test_goal_contract_comparison_labels_historical_behavior_in_plain_language() -> None:
    current = goal_document()
    previous = deepcopy(current)
    previous["train"]["environment"]["task"]["action"][  # type: ignore[index]
        "sticky_probability"
    ] = 0.25
    previous["eval"]["environment"]["task"]["action"][  # type: ignore[index]
        "sticky_probability"
    ] = 0.25

    diff, truncated = goal_contract_diff(current, previous)

    assert truncated is False
    assert diff == [
        {
            "path": "train+eval.environment.task.action.sticky_probability",
            "before": 0,
            "after": 0.25,
            "kind": "changed",
        }
    ]
    assert goal_contract_diff_labels(diff) == ["Sticky action probability 0 → 0.25"]


def test_goal_contract_comparison_expands_acceptance_rules_instead_of_json_blobs() -> None:
    current = {
        "eval": {
            "acceptance": [
                {
                    "metric": "eval/full/episode/return/shaped/mean",
                    "operator": ">=",
                    "threshold": 10.0,
                }
            ]
        }
    }
    previous = deepcopy(current)
    previous["eval"]["acceptance"][0]["threshold"] = 8.0  # type: ignore[index]

    diff, truncated = goal_contract_diff(current, previous)

    assert truncated is False
    assert diff == [
        {
            "path": "eval.acceptance.0.threshold",
            "before": 10.0,
            "after": 8.0,
            "kind": "changed",
        }
    ]
    assert goal_contract_diff_labels(diff) == ["Evaluation acceptance rule 1 threshold 10 → 8"]


def test_goal_contract_comparison_summarizes_policy_input_definitions() -> None:
    current = {
        "train": {
            "environment": {
                "task": {
                    "model_inputs": {
                        "context": {
                            "armor": {
                                "signal": "armor",
                                "encoding": {"kind": "continuous", "clip": True},
                            },
                            "selected_weapon": {
                                "signal": "selected_weapon",
                                "encoding": {"kind": "categorical", "values": [1, 2, 3]},
                            },
                        }
                    }
                },
            }
        }
    }
    previous = deepcopy(current)
    context = previous["train"]["environment"]["task"]["model_inputs"]["context"]  # type: ignore[index]
    del context["selected_weapon"]
    context["armor"]["encoding"] = {  # type: ignore[index]
        "kind": "continuous"
    }

    diff, truncated = goal_contract_diff(current, previous)

    assert truncated is False
    assert goal_contract_diff_labels(diff) == [
        "Policy input armor definition changed",
        "Policy input selected weapon removed",
    ]


def test_goal_contract_labels_vizdoom_native_horizon_in_tics() -> None:
    current = {
        "train": {
            "environment": {
                "env_config": {"env_args": {"vizdoom_config": {"episode_timeout": 300}}}
            }
        }
    }
    previous = deepcopy(current)
    previous["train"]["environment"]["env_config"]["env_args"]["vizdoom_config"][  # type: ignore[index]
        "episode_timeout"
    ] = 600

    diff, truncated = goal_contract_diff(current, previous)

    assert truncated is False
    assert goal_contract_diff_labels(diff) == ["Native episode horizon (tics) 300 → 600"]


def test_descriptor_rejects_identity_tampering() -> None:
    authored = goal_document()
    descriptor = build_goal_variant_descriptor(
        goal_slug="SuperMarioBros-Nes-v0/Level1-1",
        source_sha="c" * 40,
        authored_goal=authored,
        effective_goal=authored,
    )
    descriptor["variant_id"] = "goal-variant-" + "0" * 24

    with pytest.raises(ValueError, match="identity mismatch"):
        validate_goal_variant_descriptor(descriptor)
