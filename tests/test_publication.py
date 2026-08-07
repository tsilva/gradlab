from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from gradlab.policy_bundle import PolicyBundle, UnsupportedPolicyDocumentVersion
from gradlab.publication import (
    HASHED_RELEASE_FILES,
    RELEASE_MANIFEST_VERSION,
    REPO_NAMING_SCHEMA_VERSION,
    build_model_repo_id,
    build_release_manifest,
    policy_lineage_contract,
    publication_identity_from_policy_bundle,
    release_comparison,
    render_model_card,
    validate_release_manifest_document,
)


def bundle(*, seed: int = 7, step: int = 4_000_000) -> PolicyBundle:
    action_contract = {
        "schema_version": 1,
        "requested": {"mode": "custom_discrete", "meanings": ["noop", "attack"]},
        "provider": {
            "provider_id": "vizdoom-turbo",
            "mode": "custom_discrete",
            "semantics": {
                "status": "available",
                "encoding": "explicit",
                "entries": [
                    {"value": 0, "semantic_id": "noop"},
                    {"value": 1, "semantic_id": "attack"},
                ],
            },
        },
        "policy": {
            "codec": {"type": "identity"},
            "space": {"type": "discrete", "start": 0, "n": 2},
            "semantics": {
                "status": "available",
                "encoding": "explicit",
                "entries": [
                    {"value": 0, "semantic_id": "noop"},
                    {"value": 1, "semantic_id": "attack"},
                ],
            },
        },
    }
    task = {
        "action": {"set": "native"},
        "signals": {"kills": "killcount", "health": "health"},
        "model_inputs": {
            "schema_version": 1,
            "context": {
                "health": {
                    "signal": "health",
                    "update": "transition",
                    "history": "provider_frame_stack",
                    "encoding": {"kind": "continuous", "scale": 0.01},
                }
            },
        },
        "reward": {"mode": "native", "scale": 1.0},
        "events": {"monster_killed": {"signal": "kills", "operation": "increase"}},
        "termination": {"success": ["time_limit_reached"], "bootstrap": ["time_limit_reached"]},
    }
    recipe = {
        "document_type": "gradlab.recipe",
        "format_version": 4,
        "recipe": {
            "environment": {
                "env_id": "vizdoom-turbo:VizdoomDeathmatch-v1",
                "state": "default",
                "preprocessing": {
                    "obs_resize": [84, 84],
                    "obs_grayscale": True,
                    "obs_crop": [0, 32, 0, 0],
                    "obs_crop_mode": "mask",
                    "frame_stack": 4,
                    "policy_observation_layout": "dict_observation_context_v1",
                },
                "provider_args": {
                    "frame_stack": 4,
                    "num_threads": 32,
                    "info_filter": {"mode": "all", "keys": ["killcount", "health"]},
                    "vizdoom_config": {"episode_timeout": 4200},
                },
                "task": task,
            },
            "environment_hash": "sha256:environment",
            "train": {
                "policy_model": {
                    "schema_version": 2,
                    "encoder": {"kind": "nature_cnn", "features_dim": 512},
                }
            },
            "value_contract": {"discount": 0.995},
            "goal": {
                "goal_id": "VizdoomDeathmatch-v1",
                "title": "ViZDoom single-player Deathmatch score attack",
                "evaluation_mode": "evaluated",
                "eval": {
                    "episodes": 2,
                    "acceptance": [
                        {
                            "metric": "eval/full/progress/kills/mean",
                            "operator": ">=",
                            "threshold": 10.0,
                        }
                    ],
                },
                "objective": {
                    "rank": [
                        "max(eval/full/progress/kills/mean)",
                        "min(leader/checkpoint/step)",
                    ]
                },
            },
        },
    }
    model = {
        "policy": {
            "algorithm_id": "ppo",
            "model_class": "gradlab.ppo.GradLabPPO",
            "training_backend_id": "gradlab.ppo",
            "training_backend_config_hash": "a" * 64,
        },
        "checkpoint": {
            "step": step,
            "sha256": "b" * 64,
            "kind": "checkpoint",
        },
        "provenance": {
            "seed": seed,
            "repo_git_commit": "c" * 40,
            "wandb_run_id": "gradlab-" + "d" * 32,
            "wandb_project": "VizdoomDeathmatch-v1",
            "run_name": "deathmatch-run",
            "recipe_slug": "ppo",
            "training_metadata": {
                "action_contract": action_contract,
                "policy_execution_contract": {
                    "policy_model": deepcopy(recipe["recipe"]["train"]["policy_model"]),
                    "model_inputs": deepcopy(task["model_inputs"]),
                    "role_inputs": {"actor": ["image", "context"], "critic": ["image", "context"]},
                },
                "versions": {"vizdoom_turbo": "1.3.0.post23"},
            },
        },
    }
    return PolicyBundle(
        checkpoint_path=Path("model.zip"),
        model_path=Path("model.json"),
        recipe_path=Path("recipe.json"),
        model=model,
        recipe=recipe,
        source="fixture",
    )


def evaluation_evidence() -> dict:
    acceptance = {
        "rules": [
            {
                "metric": "eval/full/progress/kills/mean",
                "operator": ">=",
                "threshold": 10.0,
            }
        ],
        "outcomes": [
            {
                "metric": "eval/full/progress/kills/mean",
                "label": "Full-eval kills mean",
                "unit": "value",
                "value": 12.5,
                "operator": ">=",
                "threshold": 10.0,
                "passed": True,
            }
        ],
        "passed": True,
    }
    return {
        "document_type": "gradlab.evaluation_evidence",
        "format_version": 1,
        "status": "accepted",
        "identity": {
            "run_id": "gradlab-" + "d" * 32,
            "checkpoint_id": "checkpoint-4000000-" + "b" * 16,
            "checkpoint_step": 4_000_000,
            "checkpoint_sha256": "b" * 64,
            "recipe_sha256": "e" * 64,
        },
        "protocol": {
            "action_sampling": "stochastic",
            "episodes": 2,
            "seed": 10000,
            "seed_protocol": {"kind": "fixed"},
            "manifest": [{"episode": 0}, {"episode": 1}],
        },
        "episode_results": [
            {"episode": 0, "start_id": "default", "progress": {"kills": 12}},
            {"episode": 1, "start_id": "default", "progress": {"kills": 13}},
        ],
        "aggregates": {"eval/full/progress/kills/mean": 12.5},
        "acceptance": acceptance,
        "ranking": {
            "rules": [
                "max(eval/full/progress/kills/mean)",
                "min(leader/checkpoint/step)",
            ],
            "outcomes": [
                {
                    "metric": "eval/full/progress/kills/mean",
                    "label": "Full-eval kills mean",
                    "unit": "value",
                    "value": 12.5,
                    "direction": "max",
                    "rank_value": 12.5,
                },
                {
                    "metric": "leader/checkpoint/step",
                    "label": "Leader checkpoint step",
                    "unit": "steps",
                    "value": 4_000_000,
                    "direction": "min",
                    "rank_value": -4_000_000.0,
                },
            ],
        },
        "contracts": {
            "materialized_goal": bundle().recipe["recipe"]["goal"],
            "evaluation": {"episodes": 2},
            "environment": {"env_id": "vizdoom-turbo:VizdoomDeathmatch-v1"},
        },
        "authoritative_hashes": {
            "intent_sha256": "1" * 64,
            "raw_result_sha256": "2" * 64,
            "verified_result_sha256": "3" * 64,
            "checkpoint_manifest_sha256": "4" * 64,
            "recipe_sha256": "e" * 64,
            "checkpoint_sha256": "b" * 64,
            "evaluation_contract_sha256": "f" * 64,
        },
    }


def evaluation_summary() -> dict:
    return {
        "action_sampling": "stochastic",
        "protocol": "full",
        "checkpoint_step": 4_000_000,
        "checkpoint_artifact": "https://example.invalid/model.zip",
        "episodes": 2,
        "success_rate_min": 0.0,
        "success_rate_mean": 0.0,
        "return_mean": 100.0,
        "progress_max": 13.0,
        "by_start": [
            {
                "start_id": "default",
                "episode_count": 2,
                "success_count": 0,
                "success_rate": 0.0,
                "shaped_return_mean": 100.0,
                "failure_reasons": {"player_died": 2},
            }
        ],
        "checkpoint_sha256": "b" * 64,
        "recipe_sha256": "e" * 64,
        "recipe_format_version": 4,
        "evaluation_contract_sha256": "f" * 64,
        "exact_contract": True,
    }


def replay() -> dict:
    return {
        "capture_id": "capture-" + "1" * 32,
        "capture_fence_sha256": "1" * 64,
        "run_id": "gradlab-" + "d" * 32,
        "checkpoint_id": "checkpoint-4000000-" + "b" * 16,
        "checkpoint_sha256": "b" * 64,
        "recipe_sha256": "e" * 64,
        "episode": 1,
        "seed": 7,
        "start_id": "default",
        "sampling_mode": "stochastic",
        "steps": 2,
        "return_value": 100.0,
        "max_x_pos": 0,
        "outcome": "terminated",
        "success": False,
        "boundary_role": "terminal_observation",
        "contract": {"mode": "training"},
        "execution": {
            "source": {"kind": "checkout"},
            "qualified_environment_id": "vizdoom-turbo:VizdoomDeathmatch-v1",
            "provider_id": "vizdoom-turbo",
            "provider_version": "1.3.0.post23",
            "environment_hash": "sha256:environment",
            "runtime_versions": {"vizdoom_turbo": "1.3.0.post23"},
            "runtime_image_digest": "docker:image@sha256:" + "6" * 64,
            "asset": None,
            "execution_target": "local_player",
            "device_type": "cpu",
            "contract_mode": "training",
            "overrides": [],
            "seed": 7,
        },
        "media": {
            "sha256": "7" * 64,
            "size_bytes": 100,
            "frames": 3,
            "fps": 30.0,
            "width": 1280,
            "height": 720,
        },
    }


def publication() -> dict:
    return {
        "request_fingerprint": "8" * 64,
        "huggingface_username": "tsilva",
        "huggingface_namespace": "tsilva",
        "youtube_channel_id": "channel",
        "youtube_channel_title": "GradLab",
        "youtube_privacy": "public",
    }


def manifest() -> dict:
    source = {
        "repository": "https://github.com/tsilva/gradlab",
        "commit": "c" * 40,
        "run_id": "gradlab-" + "d" * 32,
        "run_name": "deathmatch-run",
        "wandb_project": "VizdoomDeathmatch-v1",
        "recipe": "ppo",
        "seed": 7,
        "checkpoint_step": 4_000_000,
        "checkpoint_artifact": "https://example.invalid/model.zip",
    }
    return build_release_manifest(
        publication_identity_from_policy_bundle("VizdoomDeathmatch-v1", bundle()),
        bundle(),
        release_version="v3",
        published_at="2026-08-07T12:00:00Z",
        source=source,
        evaluation=evaluation_summary(),
        artifacts={name: {"sha256": "9" * 64, "size_bytes": 1} for name in HASHED_RELEASE_FILES},
        youtube_url="https://www.youtube.com/watch?v=A0Id3WxTlqk",
        replay=replay(),
        publication=publication(),
        evaluation_evidence=evaluation_evidence(),
        featured=True,
    )


def test_schema_v3_identity_uses_trainer_and_full_lineage_digest() -> None:
    identity = publication_identity_from_policy_bundle("VizdoomDeathmatch-v1", bundle())
    assert identity.trainer == "GradLab"
    assert identity.trainer_slug == "gradlab"
    assert len(identity.lineage_digest) == 64
    assert build_model_repo_id(identity) == (
        f"tsilva/VizdoomDeathmatch-v1_gradlab-ppo_{identity.lineage_digest[:8]}"
    )


@pytest.mark.parametrize(
    "path,value",
    [
        (("recipe", "train", "policy_model", "encoder", "features_dim"), 256),
        (("recipe", "environment", "preprocessing", "frame_stack"), 8),
        (("recipe", "environment", "task", "reward", "scale"), 0.5),
        (("recipe", "value_contract", "discount"), 0.99),
        (("recipe", "environment", "task", "termination", "success"), ["monster_killed"]),
    ],
)
def test_material_policy_semantics_change_lineage(path: tuple[str, ...], value: object) -> None:
    original = bundle()
    changed = bundle()
    target = changed.recipe
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    assert policy_lineage_contract("VizdoomDeathmatch-v1", original) != policy_lineage_contract(
        "VizdoomDeathmatch-v1", changed
    )
    assert publication_identity_from_policy_bundle(
        "VizdoomDeathmatch-v1", original
    ).lineage_digest != publication_identity_from_policy_bundle(
        "VizdoomDeathmatch-v1", changed
    ).lineage_digest


def test_operational_and_evaluation_changes_do_not_change_lineage() -> None:
    original = bundle(seed=7, step=4_000_000)
    changed = bundle(seed=99, step=8_000_000)
    changed.model["provenance"]["repo_git_commit"] = "0" * 40
    changed.model["provenance"]["training_metadata"]["versions"]["vizdoom_turbo"] = "9.9"
    changed.recipe["recipe"]["environment"]["provider_args"]["num_threads"] = 1
    changed.recipe["recipe"]["goal"]["eval"]["acceptance"][0]["threshold"] = 999
    assert publication_identity_from_policy_bundle(
        "VizdoomDeathmatch-v1", original
    ).lineage_digest == publication_identity_from_policy_bundle(
        "VizdoomDeathmatch-v1", changed
    ).lineage_digest


def test_manifest_v3_is_the_only_supported_release_contract() -> None:
    value = manifest()
    assert value["format_version"] == RELEASE_MANIFEST_VERSION == 3
    assert value["repo_naming_schema"] == REPO_NAMING_SCHEMA_VERSION == 3
    assert value["release"]["checkpoint_tag"] == "checkpoint-4000000"
    assert value["evaluation"]["evidence_file"] == "evaluation_evidence.json"
    assert validate_release_manifest_document(value) == value
    old = deepcopy(value)
    old["format_version"] = 2
    with pytest.raises(UnsupportedPolicyDocumentVersion):
        validate_release_manifest_document(old)


def test_gradlab_card_uses_faithful_metadata_and_provider_aware_quick_start() -> None:
    card = render_model_card(manifest(), bundle())
    assert "Trainer: **GradLab**" in card
    assert "Algorithm: **PPO**" in card
    assert "`gradlab.ppo.GradLabPPO`" in card
    assert "Compatibility: **SB3-compatible**" in card
    assert "library_name: gradlab" in card
    assert "stable-baselines3" not in card.casefold()
    assert "uvx --from gradlab gradlab play hf://" in card
    assert "rom import" not in card
    assert "representative media and is not evaluation" in card
    assert "eval/full/progress/kills/mean" in card


def test_release_comparison_requires_all_four_contract_axes() -> None:
    current = manifest()
    previous = deepcopy(current)
    previous["release"]["version"] = "v2"
    assert release_comparison(current, previous)["comparable"] is True
    for section, key in (
        ("repository", "lineage_digest"),
        ("source", "run_id"),
        ("source", "seed"),
        ("evaluation", "evaluation_contract_sha256"),
    ):
        changed = deepcopy(previous)
        changed[section][key] = "different"
        assert release_comparison(current, changed)["comparable"] is False
