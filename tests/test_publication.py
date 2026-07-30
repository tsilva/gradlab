from __future__ import annotations

import hashlib
import json
import tempfile
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest
from huggingface_hub import ModelCard
from jinja2 import UndefinedError

from gradlab.preprocessing import preprocessing_contract
from gradlab.policy_bundle import (
    PolicyBundle,
    PolicyDocumentError,
    UnsupportedPolicyDocumentVersion,
    build_model_document,
    build_recipe_document,
    evaluation_contract_sha256,
    load_policy_bundle,
    sha256_file,
    write_canonical_json,
)
from gradlab.recipe_documents import compose_resolved_train_documents
from gradlab.training_backend import training_backend_config_hash
from gradlab.publication import (
    GITATTRIBUTES_TEXT,
    HUGGINGFACE_RELEASE_FILES,
    MIT_LICENSE_TEXT,
    PublicationIdentity,
    _render_model_card_template,
    build_model_repo_id,
    build_release_manifest,
    normalize_publication_evaluation,
    publication_identity_from_policy_bundle,
    publication_source_from_policy_bundle,
    release_artifact_records,
    render_model_card,
    validate_release_bundle,
)


def _render_current_model_card_fixture(
    *,
    algorithm: str = "ppo",
    model_class: str = "stable_baselines3.ppo.ppo.PPO",
    search_algorithm: str | None = None,
    youtube_url: str | None = "https://www.youtube.com/watch?v=example",
) -> str:
    raw_metadata = model_metadata(
        algorithm=algorithm,
        model_class=model_class,
        search_algorithm=search_algorithm,
    )
    bundle = policy_bundle_from_metadata(raw_metadata)
    identity = publication_identity_from_policy_bundle("Level1-1", bundle)
    raw_evaluation = evaluation_payload()
    if algorithm == "action-program":
        raw_evaluation["action_sampling"] = "program"
    evaluation = normalize_publication_evaluation(
        raw_evaluation,
        algorithm_id=algorithm,
    )
    source = publication_source_from_policy_bundle(bundle, evaluation)
    manifest = build_release_manifest(
        identity,
        bundle,
        release_version="v1",
        published_at="2026-07-14T12:00:00Z",
        source=source,
        evaluation=evaluation.as_manifest_value(),
        artifacts={},
        youtube_url=youtube_url,
    )
    return render_model_card(manifest, bundle)


def model_metadata(
    *,
    provider: str = "supermariobrosnes-turbo",
    game: str = "SuperMarioBros-Nes-v0",
    grayscale: bool = True,
    resize: tuple[int, int] = (84, 84),
    crop: list[int] | None = None,
    crop_mode: str = "mask",
    frame_stack: int = 4,
    layout: str = "channel_first",
    action_set: str = "basic",
    algorithm: str = "ppo",
    model_class: str = "stable_baselines3.ppo.ppo.PPO",
    search_algorithm: str | None = None,
) -> dict:
    if crop is None and game == "SuperMarioBros-Nes-v0":
        crop = [32, 0, 0, 0]
    action_meanings = {
        "basic": ("noop", "right", "right_b", "right_a", "right_a_b", "a", "left"),
        "right-jump": ("right", "right_b", "right_a", "right_a_b"),
        "native": ("native",),
    }
    if action_set not in action_meanings:
        raise ValueError(f"unknown action set: {action_set}")
    meanings = action_meanings[action_set]
    action_contract = {
        "schema_version": 1,
        "provider": {
            "mode": "native" if action_set == "native" else "custom_discrete",
            "preset": None if action_set == "native" else action_set,
        },
        "policy": {
            "space": {"type": "discrete", "start": 0, "n": len(meanings)},
            "semantics": {
                "status": "available",
                "encoding": "explicit",
                "entries": [
                    {"value": index, "semantic_id": meaning}
                    for index, meaning in enumerate(meanings)
                ],
            },
        },
    }
    metadata = {
        "algorithm_id": algorithm,
        "model_class": model_class,
        "training_backend_id": ("gradlab.jerk" if algorithm == "action-program" else "sb3.ppo"),
        "training_backend_config_hash": "c" * 64,
        "seed": 7,
        "repo_git_commit": "a" * 40,
        "run_name": "bx0000000000000000-release-s7-20260714T120000Z",
        "wandb_run_id": "run123",
        "wandb_project": "SuperMarioBros-Nes-v0",
        "recipe_slug": "base",
        "checkpoint_step": 4_000_000,
        "training_metadata": {
            "environment_hash": "sha256:environment",
            "environment": {
                "env_id": f"{provider}:{game}",
                "task": {"action": {"set": action_set}},
            },
            "action_contract": action_contract,
            "preprocessing": {
                "obs_resize": list(resize),
                "obs_crop": crop,
                "obs_crop_mode": crop_mode,
                "obs_grayscale": grayscale,
                "frame_stack": frame_stack,
                "policy_observation_layout": layout,
            },
        },
    }
    if search_algorithm:
        metadata["search_algorithm_id"] = search_algorithm
    return metadata


def policy_bundle_from_metadata(metadata: dict) -> PolicyBundle:
    value = deepcopy(metadata)
    training = deepcopy(value.pop("training_metadata"))
    environment = deepcopy(training["environment"])
    environment["preprocessing"] = deepcopy(training["preprocessing"])
    policy_keys = {
        "algorithm_id",
        "model_class",
        "training_backend_id",
        "training_backend_config_hash",
    }
    policy = {key: value.pop(key) for key in policy_keys}
    checkpoint_step = value.pop("checkpoint_step")
    persisted_training = {
        key: deepcopy(training[key]) for key in ("action_contract", "versions") if key in training
    }
    if persisted_training:
        value["training_metadata"] = persisted_training
    return PolicyBundle(
        checkpoint_path=Path("model.zip"),
        model_path=Path("model.json"),
        recipe_path=Path("recipe.json"),
        model={
            "policy": policy,
            "provenance": value,
            "checkpoint": {"step": checkpoint_step},
        },
        recipe={
            "recipe": {
                "environment": environment,
                "environment_hash": training["environment_hash"],
            }
        },
        source="test",
    )


def evaluation_payload() -> dict:
    return {
        "action_sampling": "stochastic",
        "protocol": "full",
        "eval/checkpoint/step": 4_000_000,
        "checkpoint_artifact": "tsilva/project/run-checkpoint:v3",
        "eval/full/episode/completed/count": 30,
        "eval/full/outcome/success/across_starts/rate/min": 0.8,
        "eval/full/outcome/success/across_starts/rate/mean": 0.9,
        "eval/full/episode/return/shaped/mean": 123.5,
        "eval/full/progress/x/max": 6256,
        "eval/full/by_start": [
            {
                "start_id": "Level1-1",
                "episodes": 15,
                "success_count": 12,
                "success_rate": 0.8,
                "return_mean": 100.0,
            },
            {
                "start_id": "Level1-2",
                "episodes": 15,
                "success_count": 15,
                "success_rate": 1.0,
                "return_mean": 147.0,
            },
        ],
    }


def test_mario_publication_identity_is_exact_and_provider_neutral() -> None:
    native = publication_identity_from_policy_bundle(
        "Level1-1", policy_bundle_from_metadata(model_metadata())
    )
    retro = publication_identity_from_policy_bundle(
        "Level1-1",
        policy_bundle_from_metadata(model_metadata(provider="stable-retro-turbo")),
    )

    assert native == retro
    assert native == PublicationIdentity(
        game_family="NES-SuperMarioBros",
        goal="Level1-1",
        policy_variant="gray84-hudmask-stack4-basic",
        algorithm="ppo",
    )
    assert build_model_repo_id(native) == (
        "tsilva/NES-SuperMarioBros_Level1-1_gray84-hudmask-stack4-basic_ppo"
    )


def test_non_default_reward_shape_gets_a_bounded_repository_suffix() -> None:
    metadata = model_metadata()
    metadata.update(
        reward_shape="score-step-0p01-v1",
        reward_shape_sha256="sha256:" + "a" * 64,
        reward_shape_is_default=False,
    )
    identity = publication_identity_from_policy_bundle(
        "Level1-1", policy_bundle_from_metadata(metadata)
    )

    assert identity.policy_variant.endswith("shape-score-step-0p01-v1-aaaaaaaa")
    assert len(identity.repo_name) <= 96


@pytest.mark.parametrize(
    ("provider", "game", "family"),
    [
        ("stable-retro-turbo", "SuperMarioBros3-Nes-v0", "NES-SuperMarioBros3"),
        ("ale-py", "breakout", "Atari2600-Breakout"),
        (
            "breakout-turbo-env",
            "Breakout-Atari2600-v0",
            "Atari2600-Breakout",
        ),
        ("stable-retro-turbo", "Breakout-Atari2600-v0", "Atari2600-Breakout"),
        ("ale-py", "ms_pacman", "Atari2600-MsPacman"),
    ],
)
def test_registered_game_families(provider: str, game: str, family: str) -> None:
    metadata = model_metadata(provider=provider, game=game, crop=[0, 0, 0, 0], action_set="native")
    identity = publication_identity_from_policy_bundle(
        "Goal1", policy_bundle_from_metadata(metadata)
    )
    assert identity.game_family == family


def test_policy_variant_records_rgb_shape_crop_stack_layout_and_action() -> None:
    metadata = model_metadata(
        grayscale=False,
        resize=(84, 96),
        crop=[8, 1, 2, 3],
        crop_mode="remove",
        frame_stack=2,
        layout="dict_image_task",
        action_set="native",
    )

    identity = publication_identity_from_policy_bundle(
        "Levels_1-1_1-2", policy_bundle_from_metadata(metadata)
    )

    assert identity.goal == "Levels-1-1-1-2"
    assert identity.policy_variant == "rgb84x96-crop-t8-r1-b2-l3-stack2-taskdict-native"


def test_policy_variant_accepts_another_registered_action_set() -> None:
    identity = publication_identity_from_policy_bundle(
        "Level1-1",
        policy_bundle_from_metadata(model_metadata(action_set="right-jump")),
    )
    assert identity.policy_variant.endswith("-right-jump")


@pytest.mark.parametrize(
    ("algorithm", "model_class"),
    [
        ("ppo", "stable_baselines3.ppo.ppo.PPO"),
        ("a2c", "stable_baselines3.a2c.a2c.A2C"),
        ("action-program", "gradlab.action_program.ActionProgramPolicy"),
        ("cell-graph", "gradlab.cell_graph.CellGraphPolicy"),
    ],
)
def test_supported_algorithms_are_the_last_axis(algorithm: str, model_class: str) -> None:
    identity = publication_identity_from_policy_bundle(
        "Level1-1",
        policy_bundle_from_metadata(model_metadata(algorithm=algorithm, model_class=model_class)),
    )
    assert build_model_repo_id(identity).endswith(f"_{algorithm}")


def test_publication_rejects_unknown_family_and_algorithm_mismatch() -> None:
    with pytest.raises(ValueError, match="no registered canonical game family"):
        publication_identity_from_policy_bundle(
            "Goal1",
            policy_bundle_from_metadata(
                model_metadata(provider="gymnasium", game="CustomVector-v0", crop=[])
            ),
        )
    with pytest.raises(ValueError, match="incompatible"):
        publication_identity_from_policy_bundle(
            "Level1-1",
            policy_bundle_from_metadata(
                model_metadata(
                    algorithm="a2c",
                    model_class="stable_baselines3.ppo.ppo.PPO",
                )
            ),
        )
    with pytest.raises(ValueError, match="unknown environment provider"):
        publication_identity_from_policy_bundle(
            "Level1-1",
            policy_bundle_from_metadata(model_metadata(provider="unregistered-mario-provider")),
        )
def test_long_repo_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="96"):
        build_model_repo_id(
            PublicationIdentity(
                game_family="NES-SuperMarioBros",
                goal="A" * 70,
                policy_variant="gray84-hudmask-stack4-basic",
                algorithm="ppo",
            )
        )


def test_preprocessing_contract_reads_provider_rgb_and_stack_arguments() -> None:
    contract = preprocessing_contract(
        {
            "env_provider": "supermariobrosnes-turbo",
            "obs_resize": [96, 96],
            "env_args": {"obs_grayscale": False, "frame_stack": 2},
        }
    )
    assert contract["obs_grayscale"] is False
    assert contract["obs_resize"] == [96, 96]
    assert contract["frame_stack"] == 2


def test_publication_evaluation_requires_stochastic_consistent_by_start() -> None:
    deterministic = evaluation_payload()
    deterministic["action_sampling"] = "deterministic"
    with pytest.raises(ValueError, match="action_sampling"):
        normalize_publication_evaluation(deterministic)

    inconsistent = evaluation_payload()
    inconsistent["eval/full/outcome/success/across_starts/rate/min"] = 0.7
    with pytest.raises(ValueError, match="success_rate_min"):
        normalize_publication_evaluation(inconsistent)


def test_publication_source_requires_explicit_seed_commit_and_matching_step() -> None:
    evaluation = normalize_publication_evaluation(evaluation_payload())
    metadata = model_metadata()
    metadata["repo_git_commit"] = ""
    with pytest.raises(ValueError, match="repo_git_commit"):
        publication_source_from_policy_bundle(policy_bundle_from_metadata(metadata), evaluation)

    metadata = model_metadata()
    metadata["checkpoint_step"] = 1
    with pytest.raises(ValueError, match="checkpoint_step disagrees"):
        publication_source_from_policy_bundle(policy_bundle_from_metadata(metadata), evaluation)


def test_model_card_template_uses_canonical_cli_commands() -> None:
    card = _render_current_model_card_fixture()

    assert "uv run gradlab rom import ~/roms" in card
    assert "uv run gradlab eval https://huggingface.co/" in card
    assert "gradlab import-roms" not in card
    assert "gradlab eval run" not in card


def test_model_card_template_preserves_current_sb3_golden_output() -> None:
    card = _render_current_model_card_fixture()

    ModelCard(card).validate(repo_type="model")
    assert hashlib.sha256(card.encode()).hexdigest() == (
        "c48895c4608bf5b34da0c33025bbd5ed8b9de8489d8c24cabfdcc5c3918f9d08"
    )


def test_model_card_template_preserves_current_action_program_golden_output() -> None:
    card = _render_current_model_card_fixture(
        algorithm="action-program",
        model_class="gradlab.action_program.ActionProgramPolicy",
        search_algorithm="jerk",
        youtube_url=None,
    )

    ModelCard(card).validate(repo_type="model")
    assert "| Producer | `jerk` |" in card
    assert "_action-program/resolve/v1/model.zip" in card
    assert hashlib.sha256(card.encode()).hexdigest() == (
        "b9325aa74ab72711dc920b28d0a754fcd9742cc8cbd3242db4bbc2eac1b15862"
    )


def test_model_card_template_rejects_missing_context() -> None:
    with pytest.raises(UndefinedError):
        _render_model_card_template({})


def test_release_bundle_has_exact_files_hashes_and_portable_identity() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        metadata = model_metadata()
        evaluation = normalize_publication_evaluation(evaluation_payload())
        contents = {
            ".gitattributes": GITATTRIBUTES_TEXT,
            "LICENSE": MIT_LICENSE_TEXT,
            "model.zip": "checkpoint",
            "replay.mp4": "video",
        }
        for filename, content in contents.items():
            (root / filename).write_text(content, encoding="utf-8")
        resolved = compose_resolved_train_documents(
            Path("experiments/goals/SuperMarioBros-Nes-v0/Level1-1/_goal.yaml"),
            Path("experiments/goals/SuperMarioBros-Nes-v0/Level1-1/recipes/ppo.yaml"),
            source_sha="a" * 40,
        )
        recipe_document = build_recipe_document(
            resolved.effective,
            repo_root=Path.cwd(),
            source_commit="a" * 40,
            run_description="release fixture",
            seed=7,
            runtime_image_ref="docker:example.invalid/gradlab@sha256:" + "b" * 64,
            base_materialized_recipe=resolved.base,
            canonical_goal=resolved.canonical_goal,
        )
        write_canonical_json(root / "recipe.json", recipe_document)
        metadata["training_backend_id"] = recipe_document["recipe"]["train_config"][
            "training_backend"
        ]["id"]
        metadata["training_backend_config_hash"] = training_backend_config_hash(
            recipe_document["recipe"]["train_config"]
        )
        metadata["training_metadata"] = {
            "environment_hash": recipe_document["recipe"]["environment_hash"],
            "environment": recipe_document["recipe"]["environment"],
            "action_contract": metadata["training_metadata"]["action_contract"],
            "preprocessing": recipe_document["recipe"]["environment"]["preprocessing"],
        }
        write_canonical_json(
            root / "model.json",
            build_model_document(root / "model.zip", root / "recipe.json", metadata),
        )
        bundle = load_policy_bundle(root, source=str(root))
        identity = publication_identity_from_policy_bundle("Level1-1", bundle)
        source = publication_source_from_policy_bundle(bundle, evaluation)
        evaluation_value = evaluation.as_manifest_value()
        evaluation_value.update(
            checkpoint_sha256=sha256_file(root / "model.zip"),
            recipe_sha256=sha256_file(root / "recipe.json"),
            recipe_format_version=recipe_document["format_version"],
            evaluation_contract_sha256=evaluation_contract_sha256(recipe_document),
            exact_contract=True,
        )
        provisional = build_release_manifest(
            identity,
            bundle,
            release_version="v1",
            published_at="2026-07-14T12:00:00Z",
            source=source,
            evaluation=evaluation_value,
            artifacts={},
            youtube_url="https://www.youtube.com/watch?v=example",
        )
        (root / "README.md").write_text(
            render_model_card(provisional, bundle),
            encoding="utf-8",
        )
        records = release_artifact_records(root)
        manifest = build_release_manifest(
            identity,
            bundle,
            release_version="v1",
            published_at="2026-07-14T12:00:00Z",
            source=source,
            evaluation=evaluation_value,
            artifacts=records,
            youtube_url="https://www.youtube.com/watch?v=example",
        )
        (root / "release_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        assert {path.name for path in root.iterdir()} == HUGGINGFACE_RELEASE_FILES
        assert validate_release_bundle(root) == manifest

        future = deepcopy(manifest)
        future["format_version"] = 999
        write_canonical_json(root / "release_manifest.json", future)
        with patch(
            "gradlab.publication.load_policy_bundle",
            side_effect=AssertionError("bundle access"),
        ):
            with pytest.raises(UnsupportedPolicyDocumentVersion, match="999"):
                validate_release_bundle(root)

        malformed = deepcopy(manifest)
        malformed["evaluation"]["unexpected_contract_field"] = True
        write_canonical_json(root / "release_manifest.json", malformed)
        with pytest.raises(PolicyDocumentError, match="unknown field"):
            validate_release_bundle(root)

        broken = deepcopy(manifest)
        broken["source"]["checkpoint_artifact"] = "/Users/example/model.zip"
        (root / "release_manifest.json").write_text(json.dumps(broken), encoding="utf-8")
        with pytest.raises(ValueError, match="absolute local path"):
            validate_release_bundle(root)


def test_release_bundle_rejects_non_file_entries() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for filename in HUGGINGFACE_RELEASE_FILES - {"replay.mp4"}:
            (root / filename).write_text("placeholder", encoding="utf-8")
        (root / "replay.mp4").mkdir()

        with pytest.raises(ValueError, match="regular files"):
            validate_release_bundle(root)
