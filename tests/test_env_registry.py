from __future__ import annotations

import unittest

import pytest

from gradlab.env_identity import environment_identity_from_train_config
from gradlab.env_registry import (
    environment_spec,
    env_supports_states,
    registered_env_ids,
    resolve_env_id,
    resolve_env_provider,
)


def test_environment_spec_owns_identity_defaults_task_and_eval_semantics() -> None:
    stable = environment_spec("stable-retro-turbo", "SuperMarioBros-Nes-v0")
    dedicated = environment_spec("supermariobrosnes-turbo", "SuperMarioBros-Nes-v0")

    assert stable is dedicated
    assert stable.game_family == "NES-SuperMarioBros"
    assert stable.wandb_project == "SuperMarioBros-Nes-v0"
    assert stable.task_id == "mario"
    assert stable.default_state == "Level1-1"
    assert stable.default_obs_crop == (32, 0, 0, 0)
    assert stable.eval_semantics.completion_reason == "level_change"


def test_resolves_registered_stable_retro_turbo_env_id() -> None:
    env_id = "stable-retro-turbo:SuperMarioBros-Nes-v0"

    resolved = resolve_env_id(env_id)

    assert env_id in registered_env_ids()
    assert resolved.qualified_id == env_id
    assert resolved.provider_id == "stable-retro-turbo"
    assert resolved.provider_env_id == "SuperMarioBros-Nes-v0"
    assert resolved.import_name == "stable_retro"


def test_resolves_registered_stable_retro_turbo_smb3_env_id() -> None:
    env_id = "stable-retro-turbo:SuperMarioBros3-Nes-v0"

    resolved = resolve_env_id(env_id)

    assert env_id in registered_env_ids()
    assert resolved.qualified_id == env_id
    assert resolved.provider_id == "stable-retro-turbo"
    assert resolved.provider_env_id == "SuperMarioBros3-Nes-v0"
    assert resolved.import_name == "stable_retro"


def test_resolves_registered_stable_retro_turbo_atari_env_ids() -> None:
    for game in ("Breakout-Atari2600-v0", "MsPacman-Atari2600-v0"):
        env_id = f"stable-retro-turbo:{game}"

        resolved = resolve_env_id(env_id)

        assert env_id in registered_env_ids()
        assert resolved.provider_id == "stable-retro-turbo"
        assert resolved.provider_env_id == game
        assert resolved.import_name == "stable_retro"
        assert env_supports_states("stable-retro-turbo", game)


def test_resolves_registered_breakout_turbo_env_id() -> None:
    env_id = "breakout-turbo-env:Breakout-Atari2600-v0"

    resolved = resolve_env_id(env_id)

    assert env_id in registered_env_ids()
    assert resolved.qualified_id == env_id
    assert resolved.provider_id == "breakout-turbo-env"
    assert resolved.provider_env_id == "Breakout-Atari2600-v0"
    assert resolved.import_name == "breakout_turbo_env"
    assert env_supports_states("breakout-turbo-env", "Breakout-Atari2600-v0")

    with pytest.raises(ValueError, match="does not register environment"):
        resolve_env_id("breakout-turbo-env:BreakoutTurbo-v0")


def test_resolves_registered_supermariobrosnes_turbo_env_id() -> None:
    env_id = "supermariobrosnes-turbo:SuperMarioBros-Nes-v0"

    resolved = resolve_env_id(env_id)

    assert env_id in registered_env_ids()
    assert resolved.qualified_id == env_id
    assert resolved.provider_id == "supermariobrosnes-turbo"
    assert resolved.provider_env_id == "SuperMarioBros-Nes-v0"
    assert resolved.import_name == "supermariobrosnes_turbo"
    provider = resolve_env_provider(resolved.provider_id)
    assert provider.external_rom_asset_strategy == "stable_retro_direct_path_v1"
    assert provider.requires_external_rom_asset
    assert provider.constructor_contract is not None
    assert "state_dir" not in provider.constructor_contract.explicit_env_args


def test_resolves_vizdoom_turbo_builtin_and_custom_scenarios() -> None:
    for game in ("VizdoomBasic-v1", "VizdoomDeathmatch-v1"):
        env_id = f"vizdoom-turbo:{game}"
        resolved = resolve_env_id(env_id)

        assert env_id in registered_env_ids()
        assert resolved.provider_id == "vizdoom-turbo"
        assert resolved.provider_env_id == game
        assert resolved.import_name == "vizdoom_turbo"
        assert env_supports_states("vizdoom-turbo", game)
    provider = resolve_env_provider(resolved.provider_id)
    assert provider.constructor_contract is not None
    assert "state_dir" not in provider.constructor_contract.explicit_env_args

    deathmatch = environment_spec("vizdoom-turbo", "VizdoomDeathmatch-v1")
    assert deathmatch.game_family == "Doom-ViZDoom-Deathmatch"
    assert deathmatch.wandb_project == "VizdoomDeathmatch-v1"

    custom = resolve_env_id("vizdoom-turbo:/tmp/custom-scenario.cfg")
    assert custom.provider_env_id == "/tmp/custom-scenario.cfg"


def test_resolves_vizdoom_turbo_augmented_environment() -> None:
    expected = {
        "VizdoomBasic-Plus-v1": "Doom-ViZDoom-Basic-Plus",
        "VizdoomDefendLine-Plus-v1": "Doom-ViZDoom-DefendLine-Plus",
    }
    for game, game_family in expected.items():
        env_id = f"vizdoom-turbo:{game}"
        resolved = resolve_env_id(env_id)

        assert env_id in registered_env_ids()
        assert resolved.provider_id == "vizdoom-turbo"
        assert resolved.provider_env_id == game
        spec = environment_spec(resolved.provider_id, resolved.provider_env_id)
        assert spec.game_family == game_family
        assert spec.wandb_project == game
    contract = resolve_env_provider(resolved.provider_id).constructor_contract
    assert contract is not None
    assert "enemy_variants" in contract.optional_env_args
    assert "surface_variants" in contract.optional_env_args


def test_resolves_registered_ale_py_env_id() -> None:
    env_id = "ale-py:breakout"

    resolved = resolve_env_id(env_id)

    assert env_id in registered_env_ids()
    assert resolved.qualified_id == env_id
    assert resolved.provider_id == "ale-py"
    assert resolved.provider_env_id == "breakout"
    assert resolved.import_name == "ale_py"


def test_resolves_registered_ale_py_ms_pacman_env_id() -> None:
    env_id = "ale-py:ms_pacman"

    resolved = resolve_env_id(env_id)

    assert env_id in registered_env_ids()
    assert resolved.qualified_id == env_id
    assert resolved.provider_id == "ale-py"
    assert resolved.provider_env_id == "ms_pacman"
    assert resolved.import_name == "ale_py"


def test_rejects_unregistered_env_id() -> None:
    with unittest.TestCase().assertRaisesRegex(ValueError, "does not register environment"):
        resolve_env_id("stable-retro-turbo:UnknownGame-v0")


def test_dynamic_native_provider_ids_are_explicit_but_not_hardcoded() -> None:
    gym_id = resolve_env_id("gymnasium:CustomNativeVector-v0")

    assert gym_id.provider_id == "gymnasium"
    assert gym_id.provider_env_id == "CustomNativeVector-v0"


def test_environment_identity_normalizes_bare_stable_retro_game() -> None:
    identity = environment_identity_from_train_config({"game": "SuperMarioBros-Nes-v0"})

    assert identity["env_id"] == "stable-retro-turbo:SuperMarioBros-Nes-v0"


def test_rejects_unknown_provider_alias() -> None:
    with unittest.TestCase().assertRaisesRegex(ValueError, "unknown environment provider"):
        environment_identity_from_train_config(
            {
                "env_provider": "stable-retro",
                "game": "SuperMarioBros-Nes-v0",
            }
        )
