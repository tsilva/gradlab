import builtins
from types import MappingProxyType, SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from gradlab.action_contract import (
    action_contract_meanings,
    action_contract_payload,
    action_index_for_controls,
    assert_action_contract_compatible,
    compile_runtime_action_contract,
    configured_action_values,
    configured_action_meanings,
    configured_action_name,
    declared_action_contract,
    provider_buttons,
    runtime_action_contract,
)
from gradlab.batch_runtime import ProviderDescriptor
from gradlab.env import EnvConfig, resolve_env_config
from gradlab.env_identity import validate_task_config


BREAKOUT_NO_NOOP_ACTIONS = [["BUTTON"], ["RIGHT"], ["LEFT"]]
BREAKOUT_NO_NOOP_HASH = "a1f69721fbf7ef8a00084b9426767b0bce61f39ee0880b932a954c7d5789ee15"


def test_gradoom_button_metadata_does_not_import_the_cuda_runtime(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "gradoom" or name.startswith("gradoom."):
            raise AssertionError("GraDOOM metadata resolution imported the CUDA runtime")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    assert provider_buttons(
        "gradoom",
        "VizdoomDeathmatch-v1",
        env_args={"scenario": "scenario"},
    ) == (
        "ATTACK",
        "SPEED",
        "STRAFE",
        "MOVE_RIGHT",
        "MOVE_LEFT",
        "MOVE_BACKWARD",
        "MOVE_FORWARD",
        "TURN_RIGHT",
        "TURN_LEFT",
        "SELECT_WEAPON1",
        "SELECT_WEAPON2",
        "SELECT_WEAPON3",
        "SELECT_WEAPON4",
        "SELECT_WEAPON5",
        "SELECT_WEAPON6",
        "SELECT_NEXT_WEAPON",
        "SELECT_PREV_WEAPON",
        "LOOK_UP_DOWN_DELTA",
        "TURN_LEFT_RIGHT_DELTA",
        "MOVE_LEFT_RIGHT_DELTA",
    )


def test_runtime_action_contract_traverses_common_environment_wrappers():
    contract = {"schema_version": 1}
    source = SimpleNamespace(
        venv=SimpleNamespace(
            env=SimpleNamespace(
                runtime=SimpleNamespace(action_contract=contract),
            )
        )
    )

    assert runtime_action_contract(source) is contract


def test_live_config_rejects_unknown_provider_and_task_action_fields():
    with pytest.raises(ValueError, match="constructor argument"):
        resolve_env_config(
            EnvConfig(
                env_provider="supermariobrosnes-turbo",
                game="SuperMarioBros-Nes-v0",
                env_args={"action_set": "basic"},
            )
        )

    with pytest.raises(ValueError, match=r"task\.action\.set must be 'native'"):
        validate_task_config(
            {
                "id": "mario",
                "action": {"set": "basic"},
                "signals": {},
                "events": {},
                "termination": {},
                "reward": {"reward_mode": "native"},
            }
        )


@pytest.mark.parametrize(
    ("provider", "game", "action_set", "expected_hash"),
    [
        (
            "stable-retro-turbo",
            "SuperMarioBros-Nes-v0",
            "basic",
            "2eaa8ce13795d654097e6fbeb16460de8ae78f0af39b7f88259bc51604504134",
        ),
        (
            "supermariobrosnes-turbo",
            "SuperMarioBros-Nes-v0",
            "basic",
            "2eaa8ce13795d654097e6fbeb16460de8ae78f0af39b7f88259bc51604504134",
        ),
        (
            "breakout-turbo-env",
            "Breakout-Atari2600-v0",
            "simple",
            "ae2fea9e05910b0db9ba3980c162573a8ad9ad562e077babfeb5f6144d94a091",
        ),
        (
            "stable-retro-turbo",
            "Breakout-Atari2600-v0",
            "simple",
            "ae2fea9e05910b0db9ba3980c162573a8ad9ad562e077babfeb5f6144d94a091",
        ),
    ],
)
def test_provider_metadata_resolves_shared_semantic_hash(provider, game, action_set, expected_hash):
    config = SimpleNamespace(
        env_provider=provider,
        game=game,
        env_args={"use_restricted_actions": action_set},
        task={"action": {"set": "native"}},
    )
    contract = declared_action_contract(config)

    assert contract["preset"] == action_set
    assert contract["table_hash"] == expected_hash
    assert configured_action_name(config) == action_set
    assert configured_action_meanings(config) == tuple(contract["meanings"])


def test_vizdoom_discrete_request_preflights_to_the_scenario_minimal_table():
    config = SimpleNamespace(
        env_provider="vizdoom-turbo",
        game="VizdoomBasic-v1",
        env_args={"use_restricted_actions": "discrete"},
        task={"action": {"set": "native"}},
    )

    assert declared_action_contract(config) == {
        "mode": "custom_discrete",
        "preset": "minimal",
        "table": [[], ["MOVE_LEFT"], ["MOVE_RIGHT"], ["ATTACK"]],
        "meanings": ["noop", "move_left", "move_right", "attack"],
        "table_hash": "427f093c24e4c5051f08479f7f9244380a028b013f802c73413720dbd691c690",
    }


def test_vizdoom_preflight_uses_configured_available_buttons():
    config = SimpleNamespace(
        env_provider="vizdoom-turbo",
        game="VizdoomBasic-v1",
        env_args={
            "use_restricted_actions": "discrete",
            "vizdoom_config": {
                "available_buttons": ["TURN_LEFT", "TURN_RIGHT", "ATTACK"],
            },
        },
        task={"action": {"set": "native"}},
    )

    contract = declared_action_contract(config)

    assert contract["table"] == [
        [],
        ["TURN_LEFT"],
        ["TURN_RIGHT"],
        ["ATTACK"],
    ]
    assert contract["meanings"] == ["noop", "turn_left", "turn_right", "attack"]


@pytest.mark.parametrize(
    ("game", "meanings"),
    [
        ("VizdoomBasic-v1", ("noop", "move_left", "move_right", "attack")),
        ("VizdoomBasic-Plus-v1", ("noop", "move_left", "move_right", "attack")),
        (
            "VizdoomDeadlyCorridor-v1",
            (
                "noop",
                "move_left",
                "move_right",
                "attack",
                "move_forward",
                "move_backward",
                "turn_left",
                "turn_right",
            ),
        ),
        ("VizdoomDefendCenter-v1", ("noop", "turn_left", "turn_right", "attack")),
        ("VizdoomDefendLine-v1", ("noop", "turn_left", "turn_right", "attack")),
        ("VizdoomDefendLine-Plus-v1", ("noop", "turn_left", "turn_right", "attack")),
        (
            "VizdoomHealthGathering-v1",
            ("noop", "turn_left", "turn_right", "move_forward"),
        ),
        (
            "VizdoomHealthGatheringSupreme-v1",
            ("noop", "turn_left", "turn_right", "move_forward"),
        ),
        (
            "VizdoomMyWayHome-v1",
            (
                "noop",
                "turn_left",
                "turn_right",
                "move_forward",
                "move_left",
                "move_right",
            ),
        ),
        ("VizdoomPredictPosition-v1", ("noop", "turn_left", "turn_right", "attack")),
        ("VizdoomTakeCover-v1", ("noop", "move_left", "move_right")),
    ],
)
def test_every_bundled_vizdoom_scenario_preflights_exact_discrete_meanings(
    game,
    meanings,
):
    config = SimpleNamespace(
        env_provider="vizdoom-turbo",
        game=game,
        env_args={"use_restricted_actions": "discrete"},
        task={"action": {"set": "native"}},
    )

    assert configured_action_meanings(config) == meanings


def test_stable_retro_mario_preset_compiles_to_native_button_masks():
    config = SimpleNamespace(
        env_provider="stable-retro-turbo",
        game="SuperMarioBros-Nes-v0",
        env_args={"players": 1, "use_restricted_actions": "basic"},
        task={"action": {"set": "native"}},
    )

    values = configured_action_values(config)

    assert values is not None
    assert len(values) == 7
    assert values[0] == (0, 0, 0, 0, 0, 0, 0, 0, 0)
    assert values[1] == (0, 0, 0, 0, 0, 0, 0, 1, 0)
    assert values[2] == (1, 0, 0, 0, 0, 0, 0, 1, 0)


@pytest.mark.parametrize("provider", ["breakout-turbo-env", "stable-retro-turbo"])
def test_breakout_inline_table_without_noop_preserves_order_and_semantic_hash(provider):
    config = SimpleNamespace(
        env_provider=provider,
        game="Breakout-Atari2600-v0",
        env_args={
            "players": 1,
            "use_restricted_actions": BREAKOUT_NO_NOOP_ACTIONS,
        },
        task={"action": {"set": "native"}},
    )

    contract = declared_action_contract(config)
    values = configured_action_values(config)

    assert contract == {
        "mode": "custom_discrete",
        "preset": None,
        "table": BREAKOUT_NO_NOOP_ACTIONS,
        "meanings": ["button", "right", "left"],
        "table_hash": BREAKOUT_NO_NOOP_HASH,
    }
    assert values == (
        (1, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 1),
        (0, 0, 0, 0, 0, 0, 1, 0),
    )
    assert configured_action_meanings(config) == ("button", "right", "left")


@pytest.mark.parametrize(
    ("action_set", "expected_meanings"),
    [
        (
            "basic",
            ("noop", "right", "right_b", "right_a", "right_a_b", "a", "left"),
        ),
        (
            "standard",
            (
                "noop",
                "right",
                "right_b",
                "right_a",
                "right_a_b",
                "a",
                "left",
                "down",
            ),
        ),
        (
            "basic-start",
            (
                "noop",
                "right",
                "right_b",
                "right_a",
                "right_a_b",
                "a",
                "left",
                "start",
            ),
        ),
        ("right-jump", ("right", "right_b", "right_a", "right_a_b")),
    ],
)
def test_mario_action_set_catalogs_stay_aligned(action_set, expected_meanings):
    contracts = []
    for provider in ("stable-retro-turbo", "supermariobrosnes-turbo"):
        config = SimpleNamespace(
            env_provider=provider,
            game="SuperMarioBros-Nes-v0",
            env_args={"use_restricted_actions": action_set},
            task={"action": {"set": "native"}},
        )
        contracts.append(declared_action_contract(config))

    assert contracts[0]["meanings"] == list(expected_meanings)
    assert contracts[0]["table_hash"] == contracts[1]["table_hash"]
    assert configured_action_meanings(config) == expected_meanings


def test_multiplayer_inline_table_is_joint_not_cartesian_and_order_stable():
    base = SimpleNamespace(
        env_provider="stable-retro-turbo",
        game="SuperMarioBros-Nes-v0",
        env_args={
            "players": 2,
            "use_restricted_actions": [
                [[], []],
                [["RIGHT", "A"], ["LEFT"]],
            ],
        },
        task={"action": {"set": "native"}},
    )
    reordered = SimpleNamespace(
        **{
            **vars(base),
            "env_args": {
                "players": 2,
                "use_restricted_actions": [
                    [[], []],
                    [["A", "RIGHT"], ["LEFT"]],
                ],
            },
        }
    )

    contract = declared_action_contract(base)
    reordered_contract = declared_action_contract(reordered)

    assert contract["table"] == [[[], []], [["RIGHT", "A"], ["LEFT"]]]
    assert contract["meanings"] == ["p1_noop__p2_noop", "p1_right_a__p2_left"]
    assert contract["table_hash"] == reordered_contract["table_hash"]


def _descriptor(
    *,
    provider_id="vizdoom-turbo",
    action_space=None,
    mode="custom_discrete",
    table=None,
    meanings=None,
    table_hash=None,
    buttons=(),
    combos=(),
):
    return ProviderDescriptor(
        provider_id=provider_id,
        native_observation_space=gym.spaces.Box(
            0,
            255,
            shape=(4, 84, 84),
            dtype=np.uint8,
        ),
        native_action_space=action_space or gym.spaces.Discrete(4),
        action_mode=mode,
        action_table=table,
        action_meanings=meanings,
        action_table_hash=table_hash,
        action_buttons=buttons,
        action_combos=combos,
    )


def test_runtime_vizdoom_contract_uses_provider_meanings_and_structured_controls():
    descriptor = _descriptor(
        table=((), ("MOVE_LEFT",), ("MOVE_RIGHT",), ("ATTACK",)),
        meanings=("noop", "move_left", "move_right", "attack"),
        table_hash="4" * 64,
        buttons=("MOVE_LEFT", "MOVE_RIGHT", "ATTACK"),
    )
    config = SimpleNamespace(
        env_provider="vizdoom-turbo",
        game="VizdoomBasic-v1",
        env_args={"use_restricted_actions": "discrete"},
        task={"action": {"set": "native"}},
    )

    contract = compile_runtime_action_contract(
        config,
        descriptor,
        descriptor.native_action_space,
    )

    assert action_contract_meanings(contract) == (
        "noop",
        "move_left",
        "move_right",
        "attack",
    )
    assert contract["provider"]["preset"] is None
    assert contract["policy"]["semantics"]["entries"][1] == {
        "value": 1,
        "semantic_id": "move_left",
        "label": "move left",
        "controls": [
            {
                "player": 1,
                "atoms": ["move_left"],
                "inputs": ["left"],
            }
        ],
    }
    assert action_index_for_controls(contract, ["LEFT"]) == 1
    assert action_index_for_controls(contract, ["a"]) == 3
    payload = action_contract_payload(MappingProxyType(contract))
    assert payload is not contract
    assert payload["policy"]["semantics"]["entries"][1] == {
        "value": 1,
        "semantic_id": "move_left",
        "label": "move left",
        "controls": [
            {
                "player": 1,
                "atoms": ["move_left"],
                "inputs": ["left"],
            }
        ],
    }


def test_runtime_discrete_cartesian_contract_is_compact_and_exact():
    descriptor = _descriptor(
        provider_id="stable-retro-turbo",
        action_space=gym.spaces.Discrete(6),
        mode="discrete",
        buttons=("A", "LEFT", "RIGHT"),
        combos=((0, 1), (0, 2, 4)),
    )
    config = SimpleNamespace(
        env_provider="stable-retro-turbo",
        game="Fixture-Nes-v0",
        env_args={"use_restricted_actions": "discrete"},
        task={"action": {"set": "native"}},
    )

    contract = compile_runtime_action_contract(
        config,
        descriptor,
        descriptor.native_action_space,
    )

    assert contract["policy"]["semantics"]["encoding"] == "mixed_radix"
    assert "entries" not in contract["policy"]["semantics"]
    assert action_contract_meanings(contract) == (
        "noop",
        "a",
        "left",
        "a_left",
        "right",
        "a_right",
    )
    payload = action_contract_payload(contract)
    assert payload["policy"]["semantics"]["axes"][1]["values"][2] == {
        "value": 4,
        "semantic_id": "right",
        "label": "right",
        "atoms": ["right"],
        "inputs": ["right"],
    }


def test_runtime_component_contract_payload_preserves_structured_labels():
    descriptor = _descriptor(
        provider_id="stable-retro-turbo",
        action_space=gym.spaces.MultiBinary(3),
        mode="all",
        buttons=("A", "LEFT", "RIGHT"),
    )
    config = SimpleNamespace(
        env_provider="stable-retro-turbo",
        game="Fixture-Nes-v0",
        env_args={"use_restricted_actions": "all"},
        task={"action": {"set": "native"}},
    )
    contract = compile_runtime_action_contract(
        config,
        descriptor,
        descriptor.native_action_space,
    )

    payload = action_contract_payload(contract)

    assert payload["policy"]["semantics"]["components"] == [
        {"index": 0, "semantic_id": "a", "label": "a", "input": "a"},
        {"index": 1, "semantic_id": "left", "label": "left", "input": "left"},
        {"index": 2, "semantic_id": "right", "label": "right", "input": "right"},
    ]


def test_runtime_action_contract_compatibility_checks_execution_and_semantics():
    descriptor = _descriptor(
        table=((), ("MOVE_LEFT",), ("MOVE_RIGHT",), ("ATTACK",)),
        meanings=("noop", "move_left", "move_right", "attack"),
        table_hash="4" * 64,
    )
    config = SimpleNamespace(
        env_provider="vizdoom-turbo",
        game="VizdoomBasic-v1",
        env_args={"use_restricted_actions": "discrete"},
        task={"action": {"set": "native"}},
    )
    contract = compile_runtime_action_contract(
        config,
        descriptor,
        descriptor.native_action_space,
    )

    assert assert_action_contract_compatible(contract, contract)["status"] == "compatible"
    with pytest.raises(ValueError, match="no saved runtime action contract"):
        assert_action_contract_compatible(None, contract)

    changed = compile_runtime_action_contract(
        config,
        _descriptor(
            table=((), ("MOVE_RIGHT",), ("MOVE_LEFT",), ("ATTACK",)),
            meanings=("noop", "move_right", "move_left", "attack"),
            table_hash="5" * 64,
        ),
        descriptor.native_action_space,
    )
    with pytest.raises(ValueError, match="execution_hash, semantic_hash"):
        assert_action_contract_compatible(contract, changed)


def test_explicit_task_codec_derives_policy_semantics_from_native_components():
    descriptor = _descriptor(
        provider_id="stable-retro-turbo",
        action_space=gym.spaces.MultiBinary(3),
        mode="all",
        buttons=("A", "LEFT", "RIGHT"),
    )
    config = SimpleNamespace(
        env_provider="stable-retro-turbo",
        game="Fixture-Nes-v0",
        env_args={"use_restricted_actions": "all"},
        task={
            "action": {
                "set": "basic",
                "codec": {
                    "type": "discrete_lookup",
                    "values": [[0, 0, 0], [1, 0, 1]],
                },
            }
        },
    )

    contract = compile_runtime_action_contract(
        config,
        descriptor,
        gym.spaces.Discrete(2),
        policy_action_values=((0, 0, 0), (1, 0, 1)),
    )

    assert action_contract_meanings(contract) == ("noop", "a__right")
    assert contract["policy"]["codec"]["type"] == "discrete_lookup"
    assert contract["policy"]["semantics"]["entries"][1]["native_value"] == [1, 0, 1]
