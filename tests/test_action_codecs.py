from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from gradlab.action_codecs import (
    VIZDOOM_DEATHMATCH_MULTIDISCRETE_BUTTONS,
    VIZDOOM_DEATHMATCH_MULTIDISCRETE_CODEC,
    VIZDOOM_DEATHMATCH_MULTIDISCRETE_NVEC,
    VizdoomDeathmatchMultiDiscreteActionCodec,
)
from gradlab.action_contract import compile_runtime_action_contract
from gradlab.batch_runtime import ProviderDescriptor


def _descriptor(
    *,
    provider_id: str = "vizdoom-turbo",
    buttons: tuple[str, ...] = VIZDOOM_DEATHMATCH_MULTIDISCRETE_BUTTONS,
) -> ProviderDescriptor:
    low = np.zeros(len(buttons), dtype=np.float32)
    high = np.ones(len(buttons), dtype=np.float32)
    if "TURN_LEFT_RIGHT_DELTA" in buttons:
        turn_index = buttons.index("TURN_LEFT_RIGHT_DELTA")
        low[turn_index] = -180.0
        high[turn_index] = 180.0
    return ProviderDescriptor(
        provider_id=provider_id,
        native_observation_space=gym.spaces.Box(
            0,
            255,
            shape=(4, 84, 84),
            dtype=np.uint8,
        ),
        native_action_space=gym.spaces.Box(low, high, dtype=np.float32),
        action_mode="filtered",
        action_buttons=buttons,
    )


def test_deathmatch_multidiscrete_codec_maps_each_factor_into_native_box() -> None:
    buttons = tuple(reversed(VIZDOOM_DEATHMATCH_MULTIDISCRETE_BUTTONS))
    descriptor = _descriptor(buttons=buttons)
    codec = VizdoomDeathmatchMultiDiscreteActionCodec(descriptor, 2)

    assert tuple(codec.action_space.nvec) == VIZDOOM_DEATHMATCH_MULTIDISCRETE_NVEC
    native = codec.map_actions(
        np.asarray(
            [
                [0, 0, 0, 0, 0, 10],
                [1, 2, 7, 1, 1, 0],
            ],
            dtype=np.int64,
        )
    )

    np.testing.assert_array_equal(native[0], np.zeros(14, dtype=np.float32))
    selected = {button: native[1, buttons.index(button)] for button in buttons}
    assert selected["MOVE_FORWARD"] == 1.0
    assert selected["MOVE_LEFT"] == 1.0
    assert selected["SELECT_WEAPON7"] == 1.0
    assert selected["ATTACK"] == 1.0
    assert selected["SPEED"] == 1.0
    assert selected["TURN_LEFT_RIGHT_DELTA"] == -12.5
    assert selected["MOVE_BACKWARD"] == 0.0
    assert selected["MOVE_RIGHT"] == 0.0
    assert sum(selected[f"SELECT_WEAPON{slot}"] for slot in range(1, 8)) == 1.0


def test_deathmatch_multidiscrete_codec_rejects_invalid_actions_and_native_contracts() -> None:
    codec = VizdoomDeathmatchMultiDiscreteActionCodec(_descriptor(), 1)

    with pytest.raises(ValueError, match="shape"):
        codec.map_actions(np.zeros((1, 5), dtype=np.int64))
    with pytest.raises(ValueError, match="outside"):
        codec.map_actions(np.asarray([[0, 0, 8, 0, 0, 10]], dtype=np.int64))
    with pytest.raises(ValueError, match="requires provider"):
        VizdoomDeathmatchMultiDiscreteActionCodec(
            _descriptor(provider_id="stable-retro-turbo"),
            1,
        )
    with pytest.raises(ValueError, match="native buttons differ"):
        VizdoomDeathmatchMultiDiscreteActionCodec(
            _descriptor(buttons=VIZDOOM_DEATHMATCH_MULTIDISCRETE_BUTTONS[:-1] + ("TURN_LEFT",)),
            1,
        )


def test_deathmatch_multidiscrete_runtime_contract_is_exact_and_componentized() -> None:
    descriptor = _descriptor()
    codec = VizdoomDeathmatchMultiDiscreteActionCodec(descriptor, 1)
    config = SimpleNamespace(
        env_provider="vizdoom-turbo",
        game="VizdoomDeathmatch-v1",
        env_args={"use_restricted_actions": "filtered"},
        task={
            "action": {
                "set": "sample-factory-v0",
                "codec": {"type": VIZDOOM_DEATHMATCH_MULTIDISCRETE_CODEC},
            }
        },
    )

    contract = compile_runtime_action_contract(
        config,
        descriptor,
        codec.action_space,
        policy_action_codec=config.task["action"]["codec"],
    )

    assert contract["policy"]["space"]["nvec"] == list(VIZDOOM_DEATHMATCH_MULTIDISCRETE_NVEC)
    assert contract["policy"]["codec"] == {
        "type": VIZDOOM_DEATHMATCH_MULTIDISCRETE_CODEC,
        "native_buttons": list(VIZDOOM_DEATHMATCH_MULTIDISCRETE_BUTTONS),
        "axes": [
            "forward_movement",
            "strafe",
            "weapon",
            "attack",
            "speed",
            "turn_delta",
        ],
        "turn_degrees": [float(value) for value in np.linspace(-12.5, 12.5, 21)],
    }
    components = contract["policy"]["semantics"]["components"]
    assert [component["semantic_id"] for component in components] == [
        "forward_movement",
        "strafe",
        "weapon",
        "attack",
        "speed",
        "turn_delta",
    ]
    assert components[2]["values"][7]["semantic_id"] == "select_weapon7"
    assert components[5]["values"][10]["semantic_id"] == "noop"
