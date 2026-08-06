from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np


VIZDOOM_DEATHMATCH_MULTIDISCRETE_CODEC = "vizdoom_deathmatch_multidiscrete_v1"
VIZDOOM_DEATHMATCH_MULTIDISCRETE_NVEC = (3, 3, 8, 2, 2, 21)
VIZDOOM_DEATHMATCH_MULTIDISCRETE_BUTTONS = (
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "MOVE_RIGHT",
    "MOVE_LEFT",
    "SELECT_WEAPON1",
    "SELECT_WEAPON2",
    "SELECT_WEAPON3",
    "SELECT_WEAPON4",
    "SELECT_WEAPON5",
    "SELECT_WEAPON6",
    "SELECT_WEAPON7",
    "ATTACK",
    "SPEED",
    "TURN_LEFT_RIGHT_DELTA",
)
VIZDOOM_DEATHMATCH_TURN_DEGREES = tuple(
    float(value) for value in np.linspace(-12.5, 12.5, 21, dtype=np.float32)
)


def validate_task_action_codec(codec: Mapping[str, Any], *, label: str) -> None:
    codec_type = codec.get("type")
    if codec_type == "discrete_lookup":
        extra = sorted(set(codec) - {"type", "values"})
        if extra:
            raise ValueError(f"{label} has unexpected keys: {extra}")
        values = codec.get("values")
        if not isinstance(values, list | tuple) or not values:
            raise ValueError(f"{label}.values must be a non-empty list")
        return
    if codec_type == VIZDOOM_DEATHMATCH_MULTIDISCRETE_CODEC:
        extra = sorted(set(codec) - {"type"})
        if extra:
            raise ValueError(f"{label} has unexpected keys: {extra}")
        return
    raise ValueError(
        f"{label}.type must be 'discrete_lookup' or '{VIZDOOM_DEATHMATCH_MULTIDISCRETE_CODEC}'"
    )


def _validate_deathmatch_layout(
    descriptor: Any,
    policy_action_space: gym.Space | None = None,
) -> dict[str, int]:
    if str(descriptor.provider_id) != "vizdoom-turbo":
        raise ValueError(
            f"{VIZDOOM_DEATHMATCH_MULTIDISCRETE_CODEC} requires provider 'vizdoom-turbo'"
        )
    native_space = descriptor.native_action_space
    if not isinstance(native_space, gym.spaces.Box):
        raise ValueError(
            f"{VIZDOOM_DEATHMATCH_MULTIDISCRETE_CODEC} requires a native Box action space; "
            "configure vizdoom-turbo with use_restricted_actions='filtered'"
        )
    if native_space.shape != (len(VIZDOOM_DEATHMATCH_MULTIDISCRETE_BUTTONS),):
        raise ValueError(
            f"{VIZDOOM_DEATHMATCH_MULTIDISCRETE_CODEC} requires a native "
            f"Box({len(VIZDOOM_DEATHMATCH_MULTIDISCRETE_BUTTONS)},) action space"
        )
    buttons = tuple(str(button) for button in descriptor.action_buttons)
    if len(buttons) != len(set(buttons)):
        raise ValueError("ViZDoom Deathmatch native action buttons must be unique")
    missing = sorted(set(VIZDOOM_DEATHMATCH_MULTIDISCRETE_BUTTONS) - set(buttons))
    extra = sorted(set(buttons) - set(VIZDOOM_DEATHMATCH_MULTIDISCRETE_BUTTONS))
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"unexpected {extra}")
        raise ValueError(
            f"{VIZDOOM_DEATHMATCH_MULTIDISCRETE_CODEC} native buttons differ: " + "; ".join(details)
        )
    if policy_action_space is not None:
        if (
            not isinstance(policy_action_space, gym.spaces.MultiDiscrete)
            or tuple(int(value) for value in np.asarray(policy_action_space.nvec).reshape(-1))
            != VIZDOOM_DEATHMATCH_MULTIDISCRETE_NVEC
        ):
            raise ValueError(
                f"{VIZDOOM_DEATHMATCH_MULTIDISCRETE_CODEC} requires policy "
                f"MultiDiscrete{VIZDOOM_DEATHMATCH_MULTIDISCRETE_NVEC}"
            )
    indices = {button: index for index, button in enumerate(buttons)}
    representative = np.zeros(native_space.shape, dtype=native_space.dtype)
    for button in VIZDOOM_DEATHMATCH_MULTIDISCRETE_BUTTONS[:-1]:
        representative[indices[button]] = 1
    representative[indices["TURN_LEFT_RIGHT_DELTA"]] = max(VIZDOOM_DEATHMATCH_TURN_DEGREES)
    if not native_space.contains(representative):
        raise ValueError(
            f"{VIZDOOM_DEATHMATCH_MULTIDISCRETE_CODEC} outputs exceed the native Box bounds"
        )
    representative[indices["TURN_LEFT_RIGHT_DELTA"]] = min(VIZDOOM_DEATHMATCH_TURN_DEGREES)
    if not native_space.contains(representative):
        raise ValueError(
            f"{VIZDOOM_DEATHMATCH_MULTIDISCRETE_CODEC} outputs exceed the native Box bounds"
        )
    return indices


def _choice(value: int, semantic_id: str, label: str) -> dict[str, Any]:
    return {
        "value": int(value),
        "semantic_id": semantic_id,
        "label": label,
        "atoms": [] if semantic_id == "noop" else [semantic_id],
    }


def vizdoom_deathmatch_multidiscrete_semantics() -> dict[str, Any]:
    turn_values = []
    for index, degrees in enumerate(VIZDOOM_DEATHMATCH_TURN_DEGREES):
        if degrees == 0.0:
            turn_values.append(_choice(index, "noop", "no turn"))
            continue
        sign = "negative" if degrees < 0 else "positive"
        magnitude = str(abs(degrees)).replace(".", "_")
        turn_values.append(
            _choice(
                index,
                f"turn_delta_{sign}_{magnitude}_degrees",
                f"turn delta {degrees:g} degrees",
            )
        )
    return {
        "status": "available",
        "encoding": "components",
        "components": [
            {
                "index": 0,
                "semantic_id": "forward_movement",
                "label": "forward movement",
                "values": [
                    _choice(0, "noop", "no forward movement"),
                    _choice(1, "move_forward", "move forward"),
                    _choice(2, "move_backward", "move backward"),
                ],
            },
            {
                "index": 1,
                "semantic_id": "strafe",
                "label": "strafe",
                "values": [
                    _choice(0, "noop", "no strafe"),
                    _choice(1, "move_right", "move right"),
                    _choice(2, "move_left", "move left"),
                ],
            },
            {
                "index": 2,
                "semantic_id": "weapon",
                "label": "weapon selection",
                "values": [
                    _choice(0, "noop", "keep weapon"),
                    *[
                        _choice(slot, f"select_weapon{slot}", f"select weapon {slot}")
                        for slot in range(1, 8)
                    ],
                ],
            },
            {
                "index": 3,
                "semantic_id": "attack",
                "label": "attack",
                "values": [
                    _choice(0, "noop", "do not attack"),
                    _choice(1, "attack", "attack"),
                ],
            },
            {
                "index": 4,
                "semantic_id": "speed",
                "label": "speed",
                "values": [
                    _choice(0, "noop", "normal speed"),
                    _choice(1, "speed", "sprint"),
                ],
            },
            {
                "index": 5,
                "semantic_id": "turn_delta",
                "label": "turn delta",
                "values": turn_values,
            },
        ],
    }


def vizdoom_deathmatch_multidiscrete_codec_document(
    descriptor: Any,
    policy_action_space: gym.Space,
) -> dict[str, Any]:
    _validate_deathmatch_layout(descriptor, policy_action_space)
    return {
        "type": VIZDOOM_DEATHMATCH_MULTIDISCRETE_CODEC,
        "native_buttons": [str(button) for button in descriptor.action_buttons],
        "axes": [
            "forward_movement",
            "strafe",
            "weapon",
            "attack",
            "speed",
            "turn_delta",
        ],
        "turn_degrees": list(VIZDOOM_DEATHMATCH_TURN_DEGREES),
    }


class VizdoomDeathmatchMultiDiscreteActionCodec:
    def __init__(self, descriptor: Any, num_envs: int):
        self.num_envs = int(num_envs)
        self.action_space = gym.spaces.MultiDiscrete(
            np.asarray(VIZDOOM_DEATHMATCH_MULTIDISCRETE_NVEC, dtype=np.int64)
        )
        self._indices = _validate_deathmatch_layout(descriptor, self.action_space)
        self._native_space = descriptor.native_action_space
        self._buffer = np.zeros(
            (self.num_envs, *self._native_space.shape),
            dtype=self._native_space.dtype,
        )

    def map_actions(self, actions: Any) -> np.ndarray:
        selected = np.asarray(actions, dtype=np.int64)
        expected_shape = (self.num_envs, len(VIZDOOM_DEATHMATCH_MULTIDISCRETE_NVEC))
        if selected.shape != expected_shape:
            raise ValueError(
                f"expected ViZDoom Deathmatch MultiDiscrete actions with shape "
                f"{expected_shape}, got {selected.shape}"
            )
        nvec = np.asarray(VIZDOOM_DEATHMATCH_MULTIDISCRETE_NVEC, dtype=np.int64)
        if np.any(selected < 0) or np.any(selected >= nvec):
            raise ValueError("ViZDoom Deathmatch MultiDiscrete action is outside its policy space")

        output = self._buffer
        output.fill(0)
        output[:, self._indices["MOVE_FORWARD"]] = selected[:, 0] == 1
        output[:, self._indices["MOVE_BACKWARD"]] = selected[:, 0] == 2
        output[:, self._indices["MOVE_RIGHT"]] = selected[:, 1] == 1
        output[:, self._indices["MOVE_LEFT"]] = selected[:, 1] == 2
        for slot in range(1, 8):
            output[:, self._indices[f"SELECT_WEAPON{slot}"]] = selected[:, 2] == slot
        output[:, self._indices["ATTACK"]] = selected[:, 3]
        output[:, self._indices["SPEED"]] = selected[:, 4]
        turn_values = np.asarray(
            VIZDOOM_DEATHMATCH_TURN_DEGREES,
            dtype=self._native_space.dtype,
        )
        output[:, self._indices["TURN_LEFT_RIGHT_DELTA"]] = turn_values[selected[:, 5]]
        return output


__all__ = [
    "VIZDOOM_DEATHMATCH_MULTIDISCRETE_BUTTONS",
    "VIZDOOM_DEATHMATCH_MULTIDISCRETE_CODEC",
    "VIZDOOM_DEATHMATCH_MULTIDISCRETE_NVEC",
    "VIZDOOM_DEATHMATCH_TURN_DEGREES",
    "VizdoomDeathmatchMultiDiscreteActionCodec",
    "validate_task_action_codec",
    "vizdoom_deathmatch_multidiscrete_codec_document",
    "vizdoom_deathmatch_multidiscrete_semantics",
]
