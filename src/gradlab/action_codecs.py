from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
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

VIZDOOM_SHARED_MULTIDISCRETE_CODEC = "vizdoom_shared_multidiscrete_v1"
VIZDOOM_SHARED_MULTIDISCRETE_NVEC = (3, 3, 10, 2, 2, 23)
VIZDOOM_SHARED_MULTIDISCRETE_BUTTONS = (
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "MOVE_RIGHT",
    "MOVE_LEFT",
    "SELECT_NEXT_WEAPON",
    "SELECT_PREV_WEAPON",
    "SELECT_WEAPON1",
    "SELECT_WEAPON2",
    "SELECT_WEAPON3",
    "SELECT_WEAPON4",
    "SELECT_WEAPON5",
    "SELECT_WEAPON6",
    "SELECT_WEAPON7",
    "ATTACK",
    "SPEED",
    "TURN_LEFT",
    "TURN_RIGHT",
    "TURN_LEFT_RIGHT_DELTA",
)
VIZDOOM_SHARED_TURN_DEGREES = tuple(
    value for value in VIZDOOM_DEATHMATCH_TURN_DEGREES if value != 0.0
)
LEGAL_TUPLE_DISTRIBUTION = "legal_tuple_categorical_v1"
LEGAL_TUPLE_SCORING = "sum_selected_axis_logits"


class LegalTupleMultiDiscrete(gym.spaces.MultiDiscrete):
    """A fixed MultiDiscrete head whose executable support is an ordered tuple table."""

    def __init__(
        self,
        nvec: Any,
        legal_tuples: Any,
        *,
        seed: int | np.random.Generator | None = None,
        start: Any = None,
        distribution_type: str = LEGAL_TUPLE_DISTRIBUTION,
        scoring_rule: str = LEGAL_TUPLE_SCORING,
    ) -> None:
        super().__init__(nvec, seed=seed, start=start)
        normalized = _normalize_legal_tuples(
            legal_tuples,
            nvec=tuple(int(value) for value in np.asarray(self.nvec).reshape(-1)),
            label="legal_tuples",
        )
        if normalized[0] != tuple(0 for _ in normalized[0]):
            raise ValueError("legal_tuples must begin with the all-zero noop tuple")
        if distribution_type != LEGAL_TUPLE_DISTRIBUTION:
            raise ValueError(f"unsupported legal-tuple distribution {distribution_type!r}")
        if scoring_rule != LEGAL_TUPLE_SCORING:
            raise ValueError(f"unsupported legal-tuple scoring rule {scoring_rule!r}")
        self.legal_tuples = normalized
        self.distribution_type = distribution_type
        self.scoring_rule = scoring_rule
        self._legal_tuple_set = frozenset(normalized)
        nvec_array = np.asarray(self.nvec, dtype=np.int64).reshape(-1)
        self._mixed_radix_multipliers = np.concatenate(
            (np.asarray([1], dtype=np.int64), np.cumprod(nvec_array[:-1]))
        )
        self._legal_row_lookup = np.full(int(np.prod(nvec_array)), -1, dtype=np.int64)
        legal_array = np.asarray(normalized, dtype=np.int64)
        self._legal_row_lookup[legal_array @ self._mixed_radix_multipliers] = np.arange(
            len(normalized), dtype=np.int64
        )

    @property
    def legal_tuple_count(self) -> int:
        return len(self.legal_tuples)

    def sample(self, mask: Any = None, probability: Any = None) -> np.ndarray:
        if mask is not None or probability is not None:
            raise ValueError("LegalTupleMultiDiscrete does not accept per-axis sample masks")
        index = int(self.np_random.integers(len(self.legal_tuples)))
        return np.asarray(self.legal_tuples[index], dtype=self.dtype)

    def contains(self, x: Any) -> bool:
        if not super().contains(x):
            return False
        try:
            value = tuple(int(item) for item in np.asarray(x).reshape(-1))
        except TypeError, ValueError:
            return False
        return value in self._legal_tuple_set

    def legal_tuple_indices(self, actions: Any) -> np.ndarray:
        """Map exact legal tuples to their ordered joint-categorical row indices."""

        values = np.asarray(actions)
        axis_count = int(np.asarray(self.nvec).size)
        if values.ndim < 1 or values.shape[-1] != axis_count:
            raise ValueError(f"legal tuple actions must end with {axis_count} axes")
        if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
            raise ValueError("legal tuple actions must contain finite numbers")
        integers = values.astype(np.int64)
        if not np.allclose(values, integers):
            raise ValueError("legal tuple actions must contain integers")
        nvec = np.asarray(self.nvec, dtype=np.int64).reshape(-1)
        if np.any(integers < 0) or np.any(integers >= nvec):
            raise ValueError("legal tuple action is outside its MultiDiscrete head")
        rows = self._legal_row_lookup[integers @ self._mixed_radix_multipliers]
        if np.any(rows < 0):
            raise ValueError("legal tuple action is not in the configured support")
        return rows

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, LegalTupleMultiDiscrete)
            and super().__eq__(other)
            and self.legal_tuples == other.legal_tuples
            and self.distribution_type == other.distribution_type
            and self.scoring_rule == other.scoring_rule
        )


def _normalize_legal_tuples(
    value: Any,
    *,
    nvec: tuple[int, ...] = VIZDOOM_SHARED_MULTIDISCRETE_NVEC,
    label: str,
) -> tuple[tuple[int, ...], ...]:
    if isinstance(value, str | bytes) or not isinstance(value, list | tuple) or not value:
        raise ValueError(f"{label} must be a non-empty action tuple list")
    normalized: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for row_index, raw_row in enumerate(value):
        if isinstance(raw_row, str | bytes) or not isinstance(raw_row, list | tuple):
            raise ValueError(f"{label}[{row_index}] must be an action tuple")
        if len(raw_row) != len(nvec):
            raise ValueError(
                f"{label}[{row_index}] must contain {len(nvec)} axis values"
            )
        row: list[int] = []
        for axis, (raw_item, cardinality) in enumerate(zip(raw_row, nvec, strict=True)):
            if isinstance(raw_item, bool) or not isinstance(raw_item, int | np.integer):
                raise ValueError(f"{label}[{row_index}][{axis}] must be an integer")
            item = int(raw_item)
            if not 0 <= item < cardinality:
                raise ValueError(
                    f"{label}[{row_index}][{axis}] must be in [0, {cardinality})"
                )
            row.append(item)
        result = tuple(row)
        if result in seen:
            raise ValueError(f"{label} contains duplicate tuple {result}")
        seen.add(result)
        normalized.append(result)
    return tuple(normalized)


def vizdoom_action_table_hash(table: Any) -> str:
    payload = json.dumps(table, sort_keys=False, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def vizdoom_shared_legal_tuples(
    table: Any,
    *,
    label: str = "ViZDoom action table",
) -> tuple[tuple[int, ...], ...]:
    if isinstance(table, str | bytes) or not isinstance(table, list | tuple) or not table:
        raise ValueError(f"{label} must be a non-empty action table")
    result: list[tuple[int, ...]] = []
    for row_index, raw_labels in enumerate(table):
        if isinstance(raw_labels, str | bytes) or not isinstance(raw_labels, list | tuple):
            raise ValueError(f"{label}[{row_index}] must be a button-label list")
        if any(not isinstance(button, str) for button in raw_labels):
            raise ValueError(f"{label}[{row_index}] button labels must be strings")
        if len(set(raw_labels)) != len(raw_labels):
            raise ValueError(f"{label}[{row_index}] repeats a button")
        action = [0, 0, 0, 0, 0, 0]

        def select(axis: int, value: int, button: str) -> None:
            if action[axis] != 0:
                raise ValueError(
                    f"{label}[{row_index}] combines conflicting axis buttons at {button!r}"
                )
            action[axis] = value

        for button in raw_labels:
            if button == "MOVE_FORWARD":
                select(0, 1, button)
            elif button == "MOVE_BACKWARD":
                select(0, 2, button)
            elif button == "MOVE_RIGHT":
                select(1, 1, button)
            elif button == "MOVE_LEFT":
                select(1, 2, button)
            elif button == "SELECT_NEXT_WEAPON":
                select(2, 1, button)
            elif button == "SELECT_PREV_WEAPON":
                select(2, 2, button)
            elif button.startswith("SELECT_WEAPON") and button[13:].isdigit():
                slot = int(button[13:])
                if not 1 <= slot <= 7:
                    raise ValueError(f"{label}[{row_index}] uses unsupported {button!r}")
                select(2, slot + 2, button)
            elif button == "ATTACK":
                select(3, 1, button)
            elif button == "SPEED":
                select(4, 1, button)
            elif button == "TURN_LEFT":
                select(5, 1, button)
            elif button == "TURN_RIGHT":
                select(5, 2, button)
            else:
                raise ValueError(f"{label}[{row_index}] uses unsupported button {button!r}")
        result.append(tuple(action))
    normalized = _normalize_legal_tuples(result, label=f"{label} legal tuples")
    if normalized[0] != (0, 0, 0, 0, 0, 0):
        raise ValueError(f"{label} must begin with an empty noop action")
    return normalized


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
    if codec_type == VIZDOOM_SHARED_MULTIDISCRETE_CODEC:
        expected = {"type", "legal_tuples", "source_table", "source_table_hash"}
        extra = sorted(set(codec) - expected)
        missing = sorted(expected - set(codec))
        if extra:
            raise ValueError(f"{label} has unexpected keys: {extra}")
        if missing:
            raise ValueError(f"{label} is missing required keys: {missing}")
        source_table = codec["source_table"]
        derived = vizdoom_shared_legal_tuples(source_table, label=f"{label}.source_table")
        configured = _normalize_legal_tuples(
            codec["legal_tuples"],
            label=f"{label}.legal_tuples",
        )
        if configured != derived:
            raise ValueError(f"{label}.legal_tuples do not match source_table")
        source_hash = codec["source_table_hash"]
        if (
            not isinstance(source_hash, str)
            or len(source_hash) != 64
            or any(character not in "0123456789abcdef" for character in source_hash)
        ):
            raise ValueError(f"{label}.source_table_hash must be a SHA-256 hex digest")
        return
    raise ValueError(
        f"{label}.type must be 'discrete_lookup', "
        f"'{VIZDOOM_DEATHMATCH_MULTIDISCRETE_CODEC}', or "
        f"'{VIZDOOM_SHARED_MULTIDISCRETE_CODEC}'"
    )


def _validate_deathmatch_layout(
    descriptor: Any,
    policy_action_space: gym.Space | None = None,
) -> dict[str, int]:
    if str(descriptor.provider_id) != "env-vizdoom-turbo":
        raise ValueError(
            f"{VIZDOOM_DEATHMATCH_MULTIDISCRETE_CODEC} requires provider 'env-vizdoom-turbo'"
        )
    native_space = descriptor.native_action_space
    if not isinstance(native_space, gym.spaces.Box):
        raise ValueError(
            f"{VIZDOOM_DEATHMATCH_MULTIDISCRETE_CODEC} requires a native Box action space; "
            "configure env-vizdoom-turbo with use_restricted_actions='filtered'"
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


def _validate_shared_layout(
    descriptor: Any,
    policy_action_space: gym.Space | None = None,
) -> dict[str, int]:
    if str(descriptor.provider_id) != "env-vizdoom-turbo":
        raise ValueError(
            f"{VIZDOOM_SHARED_MULTIDISCRETE_CODEC} requires provider 'env-vizdoom-turbo'"
        )
    native_space = descriptor.native_action_space
    if not isinstance(native_space, gym.spaces.Box):
        raise ValueError(
            f"{VIZDOOM_SHARED_MULTIDISCRETE_CODEC} requires a native Box action space"
        )
    if native_space.shape != (len(VIZDOOM_SHARED_MULTIDISCRETE_BUTTONS),):
        raise ValueError(
            f"{VIZDOOM_SHARED_MULTIDISCRETE_CODEC} requires native "
            f"Box({len(VIZDOOM_SHARED_MULTIDISCRETE_BUTTONS)},)"
        )
    buttons = tuple(str(button) for button in descriptor.action_buttons)
    if len(buttons) != len(set(buttons)):
        raise ValueError("shared ViZDoom native action buttons must be unique")
    if set(buttons) != set(VIZDOOM_SHARED_MULTIDISCRETE_BUTTONS):
        missing = sorted(set(VIZDOOM_SHARED_MULTIDISCRETE_BUTTONS) - set(buttons))
        extra = sorted(set(buttons) - set(VIZDOOM_SHARED_MULTIDISCRETE_BUTTONS))
        raise ValueError(
            f"{VIZDOOM_SHARED_MULTIDISCRETE_CODEC} native buttons differ: "
            f"missing={missing}, unexpected={extra}"
        )
    if policy_action_space is not None:
        if not isinstance(policy_action_space, LegalTupleMultiDiscrete):
            raise ValueError(
                f"{VIZDOOM_SHARED_MULTIDISCRETE_CODEC} requires LegalTupleMultiDiscrete"
            )
        if tuple(int(value) for value in np.asarray(policy_action_space.nvec).reshape(-1)) != (
            VIZDOOM_SHARED_MULTIDISCRETE_NVEC
        ):
            raise ValueError(
                f"{VIZDOOM_SHARED_MULTIDISCRETE_CODEC} requires policy "
                f"MultiDiscrete{VIZDOOM_SHARED_MULTIDISCRETE_NVEC}"
            )
    indices = {button: index for index, button in enumerate(buttons)}
    representative = np.zeros(native_space.shape, dtype=native_space.dtype)
    for button in VIZDOOM_SHARED_MULTIDISCRETE_BUTTONS[:-1]:
        representative[indices[button]] = 1
    representative[indices["TURN_LEFT_RIGHT_DELTA"]] = max(VIZDOOM_SHARED_TURN_DEGREES)
    if not native_space.contains(representative):
        raise ValueError(f"{VIZDOOM_SHARED_MULTIDISCRETE_CODEC} exceeds native Box bounds")
    representative[indices["TURN_LEFT_RIGHT_DELTA"]] = min(VIZDOOM_SHARED_TURN_DEGREES)
    if not native_space.contains(representative):
        raise ValueError(f"{VIZDOOM_SHARED_MULTIDISCRETE_CODEC} exceeds native Box bounds")
    return indices


def vizdoom_shared_multidiscrete_semantics() -> dict[str, Any]:
    turn_values = [
        _choice(0, "noop", "no turn"),
        _choice(1, "turn_left", "turn left"),
        _choice(2, "turn_right", "turn right"),
    ]
    for index, degrees in enumerate(VIZDOOM_SHARED_TURN_DEGREES, start=3):
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
                    _choice(1, "select_next_weapon", "select next weapon"),
                    _choice(2, "select_prev_weapon", "select previous weapon"),
                    *[
                        _choice(slot + 2, f"select_weapon{slot}", f"select weapon {slot}")
                        for slot in range(1, 8)
                    ],
                ],
            },
            {
                "index": 3,
                "semantic_id": "attack",
                "label": "attack",
                "values": [_choice(0, "noop", "do not attack"), _choice(1, "attack", "attack")],
            },
            {
                "index": 4,
                "semantic_id": "speed",
                "label": "speed",
                "values": [_choice(0, "noop", "normal speed"), _choice(1, "speed", "sprint")],
            },
            {
                "index": 5,
                "semantic_id": "turn",
                "label": "turn",
                "values": turn_values,
            },
        ],
    }


def vizdoom_shared_multidiscrete_codec_document(
    descriptor: Any,
    policy_action_space: gym.Space,
    configured_codec: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_shared_layout(descriptor, policy_action_space)
    validate_task_action_codec(configured_codec, label="task.action.codec")
    assert isinstance(policy_action_space, LegalTupleMultiDiscrete)
    return {
        "type": VIZDOOM_SHARED_MULTIDISCRETE_CODEC,
        "native_buttons": [str(button) for button in descriptor.action_buttons],
        "axes": ["forward_movement", "strafe", "weapon", "attack", "speed", "turn"],
        "turn_degrees": list(VIZDOOM_SHARED_TURN_DEGREES),
        "legal_tuples": [list(row) for row in policy_action_space.legal_tuples],
        "source_table": [list(row) for row in configured_codec["source_table"]],
        "source_table_hash": str(configured_codec["source_table_hash"]),
        "distribution": {
            "type": LEGAL_TUPLE_DISTRIBUTION,
            "scoring": LEGAL_TUPLE_SCORING,
        },
    }


class VizdoomSharedMultiDiscreteActionCodec:
    def __init__(self, descriptor: Any, num_envs: int, configured_codec: Mapping[str, Any]):
        validate_task_action_codec(configured_codec, label="task.action.codec")
        self.num_envs = int(num_envs)
        self.action_space = LegalTupleMultiDiscrete(
            np.asarray(VIZDOOM_SHARED_MULTIDISCRETE_NVEC, dtype=np.int64),
            configured_codec["legal_tuples"],
        )
        self._indices = _validate_shared_layout(descriptor, self.action_space)
        self._native_space = descriptor.native_action_space
        self._buffer = np.zeros(
            (self.num_envs, *self._native_space.shape), dtype=self._native_space.dtype
        )

    def map_actions(self, actions: Any) -> np.ndarray:
        selected = np.asarray(actions, dtype=np.int64)
        expected_shape = (self.num_envs, len(VIZDOOM_SHARED_MULTIDISCRETE_NVEC))
        if selected.shape != expected_shape:
            raise ValueError(
                f"expected shared ViZDoom actions with shape {expected_shape}, got {selected.shape}"
            )
        nvec = np.asarray(VIZDOOM_SHARED_MULTIDISCRETE_NVEC, dtype=np.int64)
        if np.any(selected < 0) or np.any(selected >= nvec):
            raise ValueError("shared ViZDoom action is outside its MultiDiscrete head")
        try:
            self.action_space.legal_tuple_indices(selected)
        except ValueError as exc:
            raise ValueError("shared ViZDoom action is not legal for this task") from exc

        output = self._buffer
        output.fill(0)
        output[:, self._indices["MOVE_FORWARD"]] = selected[:, 0] == 1
        output[:, self._indices["MOVE_BACKWARD"]] = selected[:, 0] == 2
        output[:, self._indices["MOVE_RIGHT"]] = selected[:, 1] == 1
        output[:, self._indices["MOVE_LEFT"]] = selected[:, 1] == 2
        output[:, self._indices["SELECT_NEXT_WEAPON"]] = selected[:, 2] == 1
        output[:, self._indices["SELECT_PREV_WEAPON"]] = selected[:, 2] == 2
        for slot in range(1, 8):
            output[:, self._indices[f"SELECT_WEAPON{slot}"]] = selected[:, 2] == slot + 2
        output[:, self._indices["ATTACK"]] = selected[:, 3]
        output[:, self._indices["SPEED"]] = selected[:, 4]
        output[:, self._indices["TURN_LEFT"]] = selected[:, 5] == 1
        output[:, self._indices["TURN_RIGHT"]] = selected[:, 5] == 2
        delta_mask = selected[:, 5] >= 3
        if np.any(delta_mask):
            turn_values = np.asarray(VIZDOOM_SHARED_TURN_DEGREES, dtype=self._native_space.dtype)
            output[delta_mask, self._indices["TURN_LEFT_RIGHT_DELTA"]] = turn_values[
                selected[delta_mask, 5] - 3
            ]
        return output


__all__ = [
    "LEGAL_TUPLE_DISTRIBUTION",
    "LEGAL_TUPLE_SCORING",
    "LegalTupleMultiDiscrete",
    "VIZDOOM_DEATHMATCH_MULTIDISCRETE_BUTTONS",
    "VIZDOOM_DEATHMATCH_MULTIDISCRETE_CODEC",
    "VIZDOOM_DEATHMATCH_MULTIDISCRETE_NVEC",
    "VIZDOOM_DEATHMATCH_TURN_DEGREES",
    "VIZDOOM_SHARED_MULTIDISCRETE_BUTTONS",
    "VIZDOOM_SHARED_MULTIDISCRETE_CODEC",
    "VIZDOOM_SHARED_MULTIDISCRETE_NVEC",
    "VIZDOOM_SHARED_TURN_DEGREES",
    "VizdoomDeathmatchMultiDiscreteActionCodec",
    "VizdoomSharedMultiDiscreteActionCodec",
    "validate_task_action_codec",
    "vizdoom_action_table_hash",
    "vizdoom_deathmatch_multidiscrete_codec_document",
    "vizdoom_deathmatch_multidiscrete_semantics",
    "vizdoom_shared_legal_tuples",
    "vizdoom_shared_multidiscrete_codec_document",
    "vizdoom_shared_multidiscrete_semantics",
]
