from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from gradlab.batch_runtime import ProviderDescriptor, SignalSpec
from gradlab.model_inputs import ContextTaskKernel
from gradlab.task_kernels import IdentityTaskDefinition


def _descriptor() -> ProviderDescriptor:
    return ProviderDescriptor(
        provider_id="test",
        native_observation_space=gym.spaces.Box(
            0,
            255,
            shape=(4, 84, 84),
            dtype=np.uint8,
        ),
        native_action_space=gym.spaces.Discrete(2),
        signal_schema={
            "health": SignalSpec("health", np.float32),
            "level_hi": SignalSpec("level_hi", np.int64),
            "level_lo": SignalSpec("level_lo", np.int64),
            "reset_task": SignalSpec(
                "reset_task",
                np.int64,
                available_on_step=False,
            ),
        },
        observation_ownership="owned",
        observation_buffer_depth=None,
    )


def _task() -> dict:
    return {
        "signals": {
            "health": "health",
            "level": ["level_hi", "level_lo"],
        },
        "model_inputs": {
            "schema_version": 1,
            "context": {
                "health": {
                    "signal": "health",
                    "update": "transition",
                    "encoding": {
                        "kind": "continuous",
                        "scale": 0.01,
                        "offset": 0.0,
                        "low": -1.0,
                        "high": 2.0,
                    },
                },
                "task_id": {
                    "signal": "level",
                    "update": "episode",
                    "encoding": {
                        "kind": "categorical",
                        "values": [[1, 1], [1, 2]],
                    },
                },
            },
        },
    }


def _kernel(num_envs: int = 2) -> ContextTaskKernel:
    descriptor = _descriptor()
    base = IdentityTaskDefinition(signals=_task()["signals"]).bind(
        descriptor,
        num_envs,
    )
    return ContextTaskKernel(base, descriptor, _task())


def _signals(
    health: tuple[float, float],
    levels: tuple[tuple[int, int], tuple[int, int]],
) -> dict[str, np.ndarray]:
    return {
        "health": np.asarray(health, dtype=np.float32),
        "level_hi": np.asarray([value[0] for value in levels], dtype=np.int64),
        "level_lo": np.asarray([value[1] for value in levels], dtype=np.int64),
    }


def test_context_kernel_emits_typed_flat_dict_and_updates_transition_context() -> None:
    kernel = _kernel()
    observations = np.zeros((2, 4, 84, 84), dtype=np.uint8)
    kernel.on_reset(
        observations,
        _signals((100.0, 25.0), ((1, 1), (1, 2))),
        np.ones(2, dtype=np.bool_),
    )

    encoded = kernel.encode_observations(observations)
    assert tuple(encoded) == ("observation", "context/health", "context/task_id")
    np.testing.assert_array_equal(encoded["context/health"], [[1.0], [0.25]])
    np.testing.assert_array_equal(encoded["context/task_id"], [0, 1])
    assert encoded["context/health"].dtype == np.float32
    assert encoded["context/task_id"].dtype == np.int64

    kernel.process(
        np.zeros(2, dtype=np.float32),
        np.zeros(2, dtype=np.bool_),
        np.zeros(2, dtype=np.bool_),
        _signals((70.0, 10.0), ((1, 1), (1, 2))),
    )
    np.testing.assert_allclose(
        kernel.encode_observations(observations)["context/health"],
        [[0.7], [0.1]],
    )


def test_episode_context_rejects_mid_episode_changes_but_keeps_terminal_identity() -> None:
    kernel = _kernel()
    observations = np.zeros((2, 4, 84, 84), dtype=np.uint8)
    kernel.on_reset(
        observations,
        _signals((100.0, 100.0), ((1, 1), (1, 2))),
        np.ones(2, dtype=np.bool_),
    )
    with pytest.raises(ValueError, match="changed without a boundary"):
        kernel.process(
            np.zeros(2, dtype=np.float32),
            np.zeros(2, dtype=np.bool_),
            np.zeros(2, dtype=np.bool_),
            _signals((99.0, 99.0), ((1, 2), (1, 2))),
        )

    kernel.process(
        np.zeros(2, dtype=np.float32),
        np.asarray([True, False]),
        np.zeros(2, dtype=np.bool_),
        _signals((-4.0, 99.0), ((1, 2), (1, 2))),
    )
    terminal = kernel.encode_observations(observations)
    np.testing.assert_array_equal(terminal["context/task_id"], [0, 1])
    assert terminal["context/health"][0, 0] == pytest.approx(-0.04)
    kernel.on_reset(
        observations,
        _signals((100.0, 99.0), ((1, 2), (1, 2))),
        np.asarray([True, False]),
    )
    np.testing.assert_array_equal(
        kernel.encode_observations(observations)["context/task_id"],
        [1, 1],
    )


def test_episode_context_round_trips_through_task_lane_state() -> None:
    kernel = _kernel()
    observations = np.zeros((2, 4, 84, 84), dtype=np.uint8)
    selected = np.asarray([True, False])
    kernel.on_reset(
        observations,
        _signals((80.0, 60.0), ((1, 2), (1, 1))),
        np.ones(2, dtype=np.bool_),
    )
    states = kernel.capture_lane_states(selected)
    kernel.on_reset(
        observations,
        _signals((100.0, 60.0), ((1, 1), (1, 1))),
        selected,
    )
    kernel.restore_lane_states(states, selected)

    encoded = kernel.encode_observations(observations)
    assert encoded["context/task_id"][0] == 1
    # Transition context intentionally comes from the restored provider reset.
    assert encoded["context/health"][0, 0] == pytest.approx(1.0)


def test_episode_context_may_come_from_a_reset_only_provider_signal() -> None:
    descriptor = _descriptor()
    task = {
        "signals": {"task": "reset_task"},
        "model_inputs": {
            "schema_version": 1,
            "context": {
                "task_id": {
                    "signal": "task",
                    "update": "episode",
                    "encoding": {"kind": "categorical", "values": [10, 20]},
                }
            },
        },
    }
    base = IdentityTaskDefinition(signals=task["signals"]).bind(descriptor, 2)
    kernel = ContextTaskKernel(base, descriptor, task)
    observations = np.zeros((2, 4, 84, 84), dtype=np.uint8)

    kernel.on_reset(
        observations,
        {"reset_task": np.asarray([10, 20], dtype=np.int64)},
        np.ones(2, dtype=np.bool_),
    )
    kernel.process(
        np.zeros(2, dtype=np.float32),
        np.zeros(2, dtype=np.bool_),
        np.zeros(2, dtype=np.bool_),
        {},
    )

    np.testing.assert_array_equal(
        kernel.encode_observations(observations)["context/task_id"],
        [0, 1],
    )
