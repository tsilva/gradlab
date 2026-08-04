from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from gradlab.batch_runtime import ProviderDescriptor, SignalSpec
from gradlab.env_identity import validate_task_config
from gradlab.state_archive import ArchiveCellConfig, ArchiveCellDetector
from gradlab.task_kernels import (
    CELL_NOVELTY_EPISODE_UNIQUE_CELLS,
    CellNoveltyTaskKernel,
    IdentityTaskDefinition,
    default_task_document,
    normalize_cell_novelty_config,
    with_cell_novelty,
    with_reward_transform,
)


def _novelty_config(
    *,
    first_visit_bonus: float = 0.005,
    episode_bonus_cap: float = 0.2,
) -> dict:
    return {
        "cell": {
            "dimensions": [
                {"signal": "position_x", "bucket_size": 64},
                {"signal": "position_y", "bucket_size": 64},
            ]
        },
        "first_visit_bonus": first_visit_bonus,
        "episode_bonus_cap": episode_bonus_cap,
    }


def _descriptor(
    *,
    position_dtype: np.dtype | type = np.float64,
    available_on_reset: bool = True,
) -> ProviderDescriptor:
    return ProviderDescriptor(
        provider_id="cell-novelty-test",
        native_observation_space=gym.spaces.Box(
            0,
            255,
            shape=(4, 84, 84),
            dtype=np.uint8,
        ),
        native_action_space=gym.spaces.Discrete(2),
        signal_schema={
            name: SignalSpec(
                name,
                position_dtype,
                available_on_reset=available_on_reset,
            )
            for name in ("position_x", "position_y")
        },
    )


def _kernel(
    *,
    num_envs: int = 2,
    first_visit_bonus: float = 0.005,
    episode_bonus_cap: float = 0.2,
) -> CellNoveltyTaskKernel:
    inner = IdentityTaskDefinition(
        signals={"position_x": "position_x", "position_y": "position_y"}
    ).bind(_descriptor(), num_envs)
    kernel = with_cell_novelty(
        inner,
        _novelty_config(
            first_visit_bonus=first_visit_bonus,
            episode_bonus_cap=episode_bonus_cap,
        ),
    )
    assert isinstance(kernel, CellNoveltyTaskKernel)
    return kernel


def _signals(xs: tuple[float, ...], ys: tuple[float, ...]) -> dict[str, np.ndarray]:
    return {
        "position_x": np.asarray(xs, dtype=np.float64),
        "position_y": np.asarray(ys, dtype=np.float64),
    }


def _reset(
    kernel: CellNoveltyTaskKernel,
    xs: tuple[float, ...],
    ys: tuple[float, ...],
    mask: tuple[bool, ...] | None = None,
) -> None:
    selected = np.asarray(mask or ((True,) * kernel.num_envs), dtype=np.bool_)
    kernel.on_reset(
        np.zeros((kernel.num_envs, 4, 84, 84), dtype=np.uint8),
        _signals(xs, ys),
        selected,
    )


def _step(
    kernel,
    xs: tuple[float, ...],
    ys: tuple[float, ...],
    *,
    native_rewards: tuple[float, ...] | None = None,
    provider_terminated: tuple[bool, ...] | None = None,
):
    num_envs = kernel.num_envs
    return kernel.process(
        np.asarray(native_rewards or ((0.0,) * num_envs), dtype=np.float32),
        np.asarray(provider_terminated or ((False,) * num_envs), dtype=np.bool_),
        np.zeros(num_envs, dtype=np.bool_),
        _signals(xs, ys),
    )


def test_cell_novelty_config_is_strict_and_requires_declared_semantic_signals() -> None:
    expected = _novelty_config()
    assert normalize_cell_novelty_config(expected, label="novelty") == {
        "cell": {
            "dimensions": [
                {"signal": "position_x", "bucket_size": 64.0},
                {"signal": "position_y", "bucket_size": 64.0},
            ]
        },
        "first_visit_bonus": 0.005,
        "episode_bonus_cap": 0.2,
    }

    with pytest.raises(ValueError, match="semantic signals"):
        normalize_cell_novelty_config(
            {
                **expected,
                "cell": {"dimensions": [{"source": "position_x", "bucket_size": 64}]},
            },
            label="novelty",
        )
    with pytest.raises(ValueError, match="at least first_visit_bonus"):
        normalize_cell_novelty_config(
            _novelty_config(first_visit_bonus=0.1, episode_bonus_cap=0.05),
            label="novelty",
        )
    with pytest.raises(ValueError, match="unexpected fields"):
        normalize_cell_novelty_config({**expected, "decay": 0.5}, label="novelty")

    task = default_task_document("identity")
    task["signals"] = {"position_x": "position_x"}
    task["reward"]["cell_novelty"] = expected
    with pytest.raises(ValueError, match="position_y"):
        validate_task_config(task)


def test_cell_novelty_requires_numeric_signals_available_on_reset() -> None:
    base = IdentityTaskDefinition(
        signals={"position_x": "position_x", "position_y": "position_y"}
    )
    with pytest.raises(ValueError, match="must be numeric"):
        with_cell_novelty(
            base.bind(_descriptor(position_dtype=np.str_), 1),
            _novelty_config(),
        )
    with pytest.raises(ValueError, match="available on reset and step"):
        with_cell_novelty(
            base.bind(_descriptor(available_on_reset=False), 1),
            _novelty_config(),
        )


def test_archive_bucketing_is_deterministic_and_floors_negative_coordinates() -> None:
    detector = ArchiveCellDetector(
        ArchiveCellConfig.from_mapping(
            _novelty_config()["cell"],
            label="cell",
        )
    )
    values = {
        ("signal", "position_x"): np.asarray([-0.1, 0.0, -64.0, -64.1]),
        ("signal", "position_y"): np.asarray([-1.0, 63.9, -128.0, 64.0]),
    }
    first = detector.keys(values, n_envs=4)
    second = detector.keys(values, n_envs=4)
    assert first == second
    assert first == (b"[-1,-1]", b"[0,0]", b"[-1,-2]", b"[-2,1]")


def test_starting_cells_revisits_and_lanes_are_independent() -> None:
    kernel = _kernel()
    _reset(kernel, (0.0, 0.0), (0.0, 0.0))

    starting = _step(kernel, (0.0, 0.0), (0.0, 0.0))
    np.testing.assert_array_equal(starting.rewards, (0.0, 0.0))
    np.testing.assert_array_equal(
        starting.metrics[CELL_NOVELTY_EPISODE_UNIQUE_CELLS],
        (1, 1),
    )

    lane_zero_moves = _step(kernel, (64.0, 0.0), (0.0, 0.0))
    np.testing.assert_allclose(lane_zero_moves.rewards, (0.005, 0.0))
    lane_one_enters_same_cell = _step(kernel, (64.0, 64.0), (0.0, 0.0))
    np.testing.assert_allclose(lane_one_enters_same_cell.rewards, (0.0, 0.005))
    revisited = _step(kernel, (0.0, 0.0), (0.0, 0.0))
    np.testing.assert_array_equal(revisited.rewards, (0.0, 0.0))
    np.testing.assert_array_equal(
        revisited.metrics[CELL_NOVELTY_EPISODE_UNIQUE_CELLS],
        (2, 2),
    )


def test_bonus_cap_terminal_steps_and_masked_resets() -> None:
    kernel = _kernel(first_visit_bonus=0.005, episode_bonus_cap=0.01)
    _reset(kernel, (0.0, 0.0), (0.0, 0.0))

    np.testing.assert_allclose(_step(kernel, (64.0, 0.0), (0.0, 0.0)).rewards, (0.005, 0.0))
    terminal = _step(
        kernel,
        (128.0, 0.0),
        (0.0, 0.0),
        provider_terminated=(True, False),
    )
    np.testing.assert_allclose(terminal.rewards, (0.005, 0.0))
    capped = _step(kernel, (192.0, 0.0), (0.0, 0.0))
    np.testing.assert_array_equal(capped.rewards, (0.0, 0.0))
    np.testing.assert_array_equal(
        capped.metrics[CELL_NOVELTY_EPISODE_UNIQUE_CELLS],
        (4, 1),
    )

    _reset(kernel, (256.0, float("nan")), (0.0, float("nan")), (True, False))
    after_partial_reset = _step(kernel, (256.0, 64.0), (0.0, 0.0))
    np.testing.assert_allclose(after_partial_reset.rewards, (0.0, 0.005))
    np.testing.assert_array_equal(
        after_partial_reset.metrics[CELL_NOVELTY_EPISODE_UNIQUE_CELLS],
        (1, 2),
    )


def test_cell_novelty_state_round_trip_preserves_visits_and_cap_progress() -> None:
    original = _kernel()
    _reset(original, (0.0, 0.0), (0.0, 0.0))
    _step(original, (64.0, 0.0), (0.0, 0.0))
    captured = original.capture_lane_states(np.asarray((True, False)))

    restored = _kernel()
    _reset(restored, (0.0, 0.0), (0.0, 0.0))
    restored.restore_lane_states(captured, np.asarray((True, False)))
    revisit = _step(restored, (64.0, 0.0), (0.0, 0.0))
    np.testing.assert_array_equal(revisit.rewards, (0.0, 0.0))
    new_cell = _step(restored, (128.0, 0.0), (0.0, 0.0))
    np.testing.assert_allclose(new_cell.rewards, (0.005, 0.0))
    np.testing.assert_array_equal(
        new_cell.metrics[CELL_NOVELTY_EPISODE_UNIQUE_CELLS],
        (3, 1),
    )


def test_cell_bonus_is_added_before_reward_scaling_and_clipping() -> None:
    novelty = _kernel(
        num_envs=1,
        first_visit_bonus=0.2,
        episode_bonus_cap=0.4,
    )
    kernel = with_reward_transform(
        novelty,
        {"reward_scale": 2.0, "reward_clip": [-0.05, 0.05]},
    )
    _reset(kernel, (0.0,), (0.0,))
    result = _step(kernel, (64.0,), (0.0,))

    np.testing.assert_allclose(result.metrics["native_reward_component"], (0.0,))
    np.testing.assert_allclose(result.metrics["cell_novelty_reward_component"], (0.2,))
    np.testing.assert_allclose(result.metrics["raw_reward"], (0.2,))
    np.testing.assert_allclose(result.metrics["shaped_reward"], (0.05,))
    np.testing.assert_allclose(result.rewards, (0.05,))
