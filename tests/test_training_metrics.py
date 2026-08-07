from types import SimpleNamespace

import pytest

from gradlab.metric_names import (
    EPISODE_METRIC_WINDOW_SIZE,
    TRAIN_EXPLORATION_CELL_UNIQUE_ORIGIN_TARGET_ROLLING_MEAN,
    TRAIN_EPISODE_RETURN_SHAPED_ORIGIN_TARGET_ROLLING_MEAN,
    TRAIN_PROGRESS_KILLS_ORIGIN_TARGET_ROLLING_MEAN,
    validate_metric_name,
)
from gradlab.training_metrics import EpisodeMetricsReducer


def _episode(
    episode_return: float,
    *,
    origin: str = "target",
    kills: float | None = None,
    unique_cells: int | None = None,
) -> SimpleNamespace:
    metrics = {}
    if unique_cells is not None:
        metrics["cell_novelty_episode_unique_cells"] = unique_cells
    if kills is not None:
        metrics["kills"] = kills
    return SimpleNamespace(
        episode_return=episode_return,
        episode_length=1,
        start_origin=origin,
        start_id="default",
        outcome="success",
        events=(),
        terminated=True,
        truncated=False,
        metrics=metrics,
    )


def test_target_return_mean_uses_the_one_canonical_rolling_window() -> None:
    reducer = EpisodeMetricsReducer(track_success=False)
    assert reducer.window_size == EPISODE_METRIC_WINDOW_SIZE == 100

    partial = reducer.consume(_episode(1.0) for _ in range(99))
    assert partial[TRAIN_EPISODE_RETURN_SHAPED_ORIGIN_TARGET_ROLLING_MEAN] == 1.0

    mature = reducer.consume((_episode(3.0),))
    assert mature[TRAIN_EPISODE_RETURN_SHAPED_ORIGIN_TARGET_ROLLING_MEAN] == pytest.approx(1.02)

    archive_only = reducer.consume((_episode(1000.0, origin="curriculum"),))
    assert archive_only[TRAIN_EPISODE_RETURN_SHAPED_ORIGIN_TARGET_ROLLING_MEAN] == pytest.approx(
        1.02
    )

    rolled = reducer.consume((_episode(-1.0),))
    assert rolled[TRAIN_EPISODE_RETURN_SHAPED_ORIGIN_TARGET_ROLLING_MEAN] == 1.0


def test_target_return_rolling_mean_is_registered() -> None:
    assert (
        validate_metric_name(TRAIN_EPISODE_RETURN_SHAPED_ORIGIN_TARGET_ROLLING_MEAN)
        == TRAIN_EPISODE_RETURN_SHAPED_ORIGIN_TARGET_ROLLING_MEAN
    )


def test_unique_cell_mean_uses_only_completed_target_origin_episodes() -> None:
    reducer = EpisodeMetricsReducer(track_success=False)

    payload = reducer.consume(
        (
            _episode(0.0, unique_cells=2),
            _episode(0.0, unique_cells=6),
            _episode(0.0, origin="curriculum", unique_cells=100),
            _episode(0.0),
        )
    )

    assert payload[TRAIN_EXPLORATION_CELL_UNIQUE_ORIGIN_TARGET_ROLLING_MEAN] == 4.0
    assert (
        validate_metric_name(TRAIN_EXPLORATION_CELL_UNIQUE_ORIGIN_TARGET_ROLLING_MEAN)
        == TRAIN_EXPLORATION_CELL_UNIQUE_ORIGIN_TARGET_ROLLING_MEAN
    )


def test_configured_frag_mean_rolls_over_latest_100_target_episodes() -> None:
    reducer = EpisodeMetricsReducer(progress_fields=("kills",), track_success=False)

    partial = reducer.consume(_episode(1.0, kills=1) for _ in range(99))
    assert partial[TRAIN_PROGRESS_KILLS_ORIGIN_TARGET_ROLLING_MEAN] == 1.0

    mature = reducer.consume((_episode(3.0, kills=3),))
    assert mature[TRAIN_PROGRESS_KILLS_ORIGIN_TARGET_ROLLING_MEAN] == pytest.approx(1.02)

    archive_only = reducer.consume((_episode(1000.0, origin="curriculum", kills=1000),))
    assert archive_only[TRAIN_PROGRESS_KILLS_ORIGIN_TARGET_ROLLING_MEAN] == pytest.approx(1.02)

    rolled = reducer.consume((_episode(-1.0, kills=-1),))
    assert rolled[TRAIN_PROGRESS_KILLS_ORIGIN_TARGET_ROLLING_MEAN] == 1.0
    assert (
        validate_metric_name(TRAIN_PROGRESS_KILLS_ORIGIN_TARGET_ROLLING_MEAN)
        == TRAIN_PROGRESS_KILLS_ORIGIN_TARGET_ROLLING_MEAN
    )


def test_configured_progress_field_requires_a_finite_episode_value() -> None:
    reducer = EpisodeMetricsReducer(progress_fields=("kills",), track_success=False)

    with pytest.raises(ValueError, match="kills.*finite number"):
        reducer.consume((_episode(0.0),))
