from types import SimpleNamespace

import pytest

from gradlab.metric_names import (
    TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_MEAN,
    TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_WINDOW_100_MEAN,
    validate_metric_name,
)
from gradlab.training_metrics import EpisodeMetricsReducer


def _episode(episode_return: float, *, origin: str = "target") -> SimpleNamespace:
    return SimpleNamespace(
        episode_return=episode_return,
        episode_length=1,
        start_origin=origin,
        start_id="default",
        outcome="success",
        events=(),
        terminated=True,
        truncated=False,
        metrics={},
    )


def test_mature_target_return_mean_requires_and_rolls_a_full_target_window() -> None:
    reducer = EpisodeMetricsReducer(track_success=False)

    partial = reducer.consume(_episode(1.0) for _ in range(99))
    assert partial[TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_MEAN] == 1.0
    assert TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_WINDOW_100_MEAN not in partial

    mature = reducer.consume((_episode(3.0),))
    assert mature[TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_MEAN] == pytest.approx(1.02)
    assert mature[TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_WINDOW_100_MEAN] == pytest.approx(
        1.02
    )

    archive_only = reducer.consume((_episode(1000.0, origin="curriculum"),))
    assert archive_only[TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_MEAN] == pytest.approx(1.02)
    assert archive_only[
        TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_WINDOW_100_MEAN
    ] == pytest.approx(1.02)

    rolled = reducer.consume((_episode(-1.0),))
    assert rolled[TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_MEAN] == 1.0
    assert rolled[TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_WINDOW_100_MEAN] == 1.0


def test_mature_target_return_mean_is_registered() -> None:
    assert (
        validate_metric_name(TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_WINDOW_100_MEAN)
        == TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_WINDOW_100_MEAN
    )
