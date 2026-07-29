from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from gradlab.env import EnvConfig
from gradlab.play_session import (
    optional_fast_env_frames,
    playback_model_observation,
    vector_env_frame,
)


def test_generic_vector_observation_reaches_policy_unchanged() -> None:
    observation = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)

    result = playback_model_observation(
        type("Model", (), {"observation_space": object()})(),
        observation,
        EnvConfig(game="CartPole-v1"),
        active_task_state=None,
        active_info_value=None,
    )

    assert result is observation
    assert optional_fast_env_frames(result) is None


def test_vector_env_frame_returns_owned_lane_frame() -> None:
    source = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    env = SimpleNamespace(get_images=lambda: [source])

    frame = vector_env_frame(env)
    source.fill(0)

    assert frame.shape == (2, 2, 3)
    assert frame.any()
