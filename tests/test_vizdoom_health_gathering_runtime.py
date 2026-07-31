from pathlib import Path

import numpy as np
import pytest

from gradlab.env import make_training_vec_env, resolve_env_config
from gradlab.env_config import env_config_from_mapping
from gradlab.recipe_documents import compose_train_document


GOALS_ROOT = Path("experiments/goals")


@pytest.mark.parametrize(
    "goal_id",
    ["VizdoomHealthGathering-v1", "VizdoomHealthGatheringSupreme-v1"],
)
def test_health_gathering_recipe_runs_through_the_real_vector_runtime(
    goal_id: str,
) -> None:
    goal_root = GOALS_ROOT / goal_id
    document = compose_train_document(
        goal_root / "_goal.yaml",
        goal_root / "recipes/ppo.yaml",
    )
    config = resolve_env_config(env_config_from_mapping(document["train_config"]))
    env = make_training_vec_env(config, n_envs=2, seed=1701)

    try:
        observations = env.reset()
        assert set(observations) == {
            "observation",
            "context/health",
            "context/remaining_time",
        }
        assert observations["observation"].shape == (2, 4, 84, 84)
        assert observations["context/health"].shape == (2, 1)
        assert observations["context/health"].dtype == np.float32
        assert np.all(
            (-1.0 <= observations["context/health"])
            & (observations["context/health"] <= 2.0)
        )
        np.testing.assert_array_equal(
            observations["context/remaining_time"],
            np.ones((2, 1), dtype=np.float32),
        )

        next_observations, rewards, dones, infos = env.step(np.zeros(2, dtype=np.int64))
        assert set(next_observations) == {
            "observation",
            "context/health",
            "context/remaining_time",
        }
        np.testing.assert_allclose(
            next_observations["context/remaining_time"],
            np.full((2, 1), 524.0 / 525.0, dtype=np.float32),
        )
        assert rewards.shape == (2,)
        assert dones.shape == (2,)
        assert len(infos) == 2
        assert all(isinstance(info, dict) for info in infos)
    finally:
        env.close()
