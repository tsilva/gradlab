from pathlib import Path

import numpy as np

from gradlab.env import make_training_vec_env, resolve_env_config
from gradlab.env_config import env_config_from_mapping
from gradlab.recipe_documents import compose_train_document


GOAL_ROOT = Path("experiments/goals/VizdoomDeathmatch-v1")


def test_deathmatch_recipe_runs_through_the_real_single_player_vector_runtime() -> None:
    document = compose_train_document(
        GOAL_ROOT / "_goal.yaml",
        GOAL_ROOT / "recipes/ppo.yaml",
    )
    config = resolve_env_config(env_config_from_mapping(document["train_config"]))
    env = make_training_vec_env(config, n_envs=2, seed=1701)

    try:
        observations = env.reset()
        assert set(observations) == {"observation", "context/health"}
        assert observations["observation"].shape == (2, 4, 84, 84)
        assert observations["context/health"].shape == (2, 1)
        assert observations["context/health"].dtype == np.float32
        assert np.all(
            (0.0 <= observations["context/health"])
            & (observations["context/health"] <= 2.0)
        )
        assert env.action_space.n == 17
        assert {
            "killcount",
            "health",
            "armor",
            "selected_weapon",
            "selected_weapon_ammo",
            "player_dead",
            "pending_reset",
        } <= set(env.reset_infos[0])

        next_observations, rewards, dones, infos = env.step(
            np.asarray([0, 9], dtype=np.int64)
        )
        assert set(next_observations) == {"observation", "context/health"}
        assert next_observations["context/health"].shape == (2, 1)
        assert rewards.shape == (2,)
        assert dones.shape == (2,)
        assert len(infos) == 2
        assert all(isinstance(info, dict) for info in infos)

        contract = env.runtime.action_contract
        assert contract["provider"]["mode"] == "custom_discrete"
        assert contract["provider"]["space"] == {
            "type": "discrete",
            "n": 17,
            "start": 0,
            "dtype": "int64",
        }
        assert contract["requested"]["meanings"][8:10] == [
            "speed_move_forward",
            "attack_move_forward",
        ]
    finally:
        env.close()
