from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from gradlab.batch_runtime import ProviderDescriptor, SignalSpec
from gradlab.recipe_documents import compose_train_document
from gradlab.play_catalog import PlayCatalog
from gradlab.task_kernels import IdentityTaskDefinition, with_event_rewards


ROOT = Path("experiments/goals/Breakout-Atari2600-v0")


def test_breakout_catalog_has_two_separate_goals():
    page = PlayCatalog(repo_root=Path.cwd()).goals(
        environment_id="Breakout-Atari2600-v0", include_evidence=False,
    )
    assert [(item["goal_id"], item["title"], item["recipe_count"]) for item in page.items] == [
        ("FirstWall", "Clear the first Breakout wall", 2),
        ("TwoWalls", "Clear both Breakout walls", 1),
    ]


def test_two_wall_goal_preserves_ppo_settings():
    base = compose_train_document(ROOT / "FirstWall/_goal.yaml", ROOT / "FirstWall/recipes/ppo.yaml")
    two_walls = compose_train_document(ROOT / "TwoWalls/_goal.yaml", ROOT / "TwoWalls/recipes/ppo-two-walls.yaml")
    train = two_walls["train_config"]
    for key in ("training_backend", "policy_model", "timesteps", "n_envs", "frame_skip"):
        assert train[key] == base["train_config"][key]
    assert train["task"]["model_inputs"] == base["train_config"]["task"]["model_inputs"]
    assert train["checkpoint_eval_backend"] == "none"
    assert two_walls["goal_variant"]["goal_slug"] != base["goal_variant"]["goal_slug"]
    assert train["occupancy"]["domains"] == [list(range(8))]


@pytest.mark.parametrize("goal,recipe,event,steps", [
    ("FirstWall", "ppo", "one_wall_cleared", [(0, 107, 107, False), (1, 108, 21, True)]),
    ("TwoWalls", "ppo-two-walls", "two_walls_cleared", [(1, 108, 108, False), (2, 216, 128, True)]),
])
def test_goal_owns_wall_success_and_recipe_rewards(goal, recipe, event, steps):
    document = compose_train_document(ROOT / goal / "_goal.yaml", ROOT / goal / "recipes" / f"{recipe}.yaml")
    train = document["train_config"]
    goal_task = document["goal"]["train"]["environment"]["task"]
    assert goal_task["termination"]["success"] == [event]
    assert train["task"]["termination"]["success"] == [event]
    assert train["task"]["events"][event] == goal_task["events"][event]

    task = train["task"]
    descriptor = ProviderDescriptor(
        provider_id="two-wall-contract-test",
        native_observation_space=gym.spaces.Box(0, 255, shape=(1, 8, 8), dtype=np.uint8),
        native_action_space=gym.spaces.Discrete(3),
        signal_schema={name: SignalSpec(name, np.float64) for name in task["signals"].values()},
    )
    kernel = with_event_rewards(
        IdentityTaskDefinition(
            signals=task["signals"], events=task["events"], termination=task["termination"],
        ).bind(descriptor, 1),
        task["reward"]["event_rewards"],
        event_delta_rewards=task["reward"]["event_delta_rewards"],
        include_native=False,
    )
    signals = {name: np.zeros(1) for name in descriptor.signal_schema}
    signals["ball_y"][:] = 1
    signals["lives"][:] = 5
    kernel.on_reset(np.zeros((1, 1, 8, 8), dtype=np.uint8), signals, np.ones(1, dtype=bool))
    flags = np.zeros(1, dtype=bool)
    for walls, bricks, expected_reward, done in steps:
        signals = {name: value.copy() for name, value in signals.items()}
        signals["walls_cleared"][:] = walls
        signals["bricks_destroyed"][:] = bricks
        result = kernel.process(np.array([999.0]), flags, flags, signals)
        np.testing.assert_allclose(result.rewards, [expected_reward])
        assert result.terminated.tolist() == [done]
        assert not result.truncated.any()
