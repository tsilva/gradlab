from pathlib import Path
import json

import gymnasium as gym
import numpy as np

from gradlab.batch_runtime import ProviderDescriptor, SignalSpec
from gradlab.recipe_documents import compose_train_document, load_recipe_source_document
from gradlab.task_kernels import IdentityTaskDefinition, with_event_rewards


ROOT = Path("experiments/goals/Breakout-Atari2600-v0")


def test_two_wall_variant_preserves_ppo_and_only_finishes_at_second_wall():
    base = compose_train_document(ROOT / "_goal.yaml", ROOT / "recipes/ppo.yaml")
    variant = compose_train_document(ROOT / "_goal.yaml", ROOT / "recipes/ppo-two-walls.yaml")
    environment = load_recipe_source_document(
        ROOT / "recipes/ppo-two-walls.yaml"
    ).document["train"]["environment"]
    # Explicit launch overrides bind recipe-authored policy semantics into the goal variant.
    launched = compose_train_document(
        ROOT / "_goal.yaml", ROOT / "recipes/ppo-two-walls.yaml",
        recipe_overrides=[
            f"train.environment.{key}=" + json.dumps(environment[key])
            for key in ("task", "preprocessing")
        ],
    )
    assert launched["train_config"]["task"] == variant["train_config"]["task"]
    train = variant["train_config"]
    for key in ("training_backend", "policy_model", "timesteps", "n_envs", "frame_skip"):
        assert train[key] == base["train_config"][key]
    assert train["task"]["model_inputs"] == base["train_config"]["task"]["model_inputs"]
    assert train["checkpoint_eval_backend"] == "none"
    assert launched["goal_variant"]["variant_id"] != base["goal_variant"]["variant_id"]
    assert launched["goal"]["train"]["environment"]["task"]["termination"]["success"] == [
        "two_walls_cleared"
    ]
    assert train["occupancy"]["domains"] == [list(range(8))]

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
    for walls, bricks, expected_reward, done in ((1, 108, 108, False), (2, 216, 128, True)):
        signals = {name: value.copy() for name, value in signals.items()}
        signals["walls_cleared"][:] = walls
        signals["bricks_destroyed"][:] = bricks
        result = kernel.process(np.array([999.0]), flags, flags, signals)
        np.testing.assert_allclose(result.rewards, [expected_reward])
        assert result.terminated.tolist() == [done]
        assert not result.truncated.any()
