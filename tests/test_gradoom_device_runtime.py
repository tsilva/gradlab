from types import SimpleNamespace

import numpy as np
import pytest
import torch
import gymnasium as gym

from gradlab.gradoom_device_runtime import GraDoomDeviceRuntime, _DeviceContextEncoder
from gradlab.task_kernels import Outcome


def _field(
    name: str,
    source_names: tuple[str, ...],
    *,
    encoding: str,
    scale: np.ndarray | None = None,
    offset: np.ndarray | None = None,
    low: np.ndarray | None = None,
    high: np.ndarray | None = None,
    clip: bool = False,
    categories: tuple[int, ...] = (),
    history: str | None = None,
    history_depth: int = 1,
):
    return SimpleNamespace(
        name=name,
        source_names=source_names,
        encoding=encoding,
        scale=scale,
        offset=offset,
        low=low,
        high=high,
        clip=clip,
        categories=categories,
        history=history,
        history_depth=history_depth,
    )


def test_device_context_encoder_keeps_context_on_the_observation_device() -> None:
    env = SimpleNamespace(device_signal_names=("health", "weapon", "ammo2", "ammo4"))
    kernel = SimpleNamespace(
        fields=(
            _field(
                "health",
                ("health",),
                encoding="continuous",
                scale=np.asarray([0.01], dtype=np.float32),
                offset=np.asarray([0.0], dtype=np.float32),
                low=np.asarray([0.0], dtype=np.float32),
                high=np.asarray([2.0], dtype=np.float32),
                clip=True,
            ),
            _field(
                "selected_weapon",
                ("weapon",),
                encoding="categorical",
                categories=(1, 2, 3),
            ),
            _field(
                "shared_bullets",
                ("ammo2", "ammo4"),
                encoding="continuous",
                scale=np.asarray([0.005, 0.005], dtype=np.float32),
                offset=np.asarray([0.0, 0.0], dtype=np.float32),
                low=np.asarray([0.0, 0.0], dtype=np.float32),
                high=np.asarray([1.0, 1.0], dtype=np.float32),
            ),
        )
    )
    observations = torch.zeros((2, 4, 84, 84), dtype=torch.uint8)
    signals = torch.tensor(
        [[250.0, 1.0, 50.0, 50.0], [75.0, 3.0, 200.0, 200.0]],
        dtype=torch.float32,
    )

    encoded = _DeviceContextEncoder(env, kernel, observations.device).encode(
        observations,
        signals,
    )

    assert tuple(encoded) == (
        "observation",
        "context/health",
        "context/selected_weapon",
        "context/shared_bullets",
    )
    assert all(value.device == observations.device for value in encoded.values())
    torch.testing.assert_close(encoded["context/health"], torch.tensor([[2.0], [0.75]]))
    torch.testing.assert_close(
        encoded["context/selected_weapon"],
        torch.tensor([0, 2], dtype=torch.int64),
    )
    torch.testing.assert_close(
        encoded["context/shared_bullets"],
        torch.tensor([[0.25, 0.25], [1.0, 1.0]]),
    )


def test_device_context_encoder_keeps_provider_histories_on_device() -> None:
    env = SimpleNamespace(
        device_signal_names=("health",),
        device_info_history_names=("health",),
    )
    field = _field(
        "health",
        ("health_frame_stack",),
        encoding="continuous",
        scale=np.asarray([0.01], dtype=np.float32),
        offset=np.asarray([0.0], dtype=np.float32),
        low=np.asarray([0.0], dtype=np.float32),
        high=np.asarray([2.0], dtype=np.float32),
        history="provider_frame_stack",
        history_depth=4,
    )
    histories = torch.tensor(
        [
            [[100.0, 90.0, 80.0, 70.0]],
            [[50.0, 40.0, 30.0, 20.0]],
        ]
    )

    encoded = _DeviceContextEncoder(
        env,
        SimpleNamespace(fields=(field,)),
        torch.device("cpu"),
    ).encode(torch.zeros((2, 4, 84, 84)), torch.zeros((2, 1)), histories)

    torch.testing.assert_close(
        encoded["context/health"],
        torch.tensor(
            [
                [[1.0], [0.9], [0.8], [0.7]],
                [[0.5], [0.4], [0.3], [0.2]],
            ]
        ),
    )


class _RecordProvider:
    num_envs = 2
    device = torch.device("cpu")
    device_signal_names = ("killcount",)

    def __init__(self) -> None:
        self.calls = 0

    def reset_device(self, mask, seeds):
        del mask, seeds
        return torch.zeros((2, 1)), torch.zeros((2, 1))

    def step_and_reset_device(self, actions, reset_seeds):
        del actions, reset_seeds
        self.calls += 1
        if self.calls == 1:
            terminated = torch.tensor([False, False])
            truncated = torch.tensor([False, False])
            rewards = torch.tensor([1.0, 2.0])
            kills = torch.tensor([[0.0], [1.0]])
        else:
            terminated = torch.tensor([True, False])
            truncated = torch.tensor([False, True])
            rewards = torch.tensor([3.0, 5.0])
            kills = torch.tensor([[2.0], [4.0]])
        return SimpleNamespace(
            observations=torch.zeros((2, 1)),
            signals=torch.zeros((2, 1)),
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            final_observations=torch.zeros((2, 1)),
            final_signals=kills,
        )

    def close(self):
        pass


def test_device_runtime_drains_episode_telemetry_once_per_rollout() -> None:
    provider = _RecordProvider()
    kernel = SimpleNamespace(
        fields=(),
        observation_space=gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
        action_space=gym.spaces.Discrete(2),
    )
    runtime = GraDoomDeviceRuntime(
        provider,
        descriptor=SimpleNamespace(),
        kernel=kernel,
        action_contract={"provider_id": "env-doom-turbo-torch"},
        run_seed=7,
    )
    runtime.reset()

    runtime.step(torch.zeros(2, dtype=torch.int64))
    runtime.step(torch.ones(2, dtype=torch.int64))
    records = runtime.drain_records()

    assert runtime.drain_records() == []
    assert len(records) == 2
    assert records[0].episode_return == pytest.approx(4.0)
    assert records[0].episode_length == 2
    assert records[0].outcome is Outcome.FAILURE
    assert records[0].events == ("player_died",)
    assert records[0].metrics == {"kills": 2.0}
    assert records[1].episode_return == pytest.approx(7.0)
    assert records[1].outcome is Outcome.SUCCESS
    assert records[1].events == ("time_limit_reached",)
    assert records[1].metrics == {"kills": 4.0}
