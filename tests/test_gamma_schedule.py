from types import SimpleNamespace

import numpy as np
import pytest
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import BaseCallback

from gradlab.policy_bundle import _derive_critic_value_contract
from gradlab.schedules import GammaScheduleHelper, apply_rollout_gamma, scheduled_scalar
from gradlab.training.sb3 import PPO_DEFAULT_CONFIG, normalize_on_policy_config


def test_schedule_endpoints_resume_and_constant():
    assert scheduled_scalar(0.9, 0.99, 0, 100) == 0.9
    assert scheduled_scalar(0.9, 0.99, 50, 100) == pytest.approx(0.945)
    assert scheduled_scalar(0.9, 0.99, 200, 100) == 0.99
    assert scheduled_scalar(0.9, None, 200, 0) == 0.9
    model = SimpleNamespace(num_timesteps=50, rollout_buffer=SimpleNamespace(gamma=0.5))
    config = dict(gamma=0.9, gamma_final=0.99, gamma_schedule_timesteps=0)
    assert apply_rollout_gamma(model, config, 100) == pytest.approx(0.945)
    assert model.rollout_buffer.gamma == model.gamma


@pytest.mark.parametrize(
    "values",
    [
        dict(gamma_final=float("nan")),
        dict(gamma_final=1.1),
        dict(gamma_final=True),
        dict(gamma=None),
        dict(gamma_schedule_timesteps=-1),
        dict(gamma_schedule_timesteps=10),
    ],
)
def test_invalid_gamma_config(values):
    with pytest.raises(ValueError):
        normalize_on_policy_config(values, defaults=PPO_DEFAULT_CONFIG, label="ppo")


def test_scheduled_critic_preserves_schedule_without_claiming_stationary_discount():
    config = dict(
        timesteps=100,
        training_backend=dict(
            id="gradlab.ppo", config=dict(gamma=0.9, gamma_final=0.99, gamma_schedule_timesteps=0)
        ),
    )
    contract = _derive_critic_value_contract(config, policy_environment_hash="hash")
    assert contract["discount"] is None
    assert contract["discount_schedule"] == dict(initial=0.9, final=0.99, timesteps=100)
    config["training_backend"]["config"]["gamma_final"] = None
    contract = _derive_critic_value_contract(config, policy_environment_hash="hash")
    assert contract["discount"] == 0.9
    assert "discount_schedule" not in contract


class ScheduleCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.helper = GammaScheduleHelper(
            dict(gamma=0.9, gamma_final=0.99, gamma_schedule_timesteps=16), 32
        )
        self.gammas = []

    def _on_rollout_start(self):
        self.helper.bind(self)
        self.helper._on_rollout_start()
        self.gammas.append(self.model.gamma)

    def _on_step(self):
        assert self.model.gamma == self.gammas[-1]
        assert self.model.rollout_buffer.gamma == self.model.gamma
        return True

    def _on_rollout_end(self):
        self.helper.bind(self)
        self.helper._on_rollout_end()


@pytest.mark.parametrize("algorithm", [PPO, A2C])
def test_real_rollouts_and_checkpoint_resume(tmp_path, algorithm):
    model = algorithm(
        "MlpPolicy",
        "CartPole-v1",
        n_steps=8,
        **(dict(batch_size=8, n_epochs=1) if algorithm is PPO else {}),
        policy_kwargs=dict(net_arch=[8]),
        device="cpu",
        seed=1,
    )
    callback = ScheduleCallback()
    model.learn(16, callback=callback)
    np.testing.assert_allclose(callback.gammas, [0.9, 0.945])
    path = tmp_path / "policy.zip"
    model.save(path)
    loaded = algorithm.load(path, env=model.env)
    assert loaded.gamma == pytest.approx(0.945)
    assert loaded.num_timesteps == 16
    resumed = ScheduleCallback()
    loaded.learn(8, reset_num_timesteps=False, callback=resumed)
    assert resumed.gammas == [0.99]
    model.env.close()
