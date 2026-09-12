import pytest
from stable_baselines3 import PPO

from gradlab.schedules import learning_rate_schedule
from gradlab.training.sb3 import A2C_DEFAULT_CONFIG, PPO_DEFAULT_CONFIG
from gradlab.training.sb3_on_policy import normalize_on_policy_config


def config(**overrides):
    return normalize_on_policy_config(
        {
            "learning_rate": 1e-3,
            "learning_rate_milestones": [
                {"step": 8, "value": 5e-4},
                {"step": 16, "value": 5e-4},
                {"step": 24, "value": 1e-4},
            ],
            **overrides,
        },
        defaults=PPO_DEFAULT_CONFIG,
        label="ppo",
    )


def test_absolute_milestones_interpolate_hold_and_ignore_changed_run_cap():
    for total in (32, 64):
        schedule = learning_rate_schedule({"timesteps": total}, config())
        for step, expected in [
            (0, 1e-3),
            (4, 7.5e-4),
            (8, 5e-4),
            (12, 5e-4),
            (20, 3e-4),
            (24, 1e-4),
            (32, 1e-4),
        ]:
            assert schedule(1 - step / total) == pytest.approx(expected)
        assert schedule(2) == 1e-3
        assert schedule(-1) == 1e-4


@pytest.mark.parametrize(
    "milestones",
    [
        [],
        "8:0.1",
        [{"step": 0, "value": 0.1}],
        [{"step": True, "value": 0.1}],
        [{"step": 1.5, "value": 0.1}],
        [{"step": 8, "value": 0.1}, {"step": 8, "value": 0.01}],
        [{"step": 8, "value": 0.1}, {"step": 4, "value": 0.01}],
        [{"step": 8, "value": float("nan")}],
        [{"step": 8, "value": float("inf")}],
        [{"step": 8, "value": -0.1}],
        [{"step": 8, "value": True}],
        [{"step": 8, "value": 0.1, "typo": 1}],
    ],
)
def test_invalid_milestones_fail_before_training(milestones):
    with pytest.raises(ValueError, match="learning_rate_milestones"):
        config(learning_rate_milestones=milestones)


@pytest.mark.parametrize(
    "overrides",
    [
        {"learning_rate_final": 1e-5},
        {"learning_rate_schedule_timesteps": 32},
    ],
)
def test_milestones_reject_ambiguous_linear_schedule(overrides):
    with pytest.raises(ValueError, match="cannot combine"):
        config(**overrides)


@pytest.mark.parametrize("defaults", [PPO_DEFAULT_CONFIG, A2C_DEFAULT_CONFIG])
def test_existing_constant_and_linear_schedules_are_preserved(defaults):
    backend = normalize_on_policy_config({}, defaults=defaults, label="policy")
    assert learning_rate_schedule({"timesteps": 100}, backend) == defaults["learning_rate"]
    backend = normalize_on_policy_config(
        {
            "learning_rate": 1e-3,
            "learning_rate_final": 1e-4,
            "learning_rate_schedule_timesteps": 50,
        },
        defaults=defaults,
        label="policy",
    )
    schedule = learning_rate_schedule({"timesteps": 100}, backend)
    assert schedule(0.75) == pytest.approx(5.5e-4)
    assert schedule(0.25) == pytest.approx(1e-4)


def test_fresh_ppo_applies_all_phases_without_restart():
    class RecordedPPO(PPO):
        def train(self):
            super().train()
            rates.append(self.policy.optimizer.param_groups[0]["lr"])

    rates = []
    model = RecordedPPO(
        "MlpPolicy",
        "CartPole-v1",
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        learning_rate=learning_rate_schedule({"timesteps": 32}, config()),
        policy_kwargs={"net_arch": [8]},
        device="cpu",
        seed=123,
    )
    try:
        model.learn(32)
        assert rates == pytest.approx([5e-4, 5e-4, 1e-4, 1e-4])
    finally:
        model.env.close()
