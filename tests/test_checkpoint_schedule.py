from __future__ import annotations

from types import SimpleNamespace

import pytest

from gradlab.checkpoint_schedule import (
    CheckpointSchedule,
    checkpoint_plan,
    normalize_checkpoint_config,
)


def test_crossed_thresholds_coalesce_at_safe_boundaries_and_final_is_separate():
    schedule = CheckpointSchedule(
        dict(timesteps=48, checkpoint_freq=10, checkpoint_steps=[5, 7, 17, 48])
    )
    assert schedule.due(16)
    assert not schedule.due(16)
    assert schedule.due(32)
    assert not schedule.due(48)
    assert not schedule.due(64)


def test_resume_uses_absolute_cadence_and_preserves_interruption_after_last_milestone():
    schedule = CheckpointSchedule(
        dict(timesteps=100, checkpoint_freq=20, checkpoint_steps=[10, 35]), initial_step=32
    )
    assert not schedule.due(32)
    assert schedule.due(40)
    assert schedule.due(64)
    milestone_only = CheckpointSchedule(
        dict(timesteps=100, checkpoint_freq=0, checkpoint_steps=[10, 20]), initial_step=32
    )
    assert not milestone_only.due(40)
    assert milestone_only.enabled


@pytest.mark.parametrize("steps", [[True], [-1], [0], [1.5], [8, 8], [16, 8], "8", None])
def test_milestones_reject_malformed_values(steps):
    with pytest.raises(ValueError, match="checkpoint_steps"):
        normalize_checkpoint_config(dict(checkpoint_steps=steps))


def candidate_config(**overrides):
    config = dict(
        timesteps=1000,
        training_backend=dict(id="sb3.ppo"),
        checkpoint_candidates=dict(
            metric="train/return/mean",
            min_delta=10,
            delta_mode="absolute",
            min_interval_steps=20,
            max_checkpoints=2,
            **overrides,
        ),
    )
    normalize_checkpoint_config(config)
    return config


def test_candidate_shortlist_requires_fresh_meaningful_improvements_and_is_bounded():
    schedule = CheckpointSchedule(candidate_config())
    assert not schedule.candidate_due(20, (100, 20))  # Establish the baseline.
    assert not schedule.candidate_due(24, (109, 24))
    assert schedule.candidate_due(40, (110, 40))
    assert not schedule.candidate_due(60, (110, 40))  # Replayed metric row.
    assert not schedule.candidate_due(64, (float("nan"), 64))
    assert not schedule.candidate_due(64, (120, 80))  # Future evidence.
    assert not schedule.candidate_due(44, (120, 44))  # Cooldown.
    assert schedule.candidate_due(60, (120, 60))
    assert not schedule.candidate_due(80, (200, 80))  # Exhausted budget.
    assert not schedule.candidate_due(1000, (300, 1000))  # Final owns the cap.


def test_negative_return_minimize_and_relative_delta():
    config = candidate_config(direction="minimize")
    config["checkpoint_candidates"].update(delta_mode="relative", min_delta=0.1)
    schedule = CheckpointSchedule(config)
    assert not schedule.candidate_due(20, (-100, 20))
    assert not schedule.candidate_due(40, (-109, 40))
    assert schedule.candidate_due(60, (-110, 60))


@pytest.mark.parametrize(
    "change",
    [
        dict(metric="eval/return/mean"),
        dict(metric="train/invented"),
        dict(direction="sideways"),
        dict(delta_mode="percent"),
        dict(min_delta=0),
        dict(min_delta=float("inf")),
        dict(min_interval_steps=0),
        dict(max_checkpoints=True),
        dict(max_checkpoints=17),
        dict(invented=1),
    ],
)
def test_candidate_contract_rejects_invalid_settings(change):
    config = dict(
        training_backend=dict(id="sb3.ppo"),
        checkpoint_candidates={"metric": "train/return/mean", **change},
    )
    with pytest.raises(ValueError):
        normalize_checkpoint_config(config)


def test_candidate_contract_rejects_search_backend():
    with pytest.raises(ValueError, match="learned PPO/A2C"):
        normalize_checkpoint_config(
            dict(
                training_backend=dict(id="gradlab.go-explore"),
                checkpoint_candidates=dict(metric="train/return/mean"),
            )
        )


def test_plan_includes_branch_points_deduplicates_final_and_budgets_candidates():
    config = candidate_config()
    config.update(
        timesteps=100,
        checkpoint_freq=25,
        checkpoint_steps=[10, 25, 100, 200],
        training_backend=dict(id="gradlab.ppo", config=dict(checkpoint_update_steps=[50])),
    )
    assert checkpoint_plan(config) == ([10, 25, 50, 75, 100], 2)


@pytest.mark.parametrize("algorithm", ["ppo", "a2c"])
def test_real_sb3_updates_save_updated_policies_once_and_exclude_runtime_hooks(
    algorithm,
    monkeypatch,
    tmp_path,
):
    import torch
    from stable_baselines3 import A2C, PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.logger import configure
    from gradlab.training import sb3_on_policy
    from gradlab.training.sb3_helpers import (
        GracefulStopHelper,
        install_on_policy_safe_boundary_stop,
    )
    from gradlab.training_backend import GracefulStopFlag

    model_class = PPO if algorithm == "ppo" else A2C
    env = make_vec_env("CartPole-v1", n_envs=2, seed=7)
    kwargs = dict(batch_size=16, n_epochs=1) if algorithm == "ppo" else {}
    model = model_class("MlpPolicy", env, n_steps=8, seed=7, device="cpu", **kwargs)
    model.set_logger(configure(folder=str(tmp_path), format_strings=[]))
    initial = next(model.policy.parameters()).detach().clone()
    context = SimpleNamespace(
        run_dir=tmp_path,
        checkpoint_dir=tmp_path / "checkpoints",
        environment=SimpleNamespace(game="CartPole-v1"),
        train_config=dict(timesteps=48, checkpoint_freq=10, checkpoint_steps=[5, 7, 17, 48]),
        stop_flag=GracefulStopFlag(),
        session=SimpleNamespace(checkpoints=SimpleNamespace(persist_intermediate=True)),
    )
    saved = []

    def save(**kwargs):
        saved.append(
            (kwargs["step"], model._n_updates, next(model.policy.parameters()).detach().clone())
        )

    monkeypatch.setattr(sb3_on_policy, "save_model_bundle", save)
    writer = sb3_on_policy.LearnedPolicyCheckpoints(model, context, algorithm_id=algorithm)
    install_on_policy_safe_boundary_stop(
        model, graceful_stop=GracefulStopHelper(context.stop_flag), after_update=writer.after_update
    )
    try:
        model.learn(total_timesteps=48)
        assert [s[0] for s in saved] == [16, 32]
        assert all(s[1] > 0 and not torch.equal(s[2], initial) for s in saved)
        model.save(tmp_path / "final.zip")
        restored = model_class.load(tmp_path / "final.zip", device="cpu")
        assert not hasattr(restored, "_gradlab_after_update")
        assert not hasattr(restored, "_gradlab_graceful_stop")
        context.stop_flag.request("cancel")
        model.num_timesteps = 40
        writer.after_update()
        assert len(saved) == 2
    finally:
        env.close()


def test_monitoring_binding_changes_when_milestones_or_candidate_budget_changes():
    from gradlab.monitor_config import calibration_binding

    config = dict(timesteps=1000, checkpoint_freq=100)
    original = calibration_binding(config, {})
    assert calibration_binding({**config, "checkpoint_steps": [50]}, {}) != original
    assert (
        calibration_binding({**config, "checkpoint_candidates": {"max_checkpoints": 3}}, {})
        != original
    )


def test_writer_uses_durable_proxy_samples_coalesces_candidate_and_periodic_saves(
    monkeypatch,
    tmp_path,
):
    from gradlab.metric_store import MetricStore, metric_store_path
    from gradlab.training import sb3_on_policy
    from gradlab.training_backend import GracefulStopFlag

    config = candidate_config()
    config["checkpoint_freq"] = 100
    model = SimpleNamespace(num_timesteps=0)
    context = SimpleNamespace(
        run_dir=tmp_path,
        checkpoint_dir=tmp_path / "checkpoints",
        environment=SimpleNamespace(game="CartPole-v1"),
        train_config=config,
        stop_flag=GracefulStopFlag(),
        session=SimpleNamespace(
            checkpoints=SimpleNamespace(persist_intermediate=True), event=lambda message: None
        ),
    )
    store = MetricStore(metric_store_path(tmp_path))
    store.init()
    saved = []
    monkeypatch.setattr(sb3_on_policy, "save_model_bundle", lambda **kw: saved.append(kw["step"]))
    writer = sb3_on_policy.LearnedPolicyCheckpoints(model, context, algorithm_id="ppo")
    for step, value in ((20, 100), (40, 120), (100, 140), (120, 300)):
        store.append_metrics({"train/return/mean": value}, step=step, source="train", publish=False)
        model.num_timesteps = step
        writer.after_update()
        writer.after_update()  # Replayed calls must not repeat the immutable save.
    assert saved == [40, 100]
    assert writer.schedule.candidate_count == 2
