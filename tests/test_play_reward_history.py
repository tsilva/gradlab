import pytest
from tests.test_play_trajectory import live_runner


def test_exact_discounted_future_and_live_cursor(tmp_path):
    runner = live_runner(tmp_path, length=140)
    try:
        for _ in range(140):
            runner._step_once()
        episode = runner.recording_status()["episode_id"]
        result = runner.reward_history(episode, 2)
        assert result["complete"]
        assert result["reward_sum"] == 139 * 0.5
        assert result["return_total"] == pytest.approx(sum(0.5 * 0.9**k for k in range(139)))
        assert result["points"][0]["weight"] == 1
        assert len(result["points"]) == 100
        page = runner.reward_history(episode, 2, result["next_last"])
        assert len(page["points"]) == 39
        assert page["points"][0]["offset"] == 100
        assert page["return_total"] == result["return_total"]
        assert runner.snapshot()["transition"]["step"] == 140
        with pytest.raises(ValueError, match="replaced"):
            runner.reward_history("old", 1)
    finally:
        runner.stop()


def test_incomplete_future_is_not_full_return(tmp_path):
    runner = live_runner(tmp_path, length=3)
    try:
        runner._step_once()
        result = runner.reward_history(runner.recording_status()["episode_id"], 1)
        assert not result["complete"]
        assert result["return_total"] is None
        assert result["discounted_reward_sum"] == 0.5
    finally:
        runner.stop()


@pytest.mark.parametrize("gamma", [0.0, 0.9, 1.0])
def test_signed_rewards_and_truncation_bootstrap(tmp_path, gamma):
    from copy import deepcopy
    from types import SimpleNamespace
    from gradlab.play_reward_history import reward_history

    runner = live_runner(tmp_path, length=3)
    try:
        runner._step_once()
        original = runner.snapshot()["transition"]

        def read(step):
            transition = deepcopy(original)
            transition.update(
                step=step, sequence=step, terminated=False, truncated=step == 3, boundary=step == 3
            )
            transition["reward"]["shaped"] = [1, 0, -2][step - 1]
            transition["return_bootstrap"] = {"source": "terminal_state_value", "value": 4}
            return {"inspection_snapshot": {"transition": transition}}

        recording = SimpleNamespace(
            root=tmp_path,
            metadata={"episode_id": "test", "first_step": 1, "discount": gamma},
            status=lambda: {"last_step": 3},
            transition=read,
        )
        result = reward_history(SimpleNamespace(recording=recording), "test", 1)
        assert [p["step"] for p in result["points"]] == [1, 3]
        assert result["discounted_reward_sum"] == pytest.approx(1 - 2 * gamma**2)
        assert result["bootstrap"]["contribution"] == pytest.approx(4 * gamma**3)
        assert result["return_total"] == pytest.approx(1 - 2 * gamma**2 + 4 * gamma**3)
        assert not result["complete"]
    finally:
        runner.stop()


def test_reward_history_worker_epoch_and_import(tmp_path, monkeypatch):
    from argparse import Namespace
    import gradlab.playback_worker as worker
    from tests.test_playback_episode_history import _episode_inspection_worker

    monkeypatch.setattr(worker, "_worker_main", _episode_inspection_worker)
    host = worker.IsolatedPlaybackHost(
        Namespace(fps=30, fixture_root=str(tmp_path)), argv=[], explicit_seed=False
    )
    try:
        host.start()
        snapshot = host.snapshot()
        epoch = snapshot["session_epoch"]
        episode = snapshot["trajectory"]["episode_id"]
        result = host.reward_history(epoch, episode, 1)
        assert result["discounted_reward_sum"] > 0
        with pytest.raises(RuntimeError, match="replaced"):
            host.reward_history(epoch + 1, episode, 1)
    finally:
        host.stop()
