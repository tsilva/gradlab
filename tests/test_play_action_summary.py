from gradlab.play_action_summary import EpisodeActionSummary


def transition(step, action=1, *, episode=1, source="policy", effective=1):
    return {
        "episode": episode,
        "step": step,
        "sequence": episode * 1000 + step,
        "action_source": source,
        "decision": {"selected_action": action},
        "effective_action": effective,
    }


def summary():
    return EpisodeActionSummary({"policy": {"space": {"type": "discrete", "n": 2}}})


def test_counts_freeze_at_cursor_exclude_humans_and_reset_by_episode():
    counts = summary()
    first = transition(1, 0)
    counts.append(first)
    frozen = counts.payload(first)
    counts.append(first)  # Repeated publication must not double count.
    counts.append(transition(2, source="human", effective=0))
    counts.append(transition(3))
    end = counts.payload(transition(3))
    assert end["policy"]["counts"] == [1, 1]
    assert end["policy"]["population_count"] == 2
    assert end["environment"]["counts"] == [1, 2]
    assert frozen["policy"]["counts"] == [1, 0]
    assert counts.payload(first) is None
    counts.append(transition(1, episode=2))
    assert counts.payload(transition(1, episode=2))["policy"]["counts"] == [0, 1]


def test_missing_actions_and_gaps_do_not_fabricate_frequencies():
    counts = summary()
    counts.append(transition(1, None, effective=99))
    payload = counts.payload(transition(1))
    assert payload["policy"]["missing_count"] == 1
    assert payload["environment"]["unmappable_count"] == 1
    counts.append(transition(3))
    assert counts.payload(transition(3))["status"] == "partial-history"


def test_legal_tuples_and_nonzero_discrete_start():
    counts = EpisodeActionSummary(
        {"policy": {"space": {"type": "multi_discrete", "legal_tuples": [[0, 0], [1, 0]]}}}
    )
    counts.append(transition(1, 1, effective=[1, 0]))
    assert counts.payload(transition(1))["environment"]["counts"] == [0, 1]
    counts = EpisodeActionSummary({"policy": {"space": {"type": "discrete", "n": 2, "start": 4}}})
    counts.append(transition(1, 5, effective=[4]))
    assert counts.payload(transition(1))["policy"]["counts"] == [0, 1]
    assert counts.payload(transition(1))["environment"]["counts"] == [1, 0]


def test_live_and_imported_cursors_use_frozen_counts_after_eviction(tmp_path, monkeypatch):
    from dataclasses import replace
    import numpy as np
    from gradlab.play_trajectory import export_trajectory
    from gradlab.play_trajectory_runner import TrajectoryPlaybackRunner
    from tests.test_play_trajectory import ScriptedSession, command, live_runner

    initialize, step = ScriptedSession.__init__, ScriptedSession.step

    def initialize_with_contract(self, length=3):
        initialize(self, length)
        self.action_contract_payload = {"policy": {"space": {"type": "discrete", "n": 2}}}

    def varied_step(self, **kwargs):
        result = step(self, **kwargs)
        action = np.array([0 if result.step <= 20 else 1])
        return replace(result, decision=replace(result.decision, raw_action=action))

    monkeypatch.setattr(ScriptedSession, "__init__", initialize_with_contract)
    monkeypatch.setattr(ScriptedSession, "step", varied_step)
    runner = live_runner(tmp_path, length=140)
    imported = None
    try:
        for _ in range(130):
            runner._step_once()
        runner.history.clear()
        latest = runner.snapshot()
        episode = runner.recording_status()["episode_id"]
        middle = runner.inspect_recorded_step(episode, 100)["snapshot"]["episode_actions"]
        assert middle["policy"]["counts"] == [20, 80]
        assert middle["environment"]["missing_count"] == 100
        first = runner.inspect_recorded_step(episode, 10)["snapshot"]["episode_actions"]
        assert first["policy"]["counts"] == [10, 0]
        assert runner.inspect_recorded_step(episode, 100)["snapshot"]["episode_actions"] == middle
        assert runner.snapshot() == latest
        assert runner.session.sequence == 130
        # Imported counts are rebuilt from evidence, never trusted from stored totals.
        monkeypatch.setattr(runner.episode_actions, "payload", lambda _: {"status": "bad"})
        runner._step_once()
        archive = export_trajectory(runner.freeze_trajectory(), tmp_path / "actions.trj")
        imported = TrajectoryPlaybackRunner(archive, runner.args)
        imported.start()
        for cursor, expected in ((100, [20, 80]), (10, [10, 0]), (131, [20, 111]), (100, [20, 80])):
            command(imported, "seek", step=cursor)
            totals = imported.snapshot()["episode_actions"]
            assert totals["policy"]["counts"] == expected
            assert totals["step"] == cursor
    finally:
        runner.stop()
        if imported is not None:
            imported.stop()
