"""Reward prefixes preserve signed activity independently of inspection windows."""

from gradlab.play_reward_summary import EpisodeRewardSummary


CONTRACT = {"status": "available", "reward_scale": 0.5, "clip_bounds": [-1, 1]}


def transition(step, raw, final, total, *, episode=1, components=None):
    return {
        "episode": episode,
        "step": step,
        "sequence": step + episode * 100,
        "reward": {
            "raw": raw,
            "shaped": final,
            "return": total,
            "components": components or {"native_reward": raw},
        },
    }


def test_prefixes_preserve_cancellation_clipping_and_are_independent_copies():
    accumulator = EpisodeRewardSummary()
    first = transition(1, 6, 1, 1)
    accumulator.append(first, CONTRACT)
    frozen = accumulator.payload(first)
    second = transition(2, -6, -1, 0)
    accumulator.append(second, CONTRACT)
    total = accumulator.payload(second)
    assert frozen["final"] == 1
    assert frozen["entries"]["native_reward"]["impact"] == 3
    assert total["final"] == 0
    assert total["entries"]["native_reward"] == {"raw": 0, "impact": 0, "magnitude": 6}
    assert total["entries"]["clip_adjustment"] == {"raw": None, "impact": 0, "magnitude": 4}
    accumulator.append(second, CONTRACT)
    assert accumulator.payload(second) == total
    assert accumulator.payload(first) is None
    reset = transition(1, 2, 1, 1, episode=2)
    accumulator.append(reset, CONTRACT)
    assert accumulator.payload(reset)["final"] == 1
    assert accumulator.payload(reset)["entries"]["native_reward"]["magnitude"] == 1


def test_missing_prefix_and_invalid_reward_never_become_valid_later():
    accumulator = EpisodeRewardSummary()
    second = transition(2, 2, 1, 2)
    accumulator.append(second, CONTRACT)
    assert accumulator.payload(second)["status"] == "partial-history"
    third = transition(3, 2, 1, 3)
    accumulator.append(third, CONTRACT)
    assert accumulator.payload(third)["status"] == "partial-history"
    bad = transition(1, 6, 3, 3, episode=2)
    accumulator.append(bad, CONTRACT)
    assert accumulator.payload(bad)["status"] == "protocol-error"
    assert "scale-then-clip" in accumulator.payload(bad)["message"]


def test_bad_episode_return_and_missing_raw_evidence_are_explicit():
    for raw, total in ((None, 1), (2, 99)):
        accumulator = EpisodeRewardSummary()
        row = transition(1, raw, 1, total)
        accumulator.append(row, CONTRACT)
        assert accumulator.payload(row)["status"] == "protocol-error"
