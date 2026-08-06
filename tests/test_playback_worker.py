from __future__ import annotations

from argparse import Namespace

import pytest

from gradlab.playback_worker import IsolatedPlaybackHost
from gradlab.play_web import PROTOCOL_VERSION


def test_isolated_playback_worker_starts_without_a_source() -> None:
    host = IsolatedPlaybackHost(
        Namespace(),
        argv=[],
        explicit_seed=False,
    )
    try:
        host.start()
        snapshot = host.snapshot()
        assert snapshot["protocol"] == PROTOCOL_VERSION
        assert snapshot["session"]["reward_accounting"]["status"] == "unavailable"
        assert snapshot["app"]["phase"] == "selecting"
        assert snapshot["app"]["source"] is None
        assert host.active_publication_context() is None
        assert (
            host.encoder.retained(
                0,
                epoch=0,
                timeout=0.0,
                kinds={1, 2},
            )
            == {}
        )
    finally:
        host.stop()


def test_isolated_playback_worker_rejects_protected_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "must-not-cross")
    host = IsolatedPlaybackHost(
        Namespace(),
        argv=[],
        explicit_seed=False,
    )
    try:
        with pytest.raises(RuntimeError, match="protected environment"):
            host.start()
    finally:
        host.stop()
