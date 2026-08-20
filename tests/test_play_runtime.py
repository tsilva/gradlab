from __future__ import annotations

from gradlab.environment_fields import EnvConfig
from gradlab.play_runtime import _with_playback_device_override


def test_with_playback_device_override_adds_gradoom_device(monkeypatch) -> None:
    monkeypatch.setattr("gradlab.play_runtime.resolve_sb3_device", lambda _value: "cpu")
    config = EnvConfig(
        env_provider="env-doom-turbo-torch",
        game="VizdoomDeathmatch-v1",
        env_args={},
    )

    patched = _with_playback_device_override(config, "auto")

    assert patched == {"device": "cpu"}
    assert config.env_args == {}


def test_with_playback_device_override_only_applies_to_gradoom(monkeypatch) -> None:
    monkeypatch.setattr("gradlab.play_runtime.resolve_sb3_device", lambda _value: "cpu")
    config = EnvConfig(
        env_provider="env-vizdoom-turbo",
        game="VizdoomDeathmatch-v1",
        env_args={},
    )

    patched = _with_playback_device_override(config, "auto")

    assert patched == {}
    assert config.env_args == {}
