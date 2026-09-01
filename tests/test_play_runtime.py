from __future__ import annotations

from gradlab.environment_fields import EnvConfig
from gradlab.play_runtime import _with_playback_device_override


def test_with_playback_device_override_enforces_cpu_for_gradoom() -> None:
    config = EnvConfig(
        env_provider="env-gradoom-turbo-torch",
        game="VizdoomDeathmatch-v1",
        env_args={},
    )

    patched = _with_playback_device_override(config)

    assert patched == {"device": "cpu"}
    assert config.env_args == {}


def test_with_playback_device_override_only_applies_to_gradoom() -> None:
    config = EnvConfig(
        env_provider="env-vizdoom-turbo",
        game="VizdoomDeathmatch-v1",
        env_args={},
    )

    patched = _with_playback_device_override(config)

    assert patched == {}
    assert config.env_args == {}
