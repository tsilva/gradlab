"""Verified native RGB and action-evidence contract."""

import importlib.metadata
from gradlab.json_utils import json_value

PROVIDER = "env-breakoutatari2600-turbo-native"
PROVIDER_VERSION = "0.5.13"


def verify_recording_provider(runtime) -> dict:
    from env_breakoutatari2600_turbo_native import BreakoutVecEnv
    from gradlab.env_providers import _StartInfoAdapter

    provider = runtime.provider
    native = provider.env if type(provider) is _StartInfoAdapter else provider
    if (
        runtime.descriptor.provider_id != PROVIDER
        or type(native) is not BreakoutVecEnv
        or importlib.metadata.version(PROVIDER) != PROVIDER_VERSION
    ):
        raise ValueError(
            "checkpoint monitoring requires the exact verified native Breakout provider"
        )
    table = runtime.descriptor.action_meanings
    expected = ("noop", "button", "right", "left")
    if (
        not table
        or any(name not in expected for name in table)
        or tuple(runtime.descriptor.action_table or ())
        != tuple(() if name == "noop" else (name.upper(),) for name in table)
    ):
        raise ValueError("checkpoint monitoring requires the verified four-action native encoding")
    # Rendering reads current unmasked native pixels; it does not change transition behavior.
    native.render_mode = "rgb_array"
    return {
        "provider": PROVIDER,
        "version": PROVIDER_VERSION,
        "execution_evidence": "verified-gradlab-submission-v1",
        "native_action_meanings": list(expected),
        "native_encoding": [expected.index(name) for name in table],
        "rgb_shape": [210, 160, 3],
        "source": "https://github.com/tsilva/env-BreakoutAtari2600-turbo-native/blob/v0.5.13/src/lib.rs",
        "action_contract": json_value(runtime.action_contract),
        "signal_metadata": json_value(native.signal_metadata),
    }
