from __future__ import annotations

from collections.abc import Iterable, Mapping


PLAYER_PROCESSING_FEATURES = frozenset(
    {
        "actions",
        "attribution",
        "cnn-inspection",
        "critic-calibration",
        "events",
        "game",
        "history",
        "observation",
        "policy",
        "raw",
        "reward-accounting",
        "rewards",
        "signals",
    }
)


def normalize_player_processing(features: object) -> frozenset[str]:
    if not isinstance(features, Iterable) or isinstance(
        features, str | bytes | bytearray | Mapping
    ):
        return frozenset()
    return frozenset(
        str(feature) for feature in features if str(feature) in PLAYER_PROCESSING_FEATURES
    )


__all__ = ["PLAYER_PROCESSING_FEATURES", "normalize_player_processing"]
