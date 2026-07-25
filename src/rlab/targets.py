from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from rlab.action_contract import MARIO_ACTION_TABLES


def target_class_name_for_game(game: str) -> str:
    parts = re.findall(r"[A-Za-z0-9]+", game)
    return "".join(part[:1].upper() + part[1:] for part in parts) + "Target"


def _button_mask(size: int, *buttons: int) -> np.ndarray:
    mask = np.zeros(size, dtype=np.int8)
    for button in buttons:
        mask[button] = 1
    return mask


_MARIO_BUTTON_INDICES = {
    name: index
    for index, name in enumerate(("B", None, "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A"))
    if name is not None
}


def _mario_action_name(buttons: tuple[str, ...]) -> str:
    return "noop" if not buttons else "_".join(button.lower() for button in buttons)


@dataclass(frozen=True)
class EvalProgressField:
    info_key: str
    result_key: str
    rank: bool = False


@dataclass(frozen=True)
class EvalSemantics:
    completion_reason: str | None = None
    completion_info_keys: tuple[str, ...] = ()
    completion_blocking_info_keys: tuple[str, ...] = ()
    progress_fields: tuple[EvalProgressField, ...] = ()
    death_flag_key: str | None = None
    death_position_key: str | None = None
    best_episode_rank: tuple[str, ...] = ("reward",)


class RetroTarget:
    game: ClassVar[str] = ""
    default_state: ClassVar[str] = ""
    default_hud_crop_top: ClassVar[int] = 0
    action_library: ClassVar[dict[str, np.ndarray]] = {}
    action_sets: ClassVar[dict[str, tuple[str, ...]]] = {}
    eval_semantics: ClassVar[EvalSemantics] = EvalSemantics()

    @classmethod
    def action_names_for_set(cls, action_set: str) -> tuple[str, ...]:
        if action_set == "native":
            return ()
        if action_set not in cls.action_sets:
            valid = ", ".join(sorted(cls.action_sets)) or "native"
            raise ValueError(f"unknown action_set {action_set!r} for {cls.game}; valid: {valid}")
        return cls.action_sets[action_set]

    @classmethod
    def action_masks_for_set(cls, action_set: str) -> tuple[np.ndarray, ...]:
        return tuple(cls.action_library[name] for name in cls.action_names_for_set(action_set))


class GenericRetroTarget(RetroTarget):
    pass


class SuperMarioBrosNesV0Target(RetroTarget):
    game = "SuperMarioBros-Nes-v0"
    default_state = "Level1-1"
    default_hud_crop_top = 32
    eval_semantics = EvalSemantics(
        completion_reason="level_change",
        completion_info_keys=("completion_event", "level_complete"),
        completion_blocking_info_keys=("died", "life_loss"),
        progress_fields=(
            EvalProgressField("max_x_pos", "max_x_pos", rank=True),
            EvalProgressField("level_max_x_pos", "max_level_x_pos"),
        ),
        death_flag_key="died",
        death_position_key="death_x_pos",
        best_episode_rank=("completion", "progress", "reward"),
    )

    action_library = {
        _mario_action_name(buttons): _button_mask(
            max(_MARIO_BUTTON_INDICES.values()) + 1,
            *(_MARIO_BUTTON_INDICES[button] for button in buttons),
        )
        for table in MARIO_ACTION_TABLES.values()
        for buttons in table
    }
    action_sets = {
        name: tuple(_mario_action_name(buttons) for buttons in table)
        for name, table in MARIO_ACTION_TABLES.items()
    }


class SuperMarioBros3NesV0Target(RetroTarget):
    game = "SuperMarioBros3-Nes-v0"
    default_state = "1Player.World1.Level1"


TARGETS: dict[str, type[RetroTarget]] = {
    SuperMarioBrosNesV0Target.game: SuperMarioBrosNesV0Target,
    SuperMarioBros3NesV0Target.game: SuperMarioBros3NesV0Target,
}


def target_for_game(game: str) -> type[RetroTarget]:
    if game not in TARGETS:
        TARGETS[game] = type(
            target_class_name_for_game(game),
            (GenericRetroTarget,),
            {"game": game, "__module__": __name__},
        )
    return TARGETS[game]
