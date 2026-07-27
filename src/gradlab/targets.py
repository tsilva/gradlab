from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar


def target_class_name_for_game(game: str) -> str:
    parts = re.findall(r"[A-Za-z0-9]+", game)
    return "".join(part[:1].upper() + part[1:] for part in parts) + "Target"


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
    eval_semantics: ClassVar[EvalSemantics] = EvalSemantics()


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
