from __future__ import annotations

import json
from pathlib import Path

import env_stableretro_turbo
import yaml

from gradlab.env import EnvConfig
from gradlab.env_providers import (
    _stable_retro_packaged_data_path,
    provider_native_vec_kwargs,
)


GOAL_PATH = Path("experiments/goals/Breakout-Atari2600-v0/_goal.yaml")
GAME = "Breakout-Atari2600-v0"


def test_breakout_goal_uses_pinned_stable_retro_data() -> None:
    goal = yaml.safe_load(GOAL_PATH.read_text(encoding="utf-8"))
    train_info = goal["train"]["environment"]["env_config"]["env_args"]["info"]
    assert train_info == "data"
    assert "eval" not in goal

    data_path = _stable_retro_packaged_data_path(GAME, f"{train_info}.json")
    assert data_path.is_file()
    assert data_path.is_relative_to(
        Path("src/gradlab/provider_data/stable_retro").resolve()
    )
    assert json.loads(data_path.read_text(encoding="utf-8")) == {
        "info": {
            "ball_y": {"address": 229, "type": "|u1"},
            "lives": {"address": 185, "type": "|u1"},
            "score": {"address": 204, "type": ">d2"},
        }
    }

    kwargs = provider_native_vec_kwargs(
        EnvConfig(
            env_provider="env-stableretro-turbo",
            game=GAME,
            state="Start",
            env_args={"info": train_info},
        ),
        n_envs=1,
        native_obs_crop=lambda _config: None,
        state_weight_mapping=lambda _config: {},
    )
    assert Path(kwargs["info"]) == data_path
    assert Path(kwargs["scenario"]).is_relative_to(
        Path(env_stableretro_turbo.__file__).resolve().parent
    )
