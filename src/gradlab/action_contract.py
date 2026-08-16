"""Resolve provider-owned actions and validate persisted action semantics."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import importlib.resources
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from gradlab.action_codecs import (
    LEGAL_TUPLE_DISTRIBUTION,
    LEGAL_TUPLE_SCORING,
    LegalTupleMultiDiscrete,
    VIZDOOM_DEATHMATCH_MULTIDISCRETE_CODEC,
    VIZDOOM_SHARED_MULTIDISCRETE_CODEC,
    vizdoom_deathmatch_multidiscrete_codec_document,
    vizdoom_deathmatch_multidiscrete_semantics,
    vizdoom_shared_multidiscrete_codec_document,
    vizdoom_shared_multidiscrete_semantics,
)


MARIO_PROVIDERS = frozenset({"stable-retro-turbo", "supermariobrosnes-turbo"})
BUILTIN_ACTION_MODES = frozenset({"all", "filtered", "discrete", "multi_discrete"})
ACTION_CONTRACT_SCHEMA_VERSION = 1
MARIO_ACTION_TABLES = {
    "basic": (
        (),
        ("RIGHT",),
        ("RIGHT", "B"),
        ("RIGHT", "A"),
        ("RIGHT", "A", "B"),
        ("A",),
        ("LEFT",),
    ),
    "standard": (
        (),
        ("RIGHT",),
        ("RIGHT", "B"),
        ("RIGHT", "A"),
        ("RIGHT", "A", "B"),
        ("A",),
        ("LEFT",),
        ("DOWN",),
    ),
    "basic-start": (
        (),
        ("RIGHT",),
        ("RIGHT", "B"),
        ("RIGHT", "A"),
        ("RIGHT", "A", "B"),
        ("A",),
        ("LEFT",),
        ("START",),
    ),
    "right-jump": (
        ("RIGHT",),
        ("RIGHT", "B"),
        ("RIGHT", "A"),
        ("RIGHT", "A", "B"),
    ),
}
BREAKOUT_ACTION_TABLES = {
    "simple": (
        (),
        ("BUTTON",),
        ("RIGHT",),
        ("LEFT",),
    ),
}
GRADOOM_DEATHMATCH_BUTTONS = (
    "ATTACK",
    "SPEED",
    "STRAFE",
    "MOVE_RIGHT",
    "MOVE_LEFT",
    "MOVE_BACKWARD",
    "MOVE_FORWARD",
    "TURN_RIGHT",
    "TURN_LEFT",
    "SELECT_WEAPON1",
    "SELECT_WEAPON2",
    "SELECT_WEAPON3",
    "SELECT_WEAPON4",
    "SELECT_WEAPON5",
    "SELECT_WEAPON6",
    "SELECT_NEXT_WEAPON",
    "SELECT_PREV_WEAPON",
    "LOOK_UP_DOWN_DELTA",
    "TURN_LEFT_RIGHT_DELTA",
    "MOVE_LEFT_RIGHT_DELTA",
)


def _mode_name(value: Any) -> str:
    name = getattr(value, "name", value)
    return str(name).split(".")[-1].strip().casefold()


def _packaged_action_sets(provider_id: str, game: str) -> Mapping[str, Any]:
    # Stable Retro does not package the dedicated Mario runtime's named action
    # sets, so GradLab supplies the same current semantic tables for that provider.
    if game == "SuperMarioBros-Nes-v0" and provider_id == "stable-retro-turbo":
        return MARIO_ACTION_TABLES
    # Original Stable Retro does not declare Breakout's provider-neutral
    # four-action table. GradLab owns this cross-provider contract and passes
    # the exact same table directly to both implementations.
    if game == "Breakout-Atari2600-v0" and provider_id == "stable-retro-turbo":
        return BREAKOUT_ACTION_TABLES
    if provider_id == "stable-retro-turbo":
        import stable_retro

        path = (
            Path(stable_retro.__file__).resolve().parent
            / "data"
            / "stable"
            / game
            / "metadata.json"
        )
        metadata = json.loads(path.read_text(encoding="utf-8"))
    else:
        package = {
            "supermariobrosnes-turbo": "supermariobrosnes_turbo",
            "breakout-turbo-env": "breakout_turbo_env",
        }.get(provider_id)
        if package is None:
            return {}
        path = importlib.resources.files(package).joinpath("data", game, "metadata.json")
        metadata = json.loads(path.read_text(encoding="utf-8"))
    action_sets = metadata.get("action_sets", {})
    return action_sets if isinstance(action_sets, Mapping) else {}


def _gradoom_scenario_buttons(
    game: str | Path | None,
    *,
    scenario: str | Path | None,
) -> tuple[str, ...]:
    """Resolve the pinned GraDOOM profile without importing its CUDA runtime."""

    requested = scenario if scenario not in (None, "scenario") else game
    if requested is None:
        return GRADOOM_DEATHMATCH_BUTTONS
    candidate = Path(str(requested)).expanduser()
    if candidate.is_file():
        if candidate.suffix.casefold() not in {".cfg", ".wad"}:
            raise ValueError("GraDOOM scenarios must be a ViZDoom .cfg or Doom .wad file")
        return GRADOOM_DEATHMATCH_BUTTONS
    alias = str(requested).strip().casefold().removesuffix(".cfg")
    if alias not in {"deathmatch", "vizdoomdeathmatch-v1"}:
        raise ValueError(
            f"unsupported GraDOOM game/scenario {requested!r}; "
            "only VizdoomDeathmatch-v1 is supported"
        )
    return GRADOOM_DEATHMATCH_BUTTONS


def provider_buttons(
    provider_id: str,
    game: str,
    *,
    env_args: Mapping[str, Any] | None = None,
) -> tuple[str | None, ...]:
    if provider_id in {"gradoom", "vizdoom-turbo"}:
        args = env_args if isinstance(env_args, Mapping) else {}
        vizdoom_config = args.get("vizdoom_config")
        if isinstance(vizdoom_config, Mapping) and "available_buttons" in vizdoom_config:
            raw_buttons = vizdoom_config["available_buttons"]
            if isinstance(raw_buttons, (str, bytes, bytearray)) or not isinstance(
                raw_buttons, list | tuple
            ):
                raise ValueError("env_args.vizdoom_config.available_buttons must be a button list")
            buttons = tuple(
                str(getattr(button, "name", button)).split(".")[-1].strip().upper()
                for button in raw_buttons
            )
            if not buttons or any(not button for button in buttons):
                raise ValueError("env_args.vizdoom_config.available_buttons must not be empty")
            if len(set(buttons)) != len(buttons):
                raise ValueError("env_args.vizdoom_config.available_buttons cannot repeat a button")
            return buttons
        if provider_id == "gradoom":
            return _gradoom_scenario_buttons(game, scenario=args.get("scenario"))
        from vizdoom_turbo import scenario_buttons

        return scenario_buttons(game, scenario=args.get("scenario"))
    if provider_id == "supermariobrosnes-turbo":
        from supermariobrosnes_turbo import NES_BUTTONS

        return tuple(NES_BUTTONS)
    if provider_id == "breakout-turbo-env":
        from breakout_turbo_env import BUTTONS

        return tuple(BUTTONS)
    if provider_id == "stable-retro-turbo":
        import stable_retro

        parts = game.rsplit("-", 2)
        if len(parts) != 3:
            raise ValueError(f"cannot infer Stable Retro system from game id {game!r}")
        return tuple(stable_retro.get_system_info(parts[-2])["buttons"])
    return ()


def declared_action_contract(config: Any) -> dict[str, Any] | None:
    """Resolve a config's provider-owned exact action table for provenance checks."""
    provider_id = str(
        config.get("env_provider", "stable-retro-turbo")
        if isinstance(config, Mapping)
        else getattr(config, "env_provider", "stable-retro-turbo")
    )
    game = str(
        config.get("game", "") if isinstance(config, Mapping) else getattr(config, "game", "")
    )
    env_args = (
        config.get("env_args", {})
        if isinstance(config, Mapping)
        else getattr(config, "env_args", {})
    )
    if not isinstance(env_args, Mapping):
        env_args = {}
    request = env_args.get("use_restricted_actions")
    if request is None:
        return None
    request_name = _mode_name(request)
    if provider_id == "vizdoom-turbo" and request_name not in {
        "all",
        "filtered",
        "multi_discrete",
    }:
        from vizdoom_turbo.action_tables import resolve_custom_action

        resolved = resolve_custom_action(
            request,
            buttons=tuple(
                str(button)
                for button in provider_buttons(provider_id, game, env_args=env_args)
                if button is not None
            ),
        )
        return {
            "mode": "custom_discrete",
            "preset": resolved.preset,
            "table": [list(action) for action in resolved.table],
            "meanings": list(resolved.meanings),
            "table_hash": resolved.table_hash,
        }
    if request_name in BUILTIN_ACTION_MODES:
        return {
            "mode": request_name,
            "preset": None,
            "table": None,
            "meanings": None,
            "table_hash": None,
        }
    preset = None
    table = request
    if isinstance(request, str):
        action_sets = _packaged_action_sets(provider_id, game)
        matches = {str(name).casefold(): (str(name), value) for name, value in action_sets.items()}
        try:
            preset, table = matches[request_name]
        except KeyError as exc:
            raise ValueError(
                f"unknown action_set {request!r}: provider {provider_id!r} "
                f"has no matching preset for {game!r}"
            ) from exc
    if (
        isinstance(table, (str, bytes, bytearray))
        or not isinstance(table, list | tuple)
        or not table
    ):
        raise ValueError("custom use_restricted_actions must be a non-empty action table")
    buttons = provider_buttons(provider_id, game, env_args=env_args)
    button_to_index = {name: index for index, name in enumerate(buttons) if name is not None}
    players = int(env_args.get("players", 1))
    if players <= 0:
        raise ValueError("env_args.players must be positive")
    normalized: list[Any] = []
    meanings: list[str] = []
    masks: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()

    def player_action(raw_player_action: Any) -> tuple[list[str], int, str]:
        if isinstance(raw_player_action, (str, bytes, bytearray)) or not isinstance(
            raw_player_action, list | tuple
        ):
            raise ValueError("custom action entries must be button-label lists")
        if not all(isinstance(label, str) for label in raw_player_action):
            raise ValueError("custom action-table button labels must be strings")
        labels = list(raw_player_action)
        if len(set(labels)) != len(labels):
            raise ValueError("custom action table entries cannot repeat a button")
        try:
            mask = sum(1 << button_to_index[label] for label in labels)
        except KeyError as exc:
            raise ValueError(f"unknown action-table button {exc.args[0]!r}") from exc
        meaning = "noop" if not labels else "_".join(label.lower() for label in labels)
        return labels, mask, meaning

    for raw_action in table:
        if players == 1:
            labels, mask, meaning = player_action(raw_action)
            public_action: Any = labels
            action_masks = (mask,)
        else:
            if isinstance(raw_action, (str, bytes, bytearray)) or not isinstance(
                raw_action, list | tuple
            ):
                raise ValueError("multiplayer action entries must contain one action per player")
            if len(raw_action) != players:
                raise ValueError(
                    f"multiplayer action entries must contain exactly {players} player actions"
                )
            public_players: list[list[str]] = []
            player_masks: list[int] = []
            player_meanings: list[str] = []
            for index, raw_player_action in enumerate(raw_action):
                labels, mask, player_meaning = player_action(raw_player_action)
                public_players.append(labels)
                player_masks.append(mask)
                player_meanings.append(f"p{index + 1}_{player_meaning}")
            public_action = public_players
            action_masks = tuple(player_masks)
            meaning = "__".join(player_meanings)
        if action_masks in seen:
            raise ValueError("custom action table contains duplicate controller actions")
        normalized.append(public_action)
        meanings.append(meaning)
        masks.append(list(action_masks))
        seen.add(action_masks)
    payload = json.dumps(masks, separators=(",", ":"), ensure_ascii=True)
    return {
        "mode": "custom_discrete",
        "preset": preset,
        "table": normalized,
        "meanings": meanings,
        "table_hash": hashlib.sha256(payload.encode("ascii")).hexdigest(),
    }


def configured_action_name(config: Any) -> str:
    contract = declared_action_contract(config)
    if contract is not None and contract.get("preset"):
        return str(contract["preset"])
    task = config.get("task", {}) if isinstance(config, Mapping) else getattr(config, "task", {})
    action = task.get("action", {}) if isinstance(task, Mapping) else {}
    return str(action.get("set", "native")) if isinstance(action, Mapping) else "native"


def configured_action_meanings(config: Any) -> tuple[str, ...]:
    contract = declared_action_contract(config)
    if contract is not None and contract.get("meanings") is not None:
        return tuple(str(value) for value in contract["meanings"])
    action_set = configured_action_name(config)
    if action_set == "native":
        return ()
    game = str(
        config.get("game", "") if isinstance(config, Mapping) else getattr(config, "game", "")
    )
    raise ValueError(f"unknown action_set {action_set!r} for {game!r}")


def configured_action_values(config: Any) -> tuple[tuple[int, ...], ...] | None:
    """Return a provider-button lookup table for an gradlab-side discrete adapter."""
    contract = declared_action_contract(config)
    if contract is None or contract.get("table") is None:
        return None
    provider_id = str(
        config.get("env_provider", "stable-retro-turbo")
        if isinstance(config, Mapping)
        else getattr(config, "env_provider", "stable-retro-turbo")
    )
    game = str(
        config.get("game", "") if isinstance(config, Mapping) else getattr(config, "game", "")
    )
    env_args = (
        config.get("env_args", {})
        if isinstance(config, Mapping)
        else getattr(config, "env_args", {})
    )
    players = int(env_args.get("players", 1)) if isinstance(env_args, Mapping) else 1
    buttons = provider_buttons(provider_id, game, env_args=env_args)
    values: list[tuple[int, ...]] = []
    for action in contract["table"]:
        player_actions = [action] if players == 1 else action
        flattened: list[int] = []
        for labels in player_actions:
            selected = set(labels)
            flattened.extend(int(button is not None and button in selected) for button in buttons)
        values.append(tuple(flattened))
    return tuple(values)


def _json_action_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_action_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {
            str(key): _json_action_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, tuple | list):
        return [_json_action_value(item) for item in value]
    if isinstance(value, float):
        if not np.isfinite(value):
            return str(value)
        return value
    if value is None or isinstance(value, str | int | bool):
        return value
    return str(value)


def _canonical_payload(value: Any) -> bytes:
    return json.dumps(
        _json_action_value(value),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _payload_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_payload(value)).hexdigest()


def action_space_document(space: gym.Space) -> dict[str, Any]:
    """Return a deterministic recursive description of one policy action space."""

    if isinstance(space, gym.spaces.Discrete):
        return {
            "type": "discrete",
            "n": int(space.n),
            "start": int(space.start),
            "dtype": str(space.dtype),
        }
    if isinstance(space, LegalTupleMultiDiscrete):
        return {
            "type": "multi_discrete",
            "nvec": _json_action_value(space.nvec),
            "start": _json_action_value(space.start),
            "shape": [int(value) for value in space.shape],
            "dtype": str(space.dtype),
            "legal_tuples": _json_action_value(space.legal_tuples),
            "distribution": {
                "type": space.distribution_type,
                "scoring": space.scoring_rule,
            },
        }
    if isinstance(space, gym.spaces.MultiDiscrete):
        return {
            "type": "multi_discrete",
            "nvec": _json_action_value(space.nvec),
            "start": _json_action_value(space.start),
            "shape": [int(value) for value in space.shape],
            "dtype": str(space.dtype),
        }
    if isinstance(space, gym.spaces.MultiBinary):
        return {
            "type": "multi_binary",
            "n": _json_action_value(space.n),
            "shape": [int(value) for value in space.shape],
            "dtype": str(space.dtype),
        }
    if isinstance(space, gym.spaces.Box):
        return {
            "type": "box",
            "shape": [int(value) for value in space.shape],
            "dtype": str(space.dtype),
            "low": _json_action_value(space.low),
            "high": _json_action_value(space.high),
        }
    if isinstance(space, gym.spaces.Tuple):
        return {
            "type": "tuple",
            "spaces": [action_space_document(child) for child in space.spaces],
        }
    if isinstance(space, gym.spaces.Dict):
        return {
            "type": "dict",
            "spaces": {
                str(key): action_space_document(child) for key, child in space.spaces.items()
            },
        }
    return {
        "type": type(space).__name__,
        "repr": str(space),
    }


def _semantic_id(value: Any) -> str:
    text = str(value).strip().casefold()
    normalized = []
    separator = False
    for character in text:
        if character.isalnum():
            normalized.append(character)
            separator = False
        elif normalized and not separator:
            normalized.append("_")
            separator = True
    return "".join(normalized).strip("_")


def _display_label(semantic_id: str) -> str:
    return semantic_id.replace("__", " · ").replace("_", " ")


def _input_atom(provider_id: str, atom: str) -> str:
    semantic = _semantic_id(atom)
    if provider_id == "gymnasium":
        return {
            "left_engine": "left",
            "main_engine": "a",
            "right_engine": "right",
            "south": "down",
            "north": "up",
            "east": "right",
            "west": "left",
            "pickup": "a",
            "dropoff": "b",
            "stick": "left",
            "hit": "right",
        }.get(semantic, semantic)
    if provider_id in {"gradoom", "vizdoom-turbo"}:
        return {
            "move_left": "left",
            "move_right": "right",
            "move_forward": "up",
            "move_backward": "down",
            "attack": "a",
            "use": "b",
            "speed": "x",
        }.get(semantic, semantic)
    return semantic


def _entry_controls(provider_id: str, table_entry: Any) -> list[dict[str, Any]] | None:
    if not isinstance(table_entry, tuple | list):
        return None
    if all(isinstance(value, str) for value in table_entry):
        atoms = [_semantic_id(value) for value in table_entry]
        return [
            {
                "player": 1,
                "atoms": atoms,
                "inputs": [_input_atom(provider_id, value) for value in table_entry],
            }
        ]
    if all(
        isinstance(player, tuple | list) and all(isinstance(value, str) for value in player)
        for player in table_entry
    ):
        return [
            {
                "player": index + 1,
                "atoms": [_semantic_id(value) for value in player],
                "inputs": [_input_atom(provider_id, value) for value in player],
            }
            for index, player in enumerate(table_entry)
        ]
    return None


def _mask_semantic(buttons: tuple[str | None, ...], mask: int) -> tuple[str, ...]:
    return tuple(
        _semantic_id(button)
        for index, button in enumerate(buttons)
        if button is not None and (int(mask) >> index) & 1
    )


def _choice(
    *,
    value: int,
    semantic_id: str,
    controls: list[dict[str, Any]] | None = None,
    native_value: Any = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "value": int(value),
        "semantic_id": semantic_id,
        "label": _display_label(semantic_id),
    }
    if controls is not None:
        result["controls"] = controls
    if native_value is not None:
        result["native_value"] = _json_action_value(native_value)
    return result


def _unavailable(reason: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "reason": str(reason),
    }


def _provider_semantics(descriptor: Any) -> dict[str, Any]:
    provider_id = str(descriptor.provider_id)
    space = descriptor.native_action_space
    meanings = tuple(descriptor.action_meanings or ())
    table = tuple(descriptor.action_table or ())
    buttons = tuple(getattr(descriptor, "action_buttons", ()) or ())
    combos = tuple(getattr(descriptor, "action_combos", ()) or ())

    if isinstance(space, gym.spaces.Discrete):
        if meanings:
            if len(meanings) != int(space.n):
                raise ValueError(
                    f"provider {provider_id!r} declares {len(meanings)} action meanings "
                    f"for Discrete({space.n})"
                )
            if table and len(table) != int(space.n):
                raise ValueError(
                    f"provider {provider_id!r} declares {len(table)} table entries "
                    f"for Discrete({space.n})"
                )
            entries = []
            ids: set[str] = set()
            for index, meaning in enumerate(meanings):
                semantic_id = _semantic_id(meaning)
                if not semantic_id:
                    raise ValueError(f"provider {provider_id!r} has an empty action meaning")
                if semantic_id in ids:
                    raise ValueError(
                        f"provider {provider_id!r} has duplicate action meaning {semantic_id!r}"
                    )
                ids.add(semantic_id)
                controls = _entry_controls(provider_id, table[index]) if table else None
                entries.append(
                    _choice(
                        value=int(space.start) + index,
                        semantic_id=semantic_id,
                        controls=controls,
                    )
                )
            return {
                "status": "available",
                "encoding": "explicit",
                "entries": entries,
            }
        if provider_id == "gradlab" and int(space.n) == 2:
            return {
                "status": "available",
                "encoding": "explicit",
                "entries": [
                    _choice(value=int(space.start) + index, semantic_id=f"arm_{index}")
                    for index in range(int(space.n))
                ],
            }
        if str(descriptor.action_mode) == "discrete" and combos and buttons:
            cardinality = 1
            axes = []
            for axis_index, raw_values in enumerate(combos):
                values = []
                for raw_value in raw_values:
                    atoms = _mask_semantic(buttons, int(raw_value))
                    semantic_id = "noop" if not atoms else "_".join(atoms)
                    values.append(
                        {
                            "value": int(raw_value),
                            "semantic_id": semantic_id,
                            "label": _display_label(semantic_id),
                            "atoms": list(atoms),
                            "inputs": [_input_atom(provider_id, atom) for atom in atoms],
                        }
                    )
                cardinality *= len(values)
                axes.append(
                    {
                        "index": axis_index,
                        "radix": len(values),
                        "values": values,
                    }
                )
            if cardinality != int(space.n):
                raise ValueError(
                    f"provider {provider_id!r} discrete action axes have cardinality "
                    f"{cardinality}, expected {space.n}"
                )
            return {
                "status": "available",
                "encoding": "mixed_radix",
                "least_significant_axis": 0,
                "axes": axes,
            }
        return _unavailable(
            f"provider {provider_id!r} did not declare meanings for its discrete actions"
        )

    if isinstance(space, gym.spaces.MultiBinary):
        count = int(np.prod(space.shape))
        if len(buttons) == count and all(button is not None for button in buttons):
            return {
                "status": "available",
                "encoding": "components",
                "components": [
                    {
                        "index": index,
                        "semantic_id": _semantic_id(button),
                        "label": _display_label(_semantic_id(button)),
                        "input": _input_atom(provider_id, str(button)),
                    }
                    for index, button in enumerate(buttons)
                ],
            }
        return _unavailable(
            f"provider {provider_id!r} did not declare one button per MultiBinary component"
        )

    if isinstance(space, gym.spaces.MultiDiscrete):
        flat_nvec = np.asarray(space.nvec).reshape(-1)
        if combos and len(combos) == len(flat_nvec) and buttons:
            axes = []
            for axis_index, (raw_values, expected) in enumerate(
                zip(combos, flat_nvec, strict=True)
            ):
                if len(raw_values) != int(expected):
                    raise ValueError(
                        f"provider {provider_id!r} action axis {axis_index} has "
                        f"{len(raw_values)} values, expected {int(expected)}"
                    )
                values = []
                for raw_value in raw_values:
                    atoms = _mask_semantic(buttons, int(raw_value))
                    semantic_id = "noop" if not atoms else "_".join(atoms)
                    values.append(
                        {
                            "value": int(raw_value),
                            "semantic_id": semantic_id,
                            "label": _display_label(semantic_id),
                            "atoms": list(atoms),
                        }
                    )
                axes.append({"index": axis_index, "values": values})
            return {
                "status": "available",
                "encoding": "components",
                "components": axes,
            }
        return _unavailable(
            f"provider {provider_id!r} did not declare MultiDiscrete component meanings"
        )

    if isinstance(space, gym.spaces.Box):
        count = int(np.prod(space.shape))
        if len(buttons) == count and all(button is not None for button in buttons):
            return {
                "status": "available",
                "encoding": "components",
                "components": [
                    {
                        "index": index,
                        "semantic_id": _semantic_id(button),
                        "label": _display_label(_semantic_id(button)),
                    }
                    for index, button in enumerate(buttons)
                ],
            }
        return _unavailable(f"provider {provider_id!r} did not declare Box component meanings")

    return _unavailable(
        f"provider {provider_id!r} action space {type(space).__name__} has no semantic adapter"
    )


def _mixed_radix_entry(
    semantics: Mapping[str, Any],
    value: int,
    *,
    start: int = 0,
) -> dict[str, Any] | None:
    if semantics.get("encoding") != "mixed_radix":
        return None
    remaining = int(value) - int(start)
    if remaining < 0:
        return None
    atoms: list[str] = []
    inputs: list[str] = []
    for axis in semantics.get("axes", ()):
        radix = int(axis["radix"])
        values = axis["values"]
        selected = values[remaining % radix]
        remaining //= radix
        for atom in selected.get("atoms", ()):
            if atom not in atoms:
                atoms.append(str(atom))
        for input_atom in selected.get("inputs", ()):
            if input_atom not in inputs:
                inputs.append(str(input_atom))
    if remaining:
        return None
    semantic_id = "noop" if not atoms else "_".join(atoms)
    return _choice(
        value=int(value),
        semantic_id=semantic_id,
        controls=[{"player": 1, "atoms": atoms, "inputs": inputs}],
    )


def action_contract_entry(contract: Mapping[str, Any], value: Any) -> dict[str, Any] | None:
    """Resolve one policy action without expanding compact encodings."""

    policy = contract.get("policy")
    if not isinstance(policy, Mapping):
        return None
    space = policy.get("space")
    if not isinstance(space, Mapping):
        return None
    semantics = policy.get("semantics")
    if not isinstance(semantics, Mapping) or semantics.get("status") != "available":
        return None
    if space.get("type") == "multi_discrete" and isinstance(
        semantics.get("legal_entries"), list
    ):
        try:
            selected = tuple(int(item) for item in np.asarray(value).reshape(-1))
        except TypeError, ValueError:
            return None
        for entry in semantics["legal_entries"]:
            if isinstance(entry, Mapping) and tuple(int(item) for item in entry["value"]) == selected:
                return dict(entry)
        return None
    try:
        scalar = int(np.asarray(value).reshape(-1)[0])
    except TypeError, ValueError:
        return None
    entries = semantics.get("entries")
    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, Mapping) and int(entry.get("value", -1)) == scalar:
                return dict(entry)
        return None
    return _mixed_radix_entry(
        semantics,
        scalar,
        start=int(space.get("start", 0)),
    )


def action_contract_meanings(contract: Mapping[str, Any]) -> tuple[str, ...]:
    """Return ordered semantic IDs for a Discrete or legal-tuple policy contract."""

    policy = contract.get("policy")
    if not isinstance(policy, Mapping):
        return ()
    space = policy.get("space")
    if not isinstance(space, Mapping):
        return ()
    semantics = policy.get("semantics")
    if space.get("type") == "multi_discrete" and isinstance(semantics, Mapping):
        entries = semantics.get("legal_entries")
        if not isinstance(entries, list):
            return ()
        ids = tuple(str(entry.get("semantic_id", "")) for entry in entries)
        return ids if all(ids) and len(ids) == len(set(ids)) else ()
    if space.get("type") != "discrete":
        return ()
    start = int(space.get("start", 0))
    count = int(space.get("n", 0))
    result = []
    for value in range(start, start + count):
        entry = action_contract_entry(contract, value)
        if entry is None:
            return ()
        result.append(str(entry["semantic_id"]))
    return tuple(result)


def action_index_for_controls(
    contract: Mapping[str, Any],
    labels: list[str] | tuple[str, ...] | set[str],
) -> int:
    """Map browser control inputs through structured policy action metadata."""

    requested = {_semantic_id(label) for label in labels if _semantic_id(label)}
    policy = contract.get("policy")
    space = policy.get("space") if isinstance(policy, Mapping) else None
    if not isinstance(space, Mapping) or space.get("type") != "discrete":
        raise ValueError("named browser controls require a Discrete policy action contract")
    start = int(space.get("start", 0))
    count = int(space.get("n", 0))
    matches = []
    for value in range(start, start + count):
        entry = action_contract_entry(contract, value)
        controls = entry.get("controls") if isinstance(entry, Mapping) else None
        if not isinstance(controls, list) or len(controls) != 1:
            continue
        inputs = {
            _semantic_id(label) for label in controls[0].get("inputs", ()) if _semantic_id(label)
        }
        if inputs == requested:
            matches.append(value)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"browser controls {sorted(requested)!r} ambiguously match actions {matches!r}"
        )
    available = action_contract_meanings(contract)
    raise ValueError(
        f"no configured discrete action matches controls {sorted(requested)!r}; "
        f"available actions: {', '.join(available) or 'none'}"
    )


def action_value_for_controls(
    contract: Mapping[str, Any],
    labels: list[str] | tuple[str, ...] | set[str],
) -> int | tuple[int, ...]:
    """Map browser control inputs to the contract's scalar or tuple policy value."""

    policy = contract.get("policy")
    space = policy.get("space") if isinstance(policy, Mapping) else None
    if not isinstance(space, Mapping) or space.get("type") == "discrete":
        return action_index_for_controls(contract, labels)
    semantics = policy.get("semantics") if isinstance(policy, Mapping) else None
    entries = semantics.get("legal_entries") if isinstance(semantics, Mapping) else None
    if space.get("type") != "multi_discrete" or not isinstance(entries, list):
        raise ValueError("named browser controls require a Discrete or legal-tuple action contract")
    requested = {_semantic_id(label) for label in labels if _semantic_id(label)}
    matches: list[tuple[int, ...]] = []
    for entry in entries:
        controls = entry.get("controls") if isinstance(entry, Mapping) else None
        if not isinstance(controls, list) or len(controls) != 1:
            continue
        inputs = {
            _semantic_id(label)
            for label in controls[0].get("inputs", ())
            if _semantic_id(label)
        }
        if inputs == requested:
            matches.append(tuple(int(value) for value in entry["value"]))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"browser controls {sorted(requested)!r} ambiguously match actions {matches!r}"
        )
    raise ValueError(
        f"no configured legal action matches controls {sorted(requested)!r}; "
        f"available actions: {', '.join(action_contract_meanings(contract)) or 'none'}"
    )


def action_value_for_semantic(
    contract: Mapping[str, Any],
    semantic_id: str,
) -> int | tuple[int, ...]:
    """Resolve an exact policy action value from its stable semantic ID."""

    requested = _semantic_id(semantic_id)
    policy = contract.get("policy")
    space = policy.get("space") if isinstance(policy, Mapping) else None
    semantics = policy.get("semantics") if isinstance(policy, Mapping) else None
    if not isinstance(space, Mapping) or not isinstance(semantics, Mapping):
        raise ValueError("runtime action contract has no policy semantics")
    if space.get("type") == "discrete":
        start = int(space.get("start", 0))
        for offset, name in enumerate(action_contract_meanings(contract)):
            if _semantic_id(name) == requested:
                return start + offset
    elif space.get("type") == "multi_discrete":
        entries = semantics.get("legal_entries")
        if isinstance(entries, list):
            for entry in entries:
                if _semantic_id(entry.get("semantic_id", "")) == requested:
                    return tuple(int(value) for value in entry["value"])
    raise ValueError(f"policy action semantics do not contain {semantic_id!r}")


def _native_action_entry(
    space: Mapping[str, Any],
    semantics: Mapping[str, Any],
    value: Any,
) -> dict[str, Any] | None:
    if space.get("type") == "discrete":
        return action_contract_entry(
            {"policy": {"space": space, "semantics": semantics}},
            value,
        )
    if semantics.get("status") != "available" or semantics.get("encoding") != "components":
        return None
    flat = np.asarray(value).reshape(-1)
    components = semantics.get("components")
    if not isinstance(components, list) or len(components) != len(flat):
        return None
    atoms: list[str] = []
    inputs: list[str] = []
    semantic_parts: list[str] = []
    for index, (component, raw_value) in enumerate(zip(components, flat, strict=True)):
        if not isinstance(component, Mapping):
            return None
        if space.get("type") == "multi_binary":
            if int(raw_value) == 0:
                continue
            semantic_id = str(component.get("semantic_id") or f"component_{index}")
            semantic_parts.append(semantic_id)
            atoms.append(semantic_id)
            input_atom = component.get("input")
            if input_atom:
                inputs.append(str(input_atom))
            continue
        values = component.get("values")
        if isinstance(values, list):
            selected_index = int(raw_value)
            if not 0 <= selected_index < len(values):
                return None
            selected = values[selected_index]
            semantic_id = str(selected.get("semantic_id") or selected_index)
        else:
            semantic_id = (
                f"{component.get('semantic_id') or f'component_{index}'}"
                f"_value_{_payload_hash(_json_action_value(raw_value))[:12]}"
            )
        if semantic_id != "noop":
            semantic_parts.append(semantic_id)
            atoms.append(semantic_id)
    semantic_id = "noop" if not semantic_parts else "__".join(semantic_parts)
    return {
        "semantic_id": semantic_id,
        "label": _display_label(semantic_id),
        "controls": [{"player": 1, "atoms": atoms, "inputs": inputs}],
    }


def _vizdoom_shared_policy_semantics(
    configured_codec: Mapping[str, Any],
) -> dict[str, Any]:
    semantics = vizdoom_shared_multidiscrete_semantics()
    table = configured_codec.get("source_table")
    tuples = configured_codec.get("legal_tuples")
    if not isinstance(table, list | tuple) or not isinstance(tuples, list | tuple):
        raise ValueError("shared ViZDoom codec requires source_table and legal_tuples")
    if len(table) != len(tuples):
        raise ValueError("shared ViZDoom source table and legal tuples differ in length")
    entries = []
    ids: set[str] = set()
    for native_labels, tuple_value in zip(table, tuples, strict=True):
        if not isinstance(native_labels, list | tuple):
            raise ValueError("shared ViZDoom source action must be a button list")
        semantic_id = (
            "noop"
            if not native_labels
            else "_".join(_semantic_id(label) for label in native_labels)
        )
        if semantic_id in ids:
            raise ValueError(f"shared ViZDoom semantic ID {semantic_id!r} is duplicated")
        ids.add(semantic_id)
        entries.append(
            {
                "value": _json_action_value(tuple_value),
                "semantic_id": semantic_id,
                "label": _display_label(semantic_id),
                "controls": _entry_controls("vizdoom-turbo", native_labels),
                "native_buttons": _json_action_value(native_labels),
            }
        )
    return {
        **semantics,
        "legal_entries": entries,
    }


def compile_runtime_action_contract(
    config: Any,
    descriptor: Any,
    policy_action_space: gym.Space,
    *,
    policy_action_values: Any = None,
    policy_action_codec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile the exact provider-native and final policy-facing action contract."""

    native_space = action_space_document(descriptor.native_action_space)
    policy_space = action_space_document(policy_action_space)
    provider_semantics = _provider_semantics(descriptor)
    provider_document = {
        "provider_id": str(descriptor.provider_id),
        "mode": descriptor.action_mode,
        "preset": descriptor.action_preset,
        "space": native_space,
        "table_hash": descriptor.action_table_hash,
        "semantics": provider_semantics,
    }

    configured_codec_type = (
        policy_action_codec.get("type") if isinstance(policy_action_codec, Mapping) else None
    )
    if configured_codec_type == VIZDOOM_DEATHMATCH_MULTIDISCRETE_CODEC:
        if policy_action_values is not None:
            raise ValueError("a policy action contract cannot combine task codecs")
        codec = vizdoom_deathmatch_multidiscrete_codec_document(
            descriptor,
            policy_action_space,
        )
        policy_semantics = vizdoom_deathmatch_multidiscrete_semantics()
    elif configured_codec_type == VIZDOOM_SHARED_MULTIDISCRETE_CODEC:
        if policy_action_values is not None:
            raise ValueError("a policy action contract cannot combine task codecs")
        codec = vizdoom_shared_multidiscrete_codec_document(
            descriptor,
            policy_action_space,
            policy_action_codec,
        )
        policy_semantics = _vizdoom_shared_policy_semantics(policy_action_codec)
    elif policy_action_values is None and configured_codec_type is None:
        codec: dict[str, Any] = {"type": "identity"}
        policy_semantics = deepcopy(provider_semantics)
    elif configured_codec_type in {None, "discrete_lookup"}:
        if configured_codec_type == "discrete_lookup" and policy_action_values is None:
            policy_action_values = policy_action_codec.get("values")
        if not isinstance(policy_action_space, gym.spaces.Discrete):
            raise ValueError("a discrete lookup codec requires a Discrete policy action space")
        values = tuple(policy_action_values)
        if len(values) != int(policy_action_space.n):
            raise ValueError(
                "policy action lookup cardinality does not match its Discrete action space"
            )
        encoded_values = [_json_action_value(value) for value in values]
        fingerprints = [_canonical_payload(value) for value in encoded_values]
        if len(set(fingerprints)) != len(fingerprints):
            raise ValueError("policy action lookup contains duplicate native actions")
        codec = {
            "type": "discrete_lookup",
            "values": encoded_values,
        }
        entries = []
        for index, native_value in enumerate(values):
            native_entry = _native_action_entry(
                native_space,
                provider_semantics,
                native_value,
            )
            if native_entry is None:
                policy_semantics = _unavailable(
                    "task action lookup targets provider actions without declared semantics"
                )
                break
            entries.append(
                _choice(
                    value=int(policy_action_space.start) + index,
                    semantic_id=str(native_entry["semantic_id"]),
                    controls=deepcopy(native_entry.get("controls")),
                    native_value=native_value,
                )
            )
        else:
            policy_semantics = {
                "status": "available",
                "encoding": "explicit",
                "entries": entries,
            }
    else:
        raise ValueError(f"unsupported policy action codec {configured_codec_type!r}")

    requested = declared_action_contract(config)
    base = {
        "schema_version": ACTION_CONTRACT_SCHEMA_VERSION,
        "requested": _json_action_value(requested),
        "provider": provider_document,
        "policy": {
            "space": policy_space,
            "codec": codec,
            "semantics": policy_semantics,
        },
    }
    execution_payload = {
        "provider": {
            "mode": provider_document["mode"],
            "space": provider_document["space"],
            "table_hash": provider_document["table_hash"],
        },
        "policy": {
            "space": policy_space,
            "codec": codec,
        },
    }
    semantic_payload = {
        "provider": provider_semantics,
        "policy": policy_semantics,
    }
    document = {
        **base,
        "execution_hash": _payload_hash(execution_payload),
        "semantic_hash": _payload_hash(semantic_payload),
    }
    document["contract_hash"] = _payload_hash(document)
    validate_runtime_action_contract(document)
    return document


def validate_runtime_action_contract(contract: Mapping[str, Any]) -> None:
    """Fail closed on malformed or internally inconsistent runtime contracts."""

    if int(contract.get("schema_version", -1)) != ACTION_CONTRACT_SCHEMA_VERSION:
        raise ValueError("unsupported runtime action contract schema_version")
    provider = contract.get("provider")
    policy = contract.get("policy")
    if not isinstance(provider, Mapping) or not isinstance(policy, Mapping):
        raise ValueError("runtime action contract requires provider and policy objects")
    if not isinstance(provider.get("space"), Mapping) or not isinstance(
        policy.get("space"), Mapping
    ):
        raise ValueError("runtime action contract requires provider and policy spaces")
    for name in ("execution_hash", "semantic_hash", "contract_hash"):
        value = contract.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"runtime action contract has invalid {name}")
    without_hash = dict(contract)
    expected = str(without_hash.pop("contract_hash"))
    if _payload_hash(without_hash) != expected:
        raise ValueError("runtime action contract hash does not match its content")
    execution_payload = {
        "provider": {
            "mode": provider.get("mode"),
            "space": provider.get("space"),
            "table_hash": provider.get("table_hash"),
        },
        "policy": {
            "space": policy.get("space"),
            "codec": policy.get("codec"),
        },
    }
    if _payload_hash(execution_payload) != contract["execution_hash"]:
        raise ValueError("runtime action execution hash does not match its content")
    semantic_payload = {
        "provider": provider.get("semantics"),
        "policy": policy.get("semantics"),
    }
    if _payload_hash(semantic_payload) != contract["semantic_hash"]:
        raise ValueError("runtime action semantic hash does not match its content")

    semantics = policy.get("semantics")
    space = policy.get("space")
    if not isinstance(semantics, Mapping):
        raise ValueError("runtime action contract policy semantics must be an object")
    if (
        semantics.get("status") == "available"
        and space.get("type") == "discrete"
        and semantics.get("encoding") == "explicit"
    ):
        entries = semantics.get("entries")
        if not isinstance(entries, list) or len(entries) != int(space.get("n", -1)):
            raise ValueError("explicit Discrete action semantics must contain one entry per action")
        values = [int(entry["value"]) for entry in entries]
        start = int(space.get("start", 0))
        if values != list(range(start, start + int(space["n"]))):
            raise ValueError("explicit Discrete action entries are out of order")
        ids = [str(entry.get("semantic_id", "")) for entry in entries]
        if any(not value for value in ids) or len(ids) != len(set(ids)):
            raise ValueError("explicit Discrete semantic IDs must be non-empty and unique")
    if (
        semantics.get("status") == "available"
        and space.get("type") == "multi_discrete"
        and semantics.get("encoding") == "components"
    ):
        nvec = np.asarray(space.get("nvec")).reshape(-1)
        components = semantics.get("components")
        if not isinstance(components, list) or len(components) != len(nvec):
            raise ValueError("MultiDiscrete action semantics must contain one component per axis")
        for index, (component, cardinality) in enumerate(zip(components, nvec, strict=True)):
            if not isinstance(component, Mapping) or int(component.get("index", -1)) != index:
                raise ValueError("MultiDiscrete action semantic components are out of order")
            values = component.get("values")
            if not isinstance(values, list) or len(values) != int(cardinality):
                raise ValueError(
                    "MultiDiscrete action semantics must contain one value per category"
                )
        legal_tuples = space.get("legal_tuples")
        if legal_tuples is not None:
            if not isinstance(legal_tuples, list) or not legal_tuples:
                raise ValueError("constrained MultiDiscrete space requires legal tuples")
            expected_width = len(nvec)
            normalized = []
            for row in legal_tuples:
                if not isinstance(row, list) or len(row) != expected_width:
                    raise ValueError("constrained MultiDiscrete legal tuple width differs")
                normalized_row = tuple(int(value) for value in row)
                if any(
                    value < 0 or value >= int(cardinality)
                    for value, cardinality in zip(normalized_row, nvec, strict=True)
                ):
                    raise ValueError("constrained MultiDiscrete legal tuple is out of bounds")
                normalized.append(normalized_row)
            if len(normalized) != len(set(normalized)):
                raise ValueError("constrained MultiDiscrete legal tuples must be unique")
            if normalized[0] != tuple(0 for _ in range(expected_width)):
                raise ValueError("constrained MultiDiscrete legal tuples must begin with noop")
            distribution = space.get("distribution")
            if distribution != {
                "type": LEGAL_TUPLE_DISTRIBUTION,
                "scoring": LEGAL_TUPLE_SCORING,
            }:
                raise ValueError("constrained MultiDiscrete distribution contract is unsupported")
            legal_entries = semantics.get("legal_entries")
            if not isinstance(legal_entries, list) or len(legal_entries) != len(normalized):
                raise ValueError("constrained MultiDiscrete semantics require one legal entry per tuple")
            entry_values = [tuple(int(value) for value in entry["value"]) for entry in legal_entries]
            if entry_values != normalized:
                raise ValueError("constrained MultiDiscrete legal entries are out of order")
            ids = [str(entry.get("semantic_id", "")) for entry in legal_entries]
            if any(not value for value in ids) or len(ids) != len(set(ids)):
                raise ValueError("constrained MultiDiscrete semantic IDs must be unique")


def action_contract_payload(
    contract: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Return one complete JSON-native validated runtime action contract.

    Action contracts are bounded protocol documents, not arbitrary provider
    diagnostics. Projecting them through a generic depth-limited serializer can
    preserve the outer ``available`` status while destroying the nested values
    and labels that make the contract usable.
    """

    if contract is None:
        return None
    if not isinstance(contract, Mapping):
        raise ValueError("runtime action contract must be an object")
    validate_runtime_action_contract(contract)
    payload = _json_action_value(contract)
    if not isinstance(payload, dict):
        raise ValueError("runtime action contract did not project to an object")
    return payload


def assert_action_contract_compatible(
    saved_contract: Mapping[str, Any] | None,
    runtime_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Require new artifacts to preserve training-time executable and semantic actions."""

    if not isinstance(runtime_contract, Mapping):
        raise ValueError("runtime did not expose its final action contract")
    validate_runtime_action_contract(runtime_contract)
    if saved_contract is None:
        raise ValueError("checkpoint has no saved runtime action contract")
    validate_runtime_action_contract(saved_contract)
    mismatches = [
        name
        for name in ("execution_hash", "semantic_hash")
        if saved_contract.get(name) != runtime_contract.get(name)
    ]
    if mismatches:
        raise ValueError(
            "runtime action contract differs from the checkpoint at " + ", ".join(mismatches)
        )
    return {
        "status": "compatible",
        "comparable": True,
        "execution_hash": str(runtime_contract["execution_hash"]),
        "semantic_hash": str(runtime_contract["semantic_hash"]),
    }


def runtime_action_contract(source: Any) -> Mapping[str, Any] | None:
    """Find a compiled contract through common policy-environment wrappers."""

    seen: set[int] = set()
    current = source
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        contract = getattr(current, "action_contract", None)
        if isinstance(contract, Mapping):
            return contract
        runtime = getattr(current, "runtime", None)
        contract = getattr(runtime, "action_contract", None)
        if isinstance(contract, Mapping):
            return contract
        current = getattr(current, "venv", None) or getattr(current, "env", None)
    return None


__all__ = [
    "ACTION_CONTRACT_SCHEMA_VERSION",
    "BUILTIN_ACTION_MODES",
    "MARIO_ACTION_TABLES",
    "MARIO_PROVIDERS",
    "action_contract_entry",
    "action_contract_meanings",
    "action_contract_payload",
    "action_index_for_controls",
    "action_value_for_controls",
    "action_value_for_semantic",
    "action_space_document",
    "assert_action_contract_compatible",
    "compile_runtime_action_contract",
    "configured_action_meanings",
    "configured_action_name",
    "configured_action_values",
    "declared_action_contract",
    "runtime_action_contract",
    "validate_runtime_action_contract",
]
