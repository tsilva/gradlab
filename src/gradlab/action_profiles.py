from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from gradlab.action_codecs import (
    VIZDOOM_SHARED_MULTIDISCRETE_BUTTONS,
    VIZDOOM_SHARED_MULTIDISCRETE_CODEC,
    VIZDOOM_SHARED_MULTIDISCRETE_NVEC,
    vizdoom_shared_legal_tuples,
)
from gradlab.action_contract import declared_action_contract
from gradlab.env_identity import train_config_from_source_environment
from gradlab.json_utils import canonical_json_sha256


VIZDOOM_SHARED_ACTION_PROFILE = "vizdoom-shared-multidiscrete-v1"
VIZDOOM_SHARED_ACTION_PROFILE_REVISION = "vizdoom-shared-actions-r1"


@dataclass(frozen=True)
class ActionProfileSelection:
    goal: dict[str, Any]
    key: str
    program_revision: str
    semantic_sha256: str
    source_table_sha256: str
    legal_tuples: tuple[tuple[int, ...], ...]


def _phase_environment(document: Mapping[str, Any], phase: str, *, label: str) -> Mapping[str, Any]:
    section = document.get(phase)
    environment = section.get("environment") if isinstance(section, Mapping) else None
    if not isinstance(environment, Mapping):
        raise ValueError(f"{label}.{phase}.environment is required for action profiles")
    return environment


def _resolved_table(
    environment: Mapping[str, Any],
    *,
    label: str,
) -> tuple[list[list[str]], str]:
    config = train_config_from_source_environment(environment)
    if config.get("env_provider") != "env-vizdoom-turbo":
        raise ValueError(f"{label} requires env_provider='env-vizdoom-turbo'")
    contract = declared_action_contract(config)
    if not isinstance(contract, Mapping) or contract.get("mode") != "custom_discrete":
        raise ValueError(f"{label} requires an exact provider-resolved discrete action table")
    table = contract.get("table")
    table_hash = contract.get("table_hash")
    if not isinstance(table, list) or not table:
        raise ValueError(f"{label} provider did not resolve an action table")
    if not isinstance(table_hash, str) or len(table_hash) != 64:
        raise ValueError(f"{label} provider did not resolve an action-table hash")
    normalized = []
    for index, row in enumerate(table):
        if not isinstance(row, list) or any(not isinstance(button, str) for button in row):
            raise ValueError(f"{label} action table row {index} is not a button-label list")
        normalized.append(list(row))
    return normalized, table_hash


def _apply_profile_to_environment(
    environment: dict[str, Any],
    *,
    table: list[list[str]],
    table_hash: str,
    legal_tuples: tuple[tuple[int, ...], ...],
) -> None:
    env_config = environment.get("env_config")
    if not isinstance(env_config, dict):
        raise ValueError("action profile environment.env_config must be an object")
    env_args = env_config.get("env_args")
    if not isinstance(env_args, dict):
        raise ValueError("action profile environment.env_config.env_args must be an object")
    env_args["use_restricted_actions"] = "filtered"
    vizdoom_config = env_args.get("vizdoom_config")
    if not isinstance(vizdoom_config, dict):
        vizdoom_config = {}
        env_args["vizdoom_config"] = vizdoom_config
    vizdoom_config["available_buttons"] = list(VIZDOOM_SHARED_MULTIDISCRETE_BUTTONS)

    task = environment.get("task")
    if not isinstance(task, dict):
        raise ValueError("action profile environment.task must be an object")
    task["action"] = {
        "set": VIZDOOM_SHARED_ACTION_PROFILE,
        "codec": {
            "type": VIZDOOM_SHARED_MULTIDISCRETE_CODEC,
            "legal_tuples": [list(row) for row in legal_tuples],
            "source_table": copy.deepcopy(table),
            "source_table_hash": table_hash,
        },
    }


def select_goal_action_profile(
    document: Mapping[str, Any],
    selector: str | None,
    *,
    label: str,
) -> ActionProfileSelection | None:
    if selector is None:
        return None
    key = str(selector).strip()
    if key != VIZDOOM_SHARED_ACTION_PROFILE:
        raise ValueError(
            f"unknown action_profile {key!r}; available: {VIZDOOM_SHARED_ACTION_PROFILE}"
        )

    phases = [
        "train",
        *(("eval",) if isinstance(document.get("eval"), Mapping) else ()),
    ]
    resolved: dict[str, tuple[list[list[str]], str]] = {}
    for phase in phases:
        resolved[phase] = _resolved_table(
            _phase_environment(document, phase, label=label),
            label=f"{label}.{phase}.environment",
        )
    train_table, train_hash = resolved["train"]
    for phase, (table, table_hash) in resolved.items():
        if table != train_table or table_hash != train_hash:
            raise ValueError(
                f"{label} action_profile requires identical train/eval action tables; "
                f"{phase} differs"
            )
    legal_tuples = vizdoom_shared_legal_tuples(
        train_table,
        label=f"{label} provider-resolved action table",
    )
    effective = copy.deepcopy(dict(document))
    for phase in phases:
        environment = effective[phase]["environment"]
        _apply_profile_to_environment(
            environment,
            table=train_table,
            table_hash=train_hash,
            legal_tuples=legal_tuples,
        )
    semantic_payload = {
        "profile": key,
        "program_revision": VIZDOOM_SHARED_ACTION_PROFILE_REVISION,
        "nvec": list(VIZDOOM_SHARED_MULTIDISCRETE_NVEC),
        "source_table": train_table,
        "source_table_sha256": train_hash,
        "legal_tuples": [list(row) for row in legal_tuples],
    }
    return ActionProfileSelection(
        goal=effective,
        key=key,
        program_revision=VIZDOOM_SHARED_ACTION_PROFILE_REVISION,
        semantic_sha256=canonical_json_sha256(semantic_payload, ensure_ascii=True),
        source_table_sha256=train_hash,
        legal_tuples=legal_tuples,
    )


__all__ = [
    "ActionProfileSelection",
    "VIZDOOM_SHARED_ACTION_PROFILE",
    "VIZDOOM_SHARED_ACTION_PROFILE_REVISION",
    "select_goal_action_profile",
]
