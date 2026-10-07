from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal

from gradlab.env_registry import STABLE_RETRO_TURBO_PROVIDER


GAME = os.environ.get("RETRO_GAME", "")
DEFAULT_OBS_RESIZE_ALGORITHM = "area"


@dataclass(frozen=True)
class EnvConfig:
    env_provider: str = STABLE_RETRO_TURBO_PROVIDER.provider_id
    game: str = GAME
    env_args: dict[str, Any] = field(default_factory=dict)
    task: dict[str, Any] = field(default_factory=dict)
    state: str = ""
    states: tuple[str, ...] = ()
    state_probs: tuple[float, ...] = ()
    frame_skip: int = 4
    max_pool_frames: bool = True
    sticky_action_prob: float = 0.0
    obs_resize: tuple[int, int] = (84, 84)
    obs_crop: tuple[int, int, int, int] | None = None
    obs_crop_mode: str = "remove"
    obs_crop_fill: int = 0
    obs_resize_algorithm: str = DEFAULT_OBS_RESIZE_ALGORITHM


IdentitySection = Literal["state", "preprocessing"]
FieldKind = Literal["value", "bool_optional"]
TypeName = Literal["str", "int", "float", "json", "obs_crop", "obs_resize"]
SequenceItemKind = Literal["str", "number", "rows"]


@dataclass(frozen=True)
class EnvironmentFieldSpec:
    dest: str
    flag: str | None = None
    kind: FieldKind = "value"
    type_name: TypeName = "str"
    cli_default: Any = None
    use_runtime_default: bool = True
    choices: tuple[str, ...] = ()
    help: str | None = None
    non_empty: bool = False
    validation_min: float | None = None
    validation_max: float | None = None
    sequence_items: SequenceItemKind | None = None
    mapping_value: bool = False
    identity_section: IdentitySection | None = None
    goal_required: bool = False
    mixed_state: bool = False


ENVIRONMENT_FIELD_SPECS = (
    EnvironmentFieldSpec(
        "env_provider",
        non_empty=True,
        help=(
            "Environment provider id. Supported: gradlab, env-stableretro-turbo, "
            "env-supermariobrosnes-turbo-emu, ale-py, gymnasium."
        ),
    ),
    EnvironmentFieldSpec(
        "game",
        non_empty=True,
        help="Provider game id. Defaults to RETRO_GAME when set.",
    ),
    EnvironmentFieldSpec(
        "env_args",
        type_name="json",
        mapping_value=True,
        help="Provider-native environment constructor arguments, serialized as a JSON object.",
    ),
    EnvironmentFieldSpec(
        "task",
        flag="--task-json",
        type_name="json",
        mapping_value=True,
        help="Canonical bound-task definition as a JSON object.",
    ),
    EnvironmentFieldSpec(
        "state",
        non_empty=True,
        identity_section="state",
        help="Provider state. If omitted, the registered environment spec may provide a default.",
    ),
    EnvironmentFieldSpec(
        "states",
        cli_default="",
        use_runtime_default=False,
        sequence_items="str",
        identity_section="state",
        mixed_state=True,
        help=(
            "Comma-separated provider states. Without --state-probs, provide exactly "
            "one state per env slot in order."
        ),
    ),
    EnvironmentFieldSpec(
        "state_probs",
        cli_default="",
        use_runtime_default=False,
        sequence_items="number",
        identity_section="state",
        mixed_state=True,
        help=(
            "Comma-separated non-negative sampling weights for --states. The native "
            "vector env normalizes weights and samples independently on each episode reset."
        ),
    ),
    EnvironmentFieldSpec(
        "frame_skip",
        type_name="int",
        validation_min=1,
        identity_section="preprocessing",
        goal_required=True,
    ),
    EnvironmentFieldSpec(
        "max_pool_frames",
        kind="bool_optional",
        identity_section="preprocessing",
        goal_required=True,
        help="Max-pool over the last two raw frames inside each frame-skip step.",
    ),
    EnvironmentFieldSpec(
        "sticky_action_prob",
        type_name="float",
        identity_section="preprocessing",
        goal_required=True,
        help=(
            "Probability of replaying the previous high-level action; 0 disables sticky actions."
        ),
    ),
    EnvironmentFieldSpec(
        "obs_resize",
        type_name="obs_resize",
        identity_section="preprocessing",
        goal_required=True,
        help="Policy observation dimensions as height,width.",
    ),
    EnvironmentFieldSpec(
        "obs_crop",
        type_name="obs_crop",
        identity_section="preprocessing",
        goal_required=True,
        help="Four-sided raw-frame crop as top,right,bottom,left before grayscale resize.",
    ),
    EnvironmentFieldSpec(
        "obs_crop_mode",
        choices=("remove", "mask"),
        non_empty=True,
        identity_section="preprocessing",
        goal_required=True,
        help="Whether obs_crop removes pixels or masks them before resize.",
    ),
    EnvironmentFieldSpec(
        "obs_crop_fill",
        type_name="int",
        validation_min=0,
        validation_max=255,
        identity_section="preprocessing",
        goal_required=True,
        help="Pixel fill value for obs_crop_mode=mask.",
    ),
    EnvironmentFieldSpec(
        "obs_resize_algorithm",
        non_empty=True,
        identity_section="preprocessing",
        goal_required=True,
        help="Resize algorithm for native frame preprocessing.",
    ),
)

ENVIRONMENT_FIELD_NAMES = tuple(spec.dest for spec in ENVIRONMENT_FIELD_SPECS)
if ENVIRONMENT_FIELD_NAMES != tuple(EnvConfig.__dataclass_fields__):
    raise RuntimeError("environment field registry must exactly match EnvConfig")
STATE_FIELD_NAMES = tuple(
    spec.dest for spec in ENVIRONMENT_FIELD_SPECS if spec.identity_section == "state"
)
PREPROCESSING_FIELD_NAMES = tuple(
    spec.dest
    for spec in ENVIRONMENT_FIELD_SPECS
    if spec.identity_section == "preprocessing"
)
GOAL_REQUIRED_ENVIRONMENT_FIELD_NAMES = frozenset(
    spec.dest for spec in ENVIRONMENT_FIELD_SPECS if spec.goal_required
)


__all__ = [
    "DEFAULT_OBS_RESIZE_ALGORITHM",
    "ENVIRONMENT_FIELD_NAMES",
    "ENVIRONMENT_FIELD_SPECS",
    "EnvConfig",
    "EnvironmentFieldSpec",
    "FieldKind",
    "GAME",
    "GOAL_REQUIRED_ENVIRONMENT_FIELD_NAMES",
    "PREPROCESSING_FIELD_NAMES",
    "SequenceItemKind",
    "STATE_FIELD_NAMES",
    "TypeName",
]
