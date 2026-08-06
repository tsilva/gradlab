from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from gradlab.json_utils import canonical_json_sha256
from gradlab.metric_names import metric_path_segment
from gradlab.state_archive import TaskLaneState
from gradlab.task_kernels import (
    RUNTIME_BOUNDARY_SIGNALS,
    BoundTaskKernel,
    SignalBindings,
    TaskStep,
)


MODEL_INPUTS_SCHEMA_VERSION = 1
CONTEXT_UPDATES = frozenset({"transition", "episode"})
CONTEXT_ENCODINGS = frozenset({"continuous", "categorical"})
CONTEXT_OBSERVATION_LAYOUT = "dict_observation_context_v1"
PROVIDER_FRAME_STACK_HISTORY = "provider_frame_stack"
EPISODE_STEP_SIGNAL = "episode_step"
NATIVE_TIME_REMAINING_SIGNAL = "native_time_remaining"
RUNTIME_CONTEXT_SIGNALS = frozenset({EPISODE_STEP_SIGNAL, NATIVE_TIME_REMAINING_SIGNAL})


def _finite_number(value: Any, *, label: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _number_or_vector(value: Any, *, label: str, allow_null: bool = False) -> Any:
    if value is None and allow_null:
        return None
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        if not value:
            raise ValueError(f"{label} must not be empty")
        return [_finite_number(item, label=f"{label}[{index}]") for index, item in enumerate(value)]
    return _finite_number(value, label=label)


def canonical_category(value: Any, *, label: str) -> int | str | tuple[int | str, ...]:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        raise ValueError(f"{label} must not be a boolean")
    if isinstance(value, int | str):
        if isinstance(value, str) and not value:
            raise ValueError(f"{label} must not be an empty string")
        return value
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        if not value:
            raise ValueError(f"{label} must not be empty")
        items: list[int | str] = []
        for index, item in enumerate(value):
            normalized = canonical_category(item, label=f"{label}[{index}]")
            if isinstance(normalized, tuple):
                raise ValueError(f"{label} must be a flat categorical identity")
            items.append(normalized)
        return tuple(items)
    raise ValueError(f"{label} must be an integer, string, or flat list")


def _canonical_runtime_category(
    value: Any,
    *,
    label: str,
) -> int | str | tuple[int | str, ...]:
    """Canonicalize categorical provider values without weakening config syntax."""

    def normalize_scalar(item: Any, *, item_label: str) -> Any:
        if isinstance(item, np.generic):
            item = item.item()
        if isinstance(item, float):
            if not math.isfinite(item) or not item.is_integer():
                raise ValueError(f"{item_label} must be a finite integral categorical value")
            return int(item)
        return item

    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        value = [
            normalize_scalar(item, item_label=f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    else:
        value = normalize_scalar(value, item_label=label)
    return canonical_category(value, label=label)


def _category_shape(value: int | str | tuple[int | str, ...]) -> tuple[str, ...]:
    if isinstance(value, tuple):
        return tuple("int" if isinstance(item, int) else "str" for item in value)
    return ("scalar-int" if isinstance(value, int) else "scalar-str",)


def _portable_category(value: int | str | tuple[int | str, ...]) -> int | str | list[int | str]:
    return list(value) if isinstance(value, tuple) else value


def normalize_model_inputs(value: Any, *, label: str = "task.model_inputs") -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    unexpected = sorted(set(value) - {"schema_version", "context"})
    if unexpected:
        raise ValueError(f"{label} has unexpected fields: {unexpected}")
    schema_version = value.get("schema_version")
    if schema_version != MODEL_INPUTS_SCHEMA_VERSION:
        raise ValueError(f"{label}.schema_version must be {MODEL_INPUTS_SCHEMA_VERSION}")
    context = value.get("context")
    if not isinstance(context, Mapping) or not context:
        raise ValueError(f"{label}.context must be a non-empty object")

    normalized_fields: dict[str, Any] = {}
    for raw_name, raw_field in context.items():
        name = str(raw_name)
        metric_path_segment(name)
        if name == "observation" or "/" in name:
            raise ValueError(f"{label}.context field {name!r} is reserved or invalid")
        field_label = f"{label}.context.{name}"
        if not isinstance(raw_field, Mapping):
            raise ValueError(f"{field_label} must be an object")
        unexpected_field = sorted(set(raw_field) - {"signal", "update", "encoding", "history"})
        if unexpected_field:
            raise ValueError(f"{field_label} has unexpected fields: {unexpected_field}")
        signal = raw_field.get("signal")
        if not isinstance(signal, str) or not signal.strip():
            raise ValueError(f"{field_label}.signal must be a non-empty string")
        update = str(raw_field.get("update") or "")
        if update not in CONTEXT_UPDATES:
            raise ValueError(f"{field_label}.update must be one of {sorted(CONTEXT_UPDATES)}")
        if signal.strip() in RUNTIME_CONTEXT_SIGNALS and update != "transition":
            raise ValueError(
                f"{field_label} uses a runtime context signal and must update on every transition"
            )
        history = raw_field.get("history")
        if history is not None:
            if history != PROVIDER_FRAME_STACK_HISTORY:
                raise ValueError(f"{field_label}.history must be {PROVIDER_FRAME_STACK_HISTORY!r}")
            if update != "transition":
                raise ValueError(
                    f"{field_label} provider frame-stack history must update on every transition"
                )
            if signal.strip() in RUNTIME_CONTEXT_SIGNALS:
                raise ValueError(
                    f"{field_label} cannot request provider history for a runtime context signal"
                )
        encoding = raw_field.get("encoding")
        if not isinstance(encoding, Mapping):
            raise ValueError(f"{field_label}.encoding must be an object")
        kind = str(encoding.get("kind") or "")
        if kind not in CONTEXT_ENCODINGS:
            raise ValueError(
                f"{field_label}.encoding.kind must be one of {sorted(CONTEXT_ENCODINGS)}"
            )
        if kind == "continuous":
            unexpected_encoding = sorted(
                set(encoding) - {"kind", "scale", "offset", "low", "high", "clip"}
            )
            if unexpected_encoding:
                raise ValueError(
                    f"{field_label}.encoding has unexpected fields: {unexpected_encoding}"
                )
            scale = _number_or_vector(
                encoding.get("scale", 1.0),
                label=f"{field_label}.encoding.scale",
            )
            scale_values = scale if isinstance(scale, list) else [scale]
            if any(value == 0.0 for value in scale_values):
                raise ValueError(f"{field_label}.encoding.scale must be non-zero")
            clip = encoding.get("clip", False)
            if not isinstance(clip, bool):
                raise ValueError(f"{field_label}.encoding.clip must be a boolean")
            normalized_encoding = {
                "kind": kind,
                "scale": scale,
                "offset": _number_or_vector(
                    encoding.get("offset", 0.0),
                    label=f"{field_label}.encoding.offset",
                ),
                "low": _number_or_vector(
                    encoding.get("low"),
                    label=f"{field_label}.encoding.low",
                    allow_null=True,
                ),
                "high": _number_or_vector(
                    encoding.get("high"),
                    label=f"{field_label}.encoding.high",
                    allow_null=True,
                ),
            }
            if clip:
                normalized_encoding["clip"] = True
        else:
            unexpected_encoding = sorted(set(encoding) - {"kind", "values"})
            if unexpected_encoding:
                raise ValueError(
                    f"{field_label}.encoding has unexpected fields: {unexpected_encoding}"
                )
            raw_values = encoding.get("values")
            if not isinstance(raw_values, Sequence) or isinstance(raw_values, str | bytes):
                raise ValueError(f"{field_label}.encoding.values must be a non-empty list")
            if not raw_values:
                raise ValueError(f"{field_label}.encoding.values must be a non-empty list")
            categories = tuple(
                canonical_category(item, label=f"{field_label}.encoding.values[{index}]")
                for index, item in enumerate(raw_values)
            )
            if len(set(categories)) != len(categories):
                raise ValueError(f"{field_label}.encoding.values contains duplicates")
            shapes = {_category_shape(item) for item in categories}
            if len(shapes) != 1:
                raise ValueError(
                    f"{field_label}.encoding.values must use one categorical identity shape"
                )
            normalized_encoding = {
                "kind": kind,
                "values": [_portable_category(item) for item in categories],
            }
        normalized_field = {
            "signal": signal.strip(),
            "update": update,
            "encoding": normalized_encoding,
        }
        if history is not None:
            normalized_field["history"] = history
        normalized_fields[name] = normalized_field
    return {
        "schema_version": MODEL_INPUTS_SCHEMA_VERSION,
        "context": {name: normalized_fields[name] for name in sorted(normalized_fields)},
    }


def normalize_task_model_inputs(task: Mapping[str, Any], *, label: str = "task") -> dict[str, Any]:
    normalized = deepcopy(dict(task))
    value = normalized.get("model_inputs")
    if value is not None:
        normalized["model_inputs"] = normalize_model_inputs(
            value,
            label=f"{label}.model_inputs",
        )
    return normalized


def model_input_fields(task: Mapping[str, Any]) -> Mapping[str, Any]:
    value = task.get("model_inputs")
    if not isinstance(value, Mapping):
        return {}
    context = value.get("context")
    return context if isinstance(context, Mapping) else {}


def has_model_inputs(task: Mapping[str, Any]) -> bool:
    return bool(model_input_fields(task))


def provider_frame_stack_info_keys(task: Mapping[str, Any]) -> tuple[str, ...]:
    """Return provider base keys requested by model-input history fields."""

    fields = model_input_fields(task)
    signals = task.get("signals")
    if not isinstance(signals, Mapping):
        return ()
    selected: list[str] = []
    for name in sorted(fields):
        field = fields[name]
        if field.get("history") != PROVIDER_FRAME_STACK_HISTORY:
            continue
        semantic_name = str(field["signal"])
        try:
            source = signals[semantic_name]
        except KeyError as exc:
            raise ValueError(
                f"provider history context {name!r} references unknown task signal "
                f"{semantic_name!r}"
            ) from exc
        source_names = (source,) if isinstance(source, str) else tuple(source)
        for source_name in source_names:
            key = str(source_name)
            if key not in selected:
                selected.append(key)
    return tuple(selected)


class ProviderFrameStackBox(gym.spaces.Box):
    """Box whose leading axis is a provider-owned policy-transition history."""

    def __init__(self, low: Any, high: Any, *, frame_stack: int) -> None:
        super().__init__(low=low, high=high, dtype=np.float32)
        self.frame_stack = int(frame_stack)


class ProviderFrameStackMultiDiscrete(gym.spaces.MultiDiscrete):
    """Categorical provider history with one category identity per transition."""

    def __init__(self, nvec: Any, *, frame_stack: int) -> None:
        super().__init__(nvec=np.asarray(nvec, dtype=np.int64), dtype=np.int64)
        self.frame_stack = int(frame_stack)


def _vector_parameter(
    value: Any,
    *,
    width: int,
    label: str,
    null_value: float | None = None,
) -> np.ndarray:
    if value is None:
        if null_value is None:
            raise ValueError(f"{label} must not be null")
        return np.full(width, null_value, dtype=np.float32)
    result = np.asarray(value, dtype=np.float32)
    if result.ndim == 0:
        return np.full(width, float(result), dtype=np.float32)
    result = result.reshape(-1)
    if result.shape != (width,):
        raise ValueError(f"{label} must be scalar or contain exactly {width} values")
    return result


@dataclass(frozen=True)
class CompiledContextField:
    name: str
    signal: str
    update: str
    encoding: str
    source_names: tuple[str, ...]
    source_shapes: tuple[tuple[int, ...], ...]
    width: int
    space: gym.Space
    history: str | None = None
    history_depth: int = 1
    scale: np.ndarray | None = None
    offset: np.ndarray | None = None
    low: np.ndarray | None = None
    high: np.ndarray | None = None
    clip: bool = False
    categories: tuple[int | str | tuple[int | str, ...], ...] = ()


@dataclass(frozen=True)
class RuntimeContextSignalSpec:
    dtype: np.dtype
    shape: tuple[int, ...] = ()
    available_on_reset: bool = True
    available_on_step: bool = True


class ContextTaskKernel:
    """Add typed task context to any provider-neutral bound task kernel."""

    def __init__(
        self,
        kernel: BoundTaskKernel,
        descriptor: Any,
        task: Mapping[str, Any],
        *,
        native_episode_horizon: Any | None = None,
    ) -> None:
        if not isinstance(kernel.observation_space, gym.spaces.Box):
            raise ValueError("model context v1 requires a Box base observation")
        self.kernel = kernel
        self.num_envs = int(kernel.num_envs)
        declarations = model_input_fields(task)
        signals = task.get("signals")
        if not isinstance(signals, Mapping):
            raise ValueError("task.signals must be an object")
        context_bindings: dict[str, Any] = {}
        history_bindings: dict[str, Any] = {}
        for name, declaration in declarations.items():
            signal = declaration["signal"]
            if signal in RUNTIME_CONTEXT_SIGNALS:
                if signal in signals:
                    raise ValueError(
                        f"task signal {signal!r} is reserved for model-input runtime context"
                    )
                if declaration["update"] != "transition":
                    raise ValueError(
                        f"runtime context field {name!r} must update on every transition"
                    )
                continue
            if signal not in signals:
                raise ValueError(
                    f"task.model_inputs.context.{name}.signal references unknown "
                    f"task signal {signal!r}"
                )
            context_bindings[signal] = signals[signal]
            if declaration.get("history") == PROVIDER_FRAME_STACK_HISTORY:
                source = signals[signal]
                source_names = (source,) if isinstance(source, str) else tuple(source)
                history_bindings[name] = tuple(
                    f"{source_name}_frame_stack" for source_name in source_names
                )
        self._bindings = SignalBindings(
            descriptor,
            context_bindings,
            self.num_envs,
            require_step=False,
        )
        self._history_bindings = SignalBindings(
            descriptor,
            history_bindings,
            self.num_envs,
            require_step=False,
        )
        compiled: list[CompiledContextField] = []
        buffers: dict[str, np.ndarray] = {}
        spaces: OrderedDict[str, gym.Space] = OrderedDict(
            [("observation", kernel.observation_space)]
        )
        contract_fields: dict[str, Any] = {}
        self._episode_steps = np.zeros(self.num_envs, dtype=np.int64)
        self._uses_episode_steps = any(
            declaration["signal"] in RUNTIME_CONTEXT_SIGNALS
            for declaration in declarations.values()
        )
        self._native_horizon_tics = (
            int(native_episode_horizon.value) if native_episode_horizon is not None else 0
        )
        self._native_action_repeat = (
            int(native_episode_horizon.action_repeat) if native_episode_horizon is not None else 0
        )
        for name in sorted(declarations):
            declaration = declarations[name]
            signal = str(declaration["signal"])
            history = declaration.get("history")
            if signal in RUNTIME_CONTEXT_SIGNALS:
                if signal == NATIVE_TIME_REMAINING_SIGNAL and native_episode_horizon is None:
                    raise ValueError(
                        f"context field {name!r} requires a provider-native episode horizon"
                    )
                source_names = (signal,)
                specs = (
                    RuntimeContextSignalSpec(
                        dtype=np.dtype(np.int64 if signal == EPISODE_STEP_SIGNAL else np.float32),
                    ),
                )
            else:
                source = self._bindings.source(signal)
                source_names = (source,) if isinstance(source, str) else tuple(source)
                specs = self._bindings.source_specs(signal)
            if any(source_name in RUNTIME_BOUNDARY_SIGNALS for source_name in source_names):
                raise ValueError(f"context field {name!r} cannot use a runtime boundary signal")
            assert all(spec is not None for spec in specs)
            if any(not spec.available_on_reset for spec in specs):
                raise ValueError(f"context field {name!r} must be available on reset")
            if declaration["update"] == "transition" and any(
                not spec.available_on_step for spec in specs
            ):
                raise ValueError(
                    f"transition context field {name!r} must be available on every step"
                )
            base_source_names = tuple(str(item) for item in source_names)
            base_specs = specs
            history_depth = 1
            if history == PROVIDER_FRAME_STACK_HISTORY:
                history_source = self._history_bindings.source(name)
                source_names = (
                    (history_source,) if isinstance(history_source, str) else tuple(history_source)
                )
                specs = self._history_bindings.source_specs(name)
                assert all(spec is not None for spec in specs)
                if len(specs) != len(base_specs):
                    raise ValueError(f"context field {name!r} history sources are inconsistent")
                depths: set[int] = set()
                for base_name, base_spec, history_name, history_spec in zip(
                    base_source_names,
                    base_specs,
                    source_names,
                    specs,
                    strict=True,
                ):
                    if not history_spec.shape:
                        raise ValueError(
                            f"provider history signal {history_name!r} has no frame-stack axis"
                        )
                    depths.add(int(history_spec.shape[0]))
                    if tuple(history_spec.shape[1:]) != tuple(base_spec.shape):
                        raise ValueError(
                            f"provider history signal {history_name!r} trailing shape does not "
                            f"match base signal {base_name!r}"
                        )
                    if history_spec.dtype != base_spec.dtype:
                        raise ValueError(
                            f"provider history signal {history_name!r} dtype does not match "
                            f"base signal {base_name!r}"
                        )
                    if not history_spec.available_on_reset or not history_spec.available_on_step:
                        raise ValueError(
                            f"provider history signal {history_name!r} must be available on "
                            "reset and step"
                        )
                if len(depths) != 1:
                    raise ValueError(
                        f"context field {name!r} provider histories have different depths"
                    )
                history_depth = depths.pop()
                if history_depth < 1:
                    raise ValueError(f"context field {name!r} history depth must be positive")
            width = sum(int(np.prod(spec.shape, dtype=np.int64)) or 1 for spec in base_specs)
            encoding = declaration["encoding"]
            if encoding["kind"] == "continuous":
                if any(
                    not np.issubdtype(spec.dtype, np.number) or np.issubdtype(spec.dtype, np.bool_)
                    for spec in specs
                ):
                    raise ValueError(f"continuous context field {name!r} must be numeric")
                scale = _vector_parameter(
                    encoding["scale"],
                    width=width,
                    label=f"context field {name!r} scale",
                )
                offset = _vector_parameter(
                    encoding["offset"],
                    width=width,
                    label=f"context field {name!r} offset",
                )
                low = _vector_parameter(
                    encoding["low"],
                    width=width,
                    label=f"context field {name!r} low",
                    null_value=-np.inf,
                )
                high = _vector_parameter(
                    encoding["high"],
                    width=width,
                    label=f"context field {name!r} high",
                    null_value=np.inf,
                )
                if np.any(low > high):
                    raise ValueError(f"context field {name!r} low exceeds high")
                if history == PROVIDER_FRAME_STACK_HISTORY:
                    history_low = np.broadcast_to(low, (history_depth, width)).copy()
                    history_high = np.broadcast_to(high, (history_depth, width)).copy()
                    space = ProviderFrameStackBox(
                        history_low,
                        history_high,
                        frame_stack=history_depth,
                    )
                    buffers[name] = np.zeros(
                        (self.num_envs, history_depth, width),
                        dtype=np.float32,
                    )
                else:
                    space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
                    buffers[name] = np.zeros((self.num_envs, width), dtype=np.float32)
                categories: tuple[Any, ...] = ()
                clip = bool(encoding.get("clip", False))
            else:
                categories = tuple(
                    canonical_category(
                        value,
                        label=f"task.model_inputs.context.{name}.encoding.values",
                    )
                    for value in encoding["values"]
                )
                expected_width = 1 if not isinstance(categories[0], tuple) else len(categories[0])
                if width != expected_width:
                    raise ValueError(
                        f"categorical context field {name!r} expects {expected_width} "
                        f"source values, provider exposes {width}"
                    )
                scale = offset = low = high = None
                clip = False
                if history == PROVIDER_FRAME_STACK_HISTORY:
                    space = ProviderFrameStackMultiDiscrete(
                        np.full(history_depth, len(categories), dtype=np.int64),
                        frame_stack=history_depth,
                    )
                    buffers[name] = np.zeros(
                        (self.num_envs, history_depth),
                        dtype=np.int64,
                    )
                else:
                    space = gym.spaces.Discrete(len(categories))
                    buffers[name] = np.zeros(self.num_envs, dtype=np.int64)
            spaces[f"context/{name}"] = space
            field = CompiledContextField(
                name=name,
                signal=signal,
                update=str(declaration["update"]),
                encoding=str(encoding["kind"]),
                source_names=tuple(str(item) for item in source_names),
                source_shapes=tuple(tuple(spec.shape) for spec in specs),
                width=width,
                space=space,
                history=str(history) if history is not None else None,
                history_depth=history_depth,
                scale=scale,
                offset=offset,
                low=low,
                high=high,
                clip=clip,
                categories=categories,
            )
            compiled.append(field)
            contract_field = {
                "signal": signal,
                "update": field.update,
                "encoding": deepcopy(dict(encoding)),
                "source": [
                    {
                        "name": str(source_name),
                        "dtype": np.dtype(spec.dtype).str,
                        "shape": list(spec.shape),
                        "available_on_reset": bool(spec.available_on_reset),
                        "available_on_step": bool(spec.available_on_step),
                        **({"origin": "runtime"} if signal in RUNTIME_CONTEXT_SIGNALS else {}),
                    }
                    for source_name, spec in zip(source_names, specs, strict=True)
                ],
                "output": (
                    {
                        "kind": "box",
                        "dtype": np.dtype(np.float32).str,
                        "shape": (
                            [history_depth, width]
                            if history == PROVIDER_FRAME_STACK_HISTORY
                            else [width]
                        ),
                    }
                    if field.encoding == "continuous"
                    else {
                        "kind": (
                            "multi_discrete"
                            if history == PROVIDER_FRAME_STACK_HISTORY
                            else "discrete"
                        ),
                        "dtype": np.dtype(np.int64).str,
                        "categories": len(categories),
                        **(
                            {"shape": [history_depth]}
                            if history == PROVIDER_FRAME_STACK_HISTORY
                            else {}
                        ),
                    }
                ),
            }
            if history is not None:
                contract_field["history"] = {
                    "kind": history,
                    "depth": history_depth,
                    "order": "oldest_to_newest",
                    "flattening": "temporal_major",
                    "base_sources": list(base_source_names),
                }
            contract_fields[name] = contract_field
        self.fields = tuple(compiled)
        self._field_by_name = {field.name: field for field in self.fields}
        self._buffers = buffers
        self._initialized = {
            field.name: np.zeros(self.num_envs, dtype=np.bool_) for field in self.fields
        }
        self.observation_space = gym.spaces.Dict(spaces)
        self.action_space = kernel.action_space
        self.event_names = kernel.event_names
        self.observation_encoding_is_view = False
        self.model_input_contract = {
            "schema_version": MODEL_INPUTS_SCHEMA_VERSION,
            "layout": CONTEXT_OBSERVATION_LAYOUT,
            "base_observation_space": {
                "kind": "box",
                "dtype": np.dtype(kernel.observation_space.dtype).str,
                "shape": list(kernel.observation_space.shape),
            },
            "context": contract_fields,
        }
        self.model_input_contract_sha256 = canonical_json_sha256(
            self.model_input_contract,
            default=str,
            ensure_ascii=True,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.kernel, name)

    def map_actions(self, actions: Any) -> Any:
        return self.kernel.map_actions(actions)

    def _columns_matrix(
        self,
        field: CompiledContextField,
        signals: Mapping[str, Any],
        *,
        mask: np.ndarray,
    ) -> np.ndarray:
        if field.signal == EPISODE_STEP_SIGNAL:
            return self._episode_steps.reshape(self.num_envs, 1)
        if field.signal == NATIVE_TIME_REMAINING_SIGNAL:
            remaining = np.maximum(
                0.0,
                (
                    self._native_horizon_tics
                    - self._episode_steps.astype(np.float64) * self._native_action_repeat
                )
                / self._native_horizon_tics,
            )
            return remaining.reshape(self.num_envs, 1)
        if field.history == PROVIDER_FRAME_STACK_HISTORY:
            columns = self._history_bindings.columns(field.name, signals, mask=mask)
            flattened = [
                np.asarray(column).reshape(self.num_envs, field.history_depth, -1)
                for column in columns
            ]
            return flattened[0] if len(flattened) == 1 else np.concatenate(flattened, axis=2)
        columns = self._bindings.columns(field.signal, signals, mask=mask)
        flattened = [np.asarray(column).reshape(self.num_envs, -1) for column in columns]
        return flattened[0] if len(flattened) == 1 else np.concatenate(flattened, axis=1)

    def _signals_present(
        self,
        field: CompiledContextField,
        signals: Mapping[str, Any],
        mask: np.ndarray,
    ) -> bool:
        if field.signal in RUNTIME_CONTEXT_SIGNALS:
            return True
        for source_name in field.source_names:
            if source_name not in signals:
                return False
            presence = signals.get(f"_{source_name}")
            if presence is not None:
                present = np.asarray(presence, dtype=np.bool_)
                if present.shape != (self.num_envs,) or np.any(mask & ~present):
                    return False
        return True

    def _encode_field(
        self,
        field: CompiledContextField,
        signals: Mapping[str, Any],
        mask: np.ndarray,
    ) -> np.ndarray:
        matrix = self._columns_matrix(field, signals, mask=mask)
        if field.encoding == "continuous":
            assert field.scale is not None
            assert field.offset is not None
            assert field.low is not None
            assert field.high is not None
            encoded = np.asarray(matrix, dtype=np.float32) * field.scale + field.offset
            selected = encoded[mask]
            if np.any(~np.isfinite(selected)):
                raise ValueError(f"context field {field.name!r} produced non-finite values")
            if field.clip:
                return np.clip(encoded, field.low, field.high)
            outside = (selected < field.low) | (selected > field.high)
            if np.any(outside):
                location = np.argwhere(outside)[0]
                selected_row = int(location[0])
                lane = int(np.flatnonzero(mask)[selected_row])
                if field.history == PROVIDER_FRAME_STACK_HISTORY:
                    timestep = int(location[1])
                    column = int(location[2])
                    raw = matrix[lane, timestep, column].item()
                    value = encoded[lane, timestep, column].item()
                    position = f", history position {timestep}"
                else:
                    column = int(location[1])
                    raw = matrix[lane, column].item()
                    value = encoded[lane, column].item()
                    position = ""
                low = field.low[column].item()
                high = field.high[column].item()
                raise ValueError(
                    f"context field {field.name!r} is outside encoded bounds in "
                    f"lane {lane}{position}, column {column}: raw={raw!r}, "
                    f"encoded={value!r}, "
                    f"bounds=[{low!r}, {high!r}]"
                )
            return encoded

        category_index = {value: index for index, value in enumerate(field.categories)}
        encoded_shape = (
            (self.num_envs, field.history_depth)
            if field.history == PROVIDER_FRAME_STACK_HISTORY
            else (self.num_envs,)
        )
        encoded = np.empty(encoded_shape, dtype=np.int64)
        for lane in np.flatnonzero(mask):
            lane_index = int(lane)
            rows = (
                matrix[lane_index]
                if field.history == PROVIDER_FRAME_STACK_HISTORY
                else matrix[lane_index].reshape(1, -1)
            )
            for timestep, row in enumerate(rows):
                raw: Any
                if field.width == 1:
                    raw = row.reshape(-1)[0]
                    if isinstance(raw, np.generic):
                        raw = raw.item()
                else:
                    raw = tuple(
                        item.item() if isinstance(item, np.generic) else item
                        for item in row.reshape(-1)
                    )
                identity = _canonical_runtime_category(
                    raw,
                    label=(
                        f"context field {field.name!r} lane {lane_index} "
                        f"history position {timestep}"
                    ),
                )
                try:
                    category = category_index[identity]
                except KeyError as exc:
                    raise ValueError(
                        f"context field {field.name!r} received unknown category {identity!r}"
                    ) from exc
                if field.history == PROVIDER_FRAME_STACK_HISTORY:
                    encoded[lane_index, timestep] = category
                else:
                    encoded[lane_index] = category
        return encoded

    def _update_field(
        self,
        field: CompiledContextField,
        signals: Mapping[str, Any],
        mask: np.ndarray,
    ) -> None:
        encoded = self._encode_field(field, signals, mask)
        self._buffers[field.name][mask] = encoded[mask]
        self._initialized[field.name][mask] = True

    def on_reset(
        self,
        reset_observations: Any,
        reset_signals: Mapping[str, Any],
        mask: np.ndarray,
    ) -> None:
        self.kernel.on_reset(reset_observations, reset_signals, mask)
        selected = np.asarray(mask, dtype=np.bool_)
        self._episode_steps[selected] = 0
        for field in self.fields:
            self._update_field(field, reset_signals, selected)

    def process(
        self,
        native_rewards: np.ndarray,
        provider_terminated: np.ndarray,
        provider_truncated: np.ndarray,
        signals: Mapping[str, Any],
    ) -> TaskStep:
        step = self.kernel.process(
            native_rewards,
            provider_terminated,
            provider_truncated,
            signals,
        )
        self._episode_steps += 1
        all_lanes = np.ones(self.num_envs, dtype=np.bool_)
        boundary = (
            np.asarray(provider_terminated, dtype=np.bool_)
            | np.asarray(provider_truncated, dtype=np.bool_)
            | np.asarray(step.terminated, dtype=np.bool_)
            | np.asarray(step.truncated, dtype=np.bool_)
        )
        for field in self.fields:
            if field.update == "transition":
                self._update_field(field, signals, all_lanes)
                continue
            specs = self._bindings.source_specs(field.signal)
            if not all(spec is not None and spec.available_on_step for spec in specs):
                continue
            if not self._signals_present(field, signals, all_lanes):
                continue
            candidate = self._encode_field(field, signals, all_lanes)
            current = self._buffers[field.name]
            changed = (
                np.any(candidate != current, axis=1)
                if candidate.ndim == 2
                else candidate != current
            )
            invalid = changed & ~boundary
            if np.any(invalid):
                lanes = np.flatnonzero(invalid).tolist()
                raise ValueError(
                    f"episode context field {field.name!r} changed without a boundary "
                    f"in lanes {lanes}"
                )
        return step

    def encode_observations(self, observations: Any) -> dict[str, np.ndarray]:
        base = self.kernel.encode_observations(observations)
        if any(np.any(~initialized) for initialized in self._initialized.values()):
            raise RuntimeError("model context is not initialized for every lane")
        result: OrderedDict[str, np.ndarray] = OrderedDict([("observation", base)])
        for field in self.fields:
            result[f"context/{field.name}"] = self._buffers[field.name]
        return dict(result)

    def validate_archive_signal(self, semantic_name: str) -> None:
        self.kernel.validate_archive_signal(semantic_name)

    def archive_signal_values(
        self,
        semantic_name: str,
        signals: Mapping[str, Any],
        *,
        mask: np.ndarray,
    ) -> np.ndarray:
        return self.kernel.archive_signal_values(semantic_name, signals, mask=mask)

    def capture_lane_states(
        self,
        mask: np.ndarray,
    ) -> tuple[TaskLaneState | None, ...]:
        selected = np.asarray(mask, dtype=np.bool_)
        inner_states = self.kernel.capture_lane_states(selected)
        states: list[TaskLaneState | None] = []
        episode_fields = tuple(field for field in self.fields if field.update == "episode")
        for lane in range(self.num_envs):
            if not bool(selected[lane]):
                states.append(None)
                continue
            inner = inner_states[lane]
            states.append(
                TaskLaneState(
                    schema_id="gradlab.context-task-lane-v1",
                    values={
                        "contract_sha256": self.model_input_contract_sha256,
                        "inner": None if inner is None else inner.to_dict(),
                        "episode_context": {
                            field.name: (
                                int(self._buffers[field.name][lane])
                                if field.encoding == "categorical"
                                else self._buffers[field.name][lane].tolist()
                            )
                            for field in episode_fields
                        },
                        **(
                            {"runtime_episode_step": int(self._episode_steps[lane])}
                            if self._uses_episode_steps
                            else {}
                        ),
                    },
                )
            )
        return tuple(states)

    def restore_lane_states(
        self,
        states: Sequence[TaskLaneState | None],
        mask: np.ndarray,
    ) -> None:
        selected = np.asarray(mask, dtype=np.bool_)
        if len(states) != self.num_envs or selected.shape != (self.num_envs,):
            raise ValueError("context task restore must contain one state per lane")
        inner_states: list[TaskLaneState | None] = [None for _ in range(self.num_envs)]
        for lane in np.flatnonzero(selected):
            lane_index = int(lane)
            state = states[lane_index]
            if state is None or state.schema_id != "gradlab.context-task-lane-v1":
                raise ValueError(f"archive lane {lane_index} has incompatible context task state")
            values = state.values
            if values.get("contract_sha256") != self.model_input_contract_sha256:
                raise ValueError(
                    f"archive lane {lane_index} context contract does not match runtime"
                )
            inner = values.get("inner")
            inner_states[lane_index] = None if inner is None else TaskLaneState.from_dict(inner)
            episode_context = values.get("episode_context")
            if not isinstance(episode_context, Mapping):
                raise ValueError(f"archive lane {lane_index} has invalid episode context state")
            expected = {field.name for field in self.fields if field.update == "episode"}
            if set(episode_context) != expected:
                raise ValueError(
                    f"archive lane {lane_index} episode context fields do not match runtime"
                )
            for field_name, encoded in episode_context.items():
                field = self._field_by_name[field_name]
                if field.encoding == "categorical":
                    value = int(encoded)
                    if not 0 <= value < len(field.categories):
                        raise ValueError(
                            f"archive lane {lane_index} category is outside vocabulary"
                        )
                    self._buffers[field_name][lane_index] = value
                else:
                    value = np.asarray(encoded, dtype=np.float32)
                    if value.shape != (field.width,) or np.any(~np.isfinite(value)):
                        raise ValueError(f"archive lane {lane_index} continuous context is invalid")
                    self._buffers[field_name][lane_index] = value
                self._initialized[field_name][lane_index] = True
            if self._uses_episode_steps:
                episode_step = values.get("runtime_episode_step")
                if (
                    not isinstance(episode_step, int)
                    or isinstance(episode_step, bool)
                    or episode_step < 0
                ):
                    raise ValueError(f"archive lane {lane_index} has invalid runtime episode step")
                self._episode_steps[lane_index] = episode_step
        self.kernel.restore_lane_states(inner_states, selected)
        for field in self.fields:
            if field.signal in RUNTIME_CONTEXT_SIGNALS:
                self._update_field(field, {}, selected)


def with_model_inputs(
    kernel: BoundTaskKernel,
    descriptor: Any,
    task: Mapping[str, Any],
    *,
    native_episode_horizon: Any | None = None,
) -> BoundTaskKernel:
    return (
        ContextTaskKernel(
            kernel,
            descriptor,
            task,
            native_episode_horizon=native_episode_horizon,
        )
        if has_model_inputs(task)
        else kernel
    )


def runtime_model_input_contract(env: Any) -> Mapping[str, Any] | None:
    runtime = getattr(env, "runtime", None)
    kernel = getattr(runtime, "kernel", None)
    contract = getattr(kernel, "model_input_contract", None)
    return deepcopy(dict(contract)) if isinstance(contract, Mapping) else None
