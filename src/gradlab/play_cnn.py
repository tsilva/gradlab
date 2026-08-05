from __future__ import annotations

import colorsys
import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from gradlab.play_attribution import (
    ActionLogProbForward,
    AttributionError,
    actor_image_feature_extractor,
)


CNN_TOP_K_DEFAULT = 12
CNN_TOP_K_MAX = 32
CNN_INTERVAL_DEFAULT = 1
CNN_RANK_BASIS = "peak_raw_positive_response"


class CNNInspectionError(RuntimeError):
    """Raised when a policy cannot provide truthful convolutional inspection."""


@dataclass(frozen=True)
class _SpatialGeometry:
    receptive_field: tuple[float, float] = (1.0, 1.0)
    jump: tuple[float, float] = (1.0, 1.0)
    start: tuple[float, float] = (0.5, 0.5)


@dataclass(frozen=True)
class _LayerTarget:
    layer_id: str
    conv: nn.Conv2d
    response_module: nn.Module
    response_stage: str
    geometry: _SpatialGeometry
    descriptor: dict[str, Any]


@dataclass(frozen=True)
class CNNInspection:
    generation: int
    layer_id: str
    input_shape: tuple[int, int]
    activation_shape: tuple[int, int]
    response_stage: str
    rank_basis: str
    filters: tuple[dict[str, Any], ...]
    atlas: np.ndarray
    atlas_columns: int
    atlas_rows: int
    atlas_tile_width: int
    atlas_tile_height: int

    def payload(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "layer_id": self.layer_id,
            "input_shape": list(self.input_shape),
            "activation_shape": list(self.activation_shape),
            "response_stage": self.response_stage,
            "rank_basis": self.rank_basis,
            "filters": [dict(item) for item in self.filters],
            "atlas": {
                "columns": self.atlas_columns,
                "rows": self.atlas_rows,
                "tile_width": self.atlas_tile_width,
                "tile_height": self.atlas_tile_height,
                "winner_tile": 0,
            },
        }


def _pair(value: Any, *, fallback: tuple[int, int] | None = None) -> tuple[int, int]:
    if value is None:
        if fallback is None:
            raise CNNInspectionError("spatial operator is missing a required size")
        return fallback
    if isinstance(value, str):
        raise CNNInspectionError(f"string-valued spatial parameter {value!r} is unsupported")
    if isinstance(value, int):
        return (value, value)
    values = tuple(int(item) for item in value)
    if len(values) != 2:
        raise CNNInspectionError(f"expected a 2D spatial parameter, got {values!r}")
    return values


def _operator_geometry(module: nn.Module, current: _SpatialGeometry) -> _SpatialGeometry:
    if isinstance(module, nn.Conv2d):
        kernel = _pair(module.kernel_size)
        stride = _pair(module.stride)
        dilation = _pair(module.dilation)
        if isinstance(module.padding, str):
            if module.padding != "same":
                raise CNNInspectionError(
                    f"convolution padding mode {module.padding!r} cannot be mapped to input pixels"
                )
            padding = tuple((dilation[index] * (kernel[index] - 1)) / 2 for index in range(2))
        else:
            padding = tuple(float(value) for value in _pair(module.padding))
    elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
        kernel = _pair(module.kernel_size)
        stride = _pair(module.stride, fallback=kernel)
        dilation = _pair(getattr(module, "dilation", 1))
        padding = tuple(float(value) for value in _pair(module.padding))
    else:
        return current

    receptive_field = tuple(
        current.receptive_field[index] + (kernel[index] - 1) * dilation[index] * current.jump[index]
        for index in range(2)
    )
    start = tuple(
        current.start[index]
        + (((kernel[index] - 1) * dilation[index]) / 2 - padding[index]) * current.jump[index]
        for index in range(2)
    )
    jump = tuple(current.jump[index] * stride[index] for index in range(2))
    return _SpatialGeometry(receptive_field=receptive_field, jump=jump, start=start)


def _leaf_modules(module: nn.Module) -> list[tuple[str, nn.Module]]:
    return [
        (name or "conv", child)
        for name, child in module.named_modules()
        if not tuple(child.children())
    ]


def _is_activation(module: nn.Module) -> bool:
    return isinstance(
        module,
        (nn.ReLU, nn.LeakyReLU, nn.ELU, nn.GELU, nn.SiLU, nn.Tanh, nn.Sigmoid),
    )


def convolutional_layer_targets(extractor: nn.Module) -> tuple[_LayerTarget, ...]:
    leaves = _leaf_modules(extractor)
    geometry = _SpatialGeometry()
    targets: list[_LayerTarget] = []
    for index, (name, module) in enumerate(leaves):
        try:
            geometry = _operator_geometry(module, geometry)
        except CNNInspectionError:
            if isinstance(module, nn.Conv2d):
                raise
            continue
        if not isinstance(module, nn.Conv2d):
            continue
        response_name = name
        response_module: nn.Module = module
        response_stage = "positive convolution output"
        if index + 1 < len(leaves) and _is_activation(leaves[index + 1][1]):
            response_name, response_module = leaves[index + 1]
            response_stage = f"post-{response_module.__class__.__name__} output"
        layer_number = len(targets) + 1
        kernel = _pair(module.kernel_size)
        stride = _pair(module.stride)
        descriptor = {
            "id": name,
            "label": (
                f"Conv {layer_number} · {module.out_channels} filters · "
                f"{kernel[0]}×{kernel[1]} / stride {stride[0]}×{stride[1]}"
            ),
            "index": layer_number - 1,
            "in_channels": int(module.in_channels),
            "out_channels": int(module.out_channels),
            "kernel_size": list(kernel),
            "stride": list(stride),
            "padding": list(_pair(module.padding))
            if not isinstance(module.padding, str)
            else module.padding,
            "dilation": list(_pair(module.dilation)),
            "groups": int(module.groups),
            "response_module": response_name,
            "response_stage": response_stage,
            "receptive_field": [float(value) for value in geometry.receptive_field],
            "feature_stride": [float(value) for value in geometry.jump],
            "first_center": [float(value) for value in geometry.start],
        }
        targets.append(
            _LayerTarget(
                layer_id=name,
                conv=module,
                response_module=response_module,
                response_stage=response_stage,
                geometry=geometry,
                descriptor=descriptor,
            )
        )
    if not targets:
        raise CNNInspectionError("policy actor image encoder has no inspectable Conv2d layer")
    return tuple(targets)


def cnn_inspection_capability(model: Any) -> dict[str, Any]:
    try:
        extractor = actor_image_feature_extractor(model.policy)
        targets = convolutional_layer_targets(extractor)
    except (AttributeError, AttributionError, CNNInspectionError) as exc:
        return {
            "layers": [],
            "default_layer_id": None,
            "rank_basis": CNN_RANK_BASIS,
            "unavailable_reason": str(exc),
        }
    return {
        "layers": [dict(target.descriptor) for target in targets],
        "default_layer_id": targets[0].layer_id,
        "rank_basis": CNN_RANK_BASIS,
        "unavailable_reason": None,
    }


def _dummy_action(policy: Any, batch_size: int) -> np.ndarray:
    space = getattr(policy, "action_space", None)
    shape = tuple(getattr(space, "shape", ()))
    dtype = getattr(space, "dtype", None)
    if dtype is None:
        dtype = (
            np.int64 if space is not None and space.__class__.__name__ == "Discrete" else np.float32
        )
    return np.zeros((batch_size, *shape), dtype=dtype)


def _resize_map(values: np.ndarray, output_shape: tuple[int, int], *, mode: str) -> np.ndarray:
    tensor = torch.as_tensor(values, dtype=torch.float32)[None, None, ...]
    resized = F.interpolate(
        tensor,
        size=output_shape,
        mode=mode,
        align_corners=False if mode in {"bilinear", "bicubic"} else None,
    )
    return resized[0, 0].cpu().numpy()


def _activation_tile(
    values: np.ndarray,
    color: tuple[int, int, int],
    output_shape: tuple[int, int],
) -> np.ndarray:
    peak = float(np.max(values)) if values.size else 0.0
    normalized = np.zeros_like(values, dtype=np.float32) if peak <= 0 else values / peak
    normalized = np.clip(_resize_map(normalized, output_shape, mode="bilinear"), 0.0, 1.0)
    rgb = np.rint(normalized[..., None] * np.asarray(color, dtype=np.float32)).astype(np.uint8)
    alpha = np.full((*output_shape, 1), 255, dtype=np.uint8)
    return np.concatenate((rgb, alpha), axis=2)


def _filter_color(filter_index: int) -> tuple[int, int, int]:
    """Assign a stable, effectively non-repeating color to a filter index."""

    hue = (int(filter_index) * 0.618033988749895) % 1.0
    saturation = 0.68 + 0.08 * (int(filter_index) % 3)
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, 0.95)
    return tuple(int(round(channel * 255)) for channel in (red, green, blue))


def _weight_colors(values: np.ndarray) -> np.ndarray:
    scale = float(np.max(np.abs(values))) if values.size else 0.0
    normalized = np.zeros_like(values, dtype=np.float32) if scale <= 0 else values / scale
    magnitude = np.abs(normalized)[..., None]
    positive = np.asarray((251, 191, 36), dtype=np.float32)
    negative = np.asarray((34, 211, 238), dtype=np.float32)
    background = np.asarray((3, 10, 14), dtype=np.float32)
    target = np.where((normalized >= 0)[..., None], positive, negative)
    return np.rint(background * (1.0 - magnitude) + target * magnitude).astype(np.uint8)


def _kernel_tile(weights: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    channels, kernel_height, kernel_width = weights.shape
    columns = max(1, math.ceil(math.sqrt(channels)))
    rows = max(1, math.ceil(channels / columns))
    height, width = output_shape
    gap = 1
    cell_width = max(1, (width - gap * (columns + 1)) // columns)
    cell_height = max(1, (height - gap * (rows + 1)) // rows)
    tile = np.zeros((height, width, 4), dtype=np.uint8)
    tile[..., :3] = (3, 10, 14)
    tile[..., 3] = 255
    colors = _weight_colors(weights)
    for channel in range(channels):
        row, column = divmod(channel, columns)
        x0 = gap + column * (cell_width + gap)
        y0 = gap + row * (cell_height + gap)
        if x0 >= width or y0 >= height:
            continue
        target_width = min(cell_width, width - x0)
        target_height = min(cell_height, height - y0)
        plane = Image.fromarray(colors[channel], mode="RGB").resize(
            (target_width, target_height),
            resample=Image.Resampling.NEAREST,
        )
        tile[y0 : y0 + target_height, x0 : x0 + target_width, :3] = np.asarray(plane)
    return tile


def _winner_tile(
    activations: np.ndarray,
    colors: tuple[tuple[int, int, int], ...],
    output_shape: tuple[int, int],
) -> np.ndarray:
    if activations.shape[0] == 0:
        return np.zeros((*output_shape, 4), dtype=np.uint8)
    winner = np.argmax(activations, axis=0).astype(np.float32)
    strength = np.max(activations, axis=0)
    winner = _resize_map(winner, output_shape, mode="nearest").astype(np.int64)
    strength = _resize_map(strength, output_shape, mode="bilinear")
    peak = float(np.max(strength)) if strength.size else 0.0
    normalized = np.zeros_like(strength) if peak <= 0 else np.clip(strength / peak, 0.0, 1.0)
    palette = np.asarray(colors, dtype=np.uint8)
    rgb = palette[np.clip(winner, 0, len(colors) - 1)]
    alpha = np.rint(220.0 * normalized).astype(np.uint8)[..., None]
    return np.concatenate((rgb, alpha), axis=2)


def _peak_region(
    row: int,
    column: int,
    geometry: _SpatialGeometry,
    input_shape: tuple[int, int],
) -> dict[str, float]:
    input_height, input_width = input_shape
    center_y = geometry.start[0] + row * geometry.jump[0]
    center_x = geometry.start[1] + column * geometry.jump[1]
    half_height = geometry.receptive_field[0] / 2
    half_width = geometry.receptive_field[1] / 2
    y0 = max(0.0, center_y - half_height)
    x0 = max(0.0, center_x - half_width)
    y1 = min(float(input_height), center_y + half_height)
    x1 = min(float(input_width), center_x + half_width)
    return {
        "x0": float(x0),
        "y0": float(y0),
        "x1": float(x1),
        "y1": float(y1),
        "center_x": float(center_x),
        "center_y": float(center_y),
    }


class PolicyCNNInspector:
    """Capture ranked convolutional responses without changing policy behavior or RNG."""

    def __init__(self, model: Any):
        self.model = model
        self.policy = model.policy
        self.image_extractor = actor_image_feature_extractor(self.policy)
        self.targets = convolutional_layer_targets(self.image_extractor)
        self.targets_by_id = {target.layer_id: target for target in self.targets}

    @property
    def capability(self) -> dict[str, Any]:
        return {
            "layers": [dict(target.descriptor) for target in self.targets],
            "default_layer_id": self.targets[0].layer_id,
            "rank_basis": CNN_RANK_BASIS,
            "unavailable_reason": None,
        }

    def inspect(
        self,
        model_obs: np.ndarray | dict[str, np.ndarray],
        *,
        layer_id: str,
        top_k: int,
        generation: int,
    ) -> CNNInspection:
        target = self.targets_by_id.get(str(layer_id))
        if target is None:
            raise CNNInspectionError(f"unknown convolutional layer {layer_id!r}")
        if isinstance(top_k, bool) or not 1 <= int(top_k) <= CNN_TOP_K_MAX:
            raise CNNInspectionError(f"top_k must be in [1, {CNN_TOP_K_MAX}]")

        captured: list[torch.Tensor] = []

        def capture(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if not isinstance(output, torch.Tensor):
                raise CNNInspectionError(
                    f"layer {layer_id!r} produced {type(output)!r}, not a tensor"
                )
            captured.append(output.detach())

        obs_tensor, _vectorized = self.policy.obs_to_tensor(model_obs)
        image_tensor = (
            next(
                (value for key, value in obs_tensor.items() if key in {"observation", "image"}),
                None,
            )
            if isinstance(obs_tensor, dict)
            else obs_tensor
        )
        if not isinstance(image_tensor, torch.Tensor) or image_tensor.ndim != 4:
            raise CNNInspectionError("policy observation does not contain a batched image tensor")
        batch_size = int(image_tensor.shape[0])
        if batch_size != 1:
            raise CNNInspectionError(
                f"player CNN inspection requires one observation, got batch size {batch_size}"
            )
        forward = ActionLogProbForward(
            self.policy,
            model_obs,
            _dummy_action(self.policy, batch_size),
        )
        input_shape = tuple(int(value) for value in forward.image_tensor.shape[-2:])

        python_rng_state = random.getstate()
        numpy_rng_state = np.random.get_state()
        torch_rng_state = torch.random.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        was_training = bool(getattr(self.policy, "training", False))
        handle = target.response_module.register_forward_hook(capture)
        self.policy.eval()
        try:
            with torch.no_grad():
                forward(forward.image_tensor)
        finally:
            handle.remove()
            self.policy.train(was_training)
            random.setstate(python_rng_state)
            np.random.set_state(numpy_rng_state)
            torch.random.set_rng_state(torch_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
        if len(captured) != 1:
            raise CNNInspectionError(
                f"layer {layer_id!r} executed {len(captured)} times; one execution is required"
            )
        response = captured[0]
        if response.ndim != 4:
            raise CNNInspectionError(
                f"layer {layer_id!r} response must be NCHW, got {tuple(response.shape)}"
            )
        positive = torch.nan_to_num(response[0].float(), nan=0.0, posinf=0.0, neginf=0.0)
        positive = positive.clamp_min(0.0).cpu().numpy()
        activation_shape = tuple(int(value) for value in positive.shape[-2:])
        flat = positive.reshape(positive.shape[0], -1)
        peaks = flat.max(axis=1)
        means = np.divide(
            flat.sum(axis=1),
            np.maximum(1, np.count_nonzero(flat > 0, axis=1)),
        )
        coverage = np.mean(flat > 0, axis=1)
        indices = np.arange(positive.shape[0])
        order = np.lexsort((indices, -peaks))[: min(int(top_k), len(indices))]
        selected = positive[order]
        colors = tuple(_filter_color(int(filter_index)) for filter_index in order)

        tiles: list[np.ndarray] = [_winner_tile(selected, colors, input_shape)]
        filters: list[dict[str, Any]] = []
        weights = target.conv.weight.detach().float().cpu().numpy()
        bias = None if target.conv.bias is None else target.conv.bias.detach().float().cpu().numpy()
        for rank_index, filter_index_value in enumerate(order):
            filter_index = int(filter_index_value)
            activation = positive[filter_index]
            peak_flat = int(np.argmax(activation))
            peak_row, peak_column = np.unravel_index(peak_flat, activation_shape)
            color = colors[rank_index]
            kernel_tile_index = len(tiles)
            tiles.append(_kernel_tile(weights[filter_index], input_shape))
            activation_tile_index = len(tiles)
            tiles.append(_activation_tile(activation, color, input_shape))
            filters.append(
                {
                    "rank": rank_index + 1,
                    "filter_index": filter_index,
                    "color": "#" + "".join(f"{channel:02x}" for channel in color),
                    "peak_response": float(peaks[filter_index]),
                    "mean_positive_response": float(means[filter_index]),
                    "positive_coverage": float(coverage[filter_index]),
                    "peak_cell": [int(peak_row), int(peak_column)],
                    "peak_input_region": _peak_region(
                        int(peak_row),
                        int(peak_column),
                        target.geometry,
                        input_shape,
                    ),
                    "kernel_l2": float(np.linalg.norm(weights[filter_index])),
                    "bias": None if bias is None else float(bias[filter_index]),
                    "kernel_tile": kernel_tile_index,
                    "activation_tile": activation_tile_index,
                }
            )

        atlas_columns = max(1, math.ceil(math.sqrt(len(tiles))))
        atlas_rows = math.ceil(len(tiles) / atlas_columns)
        tile_height, tile_width = input_shape
        atlas = np.zeros(
            (atlas_rows * tile_height, atlas_columns * tile_width, 4),
            dtype=np.uint8,
        )
        for tile_index, tile in enumerate(tiles):
            row, column = divmod(tile_index, atlas_columns)
            y0 = row * tile_height
            x0 = column * tile_width
            atlas[y0 : y0 + tile_height, x0 : x0 + tile_width] = tile

        return CNNInspection(
            generation=int(generation),
            layer_id=target.layer_id,
            input_shape=input_shape,
            activation_shape=activation_shape,
            response_stage=target.response_stage,
            rank_basis=CNN_RANK_BASIS,
            filters=tuple(filters),
            atlas=atlas,
            atlas_columns=atlas_columns,
            atlas_rows=atlas_rows,
            atlas_tile_width=tile_width,
            atlas_tile_height=tile_height,
        )


__all__ = [
    "CNN_INTERVAL_DEFAULT",
    "CNN_RANK_BASIS",
    "CNN_TOP_K_DEFAULT",
    "CNN_TOP_K_MAX",
    "CNNInspection",
    "CNNInspectionError",
    "PolicyCNNInspector",
    "cnn_inspection_capability",
    "convolutional_layer_targets",
]
