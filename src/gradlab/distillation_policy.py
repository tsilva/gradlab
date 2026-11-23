"""Data-only, start-conditioned categorical student policy."""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
from pathlib import Path
from typing import Any
import zipfile

import numpy as np
import torch
from gymnasium.spaces import Discrete
from torch import nn
from torch.nn import functional as F

from gradlab.action_contract import action_contract_meanings


def pixels(observation: Any) -> np.ndarray:
    image = np.asarray(observation["image"] if isinstance(observation, Mapping) else observation)
    if image.ndim != 4 or image.dtype != np.uint8:
        raise ValueError(f"expected batched uint8 CHW images, got {image.shape} {image.dtype}")
    return image


class Student(nn.Module):
    """One actor with a common image encoder and a level condition."""

    def __init__(self, shape: tuple[int, ...], action_count: int, level_count: int = 1) -> None:
        super().__init__()
        self.level_count = level_count
        self.encoder = nn.Sequential(
            nn.Conv2d(shape[0], 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            width = self.encoder(torch.zeros(1, *shape)).shape[1]
        self.actor = nn.Sequential(
            nn.Linear(width + level_count, 512), nn.ReLU(), nn.Linear(512, action_count)
        )

    def forward(self, images: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images.float() / 255)
        return self.actor(torch.cat((features, F.one_hot(levels, self.level_count)), dim=1))


STUDENT_MODEL_CLASS = "gradlab.distillation_policy.DistilledPolicy"
_ARCHIVE_FORMAT = "gradlab.distilled-categorical-v1"


class DistilledPolicy:
    requires_episode_start_ids = True

    def __init__(
        self,
        *,
        image_shape: Any,
        actions: Any,
        tasks: Any,
        temperature: float,
        device: str,
    ) -> None:
        self.shape = tuple(image_shape)
        self.actions = tuple(actions)
        self.tasks = tuple((task, tuple(columns)) for task, columns in tasks)
        self.temperature = float(temperature)
        if (
            len(self.shape) != 3
            or any(type(dim) is not int for dim in self.shape)
            or not 1 <= self.shape[0] <= 16
            or not all(36 <= size <= 128 for size in self.shape[1:])
        ):
            raise ValueError("unsupported student image layout")
        if (
            not 1 <= len(self.actions) <= 256
            or any(not isinstance(action, str) or not action for action in self.actions)
            or len(set(self.actions)) != len(self.actions)
        ):
            raise ValueError("student actions must be unique semantic identifiers")
        if (
            not 1 <= len(self.tasks) <= 128
            or any(not isinstance(task, str) or not task for task, _ in self.tasks)
            or len(dict(self.tasks)) != len(self.tasks)
            or any(
                not columns
                or any(type(idx) is not int or not 0 <= idx < len(self.actions) for idx in columns)
                or len(set(columns)) != len(columns)
                for _, columns in self.tasks
            )
        ):
            raise ValueError("student task ordering or action mask is invalid")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("student temperature must be finite and positive")
        self.device = device
        self.actor = Student(self.shape, len(self.actions), len(self.tasks)).to(device).eval()
        self._runtime_actions: tuple[str, ...] | None = None
        self._task_indices = {task: index for index, (task, _) in enumerate(self.tasks)}

    def bind_action_contract(self, action_contract: Any) -> None:
        names = action_contract_meanings(action_contract)
        if not names or len(names) != len(set(names)):
            raise ValueError("distilled policies require unique categorical action semantics")
        self._runtime_actions = names

    def bind_action_space(self, action_space: Any) -> None:
        if (
            not isinstance(action_space, Discrete)
            or int(action_space.start) != 0
            or len(self._runtime_actions or ()) != action_space.n
        ):
            raise ValueError("distilled policy requires an ordered, zero-based Discrete space")

    @torch.inference_mode()
    def policy_decisions(
        self,
        observation: Any,
        *,
        action_selection_mode: str,
        execution_context: tuple[str, ...] | None,
        sampling_temperature: float,
    ) -> tuple[Any, ...]:
        from gradlab.play_debug import PolicyDecision

        images = pixels(observation)
        if images.shape[1:] != self.shape:
            raise ValueError("student observation layout differs from the artifact")
        if execution_context is None or len(execution_context) != len(images):
            raise ValueError("student playback requires a declared episode start for each lane")
        if self._runtime_actions is None:
            raise ValueError("student playback requires the runtime action contract")
        unknown = set(execution_context) - set(self._task_indices)
        if unknown:
            raise ValueError(f"student has no task for episode starts: {unknown}")
        indices = [self._task_indices[task] for task in execution_context]
        logits = self.actor(
            torch.from_numpy(images).to(self.device),
            torch.tensor(indices, device=self.device),
        )
        decisions = []
        for row, idx in zip(logits, indices, strict=True):
            _, columns = self.tasks[idx]
            names = [self.actions[column] for column in columns]
            if any(name not in self._runtime_actions for name in names):
                raise ValueError(
                    "student task requires actions absent from the playback environment"
                )
            local_ids = [self._runtime_actions.index(name) for name in names]
            probabilities = (
                row[list(columns)] / (self.temperature * sampling_temperature)
            ).softmax(0)
            stochastic = action_selection_mode == "stochastic"
            choice = (
                int(torch.multinomial(probabilities, 1))
                if stochastic
                else int(probabilities.argmax())
            )
            probability = float(probabilities[choice])
            full_probs = np.zeros(len(self._runtime_actions), dtype=np.float32)
            full_probs[local_ids] = probabilities.cpu().numpy()
            action = np.asarray(local_ids[choice], dtype=np.int64)
            decisions.append(
                PolicyDecision(
                    raw_action=action,
                    executed_action=action,
                    action_selection_mode=action_selection_mode,
                    distribution_kind="categorical",
                    probabilities=full_probs,
                    log_probability=math.log(probability),
                    entropy=float(torch.special.entr(probabilities).sum()),
                    mode=np.asarray(local_ids[int(probabilities.argmax())]),
                    sampled=stochastic,
                    categorical_index=int(action),
                )
            )
        return tuple(decisions)

    def save(self, path: Path, *, report: dict[str, Any] | None = None) -> None:
        metadata = {
            "format": _ARCHIVE_FORMAT,
            "image_shape": self.shape,
            "actions": self.actions,
            "tasks": self.tasks,
            "temperature": self.temperature,
            "report": report,
        }
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(zipfile.ZipInfo("student.json"), json.dumps(metadata, sort_keys=True))
            for key, tensor in self.actor.state_dict().items():
                data = tensor.detach().cpu().numpy().astype("<f4", copy=False).tobytes()
                archive.writestr(zipfile.ZipInfo(f"weights/{key}.f32"), data)

    @classmethod
    def load(cls, path: Path, *, device: str) -> DistilledPolicy:
        with zipfile.ZipFile(path) as archive:
            if archive.getinfo("student.json").file_size > 1024**2:
                raise ValueError("student archive metadata exceeds its size limit")
            metadata = json.loads(archive.read("student.json"))
            if metadata.pop("format", None) != _ARCHIVE_FORMAT:
                raise ValueError("unsupported student archive format")
            metadata.pop("report", None)
            with torch.random.fork_rng(devices=[]):
                policy = cls(device=device, **metadata)
            expected = policy.actor.state_dict()
            members = ["student.json", *(f"weights/{key}.f32" for key in expected)]
            if sorted(archive.namelist()) != sorted(members):
                raise ValueError("student archive has an unexpected file set")
            for key, tensor in expected.items():
                name = f"weights/{key}.f32"
                if archive.getinfo(name).file_size != tensor.numel() * 4:
                    raise ValueError(f"student tensor size is invalid: {key}")
                array = np.frombuffer(archive.read(name), dtype="<f4").reshape(tensor.shape)
                if not np.isfinite(array).all():
                    raise ValueError(f"student tensor contains nonfinite weights: {key}")
                expected[key] = torch.from_numpy(array.astype(np.float32, copy=True))
            policy.actor.load_state_dict(expected, strict=True)
            return policy
