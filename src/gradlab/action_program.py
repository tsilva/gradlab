from __future__ import annotations

import json
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np


ACTION_PROGRAM_SCHEMA_VERSION = 1
ACTION_PROGRAM_MEMBER = "action_program.json"
ACTION_PROGRAM_ARTIFACT_IDENTITY_MEMBER = "artifact_identity.json"
ACTION_PROGRAM_POLICY_TYPE = "action-program"
ACTION_PROGRAM_MODEL_CLASS = "gradlab.action_program.ActionProgramPolicy"


@dataclass(frozen=True, order=True)
class ActionRun:
    """One action held for a positive number of environment steps."""

    action: int
    duration: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", int(self.action))
        object.__setattr__(self, "duration", int(self.duration))
        if self.duration < 1:
            raise ValueError("action-run durations must be positive")


def canonicalize_action_runs(runs: Sequence[ActionRun]) -> tuple[ActionRun, ...]:
    """Merge adjacent equal actions into the unique canonical run program."""

    canonical: list[ActionRun] = []
    for raw_run in runs:
        run = ActionRun(raw_run.action, raw_run.duration)
        if canonical and canonical[-1].action == run.action:
            previous = canonical[-1]
            canonical[-1] = ActionRun(previous.action, previous.duration + run.duration)
        else:
            canonical.append(run)
    return tuple(canonical)


def action_runs_from_sequence(actions: Sequence[int]) -> tuple[ActionRun, ...]:
    """Run-length encode a flat action sequence."""

    runs: list[ActionRun] = []
    for raw_action in actions:
        action = int(raw_action)
        if runs and runs[-1].action == action:
            previous = runs[-1]
            runs[-1] = ActionRun(action, previous.duration + 1)
        else:
            runs.append(ActionRun(action, 1))
    return tuple(runs)


class ActionProgramPolicy:
    """Portable deterministic open-loop action program."""

    def __init__(
        self,
        *,
        action_names: Sequence[str],
        action_runs: Sequence[ActionRun],
        fallback_action: int,
    ) -> None:
        self.action_names = tuple(str(name) for name in action_names)
        self.action_runs = canonicalize_action_runs(action_runs)
        self.fallback_action = int(fallback_action)
        self.action_space: gym.Space | None = None
        self.observation_space = None
        self._run_indices = np.zeros(1, dtype=np.int64)
        self._run_remaining = np.zeros(1, dtype=np.int64)
        self._validate_actions()

    def _validate_actions(self) -> None:
        count = len(self.action_names)
        if count < 1:
            raise ValueError("action program requires at least one action name")
        values = (*(run.action for run in self.action_runs), self.fallback_action)
        if any(action < 0 or action >= count for action in values):
            raise ValueError("action program contains an action outside its action-name table")

    @property
    def run_count(self) -> int:
        return len(self.action_runs)

    @property
    def step_count(self) -> int:
        return sum(run.duration for run in self.action_runs)

    @staticmethod
    def _batch_size(observation: Any) -> int:
        if isinstance(observation, Mapping):
            if not observation:
                return 1
            observation = next(iter(observation.values()))
        array = np.asarray(observation)
        return int(array.shape[0]) if array.ndim > 0 else 1

    def _ensure_lanes(self, count: int) -> None:
        if self._run_indices.shape != (count,):
            self._run_indices = np.zeros(count, dtype=np.int64)
            self._run_remaining = np.zeros(count, dtype=np.int64)

    def _peek(self, lane: int) -> int:
        index = int(self._run_indices[lane])
        return (
            self.action_runs[index].action
            if index < len(self.action_runs)
            else self.fallback_action
        )

    def _next_action(self, lane: int) -> int:
        index = int(self._run_indices[lane])
        if index >= len(self.action_runs):
            return self.fallback_action
        run = self.action_runs[index]
        if self._run_remaining[lane] == 0:
            self._run_remaining[lane] = run.duration
        self._run_remaining[lane] -= 1
        if self._run_remaining[lane] == 0:
            self._run_indices[lane] += 1
        return run.action

    def bind_action_space(self, action_space: gym.Space) -> None:
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError("action-program playback requires a discrete task action space")
        if int(action_space.n) != len(self.action_names):
            raise ValueError(
                "action-program action table does not match the playback environment action space"
            )
        self.action_space = action_space

    def reset_episode(self) -> None:
        self._run_indices.fill(0)
        self._run_remaining.fill(0)

    def reset_lanes(self, dones: Sequence[bool]) -> None:
        mask = np.asarray(dones, dtype=bool)
        self._ensure_lanes(int(mask.size))
        self._run_indices[mask] = 0
        self._run_remaining[mask] = 0

    def predict(self, observation: Any, deterministic: bool = False):
        # ``deterministic`` is an SB3 compatibility argument, not a semantic
        # mode for this fixed program. Both values preserve the declared runs.
        del deterministic
        count = self._batch_size(observation)
        self._ensure_lanes(count)
        actions = np.asarray(
            [self._next_action(lane) for lane in range(count)],
            dtype=np.int64,
        )
        return actions, None

    def _program_cursor(self, lane: int) -> dict[str, Any]:
        index = int(self._run_indices[lane])
        if index >= len(self.action_runs):
            action = self.fallback_action
            return {
                "run_index": index,
                "step_index": self.step_count,
                "current_run_remaining": 0,
                "remaining_steps": 0,
                "fallback": True,
                "action": action,
                "action_name": self.action_names[action],
            }
        current = self.action_runs[index]
        current_remaining = int(self._run_remaining[lane]) or current.duration
        later_remaining = sum(run.duration for run in self.action_runs[index + 1 :])
        remaining = current_remaining + later_remaining
        return {
            "run_index": index,
            "step_index": self.step_count - remaining,
            "current_run_remaining": current_remaining,
            "remaining_steps": remaining,
            "fallback": False,
            "action": current.action,
            "action_name": self.action_names[current.action],
        }

    def _decisions(self, observation: Any, *, advance: bool):
        from gradlab.play_debug import PolicyDecision

        count = self._batch_size(observation)
        self._ensure_lanes(count)
        decisions = []
        for lane in range(count):
            program = self._program_cursor(lane)
            action = self._next_action(lane) if advance else self._peek(lane)
            value = np.asarray(action, dtype=np.int64)
            decisions.append(
                PolicyDecision(
                    raw_action=value,
                    executed_action=value,
                    action_selection_mode="program",
                    distribution_kind=None,
                    mode=None,
                    program=program,
                    sampled=None,
                )
            )
        return tuple(decisions)

    def policy_decisions(
        self,
        observation: Any,
        *,
        action_selection_mode: str = "program",
    ):
        if action_selection_mode != "program":
            raise ValueError("action programs support only program action selection")
        return self._decisions(observation, advance=True)

    def inspect_policy_decisions(
        self,
        observation: Any,
        *,
        action_selection_mode: str = "program",
    ):
        if action_selection_mode != "program":
            raise ValueError("action programs support only program action selection")
        return self._decisions(observation, advance=False)

    def sample_policy_decision(self, observation: Any):
        return self.policy_decisions(
            observation,
            action_selection_mode="program",
        )[0]

    def inspect_policy_decision(self, observation: Any):
        return self.inspect_policy_decisions(
            observation,
            action_selection_mode="program",
        )[0]

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": ACTION_PROGRAM_SCHEMA_VERSION,
            "policy_type": ACTION_PROGRAM_POLICY_TYPE,
            "model_class": ACTION_PROGRAM_MODEL_CLASS,
            "action_names": list(self.action_names),
            "action_runs": [[run.action, run.duration] for run in self.action_runs],
            "fallback_action": self.fallback_action,
        }

    def save(
        self,
        path: str | Path,
        *,
        artifact_discriminator: str | None = None,
    ) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(destination, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                ACTION_PROGRAM_MEMBER,
                json.dumps(self.payload(), sort_keys=True, separators=(",", ":")) + "\n",
            )
            if artifact_discriminator is not None:
                discriminator = str(artifact_discriminator).strip()
                if not discriminator:
                    raise ValueError("action-program artifact_discriminator must not be empty")
                archive.writestr(
                    ACTION_PROGRAM_ARTIFACT_IDENTITY_MEMBER,
                    json.dumps(
                        {
                            "schema_version": 1,
                            "artifact_discriminator": discriminator,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n",
                )

    @classmethod
    def load(cls, path: str | Path) -> "ActionProgramPolicy":
        with zipfile.ZipFile(Path(path)) as archive:
            if ACTION_PROGRAM_MEMBER not in archive.namelist():
                raise ValueError(f"unsupported policy artifact: missing {ACTION_PROGRAM_MEMBER}")
            payload = json.loads(archive.read(ACTION_PROGRAM_MEMBER))
        if not isinstance(payload, Mapping):
            raise ValueError("action-program payload must be an object")
        expected_fields = {
            "schema_version",
            "policy_type",
            "model_class",
            "action_names",
            "action_runs",
            "fallback_action",
        }
        if set(payload) != expected_fields:
            missing = sorted(expected_fields - set(payload))
            unexpected = sorted(set(payload) - expected_fields)
            raise ValueError(
                f"action-program payload fields disagree; "
                f"missing={missing}, unexpected={unexpected}"
            )
        schema_version = int(payload.get("schema_version") or 0)
        if schema_version != ACTION_PROGRAM_SCHEMA_VERSION:
            raise ValueError("unsupported action-program schema version")
        if payload.get("policy_type") != ACTION_PROGRAM_POLICY_TYPE:
            raise ValueError("action-program payload has the wrong policy type")
        if payload.get("model_class") != ACTION_PROGRAM_MODEL_CLASS:
            raise ValueError("action-program payload has the wrong model class")
        return cls(
            action_names=payload["action_names"],
            action_runs=tuple(ActionRun(*run) for run in payload["action_runs"]),
            fallback_action=payload["fallback_action"],
        )
