from __future__ import annotations

import struct
import tempfile
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from rlab.batch_runtime import BatchRuntime, ProviderDescriptor, SignalSpec
from rlab.state_archive import (
    ArchiveCurriculum,
    ArchiveCurriculumConfig,
    StateArchive,
    normalize_state_archive_config,
    validate_state_archive_runtime_contract,
)
from rlab.task_kernels import IdentityTaskDefinition


def archive_config(*, n_envs: int = 3, curriculum: bool = True) -> dict[str, Any]:
    value: dict[str, Any] = {
        "semantic_id": "state-archive-v1",
        "persistence": "durable",
        "restore_semantics": "continuation",
        "recorder": (
            {
                "mode": "cell_transition",
                "cell": {"signal": "score", "bucket_size": 50},
            }
            if curriculum
            else {"mode": "backend"}
        ),
        "curriculum": (
            {
                "archive_share": 0.34,
                "priority_metric": "value_error",
                "restore_entries": True,
            }
            if curriculum
            else None
        ),
    }
    normalized = normalize_state_archive_config(value, n_envs=n_envs)
    assert normalized is not None
    return normalized


class PortableBreakoutProvider:
    supports_live_snapshots = True
    live_snapshots_deterministic = True

    def __init__(self, num_envs: int = 3) -> None:
        self.num_envs = num_envs
        self.single_observation_space = gym.spaces.Box(0, 10_000, shape=(1,), dtype=np.int64)
        self.single_action_space = gym.spaces.Discrete(2)
        self.observations = np.zeros((num_envs, 1), dtype=np.int64)
        self.score = np.zeros(num_envs, dtype=np.int64)

    def _infos(self, mask: np.ndarray | None = None) -> dict[str, Any]:
        infos: dict[str, Any] = {"score": self.score.copy()}
        if mask is not None:
            infos.update(
                {
                    "_score": mask.copy(),
                    "start_id": np.full(self.num_envs, "Start", dtype=object),
                    "_start_id": mask.copy(),
                    "start_source": np.full(self.num_envs, "snapshot", dtype=object),
                    "_start_source": mask.copy(),
                }
            )
        return infos

    def get_state(self) -> tuple[bytes, ...]:
        return tuple(
            struct.pack(
                "<qq",
                int(self.score[lane]),
                int(self.observations[lane, 0]),
            )
            for lane in range(self.num_envs)
        )

    def set_state(self, states: Sequence[bytes], *, reset_mask: np.ndarray) -> None:
        for lane in np.flatnonzero(reset_mask):
            score, observation = struct.unpack("<qq", states[int(lane)])
            self.score[int(lane)] = score
            self.observations[int(lane), 0] = observation

    def capture_snapshots(self, mask: np.ndarray) -> tuple[Any | None, ...]:
        return tuple(
            self.get_state()[lane] if bool(mask[lane]) else None for lane in range(self.num_envs)
        )

    def reset(
        self,
        *,
        seed: int | Sequence[int | None] | None = None,
        options: Mapping[str, Any] | None = None,
    ):
        del seed
        options = dict(options or {})
        mask = np.asarray(options.get("reset_mask", np.ones(self.num_envs, dtype=np.bool_)))
        snapshots = tuple(options.get("snapshots", (None,) * self.num_envs))
        for lane in np.flatnonzero(mask):
            lane_index = int(lane)
            if snapshots[lane_index] is None:
                self.score[lane_index] = 0
                self.observations[lane_index, 0] = 0
            else:
                score, observation = struct.unpack("<qq", snapshots[lane_index])
                self.score[lane_index] = score
                self.observations[lane_index, 0] = observation
        return self.observations, self._infos(mask)

    def step(self, actions: Any):
        del actions
        self.observations[:, 0] += 1
        self.score += 50
        return (
            self.observations,
            np.ones(self.num_envs, dtype=np.float32),
            np.zeros(self.num_envs, dtype=np.bool_),
            np.zeros(self.num_envs, dtype=np.bool_),
            self._infos(),
        )

    def close(self) -> None:
        return None


class StateArchiveTests(unittest.TestCase):
    def test_new_shape_rejects_removed_snapshot_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "unexpected fields"):
            normalize_state_archive_config(
                {
                    "snapshot_share": 0.2,
                    "recorder": {"mode": "backend"},
                },
                n_envs=3,
            )

    def test_backend_archive_needs_no_curriculum_priority(self) -> None:
        common = {
            "n_envs": 3,
            "state_archive": archive_config(curriculum=False),
            "task": {"signals": {}},
        }
        validate_state_archive_runtime_contract(
            common,
            backend_id="rlab.go-explore",
            supported_priority_metrics=(),
        )

    def test_curriculum_uses_entry_ids_and_bounded_probabilities(self) -> None:
        config = ArchiveCurriculumConfig.from_mapping(archive_config(), n_envs=3)
        curriculum = ArchiveCurriculum(
            config,
            n_envs=3,
            run_seed=11,
            global_lane_ids=(0, 1, 2),
        )
        for index in range(5):
            cell_id = f"score:{index}"
            self.assertTrue(curriculum.admit(cell_id, f"entry-{index}"))
            selection = curriculum.sample(lane=0, episode_index=index)
            curriculum.close_episode(selection.cell_id)
            curriculum.submit_feedback(selection.cell_id, float(index + 1))
        metrics = curriculum.complete_rollout()
        self.assertLessEqual(metrics["sampling_probability_max"], 0.25 + 1e-12)
        self.assertGreaterEqual(metrics["sampling_effective_cell_count"], 4.0)

    def test_runtime_portable_round_trip_restores_all_lanes_of_state(self) -> None:
        provider = PortableBreakoutProvider()
        descriptor = ProviderDescriptor(
            provider_id="breakout-turbo-env",
            native_observation_space=provider.single_observation_space,
            native_action_space=provider.single_action_space,
            signal_schema={"score": SignalSpec("score", np.int64)},
            start_catalog=("Start",),
            supports_live_snapshots=True,
            live_snapshots_deterministic=True,
            snapshot_codec_id="breakout-turbo-env.state-v1",
            snapshot_compatibility_id="test-environment-v1",
        )
        kernel = IdentityTaskDefinition(signals={"score": "score"}).bind(
            descriptor, provider.num_envs
        )
        with tempfile.TemporaryDirectory() as root:
            runtime = BatchRuntime(
                provider,
                descriptor,
                kernel,
                run_seed=17,
                state_archive=archive_config(curriculum=False),
                state_archive_root=root,
            )
            receipt = runtime.preflight_state_archive_round_trip(seed=17)
            self.assertTrue(receipt["observation_exact"])
            self.assertTrue(receipt["one_step_continuation_exact"])
            summary = runtime.state_archive_summary()
            assert summary is not None
            self.assertEqual(summary["persistence"], "durable")
            self.assertEqual(summary["entry_count"], 1)
            runtime.close()
            reopened = StateArchive(
                Path(root),
                provider_id="breakout-turbo-env",
                codec_id="breakout-turbo-env.state-v1",
                compatibility_id="test-environment-v1",
            )
            self.assertEqual(reopened.entry_count, 1)


if __name__ == "__main__":
    unittest.main()
