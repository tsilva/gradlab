from __future__ import annotations

import unittest
from pathlib import Path

from stable_retro_ppo.monitoring.state import (
    DeviceProbe,
    MonitorOptions,
    collect_state,
    devices_from_jobs,
    job_from_train_row,
    sample_jobs,
    target_label,
)


class MonitoringStateTests(unittest.TestCase):
    def test_target_label_summarizes_mixed_mario_state_probabilities(self) -> None:
        label = target_label(
            {
                "game": "SuperMarioBros-Nes-v0",
                "states": ["Level1-1", "Level1-2"],
                "state_probs": [0.5, 0.5],
            }
        )

        self.assertEqual(label, "Mario mixed 0.5 / 0.5")

    def test_target_label_compresses_repeated_fixed_states(self) -> None:
        label = target_label(
            {
                "game": "SuperMarioBros-Nes-v0",
                "states": ["Level1-1", "Level1-1", "Level1-2", "Level1-2"],
            }
        )

        self.assertEqual(label, "Mario L1-1 x2 + L1-2 x2")

    def test_devices_reflect_active_jobs(self) -> None:
        devices = devices_from_jobs(Path("."), sample_jobs())
        by_id = {device["id"]: device for device in devices}

        self.assertEqual(by_id["rtx4090"]["state"], "warning")
        self.assertIn("train-184", by_id["rtx4090"]["current_job"])
        self.assertEqual(by_id["modal"]["state"], "busy")
        self.assertEqual(by_id["rtx2060"]["state"], "available")

    def test_pending_jobs_do_not_make_device_busy(self) -> None:
        jobs = [
            {
                "id": "eval-4",
                "kind": "eval",
                "target": "checkpoint",
                "where": "beast-3 / RTX4090",
                "device_key": "rtx4090",
                "state": "pending",
                "progress": "",
                "attention": "",
                "details": {},
            }
        ]

        devices = devices_from_jobs(Path("."), jobs)
        by_id = {device["id"]: device for device in devices}

        self.assertEqual(by_id["rtx4090"]["state"], "available")
        self.assertEqual(by_id["rtx4090"]["current_job"], "")
        self.assertEqual(by_id["rtx4090"]["queued_job"], "eval-4")
        self.assertEqual(by_id["rtx4090"]["attention"], "1 queued")
        self.assertEqual(by_id["rtx4090"]["details"]["running jobs"], "")
        self.assertEqual(by_id["rtx4090"]["details"]["queued jobs"], "eval-4")

    def test_train_job_includes_full_payload_for_queue_inspection(self) -> None:
        job = job_from_train_row(
            {
                "id": 12,
                "goal_slug": "goal",
                "spec_slug": "spec",
                "profile_id": "rtx4090-screening",
                "train_config": {"game": "SuperMarioBros-Nes-v0", "n_envs": 32},
                "status": "pending",
                "lease_owner": None,
                "heartbeat_at": None,
                "lease_expires_at": None,
                "error": None,
                "artifact_refs": [],
                "metrics_json": {},
                "wandb_url": None,
                "run_name": "run-12",
                "job_payload": {
                    "id": 12,
                    "profile_id": "rtx4090-screening",
                    "train_config": {"game": "SuperMarioBros-Nes-v0", "n_envs": 32},
                    "status": "pending",
                },
                "result_payload": None,
            }
        )

        self.assertEqual(job["payload"]["table"], "train_jobs")
        self.assertEqual(job["payload"]["config_key"], "train_config")
        self.assertEqual(job["payload"]["job"]["train_config"]["n_envs"], 32)
        self.assertIn("train_config", job["payload"]["schema"])

    def test_offline_probe_overrides_idle_device_state(self) -> None:
        devices = devices_from_jobs(
            Path("."),
            [],
            probes={
                "rtx2060": DeviceProbe(
                    ok=False,
                    label="SSH timeout",
                    detail="connection timed out",
                )
            },
        )
        by_id = {device["id"]: device for device in devices}

        self.assertEqual(by_id["rtx2060"]["state"], "offline")
        self.assertEqual(by_id["rtx2060"]["attention"], "unreachable")
        self.assertEqual(by_id["rtx2060"]["last_check"], "unreachable")
        self.assertEqual(by_id["rtx2060"]["details"]["reachability"], "unreachable")
        self.assertEqual(by_id["rtx2060"]["details"]["health check"], "SSH timeout")

    def test_sample_state_contains_jobs_and_devices(self) -> None:
        state = collect_state(MonitorOptions(repo_root=Path("."), sample=True))

        self.assertEqual(state["source"]["campaign"], "sample")
        self.assertGreaterEqual(len(state["jobs"]), 1)
        self.assertGreaterEqual(len(state["devices"]), 1)


if __name__ == "__main__":
    unittest.main()
