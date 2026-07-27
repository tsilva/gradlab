from __future__ import annotations

import unittest

import numpy as np

from rlab.env import EnvConfig
from rlab.go_explore import GoExploreSearch
from rlab.training.go_explore import (
    GO_EXPLORE_PROVIDER_INFO_KEYS,
    _runtime_environment_config,
    normalize_config,
)


class GoExploreSearchTests(unittest.TestCase):
    @staticmethod
    def search() -> GoExploreSearch:
        return GoExploreSearch(
            n_envs=2,
            seed=17,
            action_names=("noop", "right", "jump"),
            fallback_action="noop",
            explore_steps=1,
            run_duration_mean=2.0,
            run_duration_max=4,
        )

    def test_durable_state_round_trip_preserves_exact_next_actions(self) -> None:
        search = self.search()
        search.initialize((b"initial-a", b"initial-b"), ("entry-a", "entry-b"))
        search.next_actions()
        observation = search.observe(
            rewards=np.asarray([1.0, 2.0]),
            dones=np.asarray([False, False]),
            cell_keys=(b"cell-a", b"cell-b"),
            progresses=np.asarray([8.0, 16.0]),
        )
        self.assertTrue(np.all(observation.archive_mask))
        self.assertTrue(np.all(observation.restart_mask))
        search.commit_archive(("entry-cell-a", "entry-cell-b"))
        search.take_completion_events()
        search.restart(observation.restart_mask)

        document = search.state_document(("lane-a", "lane-b"))
        restored = self.search()
        self.assertEqual(restored.restore_state(document), ("lane-a", "lane-b"))
        self.assertEqual(restored.state_document(("lane-a", "lane-b")), document)
        np.testing.assert_array_equal(restored.next_actions(), search.next_actions())

    def test_runtime_environment_selects_route_and_task_provider_info(self) -> None:
        config = EnvConfig(
            env_provider="supermariobrosnes-turbo",
            game="SuperMarioBros-Nes-v0",
            env_args={"info_filter": "all"},
            task={
                "id": "mario",
                "signals": {
                    "x": ["xscrollHi", "xscrollLo"],
                    "custom": "game_mode",
                },
            },
        )

        runtime_config = _runtime_environment_config(config)
        info_filter = runtime_config.env_args["info_filter"]

        self.assertEqual(info_filter["mode"], "all")
        self.assertEqual(
            set(info_filter["keys"]),
            set(GO_EXPLORE_PROVIDER_INFO_KEYS) | {"game_mode"},
        )
        self.assertEqual(config.env_args["info_filter"], "all")

    def test_backend_uses_compaction_without_legacy_archive_recovery(self) -> None:
        config = normalize_config(
            "rlab.go-explore",
            {"compaction_interval_steps": 500_000},
            label="backend",
        )

        self.assertEqual(config["compaction_interval_steps"], 500_000)
        with self.assertRaisesRegex(ValueError, "recovery_interval_steps"):
            normalize_config(
                "rlab.go-explore",
                {"recovery_interval_steps": 500_000},
                label="backend",
            )


if __name__ == "__main__":
    unittest.main()
