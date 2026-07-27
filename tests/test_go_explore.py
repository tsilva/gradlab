from __future__ import annotations

import unittest

import numpy as np

from rlab.go_explore import GoExploreSearch


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


if __name__ == "__main__":
    unittest.main()
