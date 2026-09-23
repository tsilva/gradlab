from __future__ import annotations

import unittest

from gradlab.goal_catalog import goal_catalog_run_success_badges
from gradlab.training_success import (
    training_success_evidence,
    validate_training_success_evidence,
)


CRITERION = {
    "metric": "train/progress/bricks_destroyed_normalized/mean",
    "operator": ">=",
    "threshold": 0.5,
}


class TrainingSuccessTests(unittest.TestCase):
    def test_success_persists_when_later_training_sample_falls(self) -> None:
        evidence = training_success_evidence(
            CRITERION,
            [
                {"step": 1, "value": 0.49, "status": "published"},
                {"step": 2, "value": 0.51, "status": "published"},
                {"step": 3, "value": 0.48, "status": "published"},
            ],
        )
        self.assertEqual(evidence["status"], "met")
        self.assertEqual(evidence["first_met"], {"step": 2, "value": 0.51})
        self.assertIn("train/success", goal_catalog_run_success_badges({
            "state": "succeeded", "training_success": evidence,
        }))
        self.assertEqual(validate_training_success_evidence(evidence), evidence)

    def test_historical_peak_below_threshold_remains_unsuccessful(self) -> None:
        evidence = training_success_evidence(CRITERION, [
            {"step": 189161472, "value": 0.4992129623889923, "status": "published"},
            {"step": 191561728, "value": 0.49819444298744203, "status": "published"},
        ])
        self.assertEqual(evidence["status"], "not_met")
        self.assertEqual(evidence["best"]["step"], 189161472)
        self.assertNotIn("train/success", goal_catalog_run_success_badges({
            "state": "succeeded", "training_success": evidence,
        }))

    def test_unpublished_frames_cannot_supply_success(self) -> None:
        evidence = training_success_evidence(CRITERION, [
            {"step": 1, "value": 0.6, "status": "pending"},
            {"step": 2, "value": 0.4, "status": "published"},
        ])
        self.assertEqual(evidence["status"], "not_met")


if __name__ == "__main__":
    unittest.main()
