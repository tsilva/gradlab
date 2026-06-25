from __future__ import annotations

import unittest

from rlab.train import checkpoint_save_frequency


class TrainTests(unittest.TestCase):
    def test_checkpoint_save_frequency_disables_zero_or_negative(self) -> None:
        self.assertIsNone(checkpoint_save_frequency(0, 2))
        self.assertIsNone(checkpoint_save_frequency(-1, 2))

    def test_checkpoint_save_frequency_scales_by_vec_envs(self) -> None:
        self.assertEqual(checkpoint_save_frequency(100_000, 2), 50_000)
        self.assertEqual(checkpoint_save_frequency(1, 32), 1)


if __name__ == "__main__":
    unittest.main()
