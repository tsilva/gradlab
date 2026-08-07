from __future__ import annotations

import signal
import unittest
from unittest import mock

from stable_baselines3.common.callbacks import BaseCallback

from gradlab.callbacks import CallbackHelper, GradLabCallback
from gradlab.schedules import EntropyCoefficientScheduleHelper
from gradlab.training.sb3_helpers import (
    GracefulStopHelper,
    install_on_policy_safe_boundary_stop,
)
from gradlab.training.sb3_on_policy import checkpoint_save_frequency
from gradlab.training_backend import GracefulStopFlag
from gradlab.train import graceful_stop_signal_scope
from gradlab.seeds import (
    DEFAULT_EVAL_SEED,
    validate_eval_seed,
)


class TrainTests(unittest.TestCase):
    def test_local_training_treats_sigint_as_a_graceful_stop(self) -> None:
        stop_flag = GracefulStopFlag()

        with (
            mock.patch("gradlab.train.signal.getsignal", return_value=signal.SIG_DFL),
            mock.patch("gradlab.train.signal.signal") as register,
        ):
            with graceful_stop_signal_scope(stop_flag, include_sigint=True):
                handlers = {call.args[0]: call.args[1] for call in register.call_args_list}
                self.assertIn(signal.SIGINT, handlers)
                handlers[signal.SIGINT](signal.SIGINT, None)
            restored = [call for call in register.call_args_list if call.args[1] == signal.SIG_DFL]

        self.assertTrue(stop_flag.requested)
        self.assertEqual(stop_flag.reason, "SIGINT")
        self.assertEqual(len(restored), len(handlers))

    def test_only_gradlab_callback_implements_the_sb3_callback_protocol(self) -> None:
        self.assertTrue(issubclass(GradLabCallback, BaseCallback))
        self.assertTrue(issubclass(GracefulStopHelper, CallbackHelper))
        self.assertFalse(issubclass(GracefulStopHelper, BaseCallback))

    def test_gradlab_callback_drives_entropy_schedule_without_emitting_a_metric(self) -> None:
        class Logger:
            def __init__(self) -> None:
                self.records: dict[str, float] = {}

            def record(self, key: str, value: float) -> None:
                self.records[key] = value

        class Model:
            def __init__(self) -> None:
                self.ent_coef = 0.0
                self.logger = Logger()

        model = Model()
        callback = GradLabCallback(
            [EntropyCoefficientScheduleHelper(0.1, 0.0, schedule_timesteps=100)]
        )
        callback.model = model  # type: ignore[assignment]
        callback._on_training_start()
        self.assertEqual(model.ent_coef, 0.1)

        callback.num_timesteps = 50
        self.assertTrue(callback._on_step())
        self.assertAlmostEqual(model.ent_coef, 0.05)
        self.assertEqual(model.logger.records, {})

    def test_checkpoint_save_frequency_disables_zero_or_negative(self) -> None:
        self.assertIsNone(checkpoint_save_frequency(0, 2))
        self.assertIsNone(checkpoint_save_frequency(-1, 2))

    def test_checkpoint_save_frequency_scales_by_vec_envs(self) -> None:
        self.assertEqual(checkpoint_save_frequency(500_000, 2), 250_000)
        self.assertEqual(checkpoint_save_frequency(1, 32), 1)

    def test_eval_seed_rejects_training_range(self) -> None:
        self.assertEqual(DEFAULT_EVAL_SEED, 10000)
        self.assertEqual(validate_eval_seed(10000), 10000)
        with self.assertRaisesRegex(ValueError, "reserved for training"):
            validate_eval_seed(9999)

    def test_graceful_stop_callback_finishes_the_current_rollout(self) -> None:
        stop_flag = GracefulStopFlag()
        callback = GracefulStopHelper(stop_flag)
        callback.num_timesteps = 123

        stop_flag.request("SIGUSR1")

        callback.acknowledge_safe_boundary(num_timesteps=123)

        self.assertTrue(callback.logged)

    def test_on_policy_loop_stops_before_collecting_another_rollout(self) -> None:
        stop_flag = GracefulStopFlag()
        graceful_stop = GracefulStopHelper(stop_flag)

        class Model:
            def __init__(self) -> None:
                self.num_timesteps = 0
                self.train_calls = 0
                self.collect_calls = 0

            def collect_rollouts(
                self,
                _env,
                _callback,
                _rollout_buffer,
                *,
                n_rollout_steps,
            ) -> bool:
                self.collect_calls += 1
                self.num_timesteps += n_rollout_steps
                stop_flag.request("SIGUSR1")
                return True

            def train(self) -> None:
                self.train_calls += 1

            def learn(self, *, total_timesteps: int) -> None:
                while self.num_timesteps < total_timesteps:
                    if not self.collect_rollouts(
                        object(),
                        object(),
                        object(),
                        n_rollout_steps=8,
                    ):
                        break
                    self.train()

        model = Model()
        install_on_policy_safe_boundary_stop(
            model,
            graceful_stop=graceful_stop,
        )
        model.learn(total_timesteps=1_000)

        self.assertEqual(model.num_timesteps, 8)
        self.assertEqual(model.train_calls, 1)
        self.assertEqual(model.collect_calls, 1)
        self.assertTrue(graceful_stop.logged)


if __name__ == "__main__":
    unittest.main()
