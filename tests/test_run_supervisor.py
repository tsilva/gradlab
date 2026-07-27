from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

from rlab.early_stop import MetricEarlyStopStateMachine, MetricSample
from rlab.eval_backend import EvalHandle, EvalPoll
from rlab.file_utils import atomic_write_json
from rlab.metric_names import TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_MEAN
from rlab.policy_bundle import (
    build_recipe_document,
    canonical_json_sha256,
    evaluation_contract_sha256,
)
from rlab.r2_store import BucketConfig, RunStorageConfig
from rlab.recipe_documents import compose_train_document
from rlab.run_authority import RunAuthority
from rlab.run_contracts import (
    CheckpointManifest,
    EarlyStopReceipt,
    EvalResult,
    PromotionReceipt,
    RunManifest,
    new_attempt_id,
    new_run_id,
    utc_now,
)
from rlab.run_supervisor import (
    RunSupervisor,
    _bind_evaluation_contract,
    _summary_scalar,
    _terminal_outcome,
)


SOURCE_SHA = "a" * 40
BUILD_SOURCE_SHA = "f" * 40
RUNTIME_INPUT_SHA256 = "e" * 64
IMAGE = "docker:registry.example/rlab@sha256:" + "b" * 64
GOAL = Path("experiments/goals/SuperMarioBros-Nes-v0/Level1-1/_goal.yaml")
RECIPE = GOAL.parent / "recipes" / "ppo.yaml"


class FailingSpawnBackend:
    def submit(self, intent):
        raise RuntimeError("connection outcome unknown")

    def poll(self, handle: EvalHandle) -> EvalPoll:
        return EvalPoll(status="running")

    def cancel(self, handle: EvalHandle) -> None:
        return None


class RunSupervisorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.storage = RunStorageConfig(
            control=BucketConfig(uri=f"file://{root}/control"),
            evaluation=BucketConfig(uri=f"file://{root}/eval"),
            models=BucketConfig(
                uri=f"file://{root}/models",
                public_base_url="https://models.example",
            ),
        )
        self.authority = RunAuthority(self.storage)
        document = compose_train_document(GOAL, RECIPE)
        self.run_id = new_run_id()
        self.asset = {
            "schema_version": 2,
            "game": "SuperMarioBros-Nes-v0",
            "filename": "mario.nes",
            "size_bytes": 1,
            "sha256": "c" * 64,
            "object_uri": self.authority.evaluation.uri("rom.nes"),
            "provider_rom_identity": "d" * 40,
            "provider_rom_identity_algorithm": "sha1-provider-body-v1",
        }
        contract_document = dict(document)
        contract_config = dict(contract_document["train_config"])
        contract_config["rom_asset_manifest"] = self.asset
        contract_config["checkpoint_eval_backend"] = "modal"
        contract_document["train_config"] = contract_config
        portable_recipe = build_recipe_document(
            contract_document,
            repo_root=Path.cwd(),
            source_commit=SOURCE_SHA,
            run_description="supervisor unit test",
            seed=123,
            runtime_image_ref=IMAGE,
        )
        self.manifest = RunManifest(
            run_id=self.run_id,
            attempt_id=new_attempt_id(),
            created_at=utc_now(),
            source_sha=SOURCE_SHA,
            image_digest=IMAGE,
            goal_slug="SuperMarioBros-Nes-v0/Level1-1",
            goal_sha256=str(document["train_config"]["effective_goal_contract_sha256"]),
            recipe_slug="ppo",
            recipe_sha256=canonical_json_sha256(portable_recipe),
            recipe_overrides=(),
            environment_sha256=str(document["environment_hash"]).removeprefix("sha256:"),
            seed=123,
            run_description="supervisor unit test",
            compute={
                "request": {
                    "kind": "local",
                    "target": "b3",
                    "max_price": None,
                    "max_cost_usd": None,
                    "allow_on_demand": False,
                    "max_duration_seconds": 3600,
                },
                "selected": {
                    "kind": "local",
                    "target": "b3",
                    "max_price": None,
                    "max_cost_usd": None,
                    "allow_on_demand": False,
                    "max_duration_seconds": 3600,
                },
                "dstack_task": self.run_id,
                "runtime_workflow_run_id": "123",
                "runtime_input_sha256": RUNTIME_INPUT_SHA256,
                "runtime_build_source_sha": BUILD_SOURCE_SHA,
            },
            wandb={
                "run_id": self.run_id,
                "entity": "entity",
                "project": "project",
                "display_name": f"Level1-1__ppo__s123__{self.run_id[5:13]}",
                "group": "cohort::SuperMarioBros-Nes-v0/Level1-1::ppo::base",
                "url": f"https://wandb.ai/entity/project/runs/{self.run_id}",
            },
            modal={
                "enabled": True,
                "environment_name": "rlab-eval",
                "app_name": f"rlab-eval-v2-{SOURCE_SHA[:12]}",
                "function_name": "evaluate_checkpoint",
                "deployment_source_sha": SOURCE_SHA,
                "rom_asset_manifest": self.asset,
            },
            storage=self.storage.manifest_locations(),
        )
        self.authority.create_manifest(self.manifest)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def supervisor(self) -> RunSupervisor:
        root = Path(self.temporary.name)
        return RunSupervisor(
            manifest_uri=self.authority.control.uri(f"runs/{self.run_id}/manifest.json"),
            storage=self.storage,
            eval_backend=FailingSpawnBackend(),
            repo_root=Path.cwd(),
            work_root=root / "work",
        )

    def test_runtime_verification_uses_build_identity_and_runtime_input(self) -> None:
        supervisor = self.supervisor()
        with (
            patch.dict("os.environ", {"RLAB_ORCHESTRATOR": "dstack"}),
            patch.object(
                supervisor.runtime,
                "runtime_contract",
                return_value={
                    "runtime_build_source_sha": BUILD_SOURCE_SHA,
                    "runtime_input_sha256": RUNTIME_INPUT_SHA256,
                },
            ),
        ):
            supervisor.validate_runtime()

    def test_startup_failure_creates_resumable_terminal_receipt(self) -> None:
        supervisor = self.supervisor()
        with patch.object(
            supervisor,
            "validate_runtime",
            side_effect=RuntimeError("invalid runtime"),
        ):
            self.assertEqual(supervisor.run(), 1)

        receipt = self.authority.control.get_json(
            f"runs/{self.run_id}/attempts/{self.manifest.attempt_id}/terminal.json"
        )
        self.assertEqual(receipt["state"], "resumable_failure")
        self.assertEqual(receipt["stop_reason"], "supervisor_startup_failure")
        self.assertEqual(receipt["drain"]["phase"], "startup")
        self.assertIn("invalid runtime", receipt["drain"]["failure"])

        with (
            patch.dict("os.environ", {"RLAB_ORCHESTRATOR": "dstack"}),
            patch.object(
                supervisor.runtime,
                "runtime_contract",
                return_value={
                    "runtime_build_source_sha": SOURCE_SHA,
                    "runtime_input_sha256": RUNTIME_INPUT_SHA256,
                },
            ),
            self.assertRaisesRegex(RuntimeError, "runtime build source SHA"),
        ):
            supervisor.validate_runtime()

    def test_recovery_failure_after_lease_creates_terminal_receipt(self) -> None:
        supervisor = self.supervisor()
        with (
            patch.object(supervisor, "validate_runtime"),
            patch.object(supervisor, "materialize"),
            patch.object(
                supervisor,
                "_recover_durable_state",
                side_effect=RuntimeError("recovery exploded"),
            ),
        ):
            self.assertEqual(supervisor.run(), 1)

        receipt = self.authority.control.get_json(
            f"runs/{self.run_id}/attempts/{self.manifest.attempt_id}/terminal.json"
        )
        self.assertEqual(receipt["state"], "resumable_failure")
        self.assertEqual(receipt["stop_reason"], "supervisor_startup_failure")
        self.assertEqual(receipt["drain"]["phase"], "startup/recovery")
        self.assertIn("recovery exploded", receipt["drain"]["failure"])

    def test_learner_log_tail_is_bounded_and_preserves_latest_failure(self) -> None:
        supervisor = self.supervisor()
        supervisor.learner_log_path.parent.mkdir(parents=True, exist_ok=True)
        supervisor.learner_log_path.write_bytes(
            b"discarded-prefix\n" + b"x" * 128 + b"\nlatest learner traceback\n"
        )

        tail = supervisor._learner_log_tail(max_bytes=64)

        self.assertNotIn("discarded-prefix", tail)
        self.assertIn("latest learner traceback", tail)

    def test_training_only_contract_omits_null_eval_contract(self) -> None:
        config = {"checkpoint_eval_contract": None}
        contract = _bind_evaluation_contract(
            config,
            recipe_document={},
            evaluation_required=False,
        )

        self.assertEqual(contract, {})
        self.assertNotIn("checkpoint_eval_contract", config)

    def test_generic_eval_receipts_do_not_hardcode_mario_episode_count(self) -> None:
        EvalResult(
            run_id=self.run_id,
            checkpoint_id="checkpoint-1-" + "a" * 16,
            idempotency_key="b" * 64,
            modal_call_id="fc-bandit",
            status="accepted",
            episode_results=[{}] * 256,
            aggregates={},
            timings={},
            evidence_sha256=[],
            completed_at=utc_now(),
        ).validate()
        PromotionReceipt(
            run_id=self.run_id,
            checkpoint_id="checkpoint-1-" + "a" * 16,
            checkpoint_step=1,
            eval_idempotency_key="b" * 64,
            eval_result_sha256="c" * 64,
            accepted_episode_count=256,
            promoted_at=utc_now(),
        ).validate()

    def test_wandb_summary_subdict_is_normalized(self) -> None:
        class SummarySubDictLike:
            def get(self, key):
                return {"max": 10}.get(key)

        self.assertEqual(_summary_scalar(SummarySubDictLike()), 10)

    def test_unaccepted_goal_is_a_clean_scientific_failure(self) -> None:
        state, stop_reason = _terminal_outcome(
            cancel_requested=False,
            failure=None,
            evaluation_required=True,
            promotion=None,
            early_stop=None,
        )

        self.assertEqual(state, "failed")
        self.assertEqual(stop_reason, "training_cap_without_acceptance")

    def test_supervisor_fault_remains_resumable(self) -> None:
        state, stop_reason = _terminal_outcome(
            cancel_requested=False,
            failure=RuntimeError("network failure"),
            evaluation_required=True,
            promotion=None,
            early_stop=None,
        )

        self.assertEqual(state, "resumable_failure")
        self.assertEqual(stop_reason, "supervisor_failure")

    def _early_stop_receipt(self, *, outcome: str = "failure") -> EarlyStopReceipt:
        return EarlyStopReceipt(
            run_id=self.run_id,
            attempt_id=self.manifest.attempt_id,
            condition_id="return_plateau",
            matched_condition_ids=("return_plateau",),
            outcome=outcome,  # type: ignore[arg-type]
            trigger="no_improvement",
            metric="train/episode/return/shaped/from/target/mean",
            metric_step=2_000_000,
            value=650.0,
            best_value=650.0,
            elapsed_steps=1_000_000,
            patience_progress=1.0,
            condition={
                "metric": "train/episode/return/shaped/from/target/mean",
                "trigger": "no_improvement",
            },
            early_stop_config_sha256="d" * 64,
            decision_sha256="e" * 64,
            recorded_at=utc_now(),
        )

    def test_failure_early_stop_is_a_clean_scientific_failure(self) -> None:
        receipt = self._early_stop_receipt()

        state, stop_reason = _terminal_outcome(
            cancel_requested=False,
            failure=None,
            evaluation_required=True,
            promotion=None,
            early_stop=receipt,
        )

        self.assertEqual(state, "failed")
        self.assertEqual(stop_reason, "early_stop_failure:return_plateau")

    def test_training_only_success_early_stop_succeeds_the_attempt(self) -> None:
        receipt = self._early_stop_receipt(outcome="success")

        state, stop_reason = _terminal_outcome(
            cancel_requested=False,
            failure=None,
            evaluation_required=False,
            promotion=None,
            early_stop=receipt,
        )

        self.assertEqual(state, "succeeded")
        self.assertEqual(stop_reason, "early_stop_success:return_plateau")

    def test_training_success_cannot_establish_evaluated_goal_acceptance(self) -> None:
        receipt = self._early_stop_receipt(outcome="success")

        state, stop_reason = _terminal_outcome(
            cancel_requested=False,
            failure=None,
            evaluation_required=True,
            promotion=None,
            early_stop=receipt,
        )

        self.assertEqual(state, "failed")
        self.assertEqual(
            stop_reason,
            "early_stop_success_without_acceptance:return_plateau",
        )

    def test_evaluation_promotion_overrides_failure_early_stop(self) -> None:
        promotion = PromotionReceipt(
            run_id=self.run_id,
            checkpoint_id="checkpoint-1-" + "a" * 16,
            checkpoint_step=1,
            eval_idempotency_key="b" * 64,
            eval_result_sha256="c" * 64,
            accepted_episode_count=100,
            promoted_at=utc_now(),
        )

        state, stop_reason = _terminal_outcome(
            cancel_requested=False,
            failure=None,
            evaluation_required=True,
            promotion=promotion,
            early_stop=self._early_stop_receipt(),
        )

        self.assertEqual(state, "succeeded")
        self.assertEqual(stop_reason, "completed_after_eval_acceptance")

    def test_supervisor_validates_and_persists_learner_early_stop_decision(self) -> None:
        supervisor = self.supervisor()
        config = {
            "conditions": {
                "return_plateau": {
                    "metric": TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_MEAN,
                    "trigger": "no_improvement",
                    "direction": "maximize",
                    "min_delta": 0.01,
                    "delta_mode": "relative",
                    "start_after_steps": 0,
                    "patience_steps": 10,
                    "outcome": "failure",
                    "action": "stop",
                }
            }
        }
        machine = MetricEarlyStopStateMachine(config)
        machine.update(
            {
                TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_MEAN: MetricSample(
                    value=100.0,
                    step=0,
                )
            }
        )
        update = machine.update(
            {
                TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_MEAN: MetricSample(
                    value=100.0,
                    step=10,
                )
            }
        )
        self.assertIsNotNone(update.stop_decision)
        supervisor.train_config = {"early_stop": machine.config}
        decision_path = (
            supervisor.run_dir
            / f"early_stop_decision-{self.manifest.attempt_id}.json"
        )
        atomic_write_json(decision_path, update.stop_decision or {})

        receipt = supervisor._resolve_early_stop_receipt()

        self.assertIsNotNone(receipt)
        assert receipt is not None
        self.assertEqual(receipt.condition_id, "return_plateau")
        self.assertEqual(receipt.outcome, "failure")
        stored = self.authority.early_stop_receipt(
            run_id=self.run_id,
            attempt_id=self.manifest.attempt_id,
        )
        self.assertEqual(stored["decision_sha256"], receipt.decision_sha256)
        tampered = dict(stored)
        tampered["outcome"] = "success"
        self.authority.control.put_json(
            (
                f"runs/{self.run_id}/attempts/"
                f"{self.manifest.attempt_id}/early-stop.json"
            ),
            tampered,
            create_only=False,
        )
        decision_path.unlink()
        with self.assertRaisesRegex(ValueError, "does not match"):
            supervisor._resolve_early_stop_receipt()

    def test_supervisor_rejects_tampered_learner_early_stop_decision(self) -> None:
        supervisor = self.supervisor()
        config = {
            "conditions": {
                "clear": {
                    "metric": "train/outcome/success/window_100/rate/min",
                    "trigger": "threshold",
                    "operator": ">=",
                    "threshold": 1.0,
                    "patience_steps": 0,
                    "outcome": "success",
                    "action": "stop",
                }
            }
        }
        machine = MetricEarlyStopStateMachine(config)
        update = machine.update(
            {
                "train/outcome/success/window_100/rate/min": MetricSample(
                    value=1.0,
                    step=10,
                )
            }
        )
        decision = dict(update.stop_decision or {})
        decision["outcome"] = "failure"
        supervisor.train_config = {"early_stop": machine.config}
        atomic_write_json(
            supervisor.run_dir
            / f"early_stop_decision-{self.manifest.attempt_id}.json",
            decision,
        )

        with self.assertRaisesRegex(ValueError, "does not match"):
            supervisor._resolve_early_stop_receipt()

    def test_drain_only_recovery_preserves_prior_early_stop_outcome(self) -> None:
        original = self.supervisor()
        config = {
            "conditions": {
                "return_plateau": {
                    "metric": TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_MEAN,
                    "trigger": "no_improvement",
                    "direction": "maximize",
                    "min_delta": 0.01,
                    "delta_mode": "relative",
                    "start_after_steps": 0,
                    "patience_steps": 10,
                    "outcome": "failure",
                    "action": "stop",
                }
            }
        }
        machine = MetricEarlyStopStateMachine(config)
        machine.update(
            {
                TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_MEAN: MetricSample(
                    value=100.0,
                    step=0,
                )
            }
        )
        update = machine.update(
            {
                TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_MEAN: MetricSample(
                    value=100.0,
                    step=10,
                )
            }
        )
        original.train_config = {"early_stop": machine.config}
        atomic_write_json(
            original.run_dir
            / f"early_stop_decision-{self.manifest.attempt_id}.json",
            update.stop_decision or {},
        )
        originating_receipt = original._resolve_early_stop_receipt()
        self.assertIsNotNone(originating_receipt)

        retry_manifest = replace(
            self.manifest,
            attempt_id=new_attempt_id(),
            created_at="9999-01-01T00:00:00Z",
            compute={**self.manifest.compute, "recovery_mode": "drain-only"},
        )
        self.authority.create_attempt_manifest(retry_manifest)
        retry = RunSupervisor(
            manifest_uri=self.authority.control.uri(
                f"runs/{self.run_id}/attempts/{retry_manifest.attempt_id}/manifest.json"
            ),
            storage=self.storage,
            eval_backend=FailingSpawnBackend(),
            repo_root=Path.cwd(),
            work_root=Path(self.temporary.name) / "retry-work",
        )
        retry.train_config = {"early_stop": machine.config}

        recovered = retry._resolve_early_stop_receipt()

        assert recovered is not None
        assert originating_receipt is not None
        self.assertEqual(recovered.condition_id, originating_receipt.condition_id)
        self.assertEqual(recovered.outcome, originating_receipt.outcome)
        self.assertEqual(recovered.decision_sha256, originating_receipt.decision_sha256)
        self.assertEqual(recovered.attempt_id, self.manifest.attempt_id)
        state, stop_reason = _terminal_outcome(
            cancel_requested=False,
            failure=None,
            evaluation_required=False,
            promotion=None,
            early_stop=recovered,
        )
        self.assertEqual(state, "failed")
        self.assertEqual(stop_reason, "early_stop_failure:return_plateau")

    def test_wandb_remote_probe_survives_sdk_finish(self) -> None:
        class RemoteRun:
            summary = {"orchestration/event_seq": {"max": 10}}

        class Api:
            def flush(self) -> None:
                return None

            def run(self, path: str) -> RemoteRun:
                self.path = path
                return RemoteRun()

        supervisor = self.supervisor()
        supervisor.store.init()
        supervisor.projector = None
        supervisor.wandb_run_path = f"entity/project/{self.run_id}"
        api = Api()
        with patch("wandb.Api", return_value=api):
            supervisor._probe_wandb_remote(
                10.0,
                local_high_water=10,
                force=True,
            )

        self.assertEqual(api.path, supervisor.wandb_run_path)
        self.assertEqual(supervisor.wandb_remote_high_water, 10)

    def test_materializes_exact_mario_acceptance_contract(self) -> None:
        supervisor = self.supervisor()
        with patch("rlab.run_supervisor.verify_rom_file"):
            supervisor.materialize()
        self.assertEqual(supervisor.train_config["timesteps"], 50_000_000)
        self.assertEqual(supervisor.train_config["checkpoint_freq"], 250_000)
        self.assertEqual(supervisor.train_config["n_envs"], 16)
        self.assertEqual(supervisor.train_config["run_name"], self.run_id)
        self.assertEqual(
            supervisor.train_config["wandb_display_name"],
            f"Level1-1__ppo__s123__{self.run_id[5:13]}",
        )
        self.assertEqual(
            supervisor.train_config["wandb_group"],
            "cohort::SuperMarioBros-Nes-v0/Level1-1::ppo::base",
        )
        self.assertEqual(supervisor.train_config["recipe_overrides"], [])
        self.assertEqual(supervisor.train_config["recipe_variant_id"], "base")
        self.assertIn("recipe_variant:base", supervisor.train_config["wandb_tags"])
        self.assertEqual(supervisor.eval_contract["episodes"], 100)
        self.assertEqual(
            supervisor.eval_contract["acceptance"],
            [
                {
                    "metric": "eval/full/outcome/success/rate/min",
                    "operator": ">=",
                    "threshold": 1.0,
                }
            ],
        )

    def test_materializes_launch_time_recipe_variant_metadata(self) -> None:
        supervisor = self.supervisor()
        overrides = ("train.timesteps=50000000",)
        document = compose_train_document(
            GOAL,
            RECIPE,
            recipe_overrides=overrides,
        )
        contract_document = dict(document)
        contract_config = dict(contract_document["train_config"])
        contract_config["rom_asset_manifest"] = self.asset
        contract_config["checkpoint_eval_backend"] = "modal"
        contract_document["train_config"] = contract_config
        portable_recipe = build_recipe_document(
            contract_document,
            repo_root=Path.cwd(),
            source_commit=SOURCE_SHA,
            run_description=self.manifest.run_description,
            seed=self.manifest.seed,
            runtime_image_ref=IMAGE,
        )
        supervisor.manifest = replace(
            supervisor.manifest,
            recipe_overrides=overrides,
            recipe_sha256=canonical_json_sha256(portable_recipe),
        )

        with patch("rlab.run_supervisor.verify_rom_file"):
            supervisor.materialize()

        self.assertEqual(supervisor.train_config["recipe_overrides"], list(overrides))
        self.assertRegex(supervisor.train_config["recipe_variant_id"], r"^v-[0-9a-f]{8}$")
        self.assertIn(
            f"recipe_variant:{supervisor.train_config['recipe_variant_id']}",
            supervisor.train_config["wandb_tags"],
        )

    def test_accepted_eval_metrics_ignore_private_r2_diagnostics(self) -> None:
        supervisor = self.supervisor()
        supervisor.store.init()
        result = EvalResult(
            run_id=self.run_id,
            checkpoint_id="checkpoint-4500000-" + "e" * 16,
            idempotency_key="a" * 64,
            modal_call_id="fc-accepted",
            status="accepted",
            episode_results=[
                {
                    "start_state": "Level1-1",
                    "return": 1.0,
                    "steps": 1,
                    "outcome": "success",
                }
            ]
            * 100,
            aggregates={"failure_count": 0},
            timings={},
            evidence_sha256=[],
            completed_at=utc_now(),
        )
        supervisor._record_eval_metrics(
            {
                "checkpoint_step": 4_500_000,
                "idempotency_key": result.idempotency_key,
                "intent": {"execution_contract": {"episodes": 100}},
            },
            result,
            {
                "duration_seconds": 1.0,
                "metrics": {
                    "death_count": 0,
                    "success_count": 100,
                    "eval/full/episode/count": 100,
                },
            },
        )

        self.assertEqual(
            supervisor.store.latest_metric("eval/full/episode/count"),
            100,
        )
        self.assertIsNone(supervisor.store.latest_metric("death_count"))
        self.assertIsNone(supervisor.store.latest_metric("success_count"))

    def test_failure_wait_renews_writer_lease(self) -> None:
        supervisor = self.supervisor()
        learner = MagicMock()
        learner.poll.side_effect = [None, 0]
        supervisor.learner = learner
        with (
            patch.object(supervisor, "_renew_lease") as renew,
            patch.object(supervisor.clock, "sleep"),
        ):
            self.assertTrue(supervisor._wait_for_learner_exit_with_lease(30))
        renew.assert_called_once()

    def test_accepted_eval_requests_stop_before_metric_projection(self) -> None:
        supervisor = self.supervisor()
        result = EvalResult(
            run_id=self.run_id,
            checkpoint_id="checkpoint-4500000-" + "e" * 16,
            idempotency_key="a" * 64,
            modal_call_id="fc-accepted",
            status="accepted",
            episode_results=[{}] * 100,
            aggregates={},
            timings={},
            evidence_sha256=[],
            completed_at=utc_now(),
        )
        row = {
            "checkpoint_step": 4_500_000,
            "idempotency_key": result.idempotency_key,
        }
        events: list[str] = []
        with (
            patch.object(supervisor.authority, "eval_result", return_value={}),
            patch.object(supervisor, "_verified_result", return_value=result),
            patch.object(supervisor.authority, "put_verified_eval_result"),
            patch.object(supervisor.store, "mark_eval_terminal"),
            patch.object(
                supervisor,
                "_request_learner_stop",
                side_effect=lambda _reason: events.append("stop"),
            ),
            patch.object(
                supervisor.store,
                "mark_stop_requested",
                return_value=0.0,
            ),
            patch.object(supervisor.store, "append_metrics"),
            patch.object(
                supervisor,
                "_record_eval_metrics",
                side_effect=lambda *_args: events.append("metrics"),
            ),
            patch.object(supervisor.clock, "time", return_value=0.0),
        ):
            self.assertTrue(supervisor._observe_result(row))

        self.assertEqual(events, ["stop", "metrics"])

    def test_ambiguous_modal_spawn_is_not_immediately_repeated(self) -> None:
        supervisor = self.supervisor()
        with patch("rlab.run_supervisor.verify_rom_file"):
            supervisor.materialize()
        supervisor.store.init()
        checkpoint = CheckpointManifest(
            run_id=self.run_id,
            checkpoint_id="checkpoint-250000-" + "e" * 16,
            step=250_000,
            purpose="periodic",
            sha256="e" * 64,
            size_bytes=10,
            public_url="https://models.example/model.zip",
            model_document_url="https://models.example/model.json",
            model_document_sha256="f" * 64,
            recipe_document_url="https://models.example/recipe.json",
            recipe_document_sha256="1" * 64,
            goal_sha256=self.manifest.goal_sha256,
            recipe_sha256=self.manifest.recipe_sha256,
            environment_sha256=self.manifest.environment_sha256,
            evaluation_contract_sha256=evaluation_contract_sha256(supervisor.recipe_document),
            recovery_sidecar_key="recovery.json",
            created_at=utc_now(),
        )
        supervisor._ensure_eval(1, checkpoint)
        self.assertEqual(supervisor._submit_pending_evals(), 0)
        row = supervisor.store.evals()[0]
        self.assertEqual(row["status"], "submitted")
        self.assertEqual(row["attempt"], 1)
        self.assertEqual(row["modal_call_id"], "")
        self.assertEqual(supervisor._submit_pending_evals(), 0)
        self.assertEqual(supervisor.store.evals()[0]["attempt"], 1)
        with supervisor.store.connection() as connection:
            connection.execute("UPDATE eval_dispatches SET attempt_expires_at = 1000")
        with patch.object(supervisor.clock, "time", return_value=1001):
            self.assertEqual(supervisor._poll_evals(10.0), 0)
        self.assertEqual(supervisor.store.evals()[0]["status"], "pending")

    def test_training_only_checkpoint_recovery_does_not_create_evals(self) -> None:
        supervisor = self.supervisor()
        supervisor.evaluation_required = False
        supervisor.eval_contract = {}
        supervisor.store.init()
        checkpoint = CheckpointManifest(
            run_id=self.run_id,
            checkpoint_id="checkpoint-250000-" + "e" * 16,
            step=250_000,
            purpose="periodic",
            sha256="e" * 64,
            size_bytes=10,
            public_url="https://models.example/model.zip",
            model_document_url="https://models.example/model.json",
            model_document_sha256="f" * 64,
            recipe_document_url="https://models.example/recipe.json",
            recipe_document_sha256="1" * 64,
            goal_sha256=self.manifest.goal_sha256,
            recipe_sha256=self.manifest.recipe_sha256,
            environment_sha256=self.manifest.environment_sha256,
            evaluation_contract_sha256="2" * 64,
            recovery_sidecar_key="recovery.json",
            created_at=utc_now(),
        )
        self.authority.models.put_json(
            f"runs/{self.run_id}/index.json",
            {"checkpoints": [checkpoint.to_dict()]},
        )

        with patch.object(supervisor, "_ensure_eval") as ensure_eval:
            supervisor._recover_durable_state()

        ensure_eval.assert_not_called()
        self.assertEqual(len(supervisor.store.checkpoint_publications()), 1)
        self.assertEqual(supervisor.store.evals(), [])

    def test_evaluated_checkpoint_recovery_recreates_eval(self) -> None:
        supervisor = self.supervisor()
        supervisor.store.init()
        checkpoint = CheckpointManifest(
            run_id=self.run_id,
            checkpoint_id="checkpoint-250000-" + "e" * 16,
            step=250_000,
            purpose="periodic",
            sha256="e" * 64,
            size_bytes=10,
            public_url="https://models.example/model.zip",
            model_document_url="https://models.example/model.json",
            model_document_sha256="f" * 64,
            recipe_document_url="https://models.example/recipe.json",
            recipe_document_sha256="1" * 64,
            goal_sha256=self.manifest.goal_sha256,
            recipe_sha256=self.manifest.recipe_sha256,
            environment_sha256=self.manifest.environment_sha256,
            evaluation_contract_sha256="2" * 64,
            recovery_sidecar_key="recovery.json",
            created_at=utc_now(),
        )
        self.authority.models.put_json(
            f"runs/{self.run_id}/index.json",
            {"checkpoints": [checkpoint.to_dict()]},
        )

        with patch.object(supervisor, "_ensure_eval") as ensure_eval:
            supervisor._recover_durable_state()

        ensure_eval.assert_called_once_with(-1, checkpoint)

    def test_rejected_final_checkpoint_does_not_displace_earlier_acceptance(
        self,
    ) -> None:
        supervisor = self.supervisor()
        supervisor.store.init()
        checkpoints = [
            {
                "checkpoint_id": "checkpoint-250000-" + "e" * 16,
                "step": 250_000,
                "purpose": "periodic",
                "sha256": "e" * 64,
            },
            {
                "checkpoint_id": "checkpoint-500000-" + "f" * 16,
                "step": 500_000,
                "purpose": "final",
                "sha256": "f" * 64,
            },
        ]
        self.authority.models.put_json(
            f"runs/{self.run_id}/index.json",
            {
                "schema_version": 1,
                "run_id": self.run_id,
                "checkpoints": checkpoints,
                "promotion": None,
            },
            create_only=True,
        )
        for ledger_id, checkpoint in enumerate(checkpoints, start=1):
            supervisor.store.record_checkpoint_publication(
                checkpoint_ledger_id=ledger_id,
                manifest=checkpoint,
            )
            key = str(ledger_id) * 64
            supervisor.store.ensure_eval(
                checkpoint_ledger_id=ledger_id,
                intent={
                    "idempotency_key": key,
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "checkpoint_step": checkpoint["step"],
                },
            )
            supervisor.store.mark_eval_terminal(
                idempotency_key=key,
                status="accepted" if ledger_id == 1 else "rejected",
                result={
                    "episode_results": [{}] * (100 if ledger_id == 1 else 1),
                    "status": "accepted" if ledger_id == 1 else "rejected",
                },
            )

        receipt = supervisor._create_promotion()

        self.assertIsNotNone(receipt)
        assert receipt is not None
        self.assertEqual(receipt.checkpoint_step, 250_000)
        self.assertEqual(receipt.checkpoint_id, checkpoints[0]["checkpoint_id"])


if __name__ == "__main__":
    unittest.main()
