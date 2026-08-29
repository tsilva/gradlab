from __future__ import annotations

import gzip
import hashlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, call, patch

from gradlab.checkpoint_contract import checkpoint_manifest_contract_sha256
from gradlab.early_stop import MetricEarlyStopStateMachine, MetricSample
from gradlab.eval_backend import EvalHandle, EvalPoll
from gradlab.file_utils import atomic_write_json
from gradlab.goal_variants import build_goal_variant_descriptor
from gradlab.metric_names import (
    ORCHESTRATION_EVENT_SEQUENCE,
    TRAIN_EPISODE_RETURN_SHAPED_ORIGIN_TARGET_ROLLING_MEAN,
    summary_metric_value,
    summary_value,
)
from gradlab.manual_evaluation import ManualEvaluationSupervisor
from gradlab.policy_bundle import (
    build_recipe_document,
    canonical_json_sha256,
    evaluation_contract_sha256,
)
from gradlab.r2_store import BucketConfig, RunStorageConfig
from gradlab.recipe_documents import (
    compose_resolved_train_documents,
    load_goal_contract,
)
from gradlab.run_authority import LeaseUnavailable, RunAuthority
from gradlab.run_contracts import (
    CheckpointManifest,
    EarlyStopReceipt,
    EvalResult,
    PromotionReceipt,
    RunManifest,
    default_liveness_policy,
    new_attempt_id,
    new_run_id,
    utc_now,
)
from gradlab.run_supervisor import (
    IncompleteEvaluationEvidence,
    LearnerExitContractMismatch,
    LearnerFailure,
    LearnerStartupTimeout,
    LearnerStateContractError,
    LearnerStopAcknowledgementTimeout,
    LearnerTeardownTimeout,
    RunSupervisor,
    _bind_evaluation_contract,
    _terminal_outcome,
)


SOURCE_SHA = "a" * 40
BUILD_SOURCE_SHA = "f" * 40
RUNTIME_INPUT_SHA256 = "e" * 64
IMAGE = "docker:registry.example/gradlab@sha256:" + "b" * 64
GOAL = Path("experiments/goals/SuperMarioBros-Nes-v0/Level1-1/_goal.yaml")
RECIPE = GOAL.parent / "recipes" / "ppo.yaml"


class FailingSpawnBackend:
    def submit(self, intent):
        raise RuntimeError("connection outcome unknown")

    def poll(self, handle: EvalHandle) -> EvalPoll:
        return EvalPoll(status="running")

    def cancel(self, handle: EvalHandle) -> None:
        return None


class CapturingEvalBackend:
    def __init__(self) -> None:
        self.payloads = []

    def submit(self, intent):
        self.payloads.append(intent)
        return EvalHandle(provider="modal", call_id=f"fc-{len(self.payloads)}")

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
        resolved_documents = compose_resolved_train_documents(
            GOAL,
            RECIPE,
            source_sha=SOURCE_SHA,
        )
        document = resolved_documents.effective
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
        contract_document["goal_variant"] = build_goal_variant_descriptor(
            goal_slug="SuperMarioBros-Nes-v0/Level1-1",
            source_sha=SOURCE_SHA,
            authored_goal=load_goal_contract(GOAL, Path.cwd()),
            effective_goal=dict(document["goal"]),
        )
        portable_recipe = build_recipe_document(
            contract_document,
            repo_root=Path.cwd(),
            source_commit=SOURCE_SHA,
            run_description="supervisor unit test",
            seed=123,
            runtime_image_ref=IMAGE,
            base_materialized_recipe={
                **resolved_documents.base,
                "train_config": {
                    **resolved_documents.base["train_config"],
                    "rom_asset_manifest": self.asset,
                    "checkpoint_eval_backend": "modal",
                },
            },
            canonical_goal=resolved_documents.canonical_goal,
        )
        self.portable_recipe = portable_recipe
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
                    "target": "local-gpu",
                    "max_price": None,
                    "max_cost_usd": None,
                    "allow_on_demand": False,
                    "max_duration_seconds": 3600,
                },
                "selected": {
                    "kind": "local",
                    "target": "local-gpu",
                    "max_price": None,
                    "max_cost_usd": None,
                    "allow_on_demand": False,
                    "max_duration_seconds": 3600,
                },
                "dstack_coordinator_id": "primary",
                "dstack_project": "main",
                "coordinator_binding_basis": "launch-selection",
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
                "environment_name": "gradlab-eval",
                "app_name": f"gradlab-eval-v3-{SOURCE_SHA[:12]}",
                "function_name": "evaluate_checkpoint",
                "deployment_source_sha": SOURCE_SHA,
                "rom_asset_manifest": self.asset,
            },
            storage=self.storage.manifest_locations(),
            goal_variant=contract_document["goal_variant"],
            liveness=default_liveness_policy(),
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

    def learner_result(
        self,
        *,
        status: str = "completed",
        terminal_reason: str = "resource_exhaustion",
        final_step: int = 125_000,
        learner_pid: int = 1234,
    ) -> dict[str, object]:
        document: dict[str, object] = {
            "document_type": "gradlab.training-result",
            "format_version": 3,
            "run_id": self.run_id,
            "attempt_id": self.manifest.attempt_id,
            "learner_pid": learner_pid,
            "training_backend_id": "sb3.ppo",
            "status": status,
            "terminal_reason": terminal_reason,
            "execution_mode": "supervised",
            "execution_policy": {},
            "first_completion_step": None,
            "final_step": final_step,
            "requested_limit": 125_000,
            "execution_limit": 125_000,
            "model_kind": None if status == "failed" else "final",
            "model": None if status == "failed" else "final_model.zip",
            "terminal_at": utc_now(),
        }
        if status == "failed":
            document.update(
                {
                    "error_type": "RuntimeError",
                    "error_message": "learner exploded",
                }
            )
        return document

    def prepare_live_learner_contract(self, supervisor: RunSupervisor) -> None:
        supervisor.run_dir.mkdir(parents=True, exist_ok=True)
        supervisor.train_config = {"training_backend": {"id": "sb3.ppo"}}
        supervisor.expected_learner_pid = 1234
        supervisor.learner_started_at = 0.0

    def test_runtime_verification_uses_build_identity_and_runtime_input(self) -> None:
        supervisor = self.supervisor()
        with (
            patch.dict("os.environ", {"GRADLAB_ORCHESTRATOR": "dstack"}),
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

    def test_manifest_v5_requires_bounded_liveness_policy(self) -> None:
        self.manifest.validate()
        self.assertEqual(self.manifest.schema_version, 5)
        self.assertEqual(self.manifest.liveness["poll_interval_seconds"], 0.25)

        missing = RunManifest(**{**self.manifest.to_dict(), "liveness": None})
        with self.assertRaisesRegex(ValueError, "liveness must be a mapping"):
            missing.validate()

        invalid_policy = {
            **dict(self.manifest.liveness),
            "startup_timeout_seconds": 3600.0,
        }
        invalid = RunManifest(**{**self.manifest.to_dict(), "liveness": invalid_policy})
        with self.assertRaisesRegex(ValueError, "below the selected max duration"):
            invalid.validate()

    def test_supervisor_observes_durable_cancel_and_requests_cooperative_stop(self) -> None:
        supervisor = self.supervisor()
        learner = MagicMock()
        learner.poll.return_value = None
        supervisor.learner = learner
        self.authority.request_cancel(
            run_id=self.run_id,
            attempt_id=self.manifest.attempt_id,
        )

        with patch.object(supervisor.runtime, "request_learner_stop") as request_stop:
            self.assertTrue(supervisor._observe_cancel_request())
            self.assertTrue(supervisor._observe_cancel_request())

        self.assertTrue(supervisor.cancel_requested)
        self.assertEqual(supervisor.stop_reason, "canceled")
        request_stop.assert_called_once_with(learner)

    def test_supervisor_retries_stop_until_safe_boundary_acknowledgement(self) -> None:
        supervisor = self.supervisor()
        self.prepare_live_learner_contract(supervisor)
        learner = MagicMock()
        learner.poll.return_value = None
        supervisor.learner = learner
        supervisor.learner_stop_requested_at = 0.0
        supervisor.last_learner_stop_signal_at = 0.0
        supervisor.stop_reason = "canceled"

        with patch.object(supervisor.runtime, "request_learner_stop") as request_stop:
            supervisor._maintain_learner_stop(2.0)
            atomic_write_json(
                supervisor.run_dir / "learner_stop_observed.json",
                {
                    "pid": 1234,
                    "boundary": "on_policy_update_end",
                    "num_timesteps": 4096,
                },
            )
            supervisor._maintain_learner_stop(2.1)

        request_stop.assert_called_once_with(learner)
        self.assertEqual(supervisor.learner_stop_signal_attempts, 1)
        self.assertTrue(supervisor.learner_stop_acknowledged)

    def test_supervisor_bounds_missing_stop_acknowledgement(self) -> None:
        supervisor = self.supervisor()
        self.prepare_live_learner_contract(supervisor)
        learner = MagicMock()
        learner.poll.return_value = None
        supervisor.learner = learner
        supervisor.learner_stop_requested_at = 0.0
        supervisor.last_learner_stop_signal_at = 9.0
        supervisor.stop_reason = "canceled"

        with self.assertRaises(LearnerStopAcknowledgementTimeout):
            supervisor._maintain_learner_stop(10.0)

    def test_manual_evaluation_queue_reuses_durable_intent_and_modal_dispatch(self) -> None:
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
            recipe_document_sha256=canonical_json_sha256(self.portable_recipe),
            goal_sha256=self.manifest.goal_sha256,
            recipe_sha256=self.manifest.recipe_sha256,
            environment_sha256=self.manifest.environment_sha256,
            evaluation_contract_sha256=evaluation_contract_sha256(self.portable_recipe),
            recovery_sidecar_key="recovery.json",
            created_at=utc_now(),
        )
        checkpoint_prefix = f"runs/{self.run_id}/checkpoints/{checkpoint.step}-{checkpoint.sha256}"
        self.authority.models.put_json(
            f"{checkpoint_prefix}/recipe.json",
            self.portable_recipe,
        )
        self.authority.models.put_json(
            f"runs/{self.run_id}/index.json",
            {
                "schema_version": 1,
                "run_id": self.run_id,
                "updated_at": utc_now(),
                "checkpoints": [checkpoint.to_dict()],
                "promotion": None,
            },
        )
        backend = CapturingEvalBackend()
        queue = ManualEvaluationSupervisor(
            authority=self.authority,
            repo_root=Path.cwd(),
            backend_factory=lambda _manifest: backend,
            project_results=False,
            holder_id="manual-eval-test",
            work_root=Path(self.temporary.name) / "manual-eval",
        )

        first = queue.advance_batch(
            job_id="job-" + "a" * 32,
            run_id=self.run_id,
            checkpoint_ids=[checkpoint.checkpoint_id],
            cancel_requested=False,
        )
        repeated = queue.advance_batch(
            job_id="job-" + "a" * 32,
            run_id=self.run_id,
            checkpoint_ids=[checkpoint.checkpoint_id],
            cancel_requested=False,
        )

        self.assertEqual(first.state, "retry_wait")
        self.assertEqual(first.subjects[0].state, "submitted")
        self.assertEqual(repeated.state, "retry_wait")
        self.assertEqual(repeated.subjects[0].state, "submitted")
        self.assertEqual(len(backend.payloads), 1)
        intent = next(
            self.authority.evaluation.get_json(key)
            for key in self.authority.evaluation.iter_keys(f"runs/{self.run_id}/evals")
            if key.endswith("/intent.json")
        )
        self.assertEqual(intent["checkpoint_id"], checkpoint.checkpoint_id)
        self.assertEqual(
            backend.payloads[0]["contract"],
            intent["execution_contract"],
        )
        self.assertIsNotNone(
            self.authority.eval_dispatch(
                run_id=self.run_id,
                idempotency_key=intent["idempotency_key"],
                attempt=1,
            )
        )

    def test_manual_evaluation_accepts_checkpoint_with_playback_contract_binding(self) -> None:
        resolved_documents = compose_resolved_train_documents(
            GOAL,
            RECIPE,
            source_sha=SOURCE_SHA,
        )
        document = resolved_documents.effective
        contract_document = dict(document)
        contract_config = dict(contract_document["train_config"])
        contract_config["rom_asset_manifest"] = self.asset
        contract_config["checkpoint_eval_backend"] = "none"
        contract_document["train_config"] = contract_config
        contract_document["goal_variant"] = build_goal_variant_descriptor(
            goal_slug="SuperMarioBros-Nes-v0/Level1-1",
            source_sha=SOURCE_SHA,
            authored_goal=load_goal_contract(GOAL, Path.cwd()),
            effective_goal=dict(document["goal"]),
        )
        recipe_document = build_recipe_document(
            contract_document,
            repo_root=Path.cwd(),
            source_commit=SOURCE_SHA,
            run_description="manual evaluation after training-only execution",
            seed=123,
            runtime_image_ref=IMAGE,
            base_materialized_recipe={
                **resolved_documents.base,
                "train_config": {
                    **resolved_documents.base["train_config"],
                    "rom_asset_manifest": self.asset,
                    "checkpoint_eval_backend": "none",
                },
            },
            canonical_goal=resolved_documents.canonical_goal,
        )
        self.assertNotIn("eval", recipe_document["recipe"])
        self.assertIn("playback", recipe_document["recipe"])

        recipe_sha256 = canonical_json_sha256(recipe_document)
        manifest = replace(self.manifest, recipe_sha256=recipe_sha256)
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
            recipe_document_sha256=recipe_sha256,
            goal_sha256=manifest.goal_sha256,
            recipe_sha256=manifest.recipe_sha256,
            environment_sha256=manifest.environment_sha256,
            evaluation_contract_sha256=checkpoint_manifest_contract_sha256(recipe_document),
            recovery_sidecar_key="recovery.json",
            created_at=utc_now(),
        )
        checkpoint_prefix = f"runs/{self.run_id}/checkpoints/{checkpoint.step}-{checkpoint.sha256}"
        self.authority.models.put_json(
            f"{checkpoint_prefix}/recipe.json",
            recipe_document,
        )
        queue = ManualEvaluationSupervisor(
            authority=self.authority,
            repo_root=Path.cwd(),
            project_results=False,
            holder_id="manual-eval-playback-contract-test",
            work_root=Path(self.temporary.name) / "manual-eval-playback-contract",
        )

        context = queue._context(
            manifest=manifest,
            checkpoint=checkpoint,
            enforce_current_protocol=False,
        )

        self.assertNotEqual(
            checkpoint.evaluation_contract_sha256,
            context.intent.evaluation_contract_sha256,
        )
        self.assertEqual(
            context.intent.evaluation_contract_sha256,
            evaluation_contract_sha256(recipe_document),
        )
        with self.assertRaisesRegex(ValueError, "checkpoint manifest contract hash mismatch"):
            queue._context(
                manifest=manifest,
                checkpoint=replace(
                    checkpoint,
                    evaluation_contract_sha256="0" * 64,
                ),
                enforce_current_protocol=False,
            )

    def test_manual_vizdoom_evaluation_rejects_stale_protocol(self) -> None:
        valid = {
            "environment": {"game": "VizdoomDefendLine-v1"},
            "episodes": 100,
            "evidence_policy": {"fail_fast": "disabled"},
            "acceptance": [
                {
                    "metric": "eval/full/episode/return/shaped/mean",
                    "operator": ">=",
                    "threshold": 5.0,
                }
            ],
        }
        ManualEvaluationSupervisor._validate_current_protocol(valid)

        for changed in (
            {**valid, "episodes": 99},
            {**valid, "evidence_policy": {"fail_fast": "first_failed_episode"}},
            {
                **valid,
                "acceptance": [
                    {
                        "metric": "eval/full/outcome/success/starts/rate/min",
                        "operator": ">=",
                        "threshold": 1.0,
                    }
                ],
            },
        ):
            with self.assertRaisesRegex(ValueError, "requires|acceptance"):
                ManualEvaluationSupervisor._validate_current_protocol(changed)

        upgraded = ManualEvaluationSupervisor._current_protocol_contract(
            {
                **valid,
                "evidence_policy": {
                    "fail_fast": "first_failed_episode",
                    "partial_rejection_metrics": True,
                },
            }
        )
        self.assertEqual(upgraded["evidence_policy"]["fail_fast"], "disabled")
        self.assertFalse(upgraded["evidence_policy"]["partial_rejection_metrics"])
        ManualEvaluationSupervisor._validate_current_protocol(upgraded)

    def test_manual_vizdoom_basic_evaluation_requires_complete_perfect_success(self) -> None:
        valid = {
            "environment": {"game": "VizdoomBasic-v1"},
            "episodes": 100,
            "evidence_policy": {"fail_fast": "disabled"},
            "acceptance": [
                {
                    "metric": "eval/full/outcome/success/starts/rate/min",
                    "operator": ">=",
                    "threshold": 1.0,
                }
            ],
        }
        ManualEvaluationSupervisor._validate_current_protocol(valid)

        stale = {
            **valid,
            "acceptance": [
                {
                    "metric": "eval/full/episode/return/shaped/mean",
                    "operator": ">=",
                    "threshold": 0.95,
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "acceptance"):
            ManualEvaluationSupervisor._validate_current_protocol(stale)

        upgraded = ManualEvaluationSupervisor._current_protocol_contract(
            {
                **valid,
                "evidence_policy": {
                    "fail_fast": "first_failed_episode",
                    "partial_rejection_metrics": True,
                },
            }
        )
        self.assertEqual(upgraded["evidence_policy"]["fail_fast"], "disabled")
        self.assertFalse(upgraded["evidence_policy"]["partial_rejection_metrics"])
        ManualEvaluationSupervisor._validate_current_protocol(upgraded)

    def test_manual_evaluation_uses_latest_attempt_manifest(self) -> None:
        latest = replace(
            self.manifest,
            attempt_id=new_attempt_id(),
            created_at="2026-12-31T23:59:59Z",
        )
        self.authority.create_attempt_manifest(latest)
        queue = ManualEvaluationSupervisor(
            authority=self.authority,
            repo_root=Path.cwd(),
            project_results=False,
            work_root=Path(self.temporary.name) / "manual-eval-latest",
        )

        self.assertEqual(
            queue._manifest(self.run_id).attempt_id,
            latest.attempt_id,
        )

    def test_manual_evaluation_sequences_after_prior_manual_jobs(self) -> None:
        queue = ManualEvaluationSupervisor(
            authority=self.authority,
            repo_root=Path.cwd(),
            project_results=False,
            work_root=Path(self.temporary.name) / "manual-eval-offset",
        )
        prefix = f"runs/{self.run_id}/manual-evals"
        self.authority.control.put_json(
            f"{prefix}/first/wandb-projection.json",
            {
                "schema_version": 1,
                "run_id": self.run_id,
                "wandb_high_water_mark": 14,
            },
        )
        self.authority.control.put_json(
            f"{prefix}/jobs/job-{'a' * 32}/terminal.json",
            {
                "schema_version": 1,
                "run_id": self.run_id,
                "wandb_high_water_mark": 19,
            },
        )

        self.assertEqual(
            queue._event_seq_offset(
                self.manifest,
                {"wandb_high_water_mark": 11},
            ),
            19,
        )

    def test_ephemeral_state_archive_is_not_recovered_or_published(self) -> None:
        supervisor = self.supervisor()
        supervisor.train_config = {
            **supervisor.train_config,
            "state_archive": {"persistence": "ephemeral"},
        }

        self.assertFalse(supervisor._durable_state_archive_enabled())
        with patch.object(supervisor.authority, "publish_state_archive") as publish:
            self.assertEqual(supervisor._publish_state_archive(), 0)
        publish.assert_not_called()

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
        self.assertIn("invalid runtime", receipt["drain"]["failure"]["message"])

        with (
            patch.dict("os.environ", {"GRADLAB_ORCHESTRATOR": "dstack"}),
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
        self.assertIn("recovery exploded", receipt["drain"]["failure"]["message"])

    def test_learner_log_tail_is_bounded_and_preserves_latest_failure(self) -> None:
        supervisor = self.supervisor()
        supervisor.learner_log_path.parent.mkdir(parents=True, exist_ok=True)
        supervisor.learner_log_path.write_bytes(
            b"discarded-prefix\n" + b"x" * 128 + b"\nlatest learner traceback\n"
        )

        tail = supervisor._learner_log_tail(max_bytes=64)

        self.assertNotIn("discarded-prefix", tail)
        self.assertIn("latest learner traceback", tail)

    def test_learner_log_evidence_archives_full_log_and_bounded_tail(self) -> None:
        supervisor = self.supervisor()
        payload = b"discarded-prefix\n" + b"x" * 20_000 + b"\nlatest native failure\n"
        supervisor.learner_log_path.parent.mkdir(parents=True, exist_ok=True)
        supervisor.learner_log_path.write_bytes(payload)

        evidence = supervisor._learner_log_evidence()

        assert evidence is not None
        self.assertEqual(evidence["size_bytes"], len(payload))
        self.assertEqual(evidence["sha256"], hashlib.sha256(payload).hexdigest())
        self.assertNotIn("discarded-prefix", evidence["tail"])
        self.assertIn("latest native failure", evidence["tail"])
        self.assertEqual(evidence["archive"]["state"], "complete")
        self.assertEqual(evidence["archive"]["attempts"], 1)
        archived = self.authority.control.get_bytes(evidence["archive"]["object_key"])
        self.assertEqual(gzip.decompress(archived), payload)

    def test_learner_log_archive_failure_is_retried_without_raising(self) -> None:
        supervisor = self.supervisor()
        supervisor.learner_log_path.parent.mkdir(parents=True, exist_ok=True)
        supervisor.learner_log_path.write_text("latest native failure\n")

        with (
            patch.object(
                supervisor.authority,
                "archive_learner_log",
                side_effect=RuntimeError("R2 archive unavailable"),
            ) as archive,
            patch.object(supervisor.clock, "sleep") as sleep,
        ):
            evidence = supervisor._learner_log_evidence()

        assert evidence is not None
        self.assertEqual(archive.call_count, 3)
        self.assertEqual(sleep.call_args_list, [call(1.0), call(2.0)])
        self.assertEqual(evidence["archive"]["state"], "failed")
        self.assertIn("R2 archive unavailable", evidence["archive"]["failure"]["message"])
        self.assertIn("latest native failure", evidence["tail"])

    def test_supervisor_starts_learner_with_explicit_execution_mode(self) -> None:
        supervisor = self.supervisor()
        with patch.object(
            supervisor.runtime,
            "start_learner",
            return_value=MagicMock(pid=1234),
        ) as start:
            supervisor._start_learner()

        command = start.call_args.args[0]
        self.assertEqual(command[-2:], ["--execution-mode", "supervised"])

    def test_fault_fixture_uses_dedicated_non_training_learner(self) -> None:
        supervisor = self.supervisor()
        with (
            patch.dict(
                "os.environ",
                {"GRADLAB_SUPERVISION_FAULT_FIXTURE": ("failed-result-live-process")},
            ),
            patch.object(
                supervisor.runtime,
                "start_learner",
                return_value=MagicMock(pid=1234),
            ) as start,
        ):
            supervisor._start_learner()

        command = start.call_args.args[0]
        self.assertIn("gradlab.supervision_fault_learner", command)
        self.assertEqual(command[-2:], ["--mode", "failed-result-live-process"])
        self.assertNotIn("gradlab.train", command)

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

    def test_wandb_summary_subdict_is_normalized_centrally(self) -> None:
        class SummarySubDict:
            def __init__(self, value: dict[str, int]) -> None:
                self.value = value

            def items(self):
                return self.value.items()

        self.assertEqual(summary_value({"max": 10}), 10)
        self.assertEqual(summary_value(SummarySubDict({"max": 10})), 10)
        self.assertEqual(
            summary_metric_value(
                {ORCHESTRATION_EVENT_SEQUENCE: SummarySubDict({"max": 10})},
                ORCHESTRATION_EVENT_SEQUENCE,
            ),
            10,
        )
        self.assertEqual(
            summary_metric_value(
                {f"{ORCHESTRATION_EVENT_SEQUENCE}.max": 777},
                ORCHESTRATION_EVENT_SEQUENCE,
            ),
            777,
        )

    def test_health_publication_keeps_only_the_v19_operational_surface(self) -> None:
        supervisor = self.supervisor()
        supervisor.store.init()

        supervisor._emit_health(15.0)

        self.assertEqual(
            set(supervisor.store.latest_metrics()),
            {
                "orchestration/outbox/pending/count",
                "orchestration/outbox/oldest/age/seconds",
                "orchestration/outbox/remote/visibility/lag/seconds",
                "orchestration/checkpoint/pending/count",
                "orchestration/eval/pending/count",
                "orchestration/scratch/used/fraction",
            },
        )
        internal = supervisor.store.state("backpressure")
        assert internal is not None
        self.assertIn("local_high_water", internal)
        self.assertIn("wandb_high_water", internal)
        self.assertNotIn("publication_capacity_sufficient", internal)

    def test_terminal_training_result_bounds_hung_learner_teardown(self) -> None:
        supervisor = self.supervisor()
        self.prepare_live_learner_contract(supervisor)
        atomic_write_json(
            supervisor.run_dir / "training-result.json",
            self.learner_result(final_step=0),
        )

        supervisor._observe_live_learner_state(10.0)
        supervisor._observe_live_learner_state(14.9)
        with self.assertRaisesRegex(
            LearnerTeardownTimeout,
            "terminal result 'resource_exhaustion'.*remained alive",
        ):
            supervisor._observe_live_learner_state(15.0)

    def test_active_iteration_repolls_before_declaring_completed_learner_hung(self) -> None:
        supervisor = self.supervisor()
        self.prepare_live_learner_contract(supervisor)
        atomic_write_json(
            supervisor.run_dir / "training-result.json",
            self.learner_result(final_step=0),
        )
        supervisor._observe_live_learner_state(10.0)
        supervisor.learner = MagicMock()
        supervisor.learner.poll.return_value = 0

        with patch.object(supervisor.clock, "monotonic", return_value=18.2):
            self.assertFalse(supervisor._observe_learner_after_active_iteration())

        supervisor.learner.poll.assert_called_once_with()

    def test_failed_learner_result_is_authoritative_immediately(self) -> None:
        supervisor = self.supervisor()
        self.prepare_live_learner_contract(supervisor)
        atomic_write_json(
            supervisor.run_dir / "training-result.json",
            self.learner_result(
                status="failed",
                terminal_reason="failed",
                final_step=17,
            ),
        )

        with self.assertRaisesRegex(LearnerFailure, "learner exploded"):
            supervisor._observe_live_learner_state(0.25)

        self.assertEqual(supervisor.learner_final_step, 17)
        self.assertEqual(supervisor.learner_result_observed_at, 0.25)

    def test_live_learner_state_rejects_noncurrent_and_identity_mismatch(self) -> None:
        supervisor = self.supervisor()
        self.prepare_live_learner_contract(supervisor)
        noncurrent = {
            "document_type": "gradlab.training-result",
            "format_version": 2,
            "status": "completed",
            "terminal_reason": "resource_exhaustion",
            "execution_mode": "supervised",
            "final_step": 1,
        }
        atomic_write_json(supervisor.run_dir / "training-result.json", noncurrent)
        with self.assertRaisesRegex(LearnerStateContractError, "format_version"):
            supervisor._observe_live_learner_state(0.25)

        mismatched = self.learner_result()
        mismatched["attempt_id"] = new_attempt_id()
        atomic_write_json(supervisor.run_dir / "training-result.json", mismatched)
        with self.assertRaisesRegex(LearnerStateContractError, "attempt_id"):
            supervisor._observe_live_learner_state(0.5)

    def test_learner_startup_deadline_uses_manifest_policy(self) -> None:
        supervisor = self.supervisor()
        self.prepare_live_learner_contract(supervisor)

        self.assertIsNone(supervisor._observe_live_learner_state(599.9))
        with self.assertRaisesRegex(LearnerStartupTimeout, "within 600.0s"):
            supervisor._observe_live_learner_state(600.0)

    def test_exit_result_contract_matrix(self) -> None:
        supervisor = self.supervisor()
        self.prepare_live_learner_contract(supervisor)
        result_path = supervisor.run_dir / "training-result.json"

        atomic_write_json(result_path, self.learner_result())
        self.assertEqual(supervisor._validate_learner_exit(0).status, "completed")
        with self.assertRaises(LearnerExitContractMismatch):
            supervisor._validate_learner_exit(9)

        atomic_write_json(
            result_path,
            self.learner_result(status="failed", terminal_reason="failed"),
        )
        with self.assertRaises(LearnerFailure):
            supervisor._validate_learner_exit(0)
        with self.assertRaises(LearnerFailure):
            supervisor._validate_learner_exit(9)

        result_path.unlink()
        with self.assertRaisesRegex(LearnerStateContractError, "without a terminal result"):
            supervisor._validate_learner_exit(0)

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

    def test_learner_failure_dominates_later_cancellation(self) -> None:
        state, stop_reason = _terminal_outcome(
            cancel_requested=True,
            failure=LearnerFailure("failed at step zero"),
            evaluation_required=True,
            promotion=None,
            early_stop=self._early_stop_receipt(),
        )

        self.assertEqual(state, "resumable_failure")
        self.assertEqual(stop_reason, "learner_failure")

    def test_cancellation_dominates_neutral_plateau(self) -> None:
        state, stop_reason = _terminal_outcome(
            cancel_requested=True,
            failure=None,
            evaluation_required=True,
            promotion=None,
            early_stop=self._early_stop_receipt(outcome="neutral"),
        )

        self.assertEqual(state, "canceled")
        self.assertEqual(stop_reason, "canceled")

    def test_incomplete_eval_evidence_has_typed_resumable_outcome(self) -> None:
        state, stop_reason = _terminal_outcome(
            cancel_requested=False,
            failure=IncompleteEvaluationEvidence("eval infrastructure failed"),
            evaluation_required=True,
            promotion=None,
            early_stop=self._early_stop_receipt(outcome="neutral"),
        )

        self.assertEqual(state, "resumable_failure")
        self.assertEqual(stop_reason, "evaluation_evidence_incomplete")

    def _early_stop_receipt(self, *, outcome: str = "failure") -> EarlyStopReceipt:
        return EarlyStopReceipt(
            run_id=self.run_id,
            attempt_id=self.manifest.attempt_id,
            condition_id="return_plateau",
            matched_condition_ids=("return_plateau",),
            outcome=outcome,  # type: ignore[arg-type]
            trigger="no_improvement",
            metric="train/episode/return/shaped/origin/target/rolling/mean",
            metric_step=2_000_000,
            value=650.0,
            best_value=650.0,
            elapsed_steps=1_000_000,
            patience_progress=1.0,
            condition={
                "metric": "train/episode/return/shaped/origin/target/rolling/mean",
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

    def test_evaluated_neutral_plateau_stops_without_scientific_outcome(self) -> None:
        receipt = self._early_stop_receipt(outcome="neutral")

        state, stop_reason = _terminal_outcome(
            cancel_requested=False,
            failure=None,
            evaluation_required=True,
            promotion=None,
            early_stop=receipt,
        )

        self.assertEqual(state, "stopped")
        self.assertEqual(stop_reason, "early_stop_neutral:return_plateau")

    def test_training_only_neutral_plateau_stops_the_attempt(self) -> None:
        receipt = self._early_stop_receipt(outcome="neutral")

        state, stop_reason = _terminal_outcome(
            cancel_requested=False,
            failure=None,
            evaluation_required=False,
            promotion=None,
            early_stop=receipt,
        )

        self.assertEqual(state, "stopped")
        self.assertEqual(stop_reason, "early_stop_neutral:return_plateau")

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

    def test_evaluation_promotion_overrides_neutral_plateau(self) -> None:
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
            early_stop=self._early_stop_receipt(outcome="neutral"),
        )

        self.assertEqual(state, "succeeded")
        self.assertEqual(stop_reason, "completed_after_eval_acceptance")

    def test_supervisor_validates_and_persists_learner_early_stop_decision(self) -> None:
        supervisor = self.supervisor()
        config = {
            "conditions": {
                "return_plateau": {
                    "metric": TRAIN_EPISODE_RETURN_SHAPED_ORIGIN_TARGET_ROLLING_MEAN,
                    "trigger": "no_improvement",
                    "direction": "maximize",
                    "min_delta": 0.01,
                    "delta_mode": "relative",
                    "start_after_steps": 0,
                    "patience_steps": 10,
                    "outcome": "neutral",
                    "action": "stop",
                }
            }
        }
        machine = MetricEarlyStopStateMachine(config)
        machine.update(
            {
                TRAIN_EPISODE_RETURN_SHAPED_ORIGIN_TARGET_ROLLING_MEAN: MetricSample(
                    value=100.0,
                    step=0,
                )
            }
        )
        update = machine.update(
            {
                TRAIN_EPISODE_RETURN_SHAPED_ORIGIN_TARGET_ROLLING_MEAN: MetricSample(
                    value=100.0,
                    step=10,
                )
            }
        )
        self.assertIsNotNone(update.stop_decision)
        supervisor.train_config = {"early_stop": machine.config}
        decision_path = supervisor.run_dir / f"early_stop_decision-{self.manifest.attempt_id}.json"
        atomic_write_json(decision_path, update.stop_decision or {})

        receipt = supervisor._resolve_early_stop_receipt()

        self.assertIsNotNone(receipt)
        assert receipt is not None
        self.assertEqual(receipt.condition_id, "return_plateau")
        self.assertEqual(receipt.outcome, "neutral")
        stored = self.authority.early_stop_receipt(
            run_id=self.run_id,
            attempt_id=self.manifest.attempt_id,
        )
        self.assertEqual(stored["decision_sha256"], receipt.decision_sha256)
        tampered = dict(stored)
        tampered["outcome"] = "failure"
        self.authority.control.put_json(
            (f"runs/{self.run_id}/attempts/{self.manifest.attempt_id}/early-stop.json"),
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
                    "metric": "train/outcome/success/starts/all/rolling/rate/min",
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
                "train/outcome/success/starts/all/rolling/rate/min": MetricSample(
                    value=1.0,
                    step=10,
                )
            }
        )
        decision = dict(update.stop_decision or {})
        decision["outcome"] = "failure"
        supervisor.train_config = {"early_stop": machine.config}
        atomic_write_json(
            supervisor.run_dir / f"early_stop_decision-{self.manifest.attempt_id}.json",
            decision,
        )

        with self.assertRaisesRegex(ValueError, "does not match"):
            supervisor._resolve_early_stop_receipt()

    def test_resume_recovery_preserves_prior_provisional_early_stop(self) -> None:
        original = self.supervisor()
        config = {
            "conditions": {
                "return_plateau": {
                    "metric": TRAIN_EPISODE_RETURN_SHAPED_ORIGIN_TARGET_ROLLING_MEAN,
                    "trigger": "no_improvement",
                    "direction": "maximize",
                    "min_delta": 0.01,
                    "delta_mode": "relative",
                    "start_after_steps": 0,
                    "patience_steps": 10,
                    "outcome": "neutral",
                    "action": "stop",
                }
            }
        }
        machine = MetricEarlyStopStateMachine(config)
        machine.update(
            {
                TRAIN_EPISODE_RETURN_SHAPED_ORIGIN_TARGET_ROLLING_MEAN: MetricSample(
                    value=100.0,
                    step=0,
                )
            }
        )
        update = machine.update(
            {
                TRAIN_EPISODE_RETURN_SHAPED_ORIGIN_TARGET_ROLLING_MEAN: MetricSample(
                    value=100.0,
                    step=10,
                )
            }
        )
        original.train_config = {"early_stop": machine.config}
        atomic_write_json(
            original.run_dir / f"early_stop_decision-{self.manifest.attempt_id}.json",
            update.stop_decision or {},
        )
        originating_receipt = original._resolve_early_stop_receipt()
        self.assertIsNotNone(originating_receipt)

        retry_manifest = replace(
            self.manifest,
            attempt_id=new_attempt_id(),
            created_at="9999-01-01T00:00:00Z",
            compute={**self.manifest.compute, "recovery_mode": "resume-training"},
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
        self.assertEqual(state, "stopped")
        self.assertEqual(stop_reason, "early_stop_neutral:return_plateau")

    def test_wandb_remote_probe_survives_sdk_finish(self) -> None:
        class SummarySubDict:
            def __init__(self, value: dict[str, int]) -> None:
                self.value = value

            def items(self):
                return self.value.items()

        class RemoteRun:
            summary = {"orchestration/event/sequence": SummarySubDict({"max": 10})}

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

    def test_wandb_remote_probe_reads_flattened_max_summary(self) -> None:
        supervisor = self.supervisor()
        supervisor.store.init()
        supervisor.wandb_run_path = f"entity/project/{self.run_id}"
        supervisor.runtime.remote_summary = MagicMock(
            return_value={"orchestration/event/sequence.max": 777}
        )

        supervisor._probe_wandb_remote(
            10.0,
            local_high_water=777,
            force=True,
        )

        self.assertEqual(supervisor.wandb_remote_high_water, 777)

    def test_wandb_delivery_drain_accepts_stale_reducer_with_current_step(self) -> None:
        supervisor = self.supervisor()
        supervisor.store.init()
        supervisor.wandb_run_path = f"entity/project/{self.run_id}"
        supervisor.runtime.remote_summary = MagicMock(
            return_value={
                "orchestration/event/sequence.max": 777,
                "_step": 778,
            }
        )

        with (
            patch.object(supervisor, "_lease_heartbeat"),
            patch.object(supervisor.clock, "monotonic", side_effect=[0.0, 2.0, 300.0]),
            patch.object(supervisor.clock, "sleep") as sleep,
        ):
            supervisor._wait_for_remote_delivery(778)

        sleep.assert_not_called()
        self.assertEqual(supervisor.wandb_remote_high_water, 778)

    def test_materializes_exact_mario_acceptance_contract(self) -> None:
        supervisor = self.supervisor()
        with patch("gradlab.run_supervisor.verify_rom_file"):
            supervisor.materialize()
        self.assertEqual(supervisor.train_config["timesteps"], 50_000_000)
        self.assertEqual(supervisor.train_config["checkpoint_freq"], 500_000)
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
                    "metric": "eval/full/outcome/success/starts/rate/min",
                    "operator": ">=",
                    "threshold": 1.0,
                }
            ],
        )

    def test_materializes_launch_time_recipe_variant_metadata(self) -> None:
        supervisor = self.supervisor()
        overrides = ("train.timesteps=50000000",)
        resolved_documents = compose_resolved_train_documents(
            GOAL,
            RECIPE,
            recipe_overrides=overrides,
            source_sha=SOURCE_SHA,
        )
        document = resolved_documents.effective
        contract_document = dict(document)
        contract_config = dict(contract_document["train_config"])
        contract_config["rom_asset_manifest"] = self.asset
        contract_config["checkpoint_eval_backend"] = "modal"
        contract_document["train_config"] = contract_config
        contract_document["goal_variant"] = build_goal_variant_descriptor(
            goal_slug="SuperMarioBros-Nes-v0/Level1-1",
            source_sha=SOURCE_SHA,
            authored_goal=load_goal_contract(GOAL, Path.cwd()),
            effective_goal=dict(document["goal"]),
        )
        portable_recipe = build_recipe_document(
            contract_document,
            repo_root=Path.cwd(),
            source_commit=SOURCE_SHA,
            run_description=self.manifest.run_description,
            seed=self.manifest.seed,
            runtime_image_ref=IMAGE,
            base_materialized_recipe={
                **resolved_documents.base,
                "train_config": {
                    **resolved_documents.base["train_config"],
                    "rom_asset_manifest": self.asset,
                    "checkpoint_eval_backend": "modal",
                },
            },
            canonical_goal=resolved_documents.canonical_goal,
        )
        supervisor.manifest = replace(
            supervisor.manifest,
            recipe_overrides=overrides,
            recipe_sha256=canonical_json_sha256(portable_recipe),
        )

        with patch("gradlab.run_supervisor.verify_rom_file"):
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
            aggregates={
                "eval/full/episode/return/shaped/mean": 1.0,
                "failure_count": 0,
            },
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
        )

        self.assertEqual(
            supervisor.store.latest_metric("eval/full/episode/return/shaped/mean"),
            1.0,
        )
        self.assertIsNone(supervisor.store.latest_metric("failure_count"))
        self.assertIsNone(supervisor.store.latest_metric("death_count"))
        self.assertIsNone(supervisor.store.latest_metric("success_count"))

    def test_start_table_requires_a_complete_accepted_or_rejected_evaluation(self) -> None:
        def result(*, episode_count: int) -> EvalResult:
            return EvalResult(
                run_id=self.run_id,
                checkpoint_id="checkpoint-100-" + "e" * 16,
                idempotency_key=("a" if episode_count == 2 else "b") * 64,
                modal_call_id="fc-rejected",
                status="rejected",
                episode_results=[
                    {
                        "start_state": "Level1-1",
                        "return": 0.0,
                        "steps": 1,
                        "outcome": "failure",
                        "terminated": True,
                    }
                ]
                * episode_count,
                aggregates={"eval/full/episode/return/shaped/mean": 0.0},
                timings={},
                evidence_sha256=[],
                completed_at=utc_now(),
            )

        complete = self.supervisor()
        complete.store.init()
        complete_result = result(episode_count=2)
        complete._record_eval_metrics(
            {
                "checkpoint_step": 100,
                "idempotency_key": complete_result.idempotency_key,
                "intent": {"execution_contract": {"episodes": 2}},
            },
            complete_result,
        )
        before = sum(
            row["kind"] == "eval_by_start" for row in complete.store.pending_metric_frames(limit=10)
        )
        self.assertEqual(before, 1)
        partial_result = result(episode_count=1)
        complete._record_eval_metrics(
            {
                "checkpoint_step": 100,
                "idempotency_key": partial_result.idempotency_key,
                "intent": {"execution_contract": {"episodes": 2}},
            },
            partial_result,
        )
        after = sum(
            row["kind"] == "eval_by_start" for row in complete.store.pending_metric_frames(limit=10)
        )
        self.assertEqual(after, before)

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

    def test_finalization_heartbeat_fails_closed_after_lease_loss(self) -> None:
        supervisor = self.supervisor()

        def lose_lease(_now: float) -> None:
            supervisor.lease_lost = True

        with patch.object(supervisor, "_renew_lease", side_effect=lose_lease):
            with self.assertRaisesRegex(LeaseUnavailable, "lost during finalization"):
                supervisor._lease_heartbeat()

    def test_accepted_eval_requests_stop_before_metric_projection(self) -> None:
        supervisor = self.supervisor()
        supervisor.store.init()
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
        self.assertTrue(supervisor.eval_admission_closed)
        self.assertEqual(
            supervisor.store.state("automatic_eval_admission")["checkpoint_id"],
            result.checkpoint_id,
        )

    def test_ambiguous_modal_spawn_is_not_immediately_repeated(self) -> None:
        supervisor = self.supervisor()
        with patch("gradlab.run_supervisor.verify_rom_file"):
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

    def test_closed_eval_admission_drains_without_new_submissions_or_retries(self) -> None:
        supervisor = self.supervisor()
        with patch("gradlab.run_supervisor.verify_rom_file"):
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
        self.assertEqual(supervisor.store.evals()[0]["status"], "submitted")

        later = replace(
            checkpoint,
            checkpoint_id="checkpoint-500000-" + "d" * 16,
            step=500_000,
            sha256="d" * 64,
        )
        supervisor._ensure_eval(2, later)
        supervisor.eval_admission_closed = True

        self.assertEqual(supervisor._submit_pending_evals(), 0)
        self.assertEqual(
            [row["status"] for row in supervisor.store.evals()],
            ["submitted", "pending"],
        )
        with supervisor.store.connection() as connection:
            connection.execute(
                "UPDATE eval_dispatches SET attempt_expires_at = 1000 WHERE status = 'submitted'"
            )
        with patch.object(supervisor.clock, "time", return_value=1001):
            self.assertEqual(supervisor._poll_evals(10.0), 2)
        rows = supervisor.store.evals()
        self.assertEqual([row["status"] for row in rows], ["expired", "deferred"])
        self.assertEqual(rows[0]["attempt"], 1)
        _checkpoints, evals = supervisor._terminal_inventory()
        self.assertEqual([row["status"] for row in evals], ["expired", "deferred"])
        self.assertIsNone(evals[1]["result_sha256"])

    def test_durable_result_is_reconciled_before_pending_eval_submission(self) -> None:
        supervisor = self.supervisor()
        supervisor.store.init()
        supervisor.store.ensure_eval(
            checkpoint_ledger_id=1,
            intent={
                "idempotency_key": "1" * 64,
                "checkpoint_id": "checkpoint-1-" + "1" * 16,
                "checkpoint_step": 1,
            },
        )
        supervisor.store.mark_eval_submitted(
            idempotency_key="1" * 64,
            attempt=1,
            modal_call_id="fc-one",
            attempt_expires_at=1_000.0,
        )
        supervisor.store.ensure_eval(
            checkpoint_ledger_id=2,
            intent={
                "idempotency_key": "2" * 64,
                "checkpoint_id": "checkpoint-2-" + "2" * 16,
                "checkpoint_step": 2,
            },
        )

        def observe(row):
            if str(row["idempotency_key"]) != "1" * 64:
                return False
            supervisor.store.mark_eval_terminal(
                idempotency_key="1" * 64,
                status="accepted",
                result={"status": "accepted"},
            )
            supervisor.eval_admission_closed = True
            return True

        with patch.object(supervisor, "_observe_result", side_effect=observe):
            self.assertEqual(supervisor._reconcile_evals_before_submission(), 2)
        self.assertEqual(supervisor._submit_pending_evals(), 0)
        self.assertEqual(
            [row["status"] for row in supervisor.store.evals()],
            ["accepted", "deferred"],
        )

    def test_failure_closes_eval_admission_and_defers_inflight_work(self) -> None:
        supervisor = self.supervisor()
        supervisor.store.init()
        supervisor.store.ensure_eval(
            checkpoint_ledger_id=1,
            intent={
                "idempotency_key": "1" * 64,
                "checkpoint_id": "checkpoint-1-" + "1" * 16,
                "checkpoint_step": 1,
            },
        )
        supervisor.store.mark_eval_submitted(
            idempotency_key="1" * 64,
            attempt=1,
            modal_call_id="fc-one",
            attempt_expires_at=10_000.0,
        )
        failure = LearnerFailure("scripted learner failure")

        supervisor._close_eval_admission_for_failure(failure)

        self.assertTrue(supervisor.eval_admission_closed)
        self.assertEqual(supervisor._submit_pending_evals(), 0)
        self.assertEqual(
            supervisor.store.state("automatic_eval_admission")["reason"],
            "learner_failure",
        )
        with (
            patch.object(supervisor, "_observe_result", return_value=False),
            patch.object(
                supervisor,
                "_reconcile_verified_eval_result",
                return_value=False,
            ),
        ):
            self.assertEqual(supervisor._defer_unsettled_evals_after_failure(), 1)
        self.assertEqual(supervisor.store.evals()[0]["status"], "deferred")

    def test_plateau_requires_valid_rejection_for_every_checkpoint(self) -> None:
        supervisor = self.supervisor()
        supervisor.store.init()
        for index, status in ((1, "rejected"), (2, "expired")):
            checkpoint_id = f"checkpoint-{index}-" + str(index) * 16
            supervisor.store.record_checkpoint_publication(
                checkpoint_ledger_id=index,
                manifest={
                    "checkpoint_id": checkpoint_id,
                    "step": index,
                    "purpose": "final" if index == 2 else "periodic",
                },
            )
            key = str(index) * 64
            supervisor.store.ensure_eval(
                checkpoint_ledger_id=index,
                intent={
                    "idempotency_key": key,
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_step": index,
                },
            )
            supervisor.store.mark_eval_terminal(
                idempotency_key=key,
                status=status,
                result={"status": status},
            )

        with self.assertRaisesRegex(
            IncompleteEvaluationEvidence,
            "valid rejected evaluation",
        ):
            supervisor._validate_no_acceptance_evidence()

        with supervisor.store.connection() as connection:
            connection.execute(
                "UPDATE eval_dispatches SET status = 'rejected', "
                'result_json = \'{"status":"rejected"}\' '
                "WHERE idempotency_key = ?",
                ("2" * 64,),
            )
        supervisor._validate_no_acceptance_evidence()

    def test_evaluation_wait_does_not_consume_delivery_drain_timeout(self) -> None:
        supervisor = self.supervisor()
        supervisor.store.init()
        with (
            patch.object(
                supervisor,
                "drain_iteration",
                side_effect=[(1, False), (0, True)],
            ),
            patch.object(
                supervisor.store,
                "all_evals_settled",
                side_effect=[False, True],
            ),
            patch.object(
                supervisor.clock,
                "monotonic",
                side_effect=[0.0, 400.0],
            ),
        ):
            supervisor._drain()

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
