from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from gradlab.r2_store import BucketConfig, ConditionalWriteConflict, RunStorageConfig
from gradlab.goal_variants import build_goal_variant_descriptor, goal_variant_scope_key
from gradlab.run_authority import LeaseUnavailable, RunAuthority
from gradlab.run_contracts import (
    DEFAULT_LIVENESS_POLICY,
    EvalIntent,
    PromotionReceipt,
    RunManifest,
    TerminalReceipt,
    checkpoint_id,
    eval_idempotency_key,
    new_attempt_id,
    new_run_id,
    utc_now,
)
from gradlab.policy_bundle import (
    build_recipe_document,
    canonical_json_sha256,
    model_document_path,
    recipe_document_path,
)
from gradlab.recipe_documents import compose_resolved_train_documents


SHA = "a" * 64


class RunAuthorityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.storage = RunStorageConfig(
            control=BucketConfig((root / "control").resolve().as_uri()),
            evaluation=BucketConfig((root / "eval").resolve().as_uri()),
            models=BucketConfig(
                (root / "models").resolve().as_uri(),
                public_base_url="https://models.example.test",
            ),
        )
        self.authority = RunAuthority(self.storage)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def manifest(self, run_id: str, attempt_id: str) -> RunManifest:
        goal_slug = "SuperMarioBros-Nes-v0/Level1-1"
        goal_variant = build_goal_variant_descriptor(
            goal_slug=goal_slug,
            source_sha="e" * 40,
            authored_goal={"goal_id": "Level1-1"},
            effective_goal={"goal_id": "Level1-1"},
        )
        return RunManifest(
            run_id=run_id,
            attempt_id=attempt_id,
            created_at=utc_now(),
            source_sha="e" * 40,
            image_digest="docker:registry.example/gradlab@sha256:" + SHA,
            goal_slug=goal_slug,
            goal_sha256=goal_variant["effective_goal_contract_sha256"],
            recipe_slug="ppo",
            recipe_sha256=SHA,
            recipe_overrides=(),
            environment_sha256=SHA,
            seed=123,
            run_description="B3 clean-slate dstack acceptance",
            compute={
                "request": {
                    "kind": "local",
                    "target": "b3",
                    "max_duration_seconds": 86_400,
                },
                "selected": {
                    "kind": "local",
                    "target": "b3",
                    "max_duration_seconds": 86_400,
                },
                "dstack_task": run_id,
                "runtime_workflow_run_id": "123",
                "runtime_input_sha256": SHA,
                "runtime_build_source_sha": "e" * 40,
            },
            wandb={
                "run_id": run_id,
                "entity": "tsilva",
                "project": "super-mario-bros",
                "url": f"https://wandb.ai/tsilva/super-mario-bros/runs/{run_id}",
            },
            modal={
                "enabled": True,
                "environment_name": "gradlab-eval",
                "app_name": "gradlab-eval-v3-" + "e" * 12,
                "function_name": "evaluate_checkpoint",
                "deployment_source_sha": "e" * 40,
                "rom_asset_manifest": {"sha256": SHA},
            },
            storage=self.storage.manifest_locations(),
            goal_variant=goal_variant,
            liveness=DEFAULT_LIVENESS_POLICY,
        )

    def test_identifiers_have_required_shapes(self) -> None:
        self.assertRegex(new_run_id(), r"^gradlab-[0-9a-f]{32}$")
        self.assertRegex(new_attempt_id(), r"^attempt-[0-9a-f]{16}$")
        self.assertEqual(
            checkpoint_id(step=250_000, sha256=SHA), "checkpoint-250000-aaaaaaaaaaaaaaaa"
        )

    def test_manifest_is_create_only_and_idempotent(self) -> None:
        run_id = new_run_id()
        manifest = self.manifest(run_id, new_attempt_id())
        first = self.authority.create_manifest(manifest)
        second = self.authority.create_manifest(manifest)
        self.assertEqual(first, second)
        changed = {**manifest.to_dict(), "seed": 124}
        with self.assertRaises(ConditionalWriteConflict):
            self.authority.control.put_json(
                f"runs/{run_id}/manifest.json",
                changed,
                create_only=True,
            )

    def test_manifest_binds_recipe_overrides(self) -> None:
        run_id = new_run_id()
        manifest = self.manifest(run_id, new_attempt_id())
        overridden = RunManifest(
            **{
                **manifest.to_dict(),
                "recipe_overrides": ["train.backend.config.learning_rate=0.0002"],
            }
        )
        overridden.validate()
        self.assertEqual(
            overridden.to_dict()["recipe_overrides"],
            ["train.backend.config.learning_rate=0.0002"],
        )
        invalid = RunManifest(**{**manifest.to_dict(), "recipe_overrides": [""]})
        with self.assertRaisesRegex(ValueError, "non-empty"):
            invalid.validate()

    def test_goal_variant_registration_is_idempotent_and_goal_scoped(self) -> None:
        authored = {
            "goal_id": "Level1-1",
            "title": "Mario Level1-1",
            "train": {"environment": {"task": {"sticky": 0}}},
            "eval": {"environment": {"task": {"sticky": 0}}},
        }
        effective = json.loads(json.dumps(authored))
        effective["train"]["environment"]["task"]["sticky"] = 0.25
        descriptor = build_goal_variant_descriptor(
            goal_slug="SuperMarioBros-Nes-v0/Level1-1",
            source_sha="e" * 40,
            authored_goal=authored,
            effective_goal=effective,
        )
        base = self.manifest(new_run_id(), new_attempt_id())
        manifest = RunManifest(
            **{
                **base.to_dict(),
                "goal_sha256": descriptor["effective_goal_contract_sha256"],
                "goal_variant": descriptor,
            }
        )

        self.authority.create_manifest(manifest)
        first = self.authority.register_goal_variant(manifest)
        second = self.authority.register_goal_variant(manifest)

        self.assertEqual(first, second)
        scope = goal_variant_scope_key(goal_slug=manifest.goal_slug)
        index = self.authority.control.get_json(f"{scope}/index.json")
        self.assertEqual(len(index["variants"]), 1)
        self.assertEqual(
            index["variants"][0]["variant_id"],
            descriptor["variant_id"],
        )
        run_index_key = f"{scope}/runs/{descriptor['variant_id']}.json"
        run_index = self.authority.control.get_json(run_index_key)
        self.assertEqual(run_index["runs"][0]["run_id"], manifest.run_id)
        self.assertEqual(run_index["runs"][0]["state"], "running")
        self.assertEqual(run_index["runs"][0]["recipe_variant_id"], "base")

        receipt = TerminalReceipt(
            run_id=manifest.run_id,
            attempt_id=manifest.attempt_id,
            state="failed",
            acceptance_required=True,
            stop_reason="training_cap_without_acceptance",
            final_step=100,
            checkpoint_inventory=[],
            eval_inventory=[],
            wandb_high_water_mark=1,
            drain={"complete": True},
            completed_at=utc_now(),
        )
        self.authority.create_attempt_terminal(
            receipt,
            metrics={"train/global_step": 100.0},
        )
        updated = self.authority.control.get_json(run_index_key)["runs"][0]
        self.assertEqual(updated["state"], "failed")
        self.assertEqual(updated["stop_reason"], "training_cap_without_acceptance")
        self.assertEqual(updated["final_step"], 100)
        self.assertEqual(updated["metrics"], {"train/global_step": 100.0})

    def test_catalog_clear_removes_current_and_noncurrent_indexes_and_projection_receipts(
        self,
    ) -> None:
        manifest = self.manifest(new_run_id(), new_attempt_id())
        self.authority.create_manifest(manifest)
        self.authority.control.put_json(
            "goal-variants/v1/scopes/obsolete/index.json",
            {"schema_version": 1},
        )
        self.authority.control.put_json(
            f"runs/{manifest.run_id}/goal-variant-registration.json",
            {"schema_version": 1},
        )

        cleared = self.authority.clear_goal_variant_catalog()

        self.assertGreaterEqual(cleared["catalog_objects"], 3)
        self.assertEqual(cleared["projection_receipts"], 2)
        self.assertEqual(list(self.authority.control.iter_keys("goal-variants/")), [])
        self.assertIsNone(
            self.authority.control.get_json_optional(
                f"runs/{manifest.run_id}/goal-variant-projection.json"
            )
        )
        self.assertIsNone(
            self.authority.control.get_json_optional(
                f"runs/{manifest.run_id}/goal-variant-registration.json"
            )
        )

    def test_v2_recipe_is_content_addressed_and_registers_exact_variant_resolution(
        self,
    ) -> None:
        goal_path = Path("experiments/goals/gradlab__bandit/_goal.yaml")
        recipe_path = goal_path.parent / "recipes/ppo.yaml"
        resolved = compose_resolved_train_documents(
            goal_path,
            recipe_path,
            source_sha="e" * 40,
        )
        recipe = build_recipe_document(
            resolved.effective,
            repo_root=Path.cwd(),
            source_commit="e" * 40,
            run_description="content-addressed resolution proof",
            seed=123,
            runtime_packages=("gradlab==0.1.0",),
            base_materialized_recipe=resolved.base,
            canonical_goal=resolved.canonical_goal,
        )
        digest = canonical_json_sha256(recipe)

        self.assertEqual(
            self.authority.put_recipe_document(recipe, expected_sha256=digest),
            digest,
        )
        self.assertEqual(self.authority.recipe_document(digest), recipe)
        # A repeated write is idempotent and verifies the existing bytes.
        self.assertEqual(
            self.authority.put_recipe_document(recipe, expected_sha256=digest),
            digest,
        )

        original = self.manifest(new_run_id(), new_attempt_id())
        manifest = RunManifest(
            **{
                **original.to_dict(),
                "goal_slug": "gradlab__bandit",
                "goal_sha256": resolved.effective["train_config"]["effective_goal_contract_sha256"],
                "recipe_sha256": digest,
                "environment_sha256": str(resolved.effective["environment_hash"]).removeprefix(
                    "sha256:"
                ),
                "wandb": {
                    **original.wandb,
                    "project": "Bandit-v0",
                },
                "goal_variant": resolved.effective["goal_variant"],
            }
        )
        self.authority.create_manifest(manifest)
        scope = goal_variant_scope_key(goal_slug="gradlab__bandit")
        index = self.authority.control.get_json(f"{scope}/index.json")
        self.assertEqual(
            index["variants"][0]["exact_resolution_run_id"],
            manifest.run_id,
        )

    def test_lease_takeover_requires_expiry_and_old_etag_cannot_renew(self) -> None:
        run_id = new_run_id()
        old_attempt = new_attempt_id()
        instant = datetime(2026, 7, 24, 10, 0, tzinfo=UTC)
        lease = self.authority.acquire_lease(
            run_id=run_id,
            attempt_id=old_attempt,
            holder_id="container-one",
            now=instant,
        )
        with self.assertRaises(LeaseUnavailable):
            self.authority.acquire_lease(
                run_id=run_id,
                attempt_id=new_attempt_id(),
                holder_id="container-two",
                now=instant + timedelta(seconds=59),
            )
        takeover = self.authority.acquire_lease(
            run_id=run_id,
            attempt_id=new_attempt_id(),
            holder_id="container-two",
            now=instant + timedelta(seconds=61),
        )
        self.assertGreater(takeover.generation, lease.generation)
        with self.assertRaises(LeaseUnavailable):
            self.authority.renew_lease(lease, now=instant + timedelta(seconds=10))

    def test_writer_lease_release_is_conditional_and_immediate(self) -> None:
        run_id = new_run_id()
        attempt_id = new_attempt_id()
        lease = self.authority.acquire_lease(
            run_id=run_id,
            attempt_id=attempt_id,
            holder_id="operator-reconcile",
        )

        self.authority.release_lease(lease)

        self.assertIsNone(
            self.authority.control.get_json_optional(
                f"runs/{run_id}/writer-lease.json"
            )
        )
        replacement = self.authority.acquire_lease(
            run_id=run_id,
            attempt_id=new_attempt_id(),
            holder_id="next-writer",
        )
        self.assertEqual(replacement.generation, 1)

    def test_metric_segments_are_ordered_and_immutable(self) -> None:
        run_id = new_run_id()
        attempt_id = new_attempt_id()
        key, digest = self.authority.seal_metric_segment(
            run_id=run_id,
            attempt_id=attempt_id,
            events=[
                {"event_seq": 1, "event_id": "one", "metrics": {"x": 1}},
                {"event_seq": 2, "event_id": "two", "metrics": {"x": 2}},
            ],
        )
        self.assertEqual(hashlib.sha256(self.authority.control.get_bytes(key)).hexdigest(), digest)
        with self.assertRaises(ValueError):
            self.authority.seal_metric_segment(
                run_id=run_id,
                attempt_id=attempt_id,
                events=[{"event_seq": 1}, {"event_seq": 1}],
            )

    def test_delivered_metric_journals_move_to_expiring_prefix(self) -> None:
        run_id = new_run_id()
        attempt_id = new_attempt_id()
        source_key, digest = self.authority.seal_metric_segment(
            run_id=run_id,
            attempt_id=attempt_id,
            events=[
                {"event_seq": 1, "event_id": "one"},
                {"event_seq": 2, "event_id": "two"},
            ],
        )
        archived = self.authority.archive_metric_journals(run_id=run_id)
        self.assertEqual(archived["segment_count"], 1)
        self.assertEqual(
            archived["prefix"],
            f"expiring-metric-journals/{run_id}/",
        )
        self.assertEqual(
            list(
                self.authority.control.iter_keys(
                    f"runs/{run_id}/attempts/{attempt_id}/metric-segments"
                )
            ),
            [],
        )
        destination_key = archived["keys"][0]
        self.assertEqual(
            hashlib.sha256(self.authority.control.get_bytes(destination_key)).hexdigest(),
            digest,
        )
        self.assertFalse(self.authority.control._file_path(source_key).exists())
        self.assertEqual(
            self.authority.archive_metric_journals(run_id=run_id),
            archived,
        )

    def test_checkpoint_is_verified_and_public_index_is_cas_updated(self) -> None:
        run_id = new_run_id()
        root = Path(self.temporary.name)
        first_model = root / "one.zip"
        second_model = root / "two.zip"
        first_model.write_bytes(b"first-model")
        second_model.write_bytes(b"second-model")
        for path in (first_model, second_model):
            model_document_path(path).write_text('{"model":true}\n', encoding="utf-8")
            recipe_document_path(path).write_text('{"recipe":true}\n', encoding="utf-8")
        hashes = {
            "goal_sha256": SHA,
            "recipe_sha256": SHA,
            "environment_sha256": SHA,
            "evaluation_contract_sha256": SHA,
        }
        second = self.authority.publish_checkpoint(
            run_id=run_id,
            model_path=second_model,
            step=500_000,
            purpose="periodic",
            contract_hashes=hashes,
            recovery_sidecar={"local_path": "checkpoints/two.zip"},
        )
        first = self.authority.publish_checkpoint(
            run_id=run_id,
            model_path=first_model,
            step=250_000,
            purpose="periodic",
            contract_hashes=hashes,
            recovery_sidecar={"local_path": "checkpoints/one.zip"},
        )
        index = self.authority.models.get_json(f"runs/{run_id}/index.json")
        self.assertEqual(
            [row["checkpoint_id"] for row in index["checkpoints"]],
            [first.checkpoint_id, second.checkpoint_id],
        )
        self.assertTrue(first.public_url.startswith("https://models.example.test/"))
        public_manifest = json.loads(
            self.authority.models.get_bytes(
                f"runs/{run_id}/checkpoints/250000-{first.sha256}/manifest.json"
            )
        )
        self.assertEqual(public_manifest["sha256"], first.sha256)
        promotion = PromotionReceipt(
            run_id=run_id,
            checkpoint_id=first.checkpoint_id,
            checkpoint_step=first.step,
            eval_idempotency_key="f" * 64,
            eval_result_sha256="f" * 64,
            accepted_episode_count=100,
            promoted_at=utc_now(),
        )
        self.authority.create_promotion(promotion)
        promoted_index = self.authority.models.get_json(f"runs/{run_id}/index.json")
        self.assertEqual(
            promoted_index["promotion"]["checkpoint_id"],
            first.checkpoint_id,
        )

    def test_state_archive_generation_is_content_addressed_and_restorable(self) -> None:
        run_id = new_run_id()
        attempt_id = new_attempt_id()
        archive_root = Path(self.temporary.name) / "local-archive"
        source = archive_root / "blobs" / "ab" / "payload"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"portable-provider-state")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        files = [
            {
                "path": "blobs/ab/payload",
                "sha256": digest,
                "size_bytes": source.stat().st_size,
            }
        ]
        inventory_sha256 = hashlib.sha256(
            json.dumps(
                files,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        (archive_root / "closure.json").write_text(
            json.dumps(
                {
                    "semantic_id": "state-archive-v1",
                    "schema_version": 1,
                    "status": "closed",
                    "step": 64,
                    "inventory_sha256": inventory_sha256,
                    "archive": {
                        "semantic_id": "state-archive-v1",
                        "entry_count": 1,
                        "blob_count": 1,
                    },
                    "files": files,
                }
            ),
            encoding="utf-8",
        )

        publication = self.authority.publish_state_archive(
            run_id=run_id,
            attempt_id=attempt_id,
            archive_root=archive_root,
        )
        restored_root = Path(self.temporary.name) / "restored-archive"
        restored = self.authority.restore_state_archive(
            run_id=run_id,
            destination=restored_root,
        )
        self.assertEqual(restored, publication)
        self.assertEqual(
            (restored_root / "blobs" / "ab" / "payload").read_bytes(),
            source.read_bytes(),
        )
        self.assertRegex(publication["generation_sha256"], r"^[0-9a-f]{64}$")

    def test_eval_intent_is_deterministic_and_private(self) -> None:
        run_id = new_run_id()
        key = eval_idempotency_key(
            run_id=run_id,
            checkpoint_sha256=SHA,
            evaluation_contract_sha256=SHA,
            episode_manifest_sha256=SHA,
            protocol="acceptance-v2",
        )
        intent = EvalIntent(
            run_id=run_id,
            checkpoint_id=checkpoint_id(step=250_000, sha256=SHA),
            idempotency_key=key,
            checkpoint_sha256=SHA,
            goal_sha256=SHA,
            recipe_sha256=SHA,
            environment_sha256=SHA,
            evaluation_contract_sha256=SHA,
            episode_manifest_sha256=SHA,
            protocol="acceptance-v2",
            execution_contract={"episodes": 100},
            result_key=f"runs/{run_id}/evals/{key}/result.json",
            timeout_seconds=1200,
            created_at=utc_now(),
            expires_at="2026-07-24T12:00:00Z",
        )
        self.authority.put_eval_intent(intent)
        stored = self.authority.evaluation.get_json(f"runs/{run_id}/evals/{key}/intent.json")
        self.assertEqual(stored["checkpoint_id"], intent.checkpoint_id)
        self.assertEqual(list(self.authority.models.iter_keys(f"runs/{run_id}/evals")), [])

    def test_canonical_terminal_requires_complete_scientific_evidence(self) -> None:
        run_id = new_run_id()
        attempt_id = new_attempt_id()
        checkpoint = "checkpoint-250000-" + "a" * 16
        promotion = PromotionReceipt(
            run_id=run_id,
            checkpoint_id=checkpoint,
            checkpoint_step=250_000,
            eval_idempotency_key="b" * 64,
            eval_result_sha256="c" * 64,
            accepted_episode_count=100,
            promoted_at=utc_now(),
        )
        self.authority.control.put_json(
            f"runs/{run_id}/promotion.json",
            promotion.to_dict(),
        )
        drain = {
            "complete": True,
            "metric_segment_high_water": 12,
            "wandb_remote_high_water_mark": 12,
            "publication_capacity_ratio": 2.5,
            "journal_archive": {
                "prefix": f"expiring-metric-journals/{run_id}/",
                "segment_count": 1,
                "keys": ["segment.jsonl"],
            },
            "journal_expires_at": utc_now(),
        }
        receipt = TerminalReceipt(
            run_id=run_id,
            attempt_id=attempt_id,
            state="succeeded",
            acceptance_required=True,
            stop_reason="eval_acceptance",
            final_step=260_000,
            checkpoint_inventory=[
                {
                    "checkpoint_id": checkpoint,
                    "step": 250_000,
                    "purpose": "periodic",
                },
                {
                    "checkpoint_id": "checkpoint-260000-" + "d" * 16,
                    "step": 260_000,
                    "purpose": "final",
                },
            ],
            eval_inventory=[
                {
                    "checkpoint_id": checkpoint,
                    "checkpoint_step": 250_000,
                    "status": "accepted",
                }
            ],
            wandb_high_water_mark=12,
            drain=drain,
            completed_at=utc_now(),
        )
        self.authority.create_terminal(receipt)
        self.assertEqual(
            self.authority.control.get_json(f"runs/{run_id}/terminal.json")["state"],
            "succeeded",
        )

        missing_remote = TerminalReceipt(
            **{
                **receipt.to_dict(),
                "run_id": new_run_id(),
                "drain": {**drain, "wandb_remote_high_water_mark": 11},
            }
        )
        with self.assertRaisesRegex(ValueError, "remotely visible"):
            self.authority.create_terminal(missing_remote)

    def test_training_only_terminal_is_attempt_scoped_and_not_scientific_success(
        self,
    ) -> None:
        run_id = new_run_id()
        attempt_id = new_attempt_id()
        receipt = TerminalReceipt(
            run_id=run_id,
            attempt_id=attempt_id,
            state="succeeded",
            acceptance_required=False,
            stop_reason="training_cap_complete",
            final_step=1_000_000,
            checkpoint_inventory=[
                {
                    "checkpoint_id": "checkpoint-1000000-" + "a" * 16,
                    "step": 1_000_000,
                    "purpose": "final",
                }
            ],
            eval_inventory=[],
            wandb_high_water_mark=25,
            drain={
                "complete": True,
                "metric_segment_high_water": 25,
                "wandb_remote_high_water_mark": 25,
            },
            completed_at=utc_now(),
        )

        self.authority.create_attempt_terminal(receipt)
        state = self.authority.semantic_state(run_id)
        self.assertEqual(state["attempt_terminals"], [receipt.to_dict()])
        self.assertIsNone(state["terminal"])
        with self.assertRaisesRegex(ValueError, "acceptance-backed"):
            self.authority.create_terminal(receipt)


if __name__ == "__main__":
    unittest.main()
