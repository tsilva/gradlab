from __future__ import annotations

import importlib.metadata
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from gradlab import runtime_refs
from gradlab.runtime_contract import (
    RUNTIME_DESCRIPTOR_SCHEMA_VERSION,
    runtime_contract,
    train_config_contract_payload,
    train_config_contract_sha256,
    validate_config_payload,
)


RUNTIME_IMAGE_REF = "docker:ghcr.io/tsilva/gradlab/gradlab-train@sha256:" + "a" * 64
SOURCE_SHA = "1" * 40
BUILD_SOURCE_SHA = "2" * 40
VIZDOOM_PROVIDER_VERSION = importlib.metadata.version("vizdoom-turbo")


def release_payload(*, source_sha: str = SOURCE_SHA) -> dict:
    return {
        "schema_version": RUNTIME_DESCRIPTOR_SCHEMA_VERSION,
        "runtime_image_ref": RUNTIME_IMAGE_REF,
        "digest": "sha256:" + "a" * 64,
        "source_sha": source_sha,
        "runtime_input_sha256": "d" * 64,
        "runtime_build_source_sha": BUILD_SOURCE_SHA,
        "overlay_key": "1" * 64,
        "dependency_key": "2" * 64,
        "gpu_key": "3" * 64,
        "train_plan_sha256": "4" * 64,
        "gpu_plan_sha256": "5" * 64,
        "tags": ["runtime-" + "d" * 64],
        "uv_lock_sha256": "e" * 64,
        "base_images": {
            "gpu": "docker:ghcr.io/tsilva/gradlab/gradlab-train-gpu@sha256:" + "9" * 64,
            "dependencies": "docker:ghcr.io/tsilva/gradlab/gradlab-train-dependencies@sha256:"
            + "f" * 64,
        },
        "workflow_run_id": "123",
        "vizdoom_smoke": {
            "contract_version": 2,
            "image_digest": "sha256:" + "a" * 64,
            "provider_distribution": "vizdoom-turbo",
            "provider_version": VIZDOOM_PROVIDER_VERSION,
            "evidence_sha256": "6" * 64,
        },
    }


def modal_readiness_payload(*, source_sha: str = SOURCE_SHA) -> dict:
    contract_sha = train_config_contract_sha256()
    app_name = "gradlab-eval-" + "a" * 12
    return {
        "schema_version": 3,
        "runtime_image_ref": RUNTIME_IMAGE_REF,
        "source_sha": source_sha,
        "runtime_input_sha256": "d" * 64,
        "runtime_build_source_sha": BUILD_SOURCE_SHA,
        "modal_app_name": app_name,
        "startup_probe": {
            "schema_version": 1,
            "app_name": app_name,
            "runtime_image_ref": RUNTIME_IMAGE_REF,
            "runtime_build_source_sha": BUILD_SOURCE_SHA,
            "runtime_input_sha256": "d" * 64,
            "train_config_contract_sha256": contract_sha,
        },
        "workflow_run_id": "123",
    }


class RuntimeContractTests(unittest.TestCase):
    def test_contract_is_stable_and_runtime_reports_source(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"GRADLAB_SOURCE_SHA": "abc123", "GRADLAB_RUNTIME_INPUT_SHA256": "d" * 64},
        ):
            receipt = runtime_contract(runtime_image_ref=RUNTIME_IMAGE_REF)

        self.assertEqual(receipt["runtime_build_source_sha"], "abc123")
        self.assertEqual(receipt["runtime_input_sha256"], "d" * 64)
        self.assertEqual(receipt["runtime_image_ref"], RUNTIME_IMAGE_REF)
        self.assertEqual(len(receipt["train_config_contract_sha256"]), 64)

    def test_runtime_contract_exposes_only_live_field_metadata(self) -> None:
        fields = train_config_contract_payload()["fields"]

        self.assertTrue(fields)
        for field in fields:
            self.assertNotIn("aliases", field)
            self.assertNotIn("flag", field)
            self.assertNotIn("help", field)
            self.assertNotIn("recipe_required", field)
            self.assertNotIn("serialize", field)
            self.assertNotEqual(field["kind"], "store_true")

    def test_runtime_payload_validation_accepts_dstack_execution_fields(self) -> None:
        receipt = validate_config_payload(
            {
                "attempt_id": "attempt-0123456789abcdef",
                "compute_target": "b3",
                "dstack_task": "gradlab-0123456789abcdef0123456789abcdef",
                "run_name": "gradlab-0123456789abcdef0123456789abcdef",
                "runtime_image_ref": RUNTIME_IMAGE_REF,
                "seed": 123,
                "training_backend": {"id": "sb3.ppo", "config": {}},
                "wandb_display_name": "Level1-1__ppo__s123__01234567",
                "wandb_group": "gradlab-0123456789abcdef0123456789abcdef",
                "wandb_run_id": "gradlab-test",
            }
        )

        self.assertTrue(receipt["validated"])
        self.assertEqual(receipt["validated_field_count"], 10)

    def test_image_receipt_rejects_noncurrent_schema_and_digest_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "schema_version must be 7"):
            runtime_refs.runtime_release_from_payload(
                {"runtime_image_ref": RUNTIME_IMAGE_REF, "source_sha": SOURCE_SHA},
                label="release",
                expected_source_sha=SOURCE_SHA,
            )
        payload = release_payload()
        payload["digest"] = "sha256:" + "b" * 64
        with self.assertRaisesRegex(ValueError, "digest does not match"):
            runtime_refs.runtime_release_from_payload(
                payload,
                label="release",
                expected_source_sha=SOURCE_SHA,
            )

    def test_receipts_require_exact_source_and_modal_probe_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "source_sha mismatch"):
            runtime_refs.runtime_release_from_payload(
                release_payload(source_sha="3" * 40),
                label="release",
                expected_source_sha="4" * 40,
            )
        payload = modal_readiness_payload()
        payload["startup_probe"]["runtime_image_ref"] = (
            "docker:ghcr.io/tsilva/gradlab/gradlab-train@sha256:" + "c" * 64
        )
        with self.assertRaisesRegex(ValueError, "startup_probe.runtime_image_ref"):
            runtime_refs.modal_readiness_from_payload(
                payload,
                label="Modal readiness",
                expected_source_sha=SOURCE_SHA,
                expected_runtime_image_ref=RUNTIME_IMAGE_REF,
                expected_runtime_input_sha256="d" * 64,
                expected_runtime_build_source_sha=BUILD_SOURCE_SHA,
            )
        payload = modal_readiness_payload()
        payload["startup_probe"]["train_config_contract_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "startup_probe.train_config_contract_sha256"):
            runtime_refs.modal_readiness_from_payload(
                payload,
                label="Modal readiness",
                expected_source_sha=SOURCE_SHA,
                expected_runtime_image_ref=RUNTIME_IMAGE_REF,
                expected_runtime_input_sha256="d" * 64,
                expected_runtime_build_source_sha=BUILD_SOURCE_SHA,
            )

    def test_version_five_receipt_rejects_invalid_runtime_identity_fields(self) -> None:
        cases = {
            "runtime fingerprint": ("runtime_input_sha256", "short", "runtime_input_sha256"),
            "build source": ("runtime_build_source_sha", "not-a-sha", "runtime_build_source_sha"),
            "dependency": (
                "base_images",
                {"dependencies": "docker:mutable"},
                "dependency image identity",
            ),
            "workflow": ("workflow_run_id", "", "workflow_run_id"),
            "GPU key": ("gpu_key", "short", "gpu_key"),
        }
        for label, (field, value, error) in cases.items():
            with self.subTest(label=label):
                payload = release_payload()
                payload[field] = value
                with self.assertRaisesRegex(ValueError, error):
                    runtime_refs.runtime_release_from_payload(
                        payload,
                        label="release",
                        expected_source_sha=SOURCE_SHA,
                    )

    def test_clean_git_source_rejects_dirty_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch("subprocess.run") as run:
                run.return_value = mock.Mock(returncode=0, stdout=" M file.py\n", stderr="")
                with self.assertRaisesRegex(RuntimeError, "clean worktree"):
                    runtime_refs.clean_git_source_sha(root)


if __name__ == "__main__":
    unittest.main()
