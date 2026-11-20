from __future__ import annotations

import importlib.metadata
import io
import json
import os
import sys
import tempfile
import unittest
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from gradlab import runtime_refs


SOURCE_SHA = "1" * 40
RUNTIME_IMAGE_REF = "docker:ghcr.io/tsilva/gradlab/gradlab-train@sha256:" + "a" * 64
VIZDOOM_PROVIDER_VERSION = importlib.metadata.version("env-vizdoom-turbo")


def image_payload() -> dict:
    return {
        "schema_version": 7,
        "runtime_image_ref": RUNTIME_IMAGE_REF,
        "digest": "sha256:" + "a" * 64,
        "source_sha": SOURCE_SHA,
        "runtime_input_sha256": "d" * 64,
        "runtime_build_source_sha": "2" * 40,
        "overlay_key": "1" * 64,
        "dependency_key": "2" * 64,
        "gpu_key": "3" * 64,
        "train_plan_sha256": "4" * 64,
        "gpu_plan_sha256": "5" * 64,
        "tags": ["runtime-" + "d" * 64],
        "uv_lock_sha256": "e" * 64,
        "base_images": {
            "gpu": (
                "docker:ghcr.io/tsilva/gradlab/gradlab-train-gpu@sha256:"
                + "9" * 64
            ),
            "dependencies": (
                "docker:ghcr.io/tsilva/gradlab/gradlab-train-dependencies@sha256:"
                + "f" * 64
            ),
        },
        "workflow_run_id": "11",
        "vizdoom_smoke": {
            "contract_version": 3,
            "image_digest": "sha256:" + "a" * 64,
            "provider_distribution": "env-vizdoom-turbo",
            "provider_version": VIZDOOM_PROVIDER_VERSION,
            "evidence_sha256": "6" * 64,
        },
    }


def image_info() -> runtime_refs.RuntimeImageInfo:
    return runtime_refs.runtime_release_from_payload(
        image_payload(),
        label="test image",
        expected_source_sha=SOURCE_SHA,
    )


class RuntimeRefsTests(unittest.TestCase):
    def test_external_dispatch_requires_its_read_only_modal_protection_inventory(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.dict(os.environ, {
                runtime_refs.ACTIONS_DISPATCH_EXECUTOR_ENV: json.dumps([sys.executable]),
                "GRADLAB_CONTROL_R2_ACCESS_KEY_ID": "synthetic-source-value",
                "HTTPS_PROXY": "synthetic-proxy-value",
            }, clear=True),
            mock.patch.object(runtime_refs, "nonterminal_modal_app_names") as native,
        ):
            arguments = dict(protected_app_names=None, protect_nonterminal_apps=True, repo_root=temporary)
            with self.assertRaisesRegex(runtime_refs.ActionsExecutorError, runtime_refs.MODAL_PROTECTION_EXECUTOR_ENV):
                runtime_refs._modal_cleanup_arguments(**arguments)

            script = Path(temporary) / "inventory.py"
            script.write_text(
                'import json, os, sys\n'
                'assert not {"GRADLAB_CONTROL_R2_ACCESS_KEY_ID", "HTTPS_PROXY"} & os.environ.keys()\n'
                'print(json.dumps({"complete": sys.argv[1] == "complete", "protected_modal_apps": ["gradlab-eval-v3-current", "gradlab-eval-v2-retained"]}))\n',
                encoding="utf-8",
            )
            command = [sys.executable, str(script), "complete"]
            os.environ[runtime_refs.MODAL_PROTECTION_EXECUTOR_ENV] = json.dumps(command)
            self.assertEqual(runtime_refs._modal_cleanup_arguments(**arguments), [
                "-f", 'protected_modal_apps_json=["gradlab-eval-v2-retained","gradlab-eval-v3-current"]',
                "-f", "cleanup_authorized=true",
            ])
            command[-1] = "incomplete"
            os.environ[runtime_refs.MODAL_PROTECTION_EXECUTOR_ENV] = json.dumps(command)
            with self.assertRaisesRegex(runtime_refs.ActionsExecutorError, "incomplete or invalid"):
                runtime_refs._modal_cleanup_arguments(**arguments)
            with (
                mock.patch.object(runtime_refs, "runtime_release_for_source", side_effect=RuntimeError),
                mock.patch.object(runtime_refs, "_matching_runs", return_value=[]),
                mock.patch.object(runtime_refs, "require_remote_source"),
                mock.patch.object(runtime_refs, "_run_gh") as dispatch,
                self.assertRaisesRegex(runtime_refs.ActionsExecutorError, "incomplete or invalid"),
            ):
                runtime_refs.wait_for_runtime_release(
                    source_sha=SOURCE_SHA, workflow=runtime_refs.DEFAULT_IMAGE_WORKFLOW,
                    branch="main", artifact_name=runtime_refs.DEFAULT_IMAGE_ARTIFACT,
                    timeout=10, protect_nonterminal_apps=True,
                )
            dispatch.assert_not_called()
        native.assert_not_called()

    def test_inventory_command_requires_a_scoped_session_and_reads_only_live_manifests(self) -> None:
        from gradlab.training_vault import TrainingVaultError

        with (
            mock.patch("gradlab.training_vault.validate_session") as session,
            mock.patch("gradlab.r2_store.BucketConfig.from_env") as config,
            mock.patch("gradlab.r2_store.R2Bucket") as bucket_type,
            mock.patch("gradlab.operator_environment.load_repository_operator_environment") as native,
            redirect_stdout(io.StringIO()) as stdout,
            redirect_stderr(io.StringIO()) as stderr,
        ):
            session.side_effect = TrainingVaultError("synthetic credential detail")
            self.assertEqual(runtime_refs._serve_modal_protection_inventory(), 1)
            self.assertNotIn("synthetic credential detail", stderr.getvalue())
            self.assertEqual(stdout.getvalue(), "")
            config.assert_not_called()

            session.side_effect = None
            bucket = bucket_type.return_value
            bucket.iter_keys.return_value = [
                "runs/live/manifest.json", "runs/active-without-modal/manifest.json",
                "runs/finished/manifest.json", "runs/finished/terminal.json",
                "runs-archive/live/terminal.json", "runs-archive/unrelated/manifest.json",
            ]
            bucket.get_json.side_effect = lambda key: (
                {"modal": {"app_name": "gradlab-eval-v2-retained"}} if key == "runs/live/manifest.json" else {}
            )
            self.assertEqual(runtime_refs._serve_modal_protection_inventory(), 0)
            self.assertEqual(json.loads(stdout.getvalue()), {
                "complete": True, "protected_modal_apps": ["gradlab-eval-v2-retained"],
            })
            self.assertCountEqual(bucket.get_json.call_args_list, [
                mock.call("runs/live/manifest.json"), mock.call("runs/active-without-modal/manifest.json"),
            ])
            native.assert_not_called()

    def test_external_executor_reads_exact_runtime_archive_and_dispatches_separately(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt = root / "receipt.zip"
            dispatch_log = root / "dispatch.json"
            with zipfile.ZipFile(receipt, mode="w") as bundle:
                bundle.writestr(runtime_refs.DEFAULT_IMAGE_ARTIFACT_FILE, json.dumps(image_payload()))
            executor = root / "executor.py"
            executor.write_text(
                """\
import json
import os
import sys
from pathlib import Path

assert not {"GH_TOKEN", "GITHUB_TOKEN", "GH_CONFIG_DIR", "HTTPS_PROXY"} & os.environ.keys()
role, path, source, *args = sys.argv[1:]
if role == "dispatch":
    assert args[:2] == ["workflow", "run"]
    Path(path).write_text(json.dumps(args))
elif args[:2] == ["run", "list"]:
    print(json.dumps([{"databaseId": 11, "headSha": source, "status": "completed"}]))
elif args == ["repo", "view", "--json", "nameWithOwner"]:
    print(json.dumps({"nameWithOwner": "tsilva/gradlab"}))
elif args == ["api", "repos/tsilva/gradlab/actions/runs/11/artifacts?per_page=100"]:
    print(json.dumps({"artifacts": [{"name": "gradlab-train-image", "id": 42}]}))
elif args == ["api", "repos/tsilva/gradlab/actions/artifacts/42/zip"]:
    if role == "rejected":
        print("synthetic-credential", file=sys.stderr)
        sys.exit(1)
    sys.stdout.buffer.write(Path(path).read_bytes())
else:
    raise AssertionError(args)
""",
                encoding="utf-8",
            )
            reader = [sys.executable, str(executor), "read", str(receipt), SOURCE_SHA]
            writer = [sys.executable, str(executor), "dispatch", str(dispatch_log), SOURCE_SHA]
            with (
                mock.patch.dict(os.environ, {
                    runtime_refs.ACTIONS_READ_EXECUTOR_ENV: json.dumps(reader),
                    runtime_refs.ACTIONS_DISPATCH_EXECUTOR_ENV: json.dumps(writer),
                    "GH_TOKEN": "synthetic",
                    "GITHUB_TOKEN": "synthetic",
                    "GH_CONFIG_DIR": "synthetic",
                    "HTTPS_PROXY": "synthetic",
                }),
                mock.patch.object(runtime_refs, "_gh_executable") as native,
                mock.patch.object(runtime_refs, "require_remote_source") as require_remote,
            ):
                release = runtime_refs.runtime_release_for_source(source_sha=SOURCE_SHA)
                self.assertEqual(release.runtime_image_ref, RUNTIME_IMAGE_REF)
                self.assertFalse(dispatch_log.exists())

                reader[2] = "rejected"
                os.environ[runtime_refs.ACTIONS_READ_EXECUTOR_ENV] = json.dumps(reader)
                with self.assertRaisesRegex(
                    runtime_refs.ActionsExecutorError, "failed to download runtime artifacts"
                ) as error:
                    runtime_refs.wait_for_runtime_release(
                        source_sha=SOURCE_SHA,
                        workflow=runtime_refs.DEFAULT_IMAGE_WORKFLOW,
                        branch="main",
                        artifact_name=runtime_refs.DEFAULT_IMAGE_ARTIFACT,
                        timeout=10,
                    )
                self.assertNotIn("synthetic-credential", str(error.exception))
                self.assertFalse(dispatch_log.exists())
                require_remote.assert_not_called()

                dispatch = ["workflow", "run", runtime_refs.DEFAULT_IMAGE_WORKFLOW, "--ref", "main"]
                runtime_refs._run_gh(["gh", *dispatch])
                self.assertEqual(json.loads(dispatch_log.read_text()), dispatch)
                native.assert_not_called()

    def test_read_only_executor_rejects_missing_dispatcher_before_dispatch_prerequisites(self) -> None:
        with (
            mock.patch.dict(os.environ, {
                runtime_refs.ACTIONS_READ_EXECUTOR_ENV: json.dumps([sys.executable]),
            }, clear=True),
            mock.patch.object(runtime_refs, "runtime_release_for_source", side_effect=RuntimeError),
            mock.patch.object(runtime_refs, "_matching_runs", return_value=[]),
            mock.patch.object(runtime_refs, "require_remote_source") as require_remote,
            mock.patch.object(runtime_refs, "_run_gh") as dispatch,
        ):
            with self.assertRaisesRegex(
                runtime_refs.ActionsExecutorError, runtime_refs.ACTIONS_DISPATCH_EXECUTOR_ENV
            ):
                runtime_refs.wait_for_runtime_release(
                    source_sha=SOURCE_SHA,
                    workflow=runtime_refs.DEFAULT_IMAGE_WORKFLOW,
                    branch="main",
                    artifact_name=runtime_refs.DEFAULT_IMAGE_ARTIFACT,
                    timeout=10,
                )
        require_remote.assert_not_called()
        dispatch.assert_not_called()

    def test_runtime_image_descriptor_requires_current_json_document(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runtime-image.json"
            path.write_text(RUNTIME_IMAGE_REF, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "must contain a JSON object"):
                runtime_refs.runtime_image_payload_from_file(path)

    def test_failed_workflow_image_receipt_is_still_usable(self) -> None:
        run = {
            "databaseId": 11,
            "headSha": SOURCE_SHA,
            "status": "completed",
            "conclusion": "failure",
            "url": "https://example.test/run/11",
            "displayTitle": "Publish immutable runtime",
            "updatedAt": "2026-07-27T12:00:00Z",
        }
        with (
            mock.patch.object(runtime_refs, "_matching_runs", return_value=[run]),
            mock.patch.object(
                runtime_refs,
                "_artifact_payload_for_run",
                return_value=image_payload(),
            ),
        ):
            release = runtime_refs.runtime_release_for_source(source_sha=SOURCE_SHA)

        self.assertEqual(release.runtime_image_ref, RUNTIME_IMAGE_REF)
        self.assertEqual(release.workflow_run_id, "11")
        self.assertEqual(release.commit_message, "Publish immutable runtime")
        self.assertEqual(release.published_at, "2026-07-27T12:00:00Z")

    def test_current_schema_rejects_unexpected_descriptive_fields(self) -> None:
        for field in ("commit_message", "published_at", "created_at", "publishedAt"):
            with self.subTest(field=field):
                payload = image_payload()
                payload[field] = "unexpected"
                with self.assertRaisesRegex(ValueError, "unknown field"):
                    runtime_refs.runtime_release_from_payload(
                        payload,
                        label="test image",
                        expected_source_sha=SOURCE_SHA,
                    )

    def test_old_schema_is_rejected(self) -> None:
        payload = image_payload()
        payload["schema_version"] = 6
        payload.pop("vizdoom_smoke")
        with self.assertRaisesRegex(ValueError, "schema_version must be 7"):
            runtime_refs.runtime_release_from_payload(
                payload,
                label="old image",
                expected_source_sha=SOURCE_SHA,
            )

    def test_smoke_evidence_must_match_the_exact_image_digest(self) -> None:
        payload = image_payload()
        payload["vizdoom_smoke"]["image_digest"] = "sha256:" + "b" * 64
        with self.assertRaisesRegex(ValueError, "smoke digest"):
            runtime_refs.runtime_release_from_payload(
                payload,
                label="test image",
                expected_source_sha=SOURCE_SHA,
            )

    def test_smoke_provider_version_must_match_installed_distribution(self) -> None:
        payload = image_payload()
        payload["vizdoom_smoke"]["provider_version"] = "0.0.0"
        with self.assertRaisesRegex(ValueError, "smoke provider version mismatch"):
            runtime_refs.runtime_release_from_payload(
                payload,
                label="test image",
                expected_source_sha=SOURCE_SHA,
            )

    def test_active_workflow_is_reused_without_dispatch(self) -> None:
        active = {
            "databaseId": 11,
            "headSha": SOURCE_SHA,
            "status": "in_progress",
            "conclusion": "",
            "url": "https://example.test/run/11",
        }
        with (
            mock.patch.object(
                runtime_refs,
                "runtime_release_for_source",
                side_effect=[RuntimeError("not yet"), image_info()],
            ),
            mock.patch.object(runtime_refs, "_matching_runs", return_value=[active]),
            mock.patch.object(runtime_refs.time, "sleep"),
            mock.patch.object(runtime_refs, "_run_gh") as dispatch,
        ):
            release = runtime_refs.wait_for_runtime_release(
                source_sha=SOURCE_SHA,
                workflow=runtime_refs.DEFAULT_IMAGE_WORKFLOW,
                branch="main",
                artifact_name=runtime_refs.DEFAULT_IMAGE_ARTIFACT,
                timeout=60,
            )

        self.assertEqual(release.runtime_image_ref, RUNTIME_IMAGE_REF)
        dispatch.assert_not_called()

    def test_pull_request_image_check_is_not_a_publishable_release(self) -> None:
        runs = [
            {
                "databaseId": 11,
                "headSha": SOURCE_SHA,
                "event": "pull_request",
                "status": "completed",
                "conclusion": "success",
            },
            {
                "databaseId": 12,
                "headSha": SOURCE_SHA,
                "event": "workflow_dispatch",
                "status": "in_progress",
                "conclusion": "",
            },
        ]
        with mock.patch.object(runtime_refs, "_workflow_runs", return_value=runs):
            assert runtime_refs._matching_runs(
                source_sha=SOURCE_SHA, workflow=runtime_refs.DEFAULT_IMAGE_WORKFLOW
            ) == [runs[1]]

    def test_missing_workflow_dispatches_exact_source_once(self) -> None:
        active = {
            "databaseId": 12,
            "headSha": SOURCE_SHA,
            "status": "queued",
            "conclusion": "",
            "url": "https://example.test/run/12",
        }
        with (
            mock.patch.object(
                runtime_refs,
                "runtime_release_for_source",
                side_effect=[RuntimeError("missing"), RuntimeError("not yet"), image_info()],
            ),
            mock.patch.object(
                runtime_refs,
                "_matching_runs",
                side_effect=[[], [active]],
            ),
            mock.patch.object(runtime_refs, "require_remote_source") as remote,
            mock.patch.object(runtime_refs, "_run_gh") as dispatch,
            mock.patch.object(runtime_refs.time, "sleep"),
        ):
            release = runtime_refs.wait_for_runtime_release(
                source_sha=SOURCE_SHA,
                workflow=runtime_refs.DEFAULT_IMAGE_WORKFLOW,
                branch="main",
                artifact_name=runtime_refs.DEFAULT_IMAGE_ARTIFACT,
                timeout=60,
            )

        self.assertEqual(release.runtime_image_ref, RUNTIME_IMAGE_REF)
        remote.assert_called_once_with(SOURCE_SHA, branch="main", repo_root=".")
        dispatch.assert_called_once()
        self.assertIn(f"source_sha={SOURCE_SHA}", dispatch.call_args.args[0])

    def test_dispatched_workflow_failure_stops_without_retry_loop(self) -> None:
        failed = {
            "databaseId": 12,
            "headSha": SOURCE_SHA,
            "status": "completed",
            "conclusion": "failure",
            "url": "https://example.test/run/12",
        }
        with (
            mock.patch.object(
                runtime_refs,
                "runtime_release_for_source",
                side_effect=RuntimeError("missing"),
            ),
            mock.patch.object(runtime_refs, "_matching_runs", side_effect=[[], [failed]]),
            mock.patch.object(runtime_refs, "require_remote_source"),
            mock.patch.object(runtime_refs, "_run_gh") as dispatch,
            mock.patch.object(runtime_refs.time, "sleep"),
        ):
            with self.assertRaisesRegex(RuntimeError, "completed without a usable image receipt"):
                runtime_refs.wait_for_runtime_release(
                    source_sha=SOURCE_SHA,
                    workflow=runtime_refs.DEFAULT_IMAGE_WORKFLOW,
                    branch="main",
                    artifact_name=runtime_refs.DEFAULT_IMAGE_ARTIFACT,
                    timeout=60,
                )

        dispatch.assert_called_once()

    def test_reused_active_workflow_failure_stops_without_new_dispatch(self) -> None:
        active = {
            "databaseId": 11,
            "headSha": SOURCE_SHA,
            "status": "in_progress",
            "conclusion": "",
            "url": "https://example.test/run/11",
        }
        failed = {
            **active,
            "status": "completed",
            "conclusion": "failure",
        }
        with (
            mock.patch.object(
                runtime_refs,
                "runtime_release_for_source",
                side_effect=RuntimeError("missing"),
            ),
            mock.patch.object(
                runtime_refs,
                "_matching_runs",
                side_effect=[[active], [failed]],
            ),
            mock.patch.object(runtime_refs, "_run_gh") as dispatch,
            mock.patch.object(runtime_refs.time, "sleep"),
        ):
            with self.assertRaisesRegex(RuntimeError, "completed without a usable image receipt"):
                runtime_refs.wait_for_runtime_release(
                    source_sha=SOURCE_SHA,
                    workflow=runtime_refs.DEFAULT_IMAGE_WORKFLOW,
                    branch="main",
                    artifact_name=runtime_refs.DEFAULT_IMAGE_ARTIFACT,
                    timeout=60,
                )

        dispatch.assert_not_called()

    def test_runtime_readiness_timeout_reports_without_canceling(self) -> None:
        with (
            mock.patch.object(
                runtime_refs,
                "runtime_release_for_source",
                side_effect=RuntimeError("missing"),
            ),
            mock.patch.object(runtime_refs, "_matching_runs", return_value=[]),
            mock.patch.object(runtime_refs, "require_remote_source"),
            mock.patch.object(runtime_refs, "_run_gh") as dispatch,
            mock.patch.object(runtime_refs.time, "monotonic", side_effect=[0.0, 61.0]),
        ):
            with self.assertRaisesRegex(TimeoutError, "timed out waiting"):
                runtime_refs.wait_for_runtime_release(
                    source_sha=SOURCE_SHA,
                    workflow=runtime_refs.DEFAULT_IMAGE_WORKFLOW,
                    branch="main",
                    artifact_name=runtime_refs.DEFAULT_IMAGE_ARTIFACT,
                    timeout=60,
                )

        dispatch.assert_called_once()

    def test_remote_source_requires_exact_branch_head(self) -> None:
        completed = mock.Mock(returncode=0, stdout=f"{'2' * 40}\trefs/heads/main\n", stderr="")
        with mock.patch("subprocess.run", return_value=completed):
            with self.assertRaisesRegex(RuntimeError, "push the commit"):
                runtime_refs.require_remote_source(SOURCE_SHA, branch="main")

    def test_modal_readiness_wait_is_one_explicit_decision(self) -> None:
        args = SimpleNamespace(
            image_workflow=runtime_refs.DEFAULT_IMAGE_WORKFLOW,
            image_branch="main",
            image_artifact=runtime_refs.DEFAULT_IMAGE_ARTIFACT,
            runtime_readiness_timeout=60,
            runtime_image_ref_file=None,
        )
        release = image_info()
        with (
            mock.patch.object(runtime_refs, "clean_git_source_sha", return_value=SOURCE_SHA),
            mock.patch.object(
                runtime_refs,
                "wait_for_runtime_release",
                return_value=release,
            ),
            mock.patch.object(
                runtime_refs,
                "wait_for_modal_readiness",
                return_value=release,
            ) as modal_wait,
        ):
            runtime_refs.runtime_release_from_args(args, wait_for_modal=False)
            runtime_refs.runtime_release_from_args(args, wait_for_modal=False)
            runtime_refs.runtime_release_from_args(args, wait_for_modal=True)

        modal_wait.assert_called_once()

    def test_missing_modal_readiness_dispatches_lifecycle_safe_deployment(self) -> None:
        release = replace(
            image_info(),
            runtime_input_sha256="b" * 64,
            runtime_build_source_sha="2" * 40,
        )
        readiness = runtime_refs.ModalReadinessInfo(
            runtime_image_ref=release.runtime_image_ref,
            source_sha=release.source_sha,
            modal_app_name="gradlab-eval-v3-111111111111",
            startup_probe={},
            workflow_run_id="13",
            runtime_input_sha256=release.runtime_input_sha256,
            runtime_build_source_sha=release.runtime_build_source_sha,
        )
        completed_image = {
            "databaseId": 12,
            "headSha": SOURCE_SHA,
            "status": "completed",
            "conclusion": "success",
            "url": "https://example.test/run/12",
        }
        with (
            mock.patch.object(
                runtime_refs,
                "modal_readiness_for_release",
                side_effect=[RuntimeError("missing"), readiness],
            ),
            mock.patch.object(
                runtime_refs,
                "_matching_runs",
                side_effect=[[completed_image], []],
            ),
            mock.patch.object(runtime_refs, "require_remote_source"),
            mock.patch.object(runtime_refs, "_run_gh") as dispatch,
            mock.patch.object(runtime_refs.time, "sleep"),
        ):
            actual = runtime_refs.wait_for_modal_readiness(
                release,
                timeout=60,
                branch="main",
                protected_app_names=("gradlab-eval-v3-222222222222",),
            )

        self.assertEqual(actual.modal_app_name, readiness.modal_app_name)
        dispatch.assert_called_once()
        command = dispatch.call_args.args[0]
        self.assertIn(runtime_refs.DEFAULT_MODAL_WORKFLOW, command)
        self.assertIn(f"source_sha={SOURCE_SHA}", command)
        self.assertIn("cleanup_authorized=true", command)
        self.assertIn(
            'protected_modal_apps_json=["gradlab-eval-v3-222222222222"]',
            command,
        )

    def test_explicit_runtime_receipt_does_not_require_git_branch(self) -> None:
        args = SimpleNamespace(
            image_workflow=runtime_refs.DEFAULT_IMAGE_WORKFLOW,
            image_branch=None,
            image_artifact=runtime_refs.DEFAULT_IMAGE_ARTIFACT,
            runtime_readiness_timeout=60,
            runtime_image_ref_file="receipt.json",
        )
        release = image_info()
        with (
            mock.patch.object(runtime_refs, "clean_git_source_sha", return_value=SOURCE_SHA),
            mock.patch.object(runtime_refs, "runtime_image_payload_from_file", return_value={}),
            mock.patch.object(runtime_refs, "runtime_release_from_payload", return_value=release),
            mock.patch.object(
                runtime_refs,
                "current_git_branch",
                side_effect=AssertionError("branch lookup is unnecessary"),
            ),
        ):
            actual = runtime_refs.runtime_release_from_args(args, wait_for_modal=False)

        self.assertEqual(actual, release)

    def test_existing_runtime_only_never_enters_dispatching_wait(self) -> None:
        args = SimpleNamespace(
            image_workflow=runtime_refs.DEFAULT_IMAGE_WORKFLOW,
            image_artifact=runtime_refs.DEFAULT_IMAGE_ARTIFACT,
            runtime_readiness_timeout=60,
            runtime_image_ref_file=None,
            existing_runtime_only=True,
            expected_runtime_image_ref=None,
            expected_runtime_input_sha256=None,
            expected_runtime_build_source_sha=None,
        )
        release = image_info()
        with (
            mock.patch.object(runtime_refs, "clean_git_source_sha", return_value=SOURCE_SHA),
            mock.patch.object(
                runtime_refs, "runtime_release_for_source", return_value=release
            ) as existing,
            mock.patch.object(runtime_refs, "wait_for_runtime_release") as wait,
        ):
            actual = runtime_refs.runtime_release_from_args(args, wait_for_modal=False)

        self.assertEqual(actual, release)
        existing.assert_called_once_with(
            source_sha=SOURCE_SHA,
            workflow=runtime_refs.DEFAULT_IMAGE_WORKFLOW,
            branch=None,
            artifact_name=runtime_refs.DEFAULT_IMAGE_ARTIFACT,
        )
        wait.assert_not_called()

    def test_expected_runtime_guards_are_all_or_none_and_exact(self) -> None:
        release = replace(
            image_info(),
            runtime_input_sha256="b" * 64,
            runtime_build_source_sha="2" * 40,
        )
        base = dict(
            image_workflow=runtime_refs.DEFAULT_IMAGE_WORKFLOW,
            image_artifact=runtime_refs.DEFAULT_IMAGE_ARTIFACT,
            runtime_readiness_timeout=60,
            runtime_image_ref_file=None,
            existing_runtime_only=True,
        )
        with (
            mock.patch.object(runtime_refs, "clean_git_source_sha", return_value=SOURCE_SHA),
            mock.patch.object(runtime_refs, "runtime_release_for_source", return_value=release),
        ):
            with self.assertRaisesRegex(ValueError, "must be supplied together"):
                runtime_refs.runtime_release_from_args(
                    SimpleNamespace(
                        **base,
                        expected_runtime_image_ref=RUNTIME_IMAGE_REF,
                        expected_runtime_input_sha256=None,
                        expected_runtime_build_source_sha=None,
                    ),
                    wait_for_modal=False,
                )
            with self.assertRaisesRegex(RuntimeError, "pinned research runtime"):
                runtime_refs.runtime_release_from_args(
                    SimpleNamespace(
                        **base,
                        expected_runtime_image_ref=RUNTIME_IMAGE_REF,
                        expected_runtime_input_sha256="c" * 64,
                        expected_runtime_build_source_sha="2" * 40,
                    ),
                    wait_for_modal=False,
                )
            actual = runtime_refs.runtime_release_from_args(
                SimpleNamespace(
                    **base,
                    expected_runtime_image_ref=RUNTIME_IMAGE_REF,
                    expected_runtime_input_sha256="b" * 64,
                    expected_runtime_build_source_sha="2" * 40,
                ),
                wait_for_modal=False,
            )

        self.assertEqual(actual, release)


if __name__ == "__main__":
    unittest.main()
