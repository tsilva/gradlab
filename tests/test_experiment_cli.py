from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from gradlab.dstack_backend import DstackTask
from gradlab.experiment_cli import (
    _bind_launch_contract,
    _bind_vizdoom_iwad_for_launch,
    _catalog_rebuild_contract_failures,
    _compute,
    _follow_fingerprint,
    _latest_attempt_terminal,
    _manifest_rom_asset,
    _manifest_vizdoom_iwad,
    _manifest_dstack_project,
    _operator_preflight,
    _poll_status,
    _project_reconciled_terminal,
    _public_dstack_state,
    _record_pre_submit_failure,
    _record_terminal_task_without_receipt,
    _require_retryable_attempt_terminal,
    _retry_compute_request,
    _required_operator_environment,
    _run_completed,
    _stage_rom,
    _stage_vizdoom_iwad,
    _task_name,
    _task_request,
    _wandb_identity,
    build_parser,
    cmd_cancel,
    cmd_follow,
    cmd_fault_test,
    cmd_launch,
    cmd_reconcile,
    cmd_resume_submit,
    cmd_wait,
    main,
)
from gradlab.operator_credentials import OperatorConfigurationError
from gradlab.goal_variants import build_goal_variant_descriptor
from gradlab.policy_bundle import build_recipe_document
from gradlab.recipe_documents import (
    compose_resolved_train_documents,
)
from gradlab.run_contracts import (
    DEFAULT_LIVENESS_POLICY,
    RunManifest,
    new_attempt_id,
    new_run_id,
)
from gradlab.vizdoom_assets import vizdoom_iwad_binding


@pytest.mark.parametrize(
    ("goal_slug", "expected_goal"),
    [
        ("SuperMarioBros-Nes-v0/Level1-1", "Level1-1"),
        ("SuperMarioBros-Nes-v0/World1/Level1-1", "World1--Level1-1"),
        ("SuperMarioBros-Nes-v0", "SuperMarioBros-Nes-v0"),
        ("custom/Level1-1", "custom--Level1-1"),
    ],
)
def test_wandb_identity_uses_project_relative_goal_display_names(
    goal_slug: str,
    expected_goal: str,
) -> None:
    run_id = "gradlab-0123456789abcdef0123456789abcdef"
    document = {
        "train_config": {
            "env_provider": "supermariobrosnes-turbo",
            "game": "SuperMarioBros-Nes-v0",
        }
    }

    with mock.patch("gradlab.experiment_cli.wandb_entity_from_env", return_value="entity"):
        identity = _wandb_identity(
            document,
            run_id,
            goal_slug=goal_slug,
            recipe_slug="ppo-local",
            recipe_variant="base",
            seed=7,
        )

    assert identity["project"] == "SuperMarioBros-Nes-v0"
    assert identity["display_name"] == f"{expected_goal}__ppo-local__s7__01234567"
    assert identity["group"] == f"cohort::{goal_slug}::ppo-local::base"
    assert identity["run_id"] == run_id
    assert identity["url"].endswith(f"/runs/{run_id}")


def test_wandb_identity_prefers_declared_campaign_group() -> None:
    document = {
        "campaign_id": "mario-local-confirmation",
        "train_config": {
            "env_provider": "supermariobrosnes-turbo",
            "game": "SuperMarioBros-Nes-v0",
        },
    }

    with mock.patch("gradlab.experiment_cli.wandb_entity_from_env", return_value="entity"):
        identity = _wandb_identity(
            document,
            "gradlab-fedcba9876543210fedcba9876543210",
            goal_slug="SuperMarioBros-Nes-v0/Level1-1",
            recipe_slug="ppo-local",
            recipe_variant="v-12345678",
            seed=123,
        )

    assert identity["group"] == "campaign::mario-local-confirmation"


def test_catalog_cli_exposes_only_the_current_rebuild_command() -> None:
    args = build_parser().parse_args(["catalog-rebuild", "--json"])

    assert args.command == "catalog-rebuild"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["unknown-catalog-command"])


def test_catalog_rebuild_rejects_conflicting_current_descriptors_before_clear() -> None:
    first = _manifest_only_run()
    second_run_id = new_run_id()
    second = replace(
        first,
        run_id=second_run_id,
        attempt_id=new_attempt_id(),
        compute={**first.compute, "dstack_task": second_run_id},
        wandb={
            **first.wandb,
            "run_id": second_run_id,
            "url": f"https://wandb.example/runs/{second_run_id}",
        },
        goal_variant={**dict(first.goal_variant or {}), "label": "Conflicting label"},
    )
    second.validate()

    failures = _catalog_rebuild_contract_failures(
        [
            (f"runs/{first.run_id}/manifest.json", first, None),
            (f"runs/{second.run_id}/manifest.json", second, None),
        ]
    )

    assert failures == [
        {
            "key": f"runs/{second.run_id}/manifest.json",
            "error_type": "ValueError",
            "error": "goal variant descriptor conflicts with another current run",
        }
    ]


def test_wandb_identity_cohort_group_includes_override_variant() -> None:
    document = {
        "train_config": {
            "env_provider": "supermariobrosnes-turbo",
            "game": "SuperMarioBros-Nes-v0",
        }
    }

    with mock.patch("gradlab.experiment_cli.wandb_entity_from_env", return_value="entity"):
        identity = _wandb_identity(
            document,
            "gradlab-fedcba9876543210fedcba9876543210",
            goal_slug="SuperMarioBros-Nes-v0/Level1-1",
            recipe_slug="ppo-local",
            recipe_variant="v-12345678",
            seed=123,
        )

    assert identity["group"] == ("cohort::SuperMarioBros-Nes-v0/Level1-1::ppo-local::v-12345678")


def _manifest_only_run() -> RunManifest:
    run_id = new_run_id()
    attempt_id = new_attempt_id()
    source_sha = "a" * 40
    compute = {
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
        "selected_offer": None,
        "dstack_project": "research",
        "dstack_task": run_id,
        "runtime_workflow_run_id": "12345",
        "runtime_input_sha256": "b" * 64,
        "runtime_build_source_sha": source_sha,
    }
    goal_variant = build_goal_variant_descriptor(
        goal_slug="example/goal",
        source_sha=source_sha,
        authored_goal={"goal_id": "goal"},
        effective_goal={"goal_id": "goal"},
    )
    manifest = RunManifest(
        run_id=run_id,
        attempt_id=attempt_id,
        created_at=(datetime.now(UTC) - timedelta(minutes=1)).isoformat().replace("+00:00", "Z"),
        source_sha=source_sha,
        image_digest="docker:example/gradlab@sha256:" + "c" * 64,
        goal_slug="example/goal",
        goal_sha256=goal_variant["effective_goal_contract_sha256"],
        recipe_slug="ppo",
        recipe_sha256="e" * 64,
        recipe_overrides=[],
        environment_sha256="f" * 64,
        seed=123,
        run_description="manifest-only recovery regression",
        compute=compute,
        wandb={
            "run_id": run_id,
            "entity": "example",
            "project": "example",
            "url": f"https://wandb.example/runs/{run_id}",
        },
        modal={
            "enabled": True,
            "environment_name": "gradlab-eval",
            "app_name": f"gradlab-eval-v3-{source_sha[:12]}",
            "function_name": "evaluate_checkpoint",
            "deployment_source_sha": source_sha,
            "rom_asset_manifest": None,
        },
        storage={
            "control": "s3://control-private",
            "evaluation": "s3://eval-private",
            "models": "s3://models-public",
            "public_models_base_url": "https://models.example",
        },
        goal_variant=goal_variant,
        liveness=DEFAULT_LIVENESS_POLICY,
    )
    manifest.validate()
    return manifest


def test_launch_parser_exposes_bounded_compute_and_hash_bound_overrides() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "launch",
            "--goal-file",
            "experiments/goals/goal/_goal.yaml",
            "--recipe-file",
            "experiments/goals/goal/recipes/ppo.yaml",
            "--seed",
            "123",
            "--run-description",
            "one isolated learning-rate ablation",
            "--set",
            "train.backend.config.learning_rate=0.0002",
            "--compute",
            "spot",
            "--target",
            "aws",
            "--max-price",
            "1.25",
            "--max-cost-usd",
            "5",
            "--max-duration",
            "8h",
        ]
    )

    assert args.recipe_overrides == ["train.backend.config.learning_rate=0.0002"]
    assert args.checkpoint_eval_backend is None
    compute = _compute(args)
    assert compute.kind == "spot"
    assert compute.target == "aws"
    assert compute.max_duration_seconds == 8 * 60 * 60
    assert compute.bounded_duration_seconds == 4 * 60 * 60


def test_fault_test_is_bounded_and_not_exposed_as_a_launch_override() -> None:
    parser = build_parser()
    args = parser.parse_args(["fault-test", "--json"])
    with mock.patch("gradlab.experiment_cli.cmd_launch", return_value=0) as launch:
        assert cmd_fault_test(args) == 0

    forwarded = launch.call_args.args[0]
    assert forwarded.max_duration == 120
    assert forwarded.compute == "local"
    assert forwarded.target is None
    assert forwarded.checkpoint_eval_backend == "none"
    assert forwarded.recipe_overrides == []
    assert forwarded.supervision_fault_fixture == "failed-result-live-process"
    launch_args = parser.parse_args(
        [
            "launch",
            "--goal-file",
            "experiments/goals/VizdoomBasic-v1/_goal.yaml",
            "--recipe-file",
            "experiments/goals/VizdoomBasic-v1/recipes/ppo.yaml",
            "--seed",
            "17",
            "--run-description",
            "normal launch",
        ]
    )
    assert not hasattr(launch_args, "supervision_fault_fixture")


def test_auto_without_cloud_budget_uses_operator_local_fleet() -> None:
    with mock.patch.dict("os.environ", {"GRADLAB_LOCAL_FLEET": "local-gpu"}):
        compute = _compute(
            SimpleNamespace(
                compute="auto",
                target=None,
                max_price=None,
                max_cost_usd=None,
                allow_on_demand=False,
                max_duration=3600,
            )
        )
    assert compute.kind == "auto"
    assert compute.target == "local-gpu"
    assert compute.max_price is None
    assert compute.bounded_duration_seconds == 3600


def test_operator_preflight_parser_defaults_to_modal() -> None:
    args = build_parser().parse_args(["operator-preflight", "--json"])

    assert args.checkpoint_eval_backend == "modal"
    assert args.target is None
    assert args.json is True


def test_operator_preflight_parser_accepts_local_target_override() -> None:
    args = build_parser().parse_args(
        ["operator-preflight", "--target", "alternate-local", "--json"]
    )

    assert args.target == "alternate-local"


def test_manifest_project_binding_wins_and_legacy_manifest_uses_operator_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DSTACK_PROJECT", "operator-project")

    assert _manifest_dstack_project({"dstack_project": "bound-project"}) == "bound-project"
    assert _manifest_dstack_project({}) == "operator-project"


def test_targetless_legacy_retry_reuses_previously_selected_local_fleet() -> None:
    request = _retry_compute_request(
        {
            "request": {
                "kind": "local",
                "target": None,
                "max_price": None,
                "max_cost_usd": None,
                "allow_on_demand": False,
                "max_duration_seconds": 3600,
            },
            "selected": {"kind": "local", "target": "recorded-local"},
        }
    )

    assert request["target"] == "recorded-local"


def test_targetless_legacy_retry_without_selected_fleet_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="no recorded local fleet"):
        _retry_compute_request(
            {
                "request": {"kind": "auto", "target": None},
                "selected": {"kind": "spot", "target": None},
            }
        )


def test_operator_preflight_reports_resolved_project_fleet_and_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in _required_operator_environment("none"):
        monkeypatch.setenv(name, "operator-value")
    monkeypatch.setenv("DSTACK_PROJECT", "research")
    monkeypatch.setenv("GRADLAB_LOCAL_FLEET", "configured-local")

    environment_report = SimpleNamespace(
        config_path=tmp_path / "operator.toml",
        config_present=True,
        source_for=lambda name, environment: (
            "operator-config" if str(environment.get(name) or "").strip() else "missing"
        ),
    )
    storage = SimpleNamespace(
        control=SimpleNamespace(),
        evaluation=SimpleNamespace(),
        models=SimpleNamespace(),
    )
    bucket = mock.MagicMock()
    bucket.iter_keys.return_value = iter(())
    with (
        mock.patch(
            "gradlab.experiment_cli._load_environment",
            return_value=environment_report,
        ),
        mock.patch(
            "gradlab.experiment_cli.RunStorageConfig.from_env",
            return_value=storage,
        ),
        mock.patch("gradlab.experiment_cli.RunAuthority", return_value=mock.MagicMock()),
        mock.patch("gradlab.experiment_cli.R2Bucket", return_value=bucket),
        mock.patch("gradlab.experiment_cli.DstackBackend.preflight"),
        mock.patch(
            "gradlab.experiment_cli.wandb_entity_from_env",
            return_value="example-entity",
        ),
    ):
        _storage, _authority, backend, report = _operator_preflight(
            tmp_path,
            checkpoint_eval_backend="none",
        )

    assert backend.project == "research"
    assert report["dstack"]["project"] == "research"
    assert report["compute"] == {
        "local_fleet": "configured-local",
        "source": "operator-config",
    }


def test_resume_submit_parser_requires_one_existing_run() -> None:
    run_id = new_run_id()
    args = build_parser().parse_args(["resume-submit", "--run", run_id, "--json"])

    assert args.run_id == run_id
    assert args.json is True


def test_launch_operator_preflight_runs_before_runtime_readiness(
    tmp_path: Path,
) -> None:
    goal = tmp_path / "experiments/goals/example/_goal.yaml"
    recipe = goal.parent / "recipes/ppo.yaml"
    args = SimpleNamespace(
        goal_file=goal,
        recipe_file=recipe,
        recipe_overrides=[],
        checkpoint_eval_backend="modal",
        compute="local",
        target="local-gpu",
        max_price=None,
        max_cost_usd=None,
        allow_on_demand=False,
        max_duration=3600,
        rom_path=None,
    )

    with (
        mock.patch("gradlab.experiment_cli.repository_root", return_value=tmp_path),
        mock.patch("gradlab.experiment_cli.clean_git_source_sha", return_value="a" * 40),
        mock.patch("gradlab.experiment_cli.current_git_branch", return_value="main"),
        mock.patch(
            "gradlab.experiment_cli._tracked_committed_path",
            side_effect=[goal, recipe],
        ),
        mock.patch(
            "gradlab.experiment_cli.compose_resolved_train_documents",
            return_value=SimpleNamespace(
                effective={
                    "train_config": {
                        "checkpoint_eval_backend": "modal",
                        "env_provider": "gradlab",
                    }
                },
            ),
        ),
        mock.patch(
            "gradlab.experiment_cli._operator_preflight",
            side_effect=OperatorConfigurationError("missing operator credentials"),
        ),
        mock.patch("gradlab.experiment_cli.runtime_release_from_args") as runtime_release,
    ):
        with pytest.raises(OperatorConfigurationError, match="missing operator"):
            cmd_launch(args)

    runtime_release.assert_not_called()


def test_operator_configuration_error_is_concise_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("GRADLAB_OPERATOR_CONFIG", str(tmp_path / "missing.toml"))
    for name in _required_operator_environment("none"):
        monkeypatch.delenv(name, raising=False)

    assert main(["operator-preflight", "--checkpoint-eval-backend", "none"]) == 2

    error = capsys.readouterr().err
    assert "operator configuration error" in error
    assert "DSTACK_TOKEN" in error
    assert "Traceback" not in error


def test_public_dstack_state_never_exposes_raw_task_environment() -> None:
    value = _public_dstack_state(
        DstackTask(
            project="research",
            name="run-one",
            status="running",
            raw={
                "fleet": {"name": "local-gpu"},
                "submitted_at": "2026-07-24T16:00:00Z",
                "run_spec": {
                    "configuration": {
                        "env": {
                            "WANDB_API_KEY": "should-never-appear",
                            "GRADLAB_CONTROL_R2_SECRET_ACCESS_KEY": "also-secret",
                        }
                    }
                },
            },
        )
    )

    encoded = json.dumps(value, sort_keys=True)
    assert value["fleet"] == "local-gpu"
    assert "raw" not in value
    assert "should-never-appear" not in encoded
    assert "also-secret" not in encoded


def test_status_poller_observes_once_before_an_immediate_timeout(tmp_path: Path) -> None:
    value = {"completed": False}
    with (
        mock.patch("gradlab.experiment_cli._status", return_value=value) as status,
        mock.patch("gradlab.experiment_cli.time.monotonic", side_effect=[10.0, 10.0]),
        mock.patch("gradlab.experiment_cli.time.sleep") as sleep,
    ):
        assert list(
            _poll_status(
                tmp_path,
                "gradlab-" + "0" * 32,
                timeout=0.0,
                poll_seconds=2.0,
            )
        ) == [(value, True)]

    status.assert_called_once()
    sleep.assert_not_called()


@pytest.mark.parametrize("command", (cmd_follow, cmd_wait))
def test_observers_prefer_completion_over_simultaneous_timeout(
    command,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    value = {
        "completed": True,
        "dstack": {"status": "done"},
        "semantic": {},
    }
    args = SimpleNamespace(
        run_id="gradlab-" + "0" * 32,
        timeout=0.0,
        poll_seconds=2.0,
        until="terminal",
    )
    with (
        mock.patch("gradlab.experiment_cli.repository_root", return_value=tmp_path),
        mock.patch("gradlab.experiment_cli._poll_status", return_value=iter([(value, True)])),
    ):
        assert command(args) == 0

    assert json.loads(capsys.readouterr().out) == value


def test_follow_fingerprint_ignores_only_poll_observation_time() -> None:
    first = {
        "run_id": "gradlab-" + "a" * 32,
        "semantic": {
            "observed_at": 1.0,
            "attempts": [{"attempt_id": "attempt-" + "b" * 16}],
        },
    }
    second = {
        **first,
        "semantic": {**first["semantic"], "observed_at": 2.0},
    }

    assert _follow_fingerprint(first) == _follow_fingerprint(second)
    second["semantic"]["attempts"] = [
        {"attempt_id": "attempt-" + "b" * 16},
        {"attempt_id": "attempt-" + "c" * 16},
    ]
    assert _follow_fingerprint(first) != _follow_fingerprint(second)


@pytest.mark.parametrize(
    ("semantic_terminal", "attempt_terminal", "dstack_terminal", "expected"),
    [
        (None, None, False, False),
        (
            None,
            {"state": "succeeded", "acceptance_required": True},
            False,
            False,
        ),
        (
            None,
            {"state": "succeeded", "acceptance_required": True},
            True,
            False,
        ),
        (
            {"state": "succeeded"},
            {"state": "succeeded", "acceptance_required": True},
            True,
            True,
        ),
        (
            None,
            {"state": "succeeded", "acceptance_required": False},
            True,
            True,
        ),
        (
            None,
            {"state": "failed", "acceptance_required": True},
            True,
            True,
        ),
        (
            None,
            {"state": "resumable_failure", "acceptance_required": True},
            True,
            True,
        ),
    ],
)
def test_run_completed_requires_receipts_expected_for_the_attempt(
    semantic_terminal: dict[str, object] | None,
    attempt_terminal: dict[str, object] | None,
    dstack_terminal: bool,
    expected: bool,
) -> None:
    assert (
        _run_completed(
            semantic_terminal=semantic_terminal,
            attempt_terminal=attempt_terminal,
            dstack_terminal=dstack_terminal,
        )
        is expected
    )


def test_designed_early_stop_failure_is_non_resumable() -> None:
    with pytest.raises(RuntimeError, match="non-resumable"):
        _require_retryable_attempt_terminal(
            {
                "state": "failed",
                "stop_reason": "early_stop_failure:return_plateau",
            }
        )

    _require_retryable_attempt_terminal(
        {
            "state": "resumable_failure",
            "stop_reason": "supervisor_failure",
        }
    )

    with pytest.raises(RuntimeError, match="non-resumable"):
        _require_retryable_attempt_terminal(
            {
                "state": "stopped",
                "stop_reason": "early_stop_neutral:return_plateau",
            }
        )


def test_on_demand_requires_explicit_permission() -> None:
    with pytest.raises(ValueError, match="requires --allow-on-demand"):
        _compute(
            SimpleNamespace(
                compute="on-demand",
                target="aws",
                max_price=2.0,
                max_cost_usd=10.0,
                allow_on_demand=False,
                max_duration=3600,
            )
        )


def test_retry_task_name_preserves_run_and_changes_attempt() -> None:
    run_id = new_run_id()
    attempt_id = new_attempt_id()

    assert _task_name(run_id, attempt_id, initial=True) == run_id
    retry_name = _task_name(run_id, attempt_id, initial=False)
    assert retry_name.startswith("gradlab-")
    assert len(retry_name) == len(run_id)
    assert _task_name(run_id, attempt_id, initial=False) == retry_name
    assert _task_name(new_run_id(), attempt_id, initial=False) != retry_name
    assert _task_name(run_id, new_attempt_id(), initial=False) != retry_name


def test_latest_attempt_terminal_does_not_reuse_prior_attempt_receipt() -> None:
    first = _manifest_only_run().to_dict()
    second = {
        **first,
        "attempt_id": new_attempt_id(),
        "created_at": "2026-07-26T00:00:01Z",
    }
    first_terminal = {
        "attempt_id": first["attempt_id"],
        "completed_at": "2026-07-26T00:00:00Z",
    }
    state = {
        "attempts": [first, second],
        "attempt_terminals": [first_terminal],
    }

    assert _latest_attempt_terminal(state) is None
    second_terminal = {
        "attempt_id": second["attempt_id"],
        "completed_at": "2026-07-26T00:00:02Z",
    }
    state["attempt_terminals"].append(second_terminal)
    assert _latest_attempt_terminal(state) == second_terminal


def test_pre_submit_failure_records_typed_attempt_evidence() -> None:
    manifest = _manifest_only_run()
    authority = mock.MagicMock()
    prefix = f"runs/{manifest.run_id}"
    authority.run_prefix.return_value = prefix
    authority.control.iter_keys.return_value = iter(
        [f"{prefix}/attempts/{manifest.attempt_id}/manifest.json"]
    )

    _record_pre_submit_failure(authority, manifest)

    receipt = authority.create_attempt_terminal.call_args.args[0]
    assert receipt.run_id == manifest.run_id
    assert receipt.attempt_id == manifest.attempt_id
    assert receipt.state == "resumable_failure"
    assert receipt.stop_reason == "pre_submit_failure"
    assert receipt.final_step == 0
    assert receipt.drain["complete"] is False


def test_terminal_task_without_receipt_records_typed_startup_failure() -> None:
    manifest = _manifest_only_run()
    authority = mock.MagicMock()
    task = DstackTask(project="research", name="gradlab-retry", status="failed")

    _record_terminal_task_without_receipt(
        authority,
        manifest,
        task,
        writer_lease=SimpleNamespace(
            run_id=manifest.run_id,
            attempt_id=manifest.attempt_id,
        ),
    )

    receipt = authority.create_attempt_terminal.call_args.args[0]
    assert receipt.run_id == manifest.run_id
    assert receipt.attempt_id == manifest.attempt_id
    assert receipt.state == "resumable_failure"
    assert receipt.stop_reason == "supervisor_startup_failure"
    assert receipt.drain["phase"] == "startup/recovery"
    assert "terminal status 'failed'" in receipt.drain["failure"]


def test_active_task_without_receipt_cannot_be_sealed() -> None:
    manifest = _manifest_only_run()
    authority = mock.MagicMock()
    task = DstackTask(project="research", name="gradlab-retry", status="running")

    with pytest.raises(RuntimeError, match="while its dstack task is active"):
        _record_terminal_task_without_receipt(
            authority,
            manifest,
            task,
            writer_lease=SimpleNamespace(
                run_id=manifest.run_id,
                attempt_id=manifest.attempt_id,
            ),
        )


def test_reconcile_acquires_lease_writes_r2_before_wandb_and_releases(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = _manifest_only_run()
    document = manifest.to_dict()
    state = {
        "run_id": manifest.run_id,
        "manifest": document,
        "attempts": [document],
        "attempt_terminals": [],
        "public_index": {
            "checkpoints": [
                {
                    "checkpoint_id": "checkpoint-17-" + "a" * 16,
                    "step": 17,
                }
            ]
        },
    }
    authority = mock.MagicMock()
    authority.semantic_state.side_effect = [state, state]
    lease = SimpleNamespace(
        run_id=manifest.run_id,
        attempt_id=manifest.attempt_id,
    )
    authority.acquire_lease.return_value = lease
    events: list[str] = []
    authority.create_attempt_terminal.side_effect = lambda _receipt: events.append("r2")
    backend = mock.MagicMock()
    backend.status.return_value = DstackTask(
        project="research",
        name=str(manifest.compute["dstack_task"]),
        status="failed",
    )
    args = SimpleNamespace(
        run_id=manifest.run_id,
        stop_reason="learner_failure",
        evidence_sha256=["f" * 64],
        json=True,
    )

    with (
        mock.patch("gradlab.experiment_cli.repository_root", return_value=tmp_path),
        mock.patch(
            "gradlab.experiment_cli._storage",
            return_value=(SimpleNamespace(), authority),
        ),
        mock.patch("gradlab.experiment_cli.DstackBackend", return_value=backend),
        mock.patch(
            "gradlab.experiment_cli._project_reconciled_terminal",
            side_effect=lambda *_args: events.append("wandb"),
        ),
    ):
        assert cmd_reconcile(args) == 0

    assert events == ["r2", "wandb"]
    authority.acquire_lease.assert_called_once()
    authority.release_lease.assert_called_once_with(lease)
    receipt = authority.create_attempt_terminal.call_args.args[0]
    assert receipt.state == "resumable_failure"
    assert receipt.stop_reason == "learner_failure"
    assert receipt.final_step == 17
    assert receipt.checkpoint_inventory == tuple(state["public_index"]["checkpoints"])
    assert receipt.drain["recovered_checkpoint_count"] == 1
    assert receipt.drain["evidence_sha256"] == ["f" * 64]
    output = json.loads(capsys.readouterr().out)
    assert output["wandb_projected"] is True


def test_reconciled_failure_closes_wandb_with_nonzero_exit() -> None:
    manifest = _manifest_only_run()
    projector = mock.MagicMock()
    receipt = SimpleNamespace(state="resumable_failure")

    with (
        mock.patch(
            "gradlab.experiment_cli.WandbProjector.resume",
            return_value=projector,
        ) as resume,
        mock.patch("gradlab.experiment_cli.publish_terminal_summary") as publish,
    ):
        _project_reconciled_terminal(manifest, receipt)

    resume.assert_called_once_with(
        {
            "wandb_run_id": manifest.run_id,
            "wandb_entity": manifest.wandb["entity"],
            "wandb_project": manifest.wandb["project"],
            "wandb_mode": "online",
            "run_name": manifest.wandb.get("display_name"),
            "wandb_group": manifest.wandb.get("group"),
        },
        update_finish_state=True,
    )
    publish.assert_called_once_with(projector.run, receipt)
    projector.close.assert_called_once_with(timeout_seconds=300, exit_code=1)


def test_reconciled_stopped_run_closes_wandb_with_zero_exit() -> None:
    manifest = _manifest_only_run()
    projector = mock.MagicMock()
    receipt = SimpleNamespace(state="stopped")

    with (
        mock.patch(
            "gradlab.experiment_cli.WandbProjector.resume",
            return_value=projector,
        ),
        mock.patch("gradlab.experiment_cli.publish_terminal_summary"),
    ):
        _project_reconciled_terminal(manifest, receipt)

    projector.close.assert_called_once_with(timeout_seconds=300, exit_code=0)


def test_resume_submit_recovers_only_the_original_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = _manifest_only_run()
    document = manifest.to_dict()
    prefix = f"runs/{manifest.run_id}"
    authority = mock.MagicMock()
    authority.run_prefix.return_value = prefix
    authority.semantic_state.return_value = {
        "run_id": manifest.run_id,
        "manifest": document,
        "terminal": None,
        "promotion": None,
        "public_index": None,
        "eval_intents": 0,
        "eval_results": 0,
        "verified_eval_results": 0,
        "attempts": [document],
        "attempt_terminals": [],
    }
    authority.control.iter_keys.return_value = iter(
        [
            f"{prefix}/manifest.json",
            f"{prefix}/attempts/{manifest.attempt_id}/manifest.json",
        ]
    )
    authority.control.uri.side_effect = lambda key: f"s3://control-private/{key}"
    authority.evaluation.iter_keys.return_value = iter(())
    authority.models.iter_keys.return_value = iter(())
    authority.models.public_url.return_value = (
        f"https://models.example/runs/{manifest.run_id}/index.json"
    )
    storage = SimpleNamespace()
    backend = mock.MagicMock()
    backend.status.side_effect = KeyError("not found")
    backend.submit.return_value = DstackTask(
        project="research",
        name=manifest.run_id,
        status="submitted",
    )

    with (
        mock.patch("gradlab.experiment_cli.repository_root", return_value=tmp_path),
        mock.patch(
            "gradlab.experiment_cli._storage",
            return_value=(storage, authority),
        ),
        mock.patch(
            "gradlab.experiment_cli._operator_preflight",
            return_value=(storage, authority, backend, {"status": "ready"}),
        ),
    ):
        assert cmd_resume_submit(SimpleNamespace(run_id=manifest.run_id, json=True)) == 0

    submitted_request = backend.submit.call_args.args[0]
    assert submitted_request.run_id == manifest.run_id
    assert submitted_request.task_name == manifest.run_id
    assert submitted_request.manifest_uri == (
        f"s3://control-private/runs/{manifest.run_id}/manifest.json"
    )
    output = json.loads(capsys.readouterr().out)
    assert output["run_id"] == manifest.run_id
    assert output["attempt_id"] == manifest.attempt_id
    assert output["resumed_submission"] is True


def test_resume_submit_rejects_any_post_manifest_run_state(
    tmp_path: Path,
) -> None:
    manifest = _manifest_only_run()
    document = manifest.to_dict()
    prefix = f"runs/{manifest.run_id}"
    authority = mock.MagicMock()
    authority.run_prefix.return_value = prefix
    authority.semantic_state.return_value = {
        "run_id": manifest.run_id,
        "manifest": document,
        "terminal": None,
        "promotion": None,
        "public_index": None,
        "eval_intents": 0,
        "eval_results": 0,
        "verified_eval_results": 0,
        "attempts": [document],
        "attempt_terminals": [],
    }
    authority.control.iter_keys.return_value = iter(
        [
            f"{prefix}/manifest.json",
            f"{prefix}/writer-lease.json",
            f"{prefix}/attempts/{manifest.attempt_id}/manifest.json",
        ]
    )

    with (
        mock.patch("gradlab.experiment_cli.repository_root", return_value=tmp_path),
        mock.patch(
            "gradlab.experiment_cli._storage",
            return_value=(SimpleNamespace(), authority),
        ),
        mock.patch("gradlab.experiment_cli._operator_preflight") as preflight,
        pytest.raises(RuntimeError, match="control state beyond"),
    ):
        cmd_resume_submit(SimpleNamespace(run_id=manifest.run_id, json=True))

    preflight.assert_not_called()


def test_launch_parser_supports_explicit_training_only_runs() -> None:
    args = build_parser().parse_args(
        [
            "launch",
            "--goal-file",
            "experiments/goals/goal/_goal.yaml",
            "--recipe-file",
            "experiments/goals/goal/recipes/ppo.yaml",
            "--seed",
            "123",
            "--run-description",
            "training-only search rung",
            "--checkpoint-eval-backend",
            "none",
            "--submission-key",
            "autoresearch-study-rung-1",
        ]
    )

    assert args.checkpoint_eval_backend == "none"
    assert args.submission_key == "autoresearch-study-rung-1"


def test_training_only_task_does_not_receive_modal_credentials() -> None:
    manifest = SimpleNamespace(
        run_id=new_run_id(),
        image_digest="docker:example/gradlab@sha256:" + "a" * 64,
        compute={
            "selected": {
                "kind": "local",
                "target": "local-gpu",
                "max_price": None,
                "max_cost_usd": None,
                "allow_on_demand": False,
                "max_duration_seconds": 3600,
            },
            "dstack_task": "training-only",
        },
        modal={"enabled": False, "environment_name": "gradlab-eval"},
    )

    task = _task_request(manifest, manifest_uri="s3://control/run/manifest.json")

    assert "MODAL_TOKEN_ID" not in task.secret_env
    assert "MODAL_TOKEN_SECRET" not in task.secret_env
    assert not any(value.startswith("MODAL_ENVIRONMENT=") for value in task.secret_env)
    assert task.rom_mount is None


def test_fault_fixture_switch_is_bound_only_through_manifest_compute() -> None:
    manifest = SimpleNamespace(
        run_id=new_run_id(),
        image_digest="docker:example/gradlab@sha256:" + "a" * 64,
        compute={
            "selected": {
                "kind": "local",
                "target": "local-gpu",
                "max_price": None,
                "max_cost_usd": None,
                "allow_on_demand": False,
                "max_duration_seconds": 120,
            },
            "dstack_task": "fault-fixture",
            "supervision_fault_fixture": "failed-result-live-process",
        },
        modal={"enabled": False, "environment_name": "gradlab-eval"},
    )

    task = _task_request(manifest, manifest_uri="s3://control/run/manifest.json")

    assert task.plain_env == {"GRADLAB_SUPERVISION_FAULT_FIXTURE": "failed-result-live-process"}
    assert "GRADLAB_SUPERVISION_FAULT_FIXTURE" not in task.secret_env


def test_rom_free_provider_does_not_require_or_stage_an_asset() -> None:
    assert (
        _stage_rom(
            SimpleNamespace(),
            env_provider="gradlab",
            game="Bandit-v0",
            rom_path=None,
        )
        is None
    )


def test_vizdoom_iwad_is_staged_by_digest_in_private_evaluation_storage(
    tmp_path: Path,
) -> None:
    iwad = tmp_path / "doom2.wad"
    iwad.write_bytes(b"IWADdoom")

    writes: list[tuple[str, object]] = []

    class EvaluationBucket:
        @staticmethod
        def uri(key: str) -> str:
            return f"s3://private-evaluation/{key}"

        @staticmethod
        def put_file(key: str, path: Path, **kwargs) -> None:
            writes.append((key, (path.read_bytes(), kwargs)))

        @staticmethod
        def put_json(key: str, value: object, **kwargs) -> None:
            writes.append((key, (value, kwargs)))

    staged = _stage_vizdoom_iwad(
        SimpleNamespace(evaluation=EvaluationBucket()),
        vizdoom_iwad_binding(iwad),
    )

    assert staged["object_uri"].startswith(
        "s3://private-evaluation/vizdoom-iwad/v1/objects/sha256/"
    )
    assert staged["sha256"] in staged["object_uri"]
    assert writes[0][1][0] == iwad.read_bytes()
    assert writes[1][0] == f"vizdoom-iwad/v1/manifests/sha256/{staged['sha256']}.json"


def test_rom_free_launch_contract_omits_null_asset() -> None:
    goal = Path("experiments/goals/gradlab__bandit/_goal.yaml")
    resolved = compose_resolved_train_documents(
        goal,
        goal.parent / "recipes/ppo.yaml",
        source_sha="a" * 40,
    )
    contract = _bind_launch_contract(
        resolved.effective,
        asset=None,
        checkpoint_eval_backend="none",
    )
    base_contract = _bind_launch_contract(
        resolved.base,
        asset=None,
        checkpoint_eval_backend="none",
    )

    assert "rom_asset_manifest" not in contract["train_config"]
    build_recipe_document(
        contract,
        repo_root=Path.cwd(),
        source_commit="a" * 40,
        run_description="ROM-free launch contract regression",
        seed=123,
        runtime_image_ref="docker:example/gradlab@sha256:" + "b" * 64,
        base_materialized_recipe=base_contract,
        canonical_goal=resolved.canonical_goal,
    )


@pytest.mark.parametrize("abort", [False, True])
def test_cancel_persists_request_before_optional_dstack_abort(
    abort: bool,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_id = "gradlab-0123456789abcdef0123456789abcdef"
    attempt_id = "attempt-0123456789abcdef"
    attempt = {
        "attempt_id": attempt_id,
        "compute": {"dstack_task": run_id},
    }
    authority = mock.Mock()
    authority.semantic_state.return_value = {
        "run_id": run_id,
        "attempts": [attempt],
        "attempt_terminals": [],
    }
    authority.request_cancel.return_value = {
        "run_id": run_id,
        "attempt_id": attempt_id,
        "requested_at": "2026-08-07T12:00:00Z",
    }
    backend = mock.Mock()

    with (
        mock.patch("gradlab.experiment_cli.repository_root", return_value=Path.cwd()),
        mock.patch(
            "gradlab.experiment_cli._storage",
            return_value=(mock.Mock(), authority),
        ),
        mock.patch(
            "gradlab.experiment_cli._dstack_backend_for_compute",
            return_value=backend,
        ),
    ):
        assert cmd_cancel(SimpleNamespace(run_id=run_id, abort=abort)) == 0

    authority.request_cancel.assert_called_once_with(run_id=run_id, attempt_id=attempt_id)
    if abort:
        backend.cancel.assert_called_once_with(run_id, abort=True)
    else:
        backend.cancel.assert_not_called()
    output = json.loads(capsys.readouterr().out)
    assert output["dstack_cancel_sent"] is abort
    assert output["cancel_requested_at"] == "2026-08-07T12:00:00Z"


def test_repair_runtime_accepts_rom_free_manifest() -> None:
    assert _manifest_rom_asset({"enabled": False, "rom_asset_manifest": None}) is None
    assert _manifest_rom_asset({"enabled": True, "rom_asset_manifest": {"game": "Example-v0"}}) == {
        "game": "Example-v0"
    }

    with pytest.raises(ValueError, match="object or null"):
        _manifest_rom_asset({"rom_asset_manifest": "invalid"})


def test_local_vizdoom_iwad_contract_is_hash_bound_and_mounts_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    iwad = tmp_path / "doom2.wad"
    iwad.write_bytes(b"IWADdoom")
    monkeypatch.setattr(
        "gradlab.vizdoom_assets.REQUIRED_VIZDOOM_IWAD_SIZE_BYTES",
        iwad.stat().st_size,
    )
    monkeypatch.setattr(
        "gradlab.vizdoom_assets.REQUIRED_VIZDOOM_IWAD_SHA256",
        hashlib.sha256(iwad.read_bytes()).hexdigest(),
    )
    binding = _bind_vizdoom_iwad_for_launch(
        env_provider="vizdoom-turbo",
        rom_path=iwad,
    )
    assert binding is not None
    assert (
        _bind_vizdoom_iwad_for_launch(
            env_provider="gradoom",
            rom_path=iwad,
        )
        == binding
    )
    goal = Path("experiments/goals/VizdoomDefendLine-v1/_goal.yaml")
    resolved = compose_resolved_train_documents(
        goal,
        goal.parent / "recipes/ppo.yaml",
        source_sha="a" * 40,
    )

    contract = _bind_launch_contract(
        resolved.effective,
        asset=None,
        vizdoom_iwad=binding,
        checkpoint_eval_backend="none",
    )

    config = contract["train_config"]
    assert config["env_args"]["rom_path"] == binding
    assert config["checkpoint_eval_environment"]["env_args"]["rom_path"] == binding
    assert _manifest_vizdoom_iwad({"vizdoom_iwad_binding": binding}) == binding

    manifest = _manifest_only_run()
    manifest = replace(
        manifest,
        modal={
            **manifest.modal,
            "rom_asset_manifest": None,
            "vizdoom_iwad_binding": binding,
        },
    )
    manifest.validate()
    task = _task_request(manifest, manifest_uri="s3://control/run/manifest.json")
    assert task.rom_mount == "/var/lib/gradlab/rom-cache:/rom-cache"
