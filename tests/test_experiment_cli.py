from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from rlab.dstack_backend import DstackTask
from rlab.experiment_cli import (
    _bind_launch_contract,
    _compute,
    _follow_fingerprint,
    _latest_attempt_terminal,
    _public_dstack_state,
    _record_pre_submit_failure,
    _record_terminal_task_without_receipt,
    _require_retryable_attempt_terminal,
    _required_operator_environment,
    _run_completed,
    _stage_rom,
    _task_name,
    _task_request,
    build_parser,
    cmd_launch,
    cmd_resume_submit,
    main,
)
from rlab.operator_credentials import OperatorConfigurationError
from rlab.policy_bundle import build_recipe_document
from rlab.recipe_documents import compose_train_document
from rlab.run_contracts import RunManifest, new_attempt_id, new_run_id


def _manifest_only_run() -> RunManifest:
    run_id = new_run_id()
    attempt_id = new_attempt_id()
    source_sha = "a" * 40
    compute = {
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
        "selected_offer": None,
        "dstack_task": run_id,
        "runtime_workflow_run_id": "12345",
        "runtime_input_sha256": "b" * 64,
        "runtime_build_source_sha": source_sha,
    }
    manifest = RunManifest(
        run_id=run_id,
        attempt_id=attempt_id,
        created_at=(datetime.now(UTC) - timedelta(minutes=1))
        .isoformat()
        .replace("+00:00", "Z"),
        source_sha=source_sha,
        image_digest="docker:example/rlab@sha256:" + "c" * 64,
        goal_slug="example/goal",
        goal_sha256="d" * 64,
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
            "environment_name": "rlab-eval",
            "app_name": f"rlab-eval-v2-{source_sha[:12]}",
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


def test_auto_without_cloud_budget_stays_local() -> None:
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
    assert compute.max_price is None
    assert compute.bounded_duration_seconds == 3600


def test_operator_preflight_parser_defaults_to_modal() -> None:
    args = build_parser().parse_args(["operator-preflight", "--json"])

    assert args.checkpoint_eval_backend == "modal"
    assert args.json is True


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
        target="b3",
        max_price=None,
        max_cost_usd=None,
        allow_on_demand=False,
        max_duration=3600,
    )

    with (
        mock.patch("rlab.experiment_cli.repository_root", return_value=tmp_path),
        mock.patch("rlab.experiment_cli.clean_git_source_sha", return_value="a" * 40),
        mock.patch("rlab.experiment_cli.current_git_branch", return_value="main"),
        mock.patch(
            "rlab.experiment_cli._tracked_committed_path",
            side_effect=[goal, recipe],
        ),
        mock.patch(
            "rlab.experiment_cli.compose_train_document",
            return_value={"train_config": {"checkpoint_eval_backend": "modal"}},
        ),
        mock.patch(
            "rlab.experiment_cli._operator_preflight",
            side_effect=OperatorConfigurationError("missing operator credentials"),
        ),
        mock.patch("rlab.experiment_cli.runtime_release_from_args") as runtime_release,
    ):
        with pytest.raises(OperatorConfigurationError, match="missing operator"):
            cmd_launch(args)

    runtime_release.assert_not_called()


def test_operator_configuration_error_is_concise_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("RLAB_OPERATOR_CONFIG", str(tmp_path / "missing.toml"))
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
            project="main",
            name="run-one",
            status="running",
            raw={
                "fleet": {"name": "b3"},
                "submitted_at": "2026-07-24T16:00:00Z",
                "run_spec": {
                    "configuration": {
                        "env": {
                            "WANDB_API_KEY": "should-never-appear",
                            "RLAB_CONTROL_R2_SECRET_ACCESS_KEY": "also-secret",
                        }
                    }
                },
            },
        )
    )

    encoded = json.dumps(value, sort_keys=True)
    assert value["fleet"] == "b3"
    assert "raw" not in value
    assert "should-never-appear" not in encoded
    assert "also-secret" not in encoded


def test_follow_fingerprint_ignores_only_poll_observation_time() -> None:
    first = {
        "run_id": "rlab-" + "a" * 32,
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
    assert retry_name.startswith("rlab-")
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
    task = DstackTask(project="main", name="rlab-retry", status="failed")

    _record_terminal_task_without_receipt(authority, manifest, task)

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
    task = DstackTask(project="main", name="rlab-retry", status="running")

    with pytest.raises(RuntimeError, match="while its dstack task is active"):
        _record_terminal_task_without_receipt(authority, manifest, task)


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
    authority.control.uri.side_effect = (
        lambda key: f"s3://control-private/{key}"
    )
    authority.evaluation.iter_keys.return_value = iter(())
    authority.models.iter_keys.return_value = iter(())
    authority.models.public_url.return_value = (
        f"https://models.example/runs/{manifest.run_id}/index.json"
    )
    storage = SimpleNamespace()
    backend = mock.MagicMock()
    backend.status.side_effect = KeyError("not found")
    backend.submit.return_value = DstackTask(
        project="main",
        name=manifest.run_id,
        status="submitted",
    )

    with (
        mock.patch("rlab.experiment_cli.repository_root", return_value=tmp_path),
        mock.patch(
            "rlab.experiment_cli._storage",
            return_value=(storage, authority),
        ),
        mock.patch(
            "rlab.experiment_cli._operator_preflight",
            return_value=(storage, authority, backend, {"status": "ready"}),
        ),
    ):
        assert (
            cmd_resume_submit(SimpleNamespace(run_id=manifest.run_id, json=True)) == 0
        )

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
        mock.patch("rlab.experiment_cli.repository_root", return_value=tmp_path),
        mock.patch(
            "rlab.experiment_cli._storage",
            return_value=(SimpleNamespace(), authority),
        ),
        mock.patch("rlab.experiment_cli._operator_preflight") as preflight,
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
        image_digest="docker:example/rlab@sha256:" + "a" * 64,
        compute={
            "selected": {
                "kind": "local",
                "target": "b3",
                "max_price": None,
                "max_cost_usd": None,
                "allow_on_demand": False,
                "max_duration_seconds": 3600,
            },
            "dstack_task": "training-only",
        },
        modal={"enabled": False, "environment_name": "rlab-eval"},
    )

    task = _task_request(manifest, manifest_uri="s3://control/run/manifest.json")

    assert "MODAL_TOKEN_ID" not in task.secret_env
    assert "MODAL_TOKEN_SECRET" not in task.secret_env
    assert not any(value.startswith("MODAL_ENVIRONMENT=") for value in task.secret_env)
    assert task.rom_mount is None


def test_rom_free_provider_does_not_require_or_stage_an_asset() -> None:
    assert (
        _stage_rom(
            SimpleNamespace(),
            env_provider="rlab",
            game="Bandit-v0",
            rom_path=None,
        )
        is None
    )


def test_rom_free_launch_contract_omits_null_asset() -> None:
    goal = Path("experiments/goals/rlab__bandit/_goal.yaml")
    document = compose_train_document(goal, goal.parent / "recipes/ppo.yaml")
    contract = _bind_launch_contract(
        document,
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
        runtime_image_ref="docker:example/rlab@sha256:" + "b" * 64,
    )
