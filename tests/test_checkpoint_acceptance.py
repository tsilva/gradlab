from __future__ import annotations

import copy

import pytest

from gradlab.checkpoint_acceptance import (
    CheckpointEvalContractCompiler,
    acceptance_aggregates,
    build_checkpoint_eval_contract,
    checkpoint_eval_watchdog_steps,
    evaluate_acceptance,
    manifest_index,
    requires_complete_evaluation,
    validate_episode_rows,
)
from gradlab.modal_eval_protocol import (
    PROTOCOL_SCHEMA_VERSION,
    SEED_PROTOCOL,
    execution_key,
    normalize_attempt_result,
    validate_attempt_result,
)


def contract(
    *,
    episodes: int = 100,
    n_envs: int = 16,
    acceptance: list[dict] | None = None,
) -> dict:
    return build_checkpoint_eval_contract(
        environment={"game": "SuperMarioBros-Nes-v0", "state": "Level1-1"},
        episodes=episodes,
        n_envs=n_envs,
        watchdog_steps=4500,
        seed=10_000,
        seed_protocol="vector-lane-v1",
        acceptance=acceptance
        or [
            {
                "metric": "eval/full/outcome/success/starts/rate/min",
                "operator": ">=",
                "threshold": 1.0,
            }
        ],
    )


def row(entry: dict, *, success: bool = True, episode_return: float = 1.0) -> dict:
    return {
        "episode_id": entry["episode_id"],
        "seed_lane": entry["lane"],
        "seed_episode_ordinal": entry["lane_episode_ordinal"],
        "seed": entry["seed"],
        "start_state": entry["start_state"],
        "outcome": "success" if success else "failure",
        "return": episode_return,
        "seed_protocol": "vector-lane-v1",
    }


def modal_contract(**kwargs) -> dict:
    value = contract(**kwargs)
    value.update(
        {
            "schema_version": PROTOCOL_SCHEMA_VERSION,
            "checkpoint_sha256": "a" * 64,
            "runtime_image_ref": "docker:example.invalid/gradlab@sha256:" + "b" * 64,
            "recipe_sha256": "c" * 64,
            "recipe_format_version": 1,
            "evaluation_contract_sha256": "d" * 64,
        }
    )
    return value


def result_identity(value: dict, *, attempt_id: str) -> dict:
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "contract_schema_version": PROTOCOL_SCHEMA_VERSION,
        "attempt_id": attempt_id,
        "execution_key": execution_key(value),
        "checkpoint_sha256": value["checkpoint_sha256"],
        "recipe_sha256": value["recipe_sha256"],
        "recipe_format_version": value["recipe_format_version"],
        "evaluation_contract_sha256": value["evaluation_contract_sha256"],
        "runtime_image_ref": value["runtime_image_ref"],
        "rom_sha256": "",
        "seed_protocol": value["seed_protocol"],
        "n_envs": value["n_envs"],
        "episodes": value["episodes"],
    }


def test_manifest_has_exact_count_unique_identities_and_fixed_quotas() -> None:
    value = contract()
    manifest = value["manifest"]

    assert manifest["lane_quotas"] == [7, 7, 7, 7] + [6] * 12
    assert len(manifest_index(value)) == 100
    assert len({entry["episode_id"] for entry in manifest["episodes"]}) == 100
    assert {entry["start_state"] for entry in manifest["episodes"]} == {"Level1-1"}


def test_rejection_is_valid_partial_evidence_only_through_first_failure() -> None:
    value = contract(episodes=4, n_envs=2)
    entries = value["manifest"]["episodes"]
    rows = [row(entries[0]), row(entries[1], success=False)]

    assert validate_episode_rows(rows, contract=value) == rows
    aggregates = acceptance_aggregates(rows, contract=value)
    assert aggregates["episodes_planned"] == 4
    assert aggregates["episodes_completed"] == 2
    assert aggregates["failure_count"] == 1
    assert not any(name.startswith("eval/full/") for name in aggregates)

    with pytest.raises(ValueError, match="after its first failure"):
        validate_episode_rows([*rows, row(entries[2])], contract=value)


@pytest.mark.parametrize(
    ("returns", "verdict"),
    [
        ([1.0, 2.0, 3.0, 4.0], "accepted"),
        ([0.0, 1.0, 2.0, 3.0], "rejected"),
    ],
)
def test_mean_return_acceptance_requires_and_uses_every_episode(
    returns: list[float],
    verdict: str,
) -> None:
    value = contract(
        episodes=4,
        n_envs=2,
        acceptance=[
            {
                "metric": "eval/full/episode/return/shaped/mean",
                "operator": ">=",
                "threshold": 2.5,
            }
        ],
    )
    assert value["evidence_policy"]["fail_fast"] == "disabled"
    rows = [
        row(entry, success=index % 2 == 0, episode_return=returns[index])
        for index, entry in enumerate(value["manifest"]["episodes"])
    ]

    with pytest.raises(ValueError, match="every planned episode"):
        validate_episode_rows(rows[:-1], contract=value)
    validate_episode_rows(rows, contract=value)
    aggregates = acceptance_aggregates(rows, contract=value)

    assert aggregates["episodes_completed"] == 4
    assert aggregates["failure_count"] == 2
    assert aggregates["eval/full/episode/return/shaped/mean"] == sum(returns) / 4
    accepted, _observed = evaluate_acceptance(aggregates, contract=value)
    assert accepted is (verdict == "accepted")


def test_vizdoom_basic_perfect_success_acceptance_requires_every_episode() -> None:
    value = build_checkpoint_eval_contract(
        environment={"game": "VizdoomBasic-v1", "state": "default"},
        episodes=4,
        n_envs=2,
        watchdog_steps=72,
        seed=10_000,
        seed_protocol="vector-lane-v1",
        acceptance=[
            {
                "metric": "eval/full/outcome/success/starts/rate/min",
                "operator": ">=",
                "threshold": 1.0,
            }
        ],
    )
    assert value["evidence_policy"]["fail_fast"] == "disabled"
    rows = [
        row(entry, success=index != 1) for index, entry in enumerate(value["manifest"]["episodes"])
    ]

    with pytest.raises(ValueError, match="every planned episode"):
        validate_episode_rows(rows[:-1], contract=value)
    validate_episode_rows(rows, contract=value)
    aggregates = acceptance_aggregates(rows, contract=value)

    assert aggregates["episodes_completed"] == 4
    assert aggregates["failure_count"] == 1
    assert aggregates["eval/full/outcome/success/starts/rate/min"] == 0.75
    accepted, _observed = evaluate_acceptance(aggregates, contract=value)
    assert accepted is False


def test_vizdoom_deathmatch_requires_complete_evaluation() -> None:
    assert requires_complete_evaluation({"game": "VizdoomDeathmatch-v1", "state": "default"})


def test_vizdoom_deathmatch_acceptance_aggregates_raw_kills() -> None:
    value = build_checkpoint_eval_contract(
        environment={"game": "VizdoomDeathmatch-v1", "state": "default"},
        episodes=4,
        n_envs=2,
        watchdog_steps=4200,
        seed=10_000,
        seed_protocol="vector-lane-v1",
        acceptance=[
            {
                "metric": "eval/full/progress/kills/mean",
                "operator": ">=",
                "threshold": 10.0,
            }
        ],
    )
    rows = [
        {**row(entry, success=False), "kills": kills}
        for entry, kills in zip(value["manifest"]["episodes"], (8, 10, 12, 20), strict=True)
    ]

    aggregates = acceptance_aggregates(rows, contract=value)

    assert aggregates["eval/full/progress/kills/mean"] == 12.5
    assert aggregates["eval/full/progress/kills/max"] == 20.0
    accepted, observed = evaluate_acceptance(aggregates, contract=value)
    assert accepted is True
    assert observed["eval/full/progress/kills/mean"] == 12.5


def test_modal_protocol_accepts_complete_mean_return_rejection() -> None:
    value = modal_contract(
        episodes=2,
        n_envs=1,
        acceptance=[
            {
                "metric": "eval/full/episode/return/shaped/mean",
                "operator": ">=",
                "threshold": 1.0,
            }
        ],
    )
    rows = [
        row(entry, success=False, episode_return=0.25) for entry in value["manifest"]["episodes"]
    ]
    attempt_id = "attempt-1"
    result = {
        **result_identity(value, attempt_id=attempt_id),
        "status": "succeeded",
        "verdict": "rejected",
        "episode_results": rows,
        "duration_seconds": 1.25,
        "evaluation_evidence": {"manifest": "verified"},
    }

    validated = validate_attempt_result(result, contract=value, attempt_id=attempt_id)
    normalized = normalize_attempt_result(result, contract=value, attempt_id=attempt_id)

    assert validated["verdict"] == "rejected"
    assert validated["aggregates"]["episodes_completed"] == 2
    assert normalized["status"] == "rejected"
    assert normalized["aggregates"]["episodes_completed"] == 2
    assert normalized["duration_seconds"] == 1.25
    assert len(normalized["evidence_sha256"]) == 2
    assert normalized["error"] is None


def test_modal_protocol_normalizes_failed_attempt_identity() -> None:
    value = modal_contract(episodes=2, n_envs=1)
    attempt_id = "attempt-2"
    result = {
        **result_identity(value, attempt_id=attempt_id),
        "status": "expired",
        "duration_seconds": 3.5,
        "error": "deadline reached",
    }

    normalized = normalize_attempt_result(result, contract=value, attempt_id=attempt_id)

    assert normalized == {
        "status": "expired",
        "episode_results": [],
        "aggregates": {},
        "duration_seconds": 3.5,
        "evidence_sha256": [],
        "error": "deadline reached",
    }
    with pytest.raises(ValueError, match="attempt id mismatch"):
        normalize_attempt_result(result, contract=value, attempt_id="different-attempt")


def test_complete_evidence_requires_every_identity_once() -> None:
    value = contract(episodes=4, n_envs=2)
    rows = [row(entry) for entry in value["manifest"]["episodes"]]

    validate_episode_rows(rows, contract=value)
    with pytest.raises(ValueError, match="unknown or duplicate"):
        validate_episode_rows([*rows[:-1], rows[0]], contract=value)
    with pytest.raises(ValueError, match="incomplete fail-fast evidence"):
        validate_episode_rows(rows[:-1], contract=value)


def test_modal_protocol_recomputes_and_rejects_a_false_worker_verdict() -> None:
    value = modal_contract(episodes=2, n_envs=1)
    rows = [row(value["manifest"]["episodes"][0], success=False)]
    attempt_id = "attempt-verdict"
    result = {
        **result_identity(value, attempt_id=attempt_id),
        "status": "succeeded",
        "verdict": "accepted",
        "episode_results": rows,
        "duration_seconds": 1.0,
    }

    with pytest.raises(ValueError, match="supervisor recomputation"):
        validate_attempt_result(result, contract=value, attempt_id=attempt_id)


@pytest.mark.parametrize("retired_field", ["metrics", "claimed_aggregates"])
def test_modal_protocol_v6_rejects_retired_worker_claims(retired_field: str) -> None:
    value = modal_contract(episodes=2, n_envs=1)
    rows = [row(entry) for entry in value["manifest"]["episodes"]]
    attempt_id = "attempt-retired"
    result = {
        **result_identity(value, attempt_id=attempt_id),
        "status": "succeeded",
        "verdict": "accepted",
        "episode_results": rows,
        "duration_seconds": 1.0,
        retired_field: {},
    }

    with pytest.raises(ValueError, match="forbids result field"):
        validate_attempt_result(result, contract=value, attempt_id=attempt_id)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["acceptance"][0].update(threshold=0.99),
        lambda value: value["manifest"]["episodes"][0].update(seed=123),
        lambda value: value["evidence_policy"].update(fail_fast="disabled"),
    ],
)
def test_execution_key_changes_for_acceptance_or_evidence_changes(mutation) -> None:
    baseline = contract(episodes=2, n_envs=1)
    changed = copy.deepcopy(baseline)
    mutation(changed)

    assert execution_key(changed) != execution_key(baseline)


def test_checkpoint_eval_watchdog_steps_requires_materialized_value() -> None:
    assert (
        checkpoint_eval_watchdog_steps(
            {
                "checkpoint_eval_watchdog_steps": 40,
            }
        )
        == 40
    )

    with pytest.raises(ValueError, match="not materialized"):
        checkpoint_eval_watchdog_steps({"checkpoint_eval_environment": {}})
    with pytest.raises(ValueError, match="not materialized"):
        checkpoint_eval_watchdog_steps({"checkpoint_eval_watchdog_steps": 0})


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("post_train_eval_episodes", "post_train_eval_episodes must be an integer"),
        ("checkpoint_eval_n_envs", "checkpoint_eval_n_envs must be an integer"),
        ("checkpoint_eval_seed", "checkpoint_eval_seed must be an integer"),
        ("checkpoint_eval_seed_protocol", "unsupported checkpoint eval seed protocol"),
    ],
)
def test_checkpoint_eval_compiler_rejects_missing_materialized_fields(
    field: str, expected: str
) -> None:
    config = {
        "checkpoint_eval_environment": {
            "env_provider": "gradlab",
            "game": "Bandit-v0",
            "task": {"termination": {"max_episode_steps": 50}},
        },
        "post_train_eval_episodes": 3,
        "checkpoint_eval_n_envs": 2,
        "checkpoint_eval_seed": 10_000,
        "checkpoint_eval_seed_protocol": SEED_PROTOCOL,
        "checkpoint_eval_watchdog_steps": 50,
    }
    config.pop(field)

    with pytest.raises(ValueError, match=expected):
        CheckpointEvalContractCompiler.from_train_config(config)


def test_checkpoint_eval_compiler_rejects_more_lanes_than_episodes() -> None:
    with pytest.raises(ValueError, match="n_envs must not exceed episodes"):
        CheckpointEvalContractCompiler.from_train_config(
            {
                "checkpoint_eval_environment": {
                    "env_provider": "gradlab",
                    "game": "Bandit-v0",
                    "task": {"termination": {"max_episode_steps": 50}},
                },
                "post_train_eval_episodes": 1,
                "checkpoint_eval_n_envs": 2,
                "checkpoint_eval_seed": 10_000,
                "checkpoint_eval_seed_protocol": SEED_PROTOCOL,
                "checkpoint_eval_watchdog_steps": 50,
            }
        )
