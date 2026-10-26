from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gradlab import autoresearch as study
from gradlab.autoresearch_reconciliation import reconcile_study
from gradlab.json_utils import canonical_json_sha256
from gradlab.r2_store import BucketConfig, ObjectNotFound, RunStorageConfig
from gradlab.run_authority import RunAuthority
from gradlab.training_backend import load_training_backend
from tests.test_autoresearch_study import base_state


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    state = base_state(tmp_path)
    state["policy"]["dstack_coordinator_id"] = "local-coordinator"
    state["policy"]["tracking"]["delivery"] = "local_only"
    study.reserve_one(
        state,
        phase="baseline-screen",
        round_number=0,
        candidate=state["candidates"][state["baseline_candidate_id"]],
        seeds=[state["screen_seed"]],
        timesteps=state["rung_caps"]["screen"],
    )
    path = tmp_path / "study/study.json"
    study.atomic_json(path, state)
    monkeypatch.setattr(study, "git_head", lambda root: state["source_sha"])
    authority = RunAuthority(
        RunStorageConfig(
            control=BucketConfig((tmp_path / "control").as_uri()),
            evaluation=BucketConfig((tmp_path / "eval").as_uri()),
            models=BucketConfig((tmp_path / "models").as_uri()),
        )
    )

    def recipe_document(digest):
        document = authority.control.get_json(f"test-recipes/{digest}.json")
        if canonical_json_sha256(document) != digest:
            raise ValueError("recipe checksum mismatch")
        return document

    monkeypatch.setattr(authority, "recipe_document", recipe_document)
    return SimpleNamespace(path=path, state=state, authority=authority)


def publish_run(campaign, *, number=1, seed=None, complete=True):
    state, authority = campaign.state, campaign.authority
    wave = state["waves"][0]
    seed = state["screen_seed"] if seed is None else seed
    run_id = f"gradlab-{number:032x}"
    attempt_id = f"attempt-{number:016x}"
    image = "docker:ghcr.io/example/gradlab@sha256:" + "d" * 64
    overrides = study.expected_recipe_overrides(state, wave)
    document = {
        "provenance": {
            "source_files": state["source_files"],
            "source_commit": state["source_sha"],
            "runtime": {"image_ref": image},
        },
        "resolution": {"recipe": {"overrides": overrides}},
        "recipe": {
            "train_config": {
                "seed": seed,
                "timesteps": wave["timesteps"],
                "training_backend": {
                    "id": state["backend"],
                    "config": load_training_backend(state["backend"]).normalize_config(
                        state["baseline"]["backend_config"], label="test backend"
                    ),
                },
            }
        },
    }
    digest = canonical_json_sha256(document)
    authority.control.put_json(f"test-recipes/{digest}.json", document)
    manifest = {
        "run_id": run_id,
        "attempt_id": attempt_id,
        "seed": seed,
        "source_sha": state["source_sha"],
        "image_digest": image,
        "recipe_sha256": digest,
        "recipe_overrides": overrides,
        "run_description": study.run_description(state, wave, seed),
        "tracking": state["policy"]["tracking"],
        "compute": {
            "submission_key": wave["submission_key"],
            "checkpoint_eval_backend": "none",
            "selected": {"kind": "local", "target": state["policy"]["compute_target"]},
            "dstack_coordinator_id": state["policy"]["dstack_coordinator_id"],
            "runtime_input_sha256": "e" * 64,
            "runtime_build_source_sha": "f" * 40,
        },
    }
    authority.control.put_json(f"runs/{run_id}/manifest.json", manifest)
    if complete:
        terminal = {
            "run_id": run_id,
            "attempt_id": attempt_id,
            "state": "complete_local",
            "tracking": state["policy"]["tracking"],
            "acceptance_required": False,
            "service_high_water_mark": 0,
            "wandb_high_water_mark": 0,
            "drain": {
                "complete": True,
                "metric_segment_high_water": 1,
                "service_remote_high_water_mark": 0,
                "journal_archive": {"retention": "durable", "segment_count": 1},
            },
        }
        authority.control.put_json(f"runs/{run_id}/attempts/{attempt_id}/terminal.json", terminal)
        authority.seal_metric_segment(
            run_id=run_id,
            attempt_id=attempt_id,
            events=[
                {
                    "event_seq": 1,
                    "event_id": f"{run_id}-1",
                    "source": "train",
                    "kind": "history",
                    "step": 20480,
                    "payload": {"train/success/Level1-3/count": 1, "train/success/min": 0.95},
                }
            ],
        )
    return manifest


def status_reader(campaign):
    def read(root, run_id):
        manifest = campaign.authority.manifest(run_id)
        terminal = campaign.authority.control.get_json_optional(
            f"runs/{run_id}/attempts/{manifest['attempt_id']}/terminal.json"
        )
        return {
            "run_id": run_id,
            "completed": terminal is not None,
            "scientific_success": False,
            "semantic": {"manifest": manifest, "terminal": None},
            "attempt_terminal": terminal,
            "dstack": {
                "status": "done" if terminal else "running",
                "terminal": terminal is not None,
            },
        }

    return read


def reconcile(campaign, **kwargs):
    return reconcile_study(
        campaign.path, authority=campaign.authority, status_reader=status_reader(campaign), **kwargs
    )


def test_lost_launch_response_recovers_terminal_and_real_journal_once(campaign):
    manifest = publish_run(campaign)
    result = reconcile(campaign)
    assert result["state"] == "ready"
    assert result["next"]["action"] == "reserve_baseline_pair"
    before = study.load_state(campaign.path)
    assert before["reserved_jobs"] == 1
    assert before["waves"][0]["run_ids"] == [manifest["run_id"]]
    record = before["waves"][0]["terminal_runs"][0]
    assert record["training_evidence"]["authority"] == "gradlab_metric_journal"
    assert record["training_evidence"]["strong"]
    assert record["tracking"]["delivery"] == "local_only"

    # A fresh invocation after a restart neither polls completed work nor duplicates evidence.
    def forbidden_poll(*args):
        raise AssertionError("completed Run was polled again")

    reconcile_study(campaign.path, authority=campaign.authority, status_reader=forbidden_poll)
    after = study.load_state(campaign.path)
    before.pop("updated_at")
    after.pop("updated_at")
    assert before == after


def test_empty_inventory_returns_reserved_commands_without_submitting(campaign):
    before = study.load_state(campaign.path)
    result = reconcile(campaign)
    assert result["state"] == "ready"
    assert result["observations"][0]["action"] == "launch_reserved_wave"
    assert result["observations"][0]["commands"][0][:3] == ["gradlab", "experiment", "launch"]
    assert study.load_state(campaign.path)["waves"] == before["waves"]
    assert list(campaign.authority.control.iter_keys("runs/")) == []


def test_live_run_waits_without_terminal_record(campaign):
    publish_run(campaign, complete=False)
    result = reconcile(campaign)
    assert result["state"] == "waiting"
    assert result["observations"][0]["action"] == "monitor"
    assert study.load_state(campaign.path)["waves"][0]["terminal_runs"] == []


def test_partial_wave_and_duplicate_seed_inventory_pause(campaign):
    campaign.state["waves"][0]["seeds"] = [123, 139]
    study.atomic_json(campaign.path, campaign.state)
    publish_run(campaign)
    assert reconcile(campaign)["state"] == "blocked"
    assert (
        study.load_state(campaign.path)["pause_reason"]["event"]
        == "submission_reconciliation_mismatch"
    )
    study.atomic_json(campaign.path, campaign.state)
    publish_run(campaign, number=2)
    assert reconcile(campaign)["state"] == "blocked"


@pytest.mark.parametrize(
    "field", ["source", "runtime", "seed", "overrides", "fleet", "tracking", "recipe"]
)
def test_contract_mismatch_pauses_without_recording_terminal(campaign, field):
    manifest = publish_run(campaign)
    changed = copy.deepcopy(manifest)
    if field == "source":
        changed["source_sha"] = "0" * 40
    elif field == "runtime":
        campaign.state["runtime"] = {
            "image_ref": changed["image_digest"],
            "input_sha256": "0" * 64,
            "build_source_sha": "f" * 40,
        }
        study.atomic_json(campaign.path, campaign.state)
    elif field == "seed":
        changed["seed"] = 999
    elif field == "overrides":
        changed["recipe_overrides"] = []
    elif field == "fleet":
        changed["compute"]["selected"]["target"] = "other"
    elif field == "tracking":
        changed["tracking"]["delivery"] = "online"
    else:
        key = f"test-recipes/{manifest['recipe_sha256']}.json"
        doc = campaign.authority.control.get_json(key)
        doc["recipe"]["train_config"]["timesteps"] += 1
        (Path(campaign.state["repo_root"]) / "control" / key).write_text(json.dumps(doc))
    manifest_path = (
        Path(campaign.state["repo_root"]) / "control" / f"runs/{manifest['run_id']}/manifest.json"
    )
    manifest_path.write_text(json.dumps(changed))
    assert reconcile(campaign)["state"] == "blocked"
    assert study.load_state(campaign.path)["waves"][0]["terminal_runs"] == []


def test_source_drift_pauses_before_any_remote_read(campaign):
    (Path(campaign.state["repo_root"]) / "recipe.yaml").write_text("changed: true\n")

    def forbidden_poll(*args):
        raise AssertionError("source drift must stop observation")

    result = reconcile_study(campaign.path, status_reader=forbidden_poll)
    assert result["state"] == "blocked"
    assert result["next"]["reason"]["event"] == "source_drift"


def test_transient_read_error_is_retryable_and_does_not_expose_urls(campaign, monkeypatch):
    original = campaign.authority.control.iter_keys

    def unavailable(prefix):
        raise OSError("https://private.example.test/?secret=hidden")

    monkeypatch.setattr(campaign.authority.control, "iter_keys", unavailable)
    result = reconcile(campaign)
    assert result["state"] == "waiting"
    assert "hidden" not in str(result)
    assert study.load_state(campaign.path)["status"] == "active"
    monkeypatch.setattr(campaign.authority.control, "iter_keys", original)
    assert reconcile(campaign)["state"] == "ready"


def test_unavailable_journal_retries_then_integrity_failure_pauses(campaign, monkeypatch):
    manifest = publish_run(campaign)
    keys = [key for key in campaign.authority.control.iter_keys("runs/") if key.endswith(".jsonl")]
    assert keys
    key = keys[0]
    content = campaign.authority.control.get_bytes(key)
    original = campaign.authority.control.get_bytes

    def unavailable(journal_key):
        if journal_key == key:
            raise ObjectNotFound("temporarily unavailable")
        return original(journal_key)

    monkeypatch.setattr(campaign.authority.control, "get_bytes", unavailable)
    assert reconcile(campaign)["state"] == "waiting"
    assert study.load_state(campaign.path)["status"] == "active"
    monkeypatch.setattr(campaign.authority.control, "get_bytes", original)
    assert reconcile(campaign)["state"] == "ready"
    # Recheck with a fresh study to exercise the corrupted-journal path.
    study.atomic_json(campaign.path, campaign.state)
    (Path(campaign.state["repo_root"]) / "control" / key).write_bytes(
        content.replace(b"0.95", b"0.55")
    )
    assert reconcile(campaign)["state"] == "blocked"
    assert manifest["run_id"] == study.load_state(campaign.path)["waves"][0]["run_ids"][0]


def test_bounded_discovery_cannot_claim_empty_inventory(campaign):
    campaign.authority.control.put_json("runs/000/unrelated.json", {})
    campaign.authority.control.put_json("runs/001/unrelated.json", {})
    result = reconcile(campaign, max_scan_keys=1)
    assert result["state"] == "waiting"
    assert result["observations"] == [
        {"state": "waiting", "action": "retry_reconcile", "error_type": "ScanLimitReached"}
    ]
    assert study.load_state(campaign.path)["status"] == "active"


def test_recovery_reads_a_real_generated_and_validated_recipe(campaign, monkeypatch):
    from gradlab.policy_bundle import build_recipe_document
    from gradlab.recipe_documents import compose_resolved_train_documents
    from tests.test_policy_bundle import GOAL, RECIPE, bind_mario_asset

    root = Path.cwd()
    baseline = compose_resolved_train_documents(
        GOAL, RECIPE, source_sha=campaign.state["source_sha"]
    )
    state = campaign.state
    state["repo_root"] = str(root)
    state["goal_path"] = str(GOAL)
    state["recipe_path"] = str(RECIPE)
    state["recipe_preimage_sha256"] = study.file_sha256(root / RECIPE)
    state["source_files"] = [
        {"path": str(Path(item["path"]).relative_to(root)), "sha256": item["sha256"]}
        for item in baseline.effective["_composition"]["source_files"]
    ]
    state["baseline"]["backend_config"] = baseline.effective["train_config"]["training_backend"][
        "config"
    ]
    wave = state["waves"][0]
    overrides = study.expected_recipe_overrides(state, wave)
    resolved = compose_resolved_train_documents(
        GOAL, RECIPE, source_sha=state["source_sha"], recipe_overrides=overrides
    )
    bind_mario_asset(resolved)
    for materialized in (resolved.effective, resolved.base):
        materialized["train_config"]["checkpoint_eval_backend"] = "none"
    manifest = publish_run(campaign)
    document = build_recipe_document(
        resolved.effective,
        repo_root=root,
        source_commit=state["source_sha"],
        run_description=manifest["run_description"],
        seed=manifest["seed"],
        runtime_image_ref=manifest["image_digest"],
        base_materialized_recipe=resolved.base,
        canonical_goal=resolved.canonical_goal,
    )
    manifest["recipe_sha256"] = campaign.authority.put_recipe_document(document)
    manifest_key = f"runs/{manifest['run_id']}/manifest.json"
    campaign.authority.control.put_json(manifest_key, manifest, create_only=False)
    study.atomic_json(campaign.path, state)
    monkeypatch.setattr(
        campaign.authority,
        "recipe_document",
        RunAuthority.recipe_document.__get__(campaign.authority),
    )
    result = reconcile(campaign)
    assert result["state"] == "ready", result
