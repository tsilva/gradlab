from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from gradlab.job_queue import JobStore, WorkerStart
from gradlab.player_publication import (
    PlayerPublicationService,
    PublicationConflict,
    generated_metadata,
)
from gradlab.policy_bundle import PolicyBundle
from gradlab.publication import PublicationIdentity


class _NormalizedEvaluation:
    @staticmethod
    def as_manifest_value() -> dict[str, object]:
        return {
            "episodes": 100,
            "success_rate_mean": 0.95,
            "success_rate_min": 0.9,
            "return_mean": 12.0,
            "action_sampling": "stochastic",
            "protocol": "full",
            "checkpoint_step": 10,
            "checkpoint_artifact": "https://example.invalid/model.zip",
            "by_start": [{"episodes": 100, "success_rate": 0.95}],
        }


def _service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PlayerPublicationService:
    model = tmp_path / "model.zip"
    model_json = tmp_path / "model.json"
    recipe = tmp_path / "recipe.json"
    for path in (model, model_json, recipe):
        path.write_bytes(b"fixture")
    bundle = PolicyBundle(
        checkpoint_path=model,
        model_path=model_json,
        recipe_path=recipe,
        model={
            "policy": {
                "algorithm_id": "ppo",
                "model_class": "gradlab.ppo.GradLabPPO",
                "training_backend_id": "gradlab.ppo",
            },
            "checkpoint": {"sha256": "a" * 64},
            "recipe": {"sha256": "b" * 64},
        },
        recipe={"format_version": 1},
        source="fixture",
    )
    capture = {
        "capture_id": "capture-" + "c" * 32,
        "capture_fence_sha256": "d" * 64,
        "run_id": "gradlab-" + "e" * 32,
        "checkpoint_id": "checkpoint-10-" + "f" * 16,
        "goal": {
            "goal_id": "Level1-1",
            "title": "Level 1-1",
            "evaluation_mode": "evaluated",
            "release": {"huggingface": {}},
        },
        "execution": {
            "qualified_environment_id": "provider:SuperMarioBros-Nes-v0"
        },
        "success": True,
        "sampling_mode": "stochastic",
    }

    class Host:
        @staticmethod
        def active_publication_context():
            return {
                "spec": SimpleNamespace(kind="public_run"),
                "bundle": bundle,
                "capture": {
                    "ready": True,
                    "recording": False,
                    "episode_in_progress": False,
                    "latest": capture,
                    "error": None,
                },
            }

    store = JobStore(tmp_path / "queue")
    service = PlayerPublicationService(
        repo_root=tmp_path,
        host=Host(),
        store=store,
        evidence_loader=lambda _run, _checkpoint: {"raw": "evidence"},
        hf_api_factory=lambda **_kwargs: object(),
        root=tmp_path / "publications",
    )
    monkeypatch.setattr(
        "gradlab.player_publication.validate_publication_evidence",
        lambda _evidence: {"evaluation": {"accepted": True}},
    )
    monkeypatch.setattr(
        "gradlab.player_publication.normalize_publication_evaluation",
        lambda *_args, **_kwargs: _NormalizedEvaluation(),
    )
    monkeypatch.setattr(
        "gradlab.player_publication.publication_identity_from_policy_bundle",
        lambda *_args: PublicationIdentity("Mario", "Level1-1", "rgb", "ppo"),
    )
    monkeypatch.setattr(
        "gradlab.player_publication.build_model_repo_id",
        lambda _identity: "tsilva/Mario_Level1-1_rgb_ppo",
    )
    monkeypatch.setattr(
        "gradlab.player_publication.resolve_huggingface_credential",
        lambda: SimpleNamespace(token="secret"),
    )
    monkeypatch.setattr(
        "gradlab.player_publication.next_release_version",
        lambda _api, _repo: ("v1", None),
    )
    monkeypatch.setattr(
        "gradlab.player_publication.ensure_flusher",
        lambda _store: WorkerStart("already_running"),
    )

    def stage(*, request, **_kwargs):
        root = service.requests_root / request["request_fingerprint"]
        root.mkdir(parents=True, exist_ok=True)
        return root

    monkeypatch.setattr(service, "_stage_snapshot", stage)
    return service


def _credentials() -> dict[str, object]:
    return {
        "ready": True,
        "huggingface": {
            "ready": True,
            "username": "tsilva",
            "namespace": "tsilva",
        },
        "youtube": {
            "ready": True,
            "channel_id": "channel-id",
            "channel_title": "Channel",
            "scopes": ["scope"],
        },
    }


def test_generated_metadata_uses_concise_task_and_actual_trainer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path, monkeypatch)
    context = service.host.active_publication_context()
    capture = context["capture"]["latest"]
    capture["goal"] = {
        "goal_id": "VizdoomDeathmatch-v1",
        "title": "ViZDoom single-player Deathmatch score attack",
    }
    capture["execution"]["qualified_environment_id"] = (
        "vizdoom-turbo:VizdoomDeathmatch-v1"
    )
    capture["success"] = False
    metadata = generated_metadata(
        capture=capture,
        bundle=context["bundle"],
        evaluation={"episodes": 100, "success_rate_mean": 0.0},
        repo_id="tsilva/VizdoomDeathmatch-v1_ppo_ae2049cb",
        settings={},
    )

    assert metadata["title"] == "ViZDoom Deathmatch — GradLab PPO"
    assert metadata["description"].startswith(
        "A PPO reinforcement-learning agent trained with GradLab plays ViZDoom Deathmatch "
        "with a 0.0% verified full-evaluation win rate."
    )
    assert "Vi ZDoom" not in metadata["description"]
    assert "stable-baselines3" not in metadata["tags"]
    assert "Stable Retro" not in metadata["tags"]
    assert "gradlab.ppo" in metadata["tags"]


def test_current_episode_must_finish_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path, monkeypatch)
    context = service.host.active_publication_context()
    context["capture"]["ready"] = False
    context["capture"]["recording"] = True
    context["capture"]["episode_in_progress"] = True
    service.host = SimpleNamespace(active_publication_context=lambda: context)

    assert service.current() == {
        "available": False,
        "message": "finish the current episode before publishing",
    }
    with pytest.raises(ValueError, match="finish the current episode"):
        service.admit({"privacy": "public", "tags": []}, credential_result=_credentials())


def test_admission_is_idempotent_for_same_capture_and_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path, monkeypatch)
    settings = {"privacy": "public", "playlist": "gradlab", "tags": []}

    first = service.admit(settings, credential_result=_credentials())
    repeated = service.admit(settings, credential_result=_credentials())

    assert first["created"] is True
    assert repeated["created"] is False
    assert repeated["job"]["job_id"] == first["job"]["job_id"]


def test_admission_rejects_changed_settings_for_same_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path, monkeypatch)
    service.admit({"privacy": "public", "tags": []}, credential_result=_credentials())

    with pytest.raises(PublicationConflict, match="immutable publication request"):
        service.admit({"privacy": "private", "tags": []}, credential_result=_credentials())


def test_exact_evidence_fallback_joins_private_results_to_active_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path, monkeypatch)
    context = service.host.active_publication_context()
    run_id = str(context["capture"]["latest"]["run_id"])
    checkpoint_id = "checkpoint-10-" + "a" * 16
    context["capture"]["latest"]["checkpoint_id"] = checkpoint_id
    context["source"] = SimpleNamespace(
        run_config={
            "checkpoint_manifest": {
                "schema_version": 2,
                "run_id": run_id,
                "checkpoint_id": checkpoint_id,
                "step": 10,
                "purpose": "periodic",
                "sha256": "a" * 64,
                "size_bytes": 7,
                "public_url": "https://example.invalid/model.zip",
                "model_document_url": "https://example.invalid/model.json",
                "model_document_sha256": "c" * 64,
                "recipe_document_url": "https://example.invalid/recipe.json",
                "recipe_document_sha256": "b" * 64,
                "goal_sha256": "d" * 64,
                "recipe_sha256": "e" * 64,
                "environment_sha256": "f" * 64,
                "evaluation_contract_sha256": "1" * 64,
                "recovery_sidecar_key": "runs/recovery-sidecar.json",
                "created_at": "2026-08-06T12:00:00Z",
            }
        }
    )
    service.host = SimpleNamespace(active_publication_context=lambda: context)
    verified = {
        "checkpoint_id": checkpoint_id,
        "idempotency_key": "eval-key",
    }

    class Evaluation:
        @staticmethod
        def iter_keys(_prefix):
            return [f"runs/{run_id}/evals/eval-key/verified-result.json"]

        @staticmethod
        def get_json(_key):
            return verified

    authority = SimpleNamespace(
        evaluation=Evaluation(),
        eval_intent=lambda **_kwargs: {"intent": True},
        eval_result=lambda **_kwargs: {"raw": True},
    )
    monkeypatch.setattr(
        "gradlab.player_publication.load_repository_operator_environment",
        lambda _root: None,
    )
    monkeypatch.setattr(
        "gradlab.player_publication.RunStorageConfig.from_env",
        lambda: object(),
    )
    monkeypatch.setattr(
        "gradlab.player_publication.RunAuthority",
        lambda _config: authority,
    )
    monkeypatch.setattr(
        "gradlab.player_publication.validate_publication_evidence",
        lambda evidence: evidence,
    )

    evidence = service._load_exact_evidence(service._active())

    assert evidence["checkpoint_manifest"]["checkpoint_id"] == checkpoint_id
    assert evidence["intent"] == {"intent": True}
    assert evidence["raw_result"] == {"raw": True}
    assert evidence["verified_result"] == verified
