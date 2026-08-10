from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from gradlab.job_queue import JobStore, WorkerStart
from gradlab.player_publication import (
    PlayerPublicationService,
    PublicationConflict,
    assert_goal_repository_compatible,
    generated_metadata,
)
from gradlab.policy_bundle import PolicyBundle
from gradlab.publication import PublicationIdentity


class _NormalizedEvaluation:
    checkpoint_step = 10

    @staticmethod
    def as_manifest_value() -> dict[str, object]:
        return {
            "episodes": 2,
            "success_rate_mean": 0.0,
            "success_rate_min": 0.0,
            "return_mean": 12.0,
            "action_sampling": "stochastic",
            "protocol": "full",
            "checkpoint_step": 10,
            "checkpoint_artifact": "https://example.invalid/model.zip",
            "by_start": [],
        }


def _bundle(tmp_path: Path) -> PolicyBundle:
    paths = [tmp_path / name for name in ("model.zip", "model.json", "recipe.json")]
    for path in paths:
        path.write_bytes(b"fixture")
    return PolicyBundle(
        checkpoint_path=paths[0],
        model_path=paths[1],
        recipe_path=paths[2],
        model={
            "policy": {
                "algorithm_id": "ppo",
                "model_class": "gradlab.ppo.GradLabPPO",
                "training_backend_id": "gradlab.ppo",
            },
            "checkpoint": {"sha256": "a" * 64, "step": 10},
            "recipe": {"sha256": "b" * 64},
        },
        recipe={"format_version": 4},
        source="fixture",
    )


def _capture() -> dict[str, object]:
    return {
        "capture_id": "capture-" + "c" * 32,
        "capture_fence_sha256": "d" * 64,
        "run_id": "gradlab-" + "e" * 32,
        "checkpoint_id": "checkpoint-10-" + "f" * 16,
        "goal": {
            "goal_id": "VizdoomDeathmatch-v1",
            "title": "ViZDoom single-player Deathmatch score attack",
            "evaluation_mode": "evaluated",
            "release": {"huggingface": {}},
        },
        "execution": {"qualified_environment_id": "vizdoom-turbo:VizdoomDeathmatch-v1"},
        "success": False,
        "sampling_mode": "stochastic",
        "episode": 1,
        "outcome": "terminated",
        "return": 96.0,
        "steps": 100,
    }


def _credentials() -> dict[str, object]:
    return {
        "ready": True,
        "huggingface": {"ready": True, "username": "tsilva", "namespace": "tsilva"},
        "youtube": {
            "ready": True,
            "channel_id": "channel-id",
            "channel_title": "Channel",
            "scopes": ["scope"],
        },
    }


def _service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PlayerPublicationService:
    bundle = _bundle(tmp_path)
    capture = _capture()

    class Host:
        @staticmethod
        def active_publication_context():
            return {
                "spec": SimpleNamespace(kind="public_run"),
                "bundle": bundle,
                "capture": {"ready": True, "latest": capture, "error": None},
            }

    service = PlayerPublicationService(
        repo_root=tmp_path,
        host=Host(),
        store=JobStore(tmp_path / "queue"),
        evidence_loader=lambda _run, _checkpoint: {"raw": "evidence"},
        hf_api_factory=lambda **_kwargs: object(),
        root=tmp_path / "publications",
    )
    monkeypatch.setattr(
        "gradlab.player_publication.validate_publication_evidence",
        lambda _value: {
            "evaluation": {"accepted": True},
            "checkpoint_manifest": {
                "model_document_url": "https://r2.invalid/model.json",
                "recipe_document_url": "https://r2.invalid/recipe.json",
            },
        },
    )
    monkeypatch.setattr(
        "gradlab.player_publication.build_evaluation_evidence_document",
        lambda _value: {
            "acceptance": {"passed": True, "outcomes": []},
            "ranking": {"outcomes": []},
        },
    )
    monkeypatch.setattr(
        "gradlab.player_publication.normalize_publication_evaluation",
        lambda *_args, **_kwargs: _NormalizedEvaluation(),
    )
    identity = PublicationIdentity(
        canonical_environment_id="VizdoomDeathmatch-v1",
        goal_id="VizdoomDeathmatch-v1",
        trainer="GradLab",
        trainer_slug="gradlab",
        algorithm="ppo",
        lineage_digest="1" * 64,
    )
    monkeypatch.setattr(
        "gradlab.player_publication.publication_identity_from_policy_bundle",
        lambda *_args: identity,
    )
    monkeypatch.setattr(
        "gradlab.player_publication.build_model_repo_id",
        lambda _identity: "tsilva/VizdoomDeathmatch-v1",
    )
    monkeypatch.setattr(
        "gradlab.player_publication.resolve_huggingface_credential",
        lambda: SimpleNamespace(token="secret"),
    )
    monkeypatch.setattr(
        "gradlab.player_publication.assert_goal_repository_compatible",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "gradlab.player_publication.next_release_version",
        lambda _api, _repo: ("v1", None),
    )
    monkeypatch.setattr(
        "gradlab.player_publication.publication_source_from_policy_bundle",
        lambda *_args: {
            "repository": "https://github.com/tsilva/gradlab",
            "commit": "a" * 40,
            "run_id": "gradlab-" + "e" * 32,
            "run_name": "run",
            "wandb_project": "VizdoomDeathmatch-v1",
            "recipe": "ppo",
            "seed": 7,
            "checkpoint_step": 10,
            "checkpoint_artifact": "https://r2.invalid/model.zip",
        },
    )
    monkeypatch.setattr(
        "gradlab.player_publication.generated_metadata",
        lambda **kwargs: {
            "title": "Generated title",
            "description": "Generated description",
            "tags": [],
            "privacy": kwargs["settings"].get("privacy", "public"),
            "container_name": "GradLab — VizdoomDeathmatch-v1",
            "thumbnail_time": 10.0,
            "operator_note": kwargs["settings"].get("operator_note", ""),
            "feature": bool(kwargs["settings"].get("feature", False)),
            "thumbnail": {"task": "Task", "trainer_algorithm": "GradLab PPO", "step": "10", "metric": "metric"},
        },
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


def test_generated_metadata_never_truncates_and_uses_goal_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = PublicationIdentity(
        "VizdoomDeathmatch-v1", "VizdoomDeathmatch-v1", "GradLab", "gradlab", "ppo", "1" * 64
    )
    monkeypatch.setattr(
        "gradlab.player_publication.publication_identity_from_policy_bundle",
        lambda *_args: identity,
    )
    metadata = generated_metadata(
        capture=_capture(),
        bundle=_bundle(tmp_path),
        evaluation={
            "checkpoint_step": 10,
            "protocol": "full",
            "action_sampling": "stochastic",
            "episodes": 2,
            "checkpoint_artifact": "https://r2.invalid/model.zip",
        },
        evaluation_evidence={
            "acceptance": {
                "outcomes": [
                    {
                        "metric": "eval/full/progress/kills/mean",
                        "label": "Full-eval kills mean",
                        "unit": "value",
                        "value": 34.29,
                        "operator": ">=",
                        "threshold": 10,
                        "passed": True,
                    }
                ]
            },
            "ranking": {"outcomes": []},
        },
        repo_id="tsilva/VizdoomDeathmatch-v1",
        release_version="v4",
        source={
            "wandb_project": "VizdoomDeathmatch-v1",
            "run_id": "gradlab-" + "e" * 32,
            "commit": "a" * 40,
            "recipe": "ppo",
            "model_document_url": "https://r2.invalid/model.json",
        },
        settings={"privacy": "public", "feature": True},
    )
    assert metadata["title"].endswith("GradLab PPO @ 10 env steps")
    assert "Full-eval kills mean" in metadata["description"]
    assert "win rate" not in metadata["description"].casefold()
    assert "{{COLLECTION_URL}}" in metadata["description"]
    assert metadata["container_name"] == "GradLab — VizdoomDeathmatch-v1"
    assert metadata["feature"] is True


def test_current_episode_must_finish_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service(tmp_path, monkeypatch)
    context = service.host.active_publication_context()
    context["capture"]["ready"] = False
    context["capture"]["error"] = "finish the current episode before publishing"
    service.host = SimpleNamespace(active_publication_context=lambda: context)
    with pytest.raises(ValueError, match="finish the current episode"):
        service.admit({"privacy": "public"}, credential_result=_credentials())


def test_publication_render_materializes_pending_capture_on_explicit_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service(tmp_path, monkeypatch)
    context = service.host.active_publication_context()
    capture = context["capture"]["latest"]
    context["capture"] = {
        "ready": True,
        "render_required": True,
        "latest": None,
        "error": None,
    }
    calls: list[str] = []

    def render() -> dict:
        calls.append("render")
        context["capture"] = {
            "ready": True,
            "render_required": False,
            "latest": capture,
            "error": None,
        }
        return capture

    service.host = SimpleNamespace(
        active_publication_context=lambda: context,
        render_publication_capture=render,
    )

    result = service.render()

    assert calls == ["render"]
    assert result["capture"]["capture_id"] == capture["capture_id"]


def test_admission_v3_is_idempotent_and_playlist_is_not_operator_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service(tmp_path, monkeypatch)
    settings = {"privacy": "public", "feature": True, "tags": []}
    first = service.admit(settings, credential_result=_credentials())
    repeated = service.admit(settings, credential_result=_credentials())
    assert first["created"] is True
    assert repeated["created"] is False
    subject = service.store.job(first["job"]["job_id"])
    assert subject["handler_version"] == 3
    assert "playlist" not in subject["payload"]


def test_admission_rejects_changed_editorial_request_for_same_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service(tmp_path, monkeypatch)
    service.admit({"privacy": "public", "feature": False}, credential_result=_credentials())
    with pytest.raises(PublicationConflict, match="immutable publication request"):
        service.admit({"privacy": "public", "feature": True}, credential_result=_credentials())


def test_full_digest_prefix_collision_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    identity = SimpleNamespace(
        canonical_environment_id="VizdoomDeathmatch-v1",
        goal_id="VizdoomDeathmatch-v1",
        lineage_digest="a" * 64,
        lineage_prefix="aaaaaaaa",
    )
    monkeypatch.setattr(
        "gradlab.player_publication._remote_release_manifest",
        lambda *_args, **_kwargs: {
            "format_version": 4,
            "repository": {
                "canonical_environment_id": "VizdoomDeathmatch-v1",
                "goal_id": "VizdoomDeathmatch-v1",
            },
            "lineage": {"digest": "a" * 8 + "b" * 56, "prefix": "aaaaaaaa"},
            "release": {"version": "v1"},
        },
    )
    monkeypatch.setattr(
        "gradlab.player_publication.validate_release_manifest_document",
        lambda value, **_kwargs: value,
    )
    with pytest.raises(ValueError, match="prefix collision"):
        assert_goal_repository_compatible(
            SimpleNamespace(
                model_info=lambda **_kwargs: object(),
                list_repo_refs=lambda **_kwargs: SimpleNamespace(
                    tags=[SimpleNamespace(name="v1")]
                ),
            ),
            repo_id="tsilva/example",
            identity=identity,
            token="secret",
        )
