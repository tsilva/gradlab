from __future__ import annotations

import json
import math
import os
import re
import shutil
import sqlite3
import uuid
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

from gradlab.file_utils import atomic_write_json, file_sha256, fsync_tree
from gradlab.goal_schema import goal_evaluation_mode
from gradlab.job_queue import JobStore, JobSubject, ensure_flusher
from gradlab.json_utils import canonical_json_sha256
from gradlab.local_paths import default_runs_dir
from gradlab.operator_environment import load_repository_operator_environment
from gradlab.policy_bundle import (
    PolicyBundle,
    build_model_document,
    evaluation_contract_sha256,
    load_policy_bundle,
    write_canonical_json,
)
from gradlab.publication import (
    GITATTRIBUTES_TEXT,
    HUGGINGFACE_NAMESPACE,
    MIT_LICENSE_TEXT,
    build_model_repo_id,
    build_release_manifest,
    normalize_publication_evaluation,
    publication_identity_from_policy_bundle,
    publication_source_from_policy_bundle,
    release_artifact_records,
    release_replay_from_capture,
    render_model_card,
    validate_release_bundle,
)
from gradlab.publication_credentials import (
    credential_lock,
    load_private_json,
    resolve_huggingface_credential,
    save_private_json,
    youtube_credential_paths,
)
from gradlab.publication_evidence import validate_publication_evidence
from gradlab.r2_store import RunStorageConfig
from gradlab.run_authority import RunAuthority
from gradlab.run_contracts import CheckpointManifest
from gradlab.youtube_publication import (
    YOUTUBE_SCOPES,
    YouTubeClient,
    YouTubePublicationError,
    refresh_access_token,
)


PLAYER_PUBLICATION_JOB_TYPE = "player-publication"
PLAYER_PUBLICATION_JOB_VERSION = 1
REQUEST_DOCUMENT_TYPE = "gradlab.player_publication_request"
REQUEST_FORMAT_VERSION = 1
REQUEST_ROOT_NAME = "player_publications"
YOUTUBE_PLACEHOLDER_URL = "https://www.youtube.com/watch?v=00000000000"
PRIVACY_VALUES = frozenset({"public", "unlisted", "private"})


class PublicationConflict(ValueError):
    def __init__(self, message: str, *, job: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.job = None if job is None else dict(job)


def publication_root() -> Path:
    return default_runs_dir() / REQUEST_ROOT_NAME


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _required_mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return deepcopy(dict(value))


def _copy_regular(source: Path, destination: Path) -> None:
    metadata = source.lstat()
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"publication input must be a regular non-symlink: {source}")
    if metadata.st_size < 1:
        raise ValueError(f"publication input is empty: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    with source.open("rb") as input_stream, temporary.open("xb") as output_stream:
        shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
        output_stream.flush()
        os.fsync(output_stream.fileno())
    os.replace(temporary, destination)


def _display_name(value: object) -> str:
    text = re.sub(r"-v[0-9]+$", "", str(value or "").strip())
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    text = re.sub(r"[_-]+", " ", text)
    text = re.sub(r"\b(?:vi\s*zdoom|vizdoom)\b", "ViZDoom", text, flags=re.I)
    text = re.sub(r"\b(nes|snes|gb|gba|n64)\b", lambda match: match.group(1).upper(), text, flags=re.I)
    text = re.sub(r"\b(Level|Stage|World)\s*(?=[0-9])", r"\1 ", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def _training_backend_display(policy: Mapping[str, Any]) -> tuple[str, str]:
    backend_id = str(policy.get("training_backend_id") or "").strip()
    model_class = str(policy.get("model_class") or "").strip()
    if backend_id.startswith("gradlab.") or model_class.startswith("gradlab."):
        return "GradLab", backend_id
    if backend_id.startswith("sb3.") or model_class.startswith("stable_baselines3."):
        return "Stable-Baselines3", backend_id
    raise ValueError("model policy does not identify a supported training backend")


def _bounded_youtube_tags(values: Sequence[object], *, max_characters: int = 500) -> list[str]:
    tags: list[str] = []
    total = 0
    for raw in values:
        value = str(raw).strip()
        if not value or value in tags:
            continue
        if len(value) > 100:
            raise ValueError("YouTube tags must be at most 100 characters each")
        added = len(value) + (1 if tags else 0)
        if total + added > max_characters:
            break
        tags.append(value)
        total += added
    return tags


def _youtube_access(
    *,
    client_factory: Callable[[str], YouTubeClient] = YouTubeClient,
) -> tuple[YouTubeClient, dict[str, Any]]:
    paths = youtube_credential_paths()
    with credential_lock(paths.lock):
        client_config = load_private_json(paths.client, root=paths.root)
        if not paths.token.exists():
            raise ValueError("YouTube authorization is missing")
        token = load_private_json(paths.token, root=paths.root)
        access_token = str(token.get("access_token") or "")
        if not access_token:
            token = refresh_access_token(client_config, token)
            save_private_json(paths.token, token, root=paths.root)
            access_token = str(token.get("access_token") or "")
    client = client_factory(access_token)
    try:
        principal = client.channel_identity()
    except YouTubePublicationError as exc:
        if exc.status != 401:
            raise
        with credential_lock(paths.lock):
            client_config = load_private_json(paths.client, root=paths.root)
            token = load_private_json(paths.token, root=paths.root)
            token = refresh_access_token(client_config, token)
            save_private_json(paths.token, token, root=paths.root)
        client = client_factory(str(token["access_token"]))
        principal = client.channel_identity()
    granted = set(token.get("scopes") or str(token.get("scope") or "").split())
    missing = sorted(set(YOUTUBE_SCOPES) - granted)
    if missing:
        raise ValueError("YouTube authorization is missing required upload scopes")
    principal["scopes"] = sorted(granted)
    return client, principal


def credential_preflight(
    *,
    hf_api_factory: Callable[..., Any] = HfApi,
    youtube_client_factory: Callable[[str], YouTubeClient] = YouTubeClient,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "huggingface": {"ready": False},
        "youtube": {"ready": False},
    }
    try:
        credential = resolve_huggingface_credential()
        api = hf_api_factory(token=credential.token)
        whoami = api.whoami()
        username = str(whoami.get("name") or "") if isinstance(whoami, Mapping) else ""
        orgs = whoami.get("orgs") if isinstance(whoami, Mapping) else ()
        namespace_allowed = username == HUGGINGFACE_NAMESPACE or any(
            isinstance(item, Mapping)
            and str(item.get("name") or "") == HUGGINGFACE_NAMESPACE
            and str(item.get("roleInOrg") or "").casefold() in {"admin", "write", "contributor"}
            for item in (orgs or ())
        )
        if not username or not namespace_allowed:
            raise ValueError(f"Hugging Face login cannot write namespace {HUGGINGFACE_NAMESPACE}")
        result["huggingface"] = {
            "ready": True,
            "username": username,
            "namespace": HUGGINGFACE_NAMESPACE,
            "credential_source": credential.source,
        }
    except Exception as exc:
        result["huggingface"] = {"ready": False, "message": str(exc)}
    try:
        _client, principal = _youtube_access(
            client_factory=youtube_client_factory,
        )
        result["youtube"] = {"ready": True, **principal}
    except Exception as exc:
        result["youtube"] = {"ready": False, "message": str(exc)}
    result["ready"] = bool(
        result["huggingface"].get("ready") and result["youtube"].get("ready")
    )
    return result


def next_release_version(api: Any, repo_id: str) -> tuple[str, str | None]:
    try:
        info = api.model_info(repo_id=repo_id)
    except HfHubHTTPError as exc:
        if getattr(exc.response, "status_code", None) == 404:
            return "v1", None
        raise
    except Exception as exc:
        if type(exc).__name__ in {"RepositoryNotFoundError"}:
            return "v1", None
        raise
    refs = api.list_repo_refs(repo_id=repo_id, repo_type="model")
    versions: dict[int, str] = {}
    for tag in getattr(refs, "tags", ()):
        match = re.fullmatch(r"v([1-9][0-9]*)", str(getattr(tag, "name", "")))
        if match:
            versions[int(match.group(1))] = str(getattr(tag, "target_commit", ""))
    if versions and sorted(versions) != list(range(1, max(versions) + 1)):
        raise ValueError("Hugging Face release tags are not contiguous")
    return f"v{max(versions, default=0) + 1}", str(getattr(info, "sha", "") or "") or None


def generated_metadata(
    *,
    capture: Mapping[str, Any],
    bundle: PolicyBundle,
    evaluation: Mapping[str, Any],
    repo_id: str,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    privacy = str(settings.get("privacy") or "public")
    if privacy not in PRIVACY_VALUES:
        raise ValueError("YouTube privacy must be public, unlisted, or private")
    goal = _required_mapping(capture.get("goal"), label="capture goal")
    goal_id = str(goal.get("goal_id") or "").strip()
    raw_goal = str(goal.get("title") or goal_id or "Goal").strip()
    goal_title = _display_name(raw_goal)
    qualified = str((capture.get("execution") or {}).get("qualified_environment_id") or "")
    provider, _separator, game = qualified.partition(":")
    raw_game = game or qualified or "Game"
    game_title = _display_name(raw_game)
    policy = _required_mapping(bundle.model.get("policy"), label="model policy")
    algorithm = str(policy.get("algorithm_id") or "policy").upper()
    trainer, backend_id = _training_backend_display(policy)
    trainer_title = "SB3" if trainer == "Stable-Baselines3" else trainer
    same_task = bool(goal_id) and goal_id.casefold() == raw_game.casefold()
    task_title = game_title if same_task else f"{game_title} — {goal_title}"
    task_description = game_title if same_task else f"{game_title} {goal_title}"
    captured_success = bool(capture.get("success"))
    success_mean = evaluation.get("success_rate_mean")
    title = f"{task_title} — {trainer_title} {algorithm}"
    title = title[:100].rstrip()
    outcome = "met" if captured_success else "did not meet"
    episode_verb = "completes" if captured_success else "plays"
    evaluation_claim = (
        f" with a {100.0 * float(success_mean):.1f}% verified full-evaluation win rate"
        if isinstance(success_mean, int | float) and not isinstance(success_mean, bool)
        else ""
    )
    description = (
        f"A {algorithm} reinforcement-learning agent trained with {trainer} "
        f"{episode_verb} {task_description}{evaluation_claim}. This exact faithful "
        f"{capture.get('sampling_mode')} "
        f"episode {outcome} the embedded goal; the separate checkpoint evaluation used "
        f"{evaluation.get('episodes')} episodes.\n\n"
        f"Model: https://huggingface.co/{repo_id}\n"
        "gradlab: https://github.com/tsilva/gradlab"
    )
    note = str(settings.get("operator_note") or "").strip()
    if len(note) > 1000:
        raise ValueError("operator note must be at most 1000 characters")
    if note:
        description += f"\n\nOperator note (not verified evidence):\n{note}"
    game_hashtag = re.sub(r"[^A-Za-z0-9]", "", game_title)[:60] or "gradlab"
    description += f"\n\n#ReinforcementLearning #{algorithm} #{game_hashtag}"
    if len(description) > 5000:
        raise ValueError("generated YouTube description exceeds 5000 characters")
    tags_value = settings.get("tags") or ()
    if isinstance(tags_value, str | bytes) or not isinstance(tags_value, Sequence):
        raise ValueError("YouTube tags must be an array")
    operator_tags = [str(value).strip() for value in tags_value if str(value).strip()]
    tags = _bounded_youtube_tags(
        [
            "reinforcement learning",
            "deep reinforcement learning",
            "AI gameplay",
            "gradlab",
            backend_id,
            trainer,
            *(["stable-baselines3"] if trainer == "Stable-Baselines3" else []),
            *(["Stable Retro"] if provider == "stable-retro-turbo" else []),
            algorithm,
            raw_game,
            game_title,
            str(goal.get("goal_id") or ""),
            goal_title,
            *operator_tags,
        ],
        max_characters=400,
    )
    playlist = str(settings.get("playlist") or "gradlab").strip() or "gradlab"
    if len(playlist) > 150:
        raise ValueError("YouTube playlist title must be at most 150 characters")
    thumbnail_value = settings.get("thumbnail_time")
    thumbnail_time = 10.0 if thumbnail_value is None else float(thumbnail_value)
    if not math.isfinite(thumbnail_time) or not 0 <= thumbnail_time <= 86_400:
        raise ValueError("YouTube thumbnail time must be finite and in [0, 86400]")
    return {
        "title": title,
        "description": description,
        "tags": tags,
        "privacy": privacy,
        "playlist": playlist,
        "thumbnail_time": thumbnail_time,
        "operator_note": note,
    }


def prepare_release_bundle(
    *,
    source_bundle: PolicyBundle,
    replay_path: Path,
    capture: Mapping[str, Any],
    evaluation_document: Mapping[str, Any],
    publication: Mapping[str, Any],
    release_version: str,
    published_at: str,
    youtube_url: str,
    output: Path,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"release output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    _copy_regular(source_bundle.checkpoint_path, output / "model.zip")
    _copy_regular(source_bundle.recipe_path, output / "recipe.json")
    _copy_regular(replay_path, output / "replay.mp4")
    (output / ".gitattributes").write_text(GITATTRIBUTES_TEXT, encoding="utf-8")
    (output / "LICENSE").write_text(MIT_LICENSE_TEXT, encoding="utf-8")
    source_model = source_bundle.model
    metadata = deepcopy(dict(source_model["provenance"]))
    metadata.update(source_model["policy"])
    metadata["checkpoint_step"] = source_model["checkpoint"].get("step")
    metadata["kind"] = source_model["checkpoint"].get("kind")
    write_canonical_json(
        output / "model.json",
        build_model_document(output / "model.zip", output / "recipe.json", metadata),
    )
    bundle = load_policy_bundle(output, source=str(output))
    goal = _required_mapping(capture.get("goal"), label="capture goal")
    identity = publication_identity_from_policy_bundle(goal.get("goal_id"), bundle)
    evaluation = normalize_publication_evaluation(
        evaluation_document,
        algorithm_id=str(bundle.model["policy"].get("algorithm_id") or ""),
    )
    source = publication_source_from_policy_bundle(bundle, evaluation)
    expected_evidence = {
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "recipe_sha256": bundle.recipe_sha256,
        "recipe_format_version": int(bundle.recipe["format_version"]),
        "evaluation_contract_sha256": evaluation_contract_sha256(bundle.recipe),
    }
    evidence = _required_mapping(
        evaluation_document.get("evaluation_evidence"),
        label="evaluation evidence",
    )
    for key, expected in expected_evidence.items():
        if evidence.get(key) != expected:
            raise ValueError(f"release evaluation {key} does not match the policy bundle")
    evaluation_value = evaluation.as_manifest_value()
    evaluation_value.update(expected_evidence, exact_contract=True)
    replay = release_replay_from_capture(capture)
    if replay["checkpoint_sha256"] != bundle.checkpoint_sha256:
        raise ValueError("captured checkpoint does not match release model.zip")
    if replay["recipe_sha256"] != bundle.recipe_sha256:
        raise ValueError("captured recipe does not match release recipe.json")
    provisional = build_release_manifest(
        identity,
        bundle,
        release_version=release_version,
        published_at=published_at,
        source=source,
        evaluation=evaluation_value,
        artifacts={},
        youtube_url=youtube_url,
        replay=replay,
        publication=publication,
    )
    (output / "README.md").write_text(render_model_card(provisional, bundle), encoding="utf-8")
    manifest = build_release_manifest(
        identity,
        bundle,
        release_version=release_version,
        published_at=published_at,
        source=source,
        evaluation=evaluation_value,
        artifacts=release_artifact_records(output),
        youtube_url=youtube_url,
        replay=replay,
        publication=publication,
    )
    write_canonical_json(output / "release_manifest.json", manifest)
    validate_release_bundle(output)
    return manifest


class PlayerPublicationService:
    def __init__(
        self,
        *,
        repo_root: Path,
        host: Any,
        store: JobStore | None = None,
        hf_api_factory: Callable[..., Any] = HfApi,
        evidence_loader: Callable[[str, str], Mapping[str, Any]] | None = None,
        root: Path | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.host = host
        self.root = Path(root or publication_root()).expanduser().resolve()
        self.requests_root = self.root / "requests"
        self.requests_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.store = store or JobStore()
        self.store.init()
        self.hf_api_factory = hf_api_factory
        self._evidence_loader = evidence_loader

    def _active(self) -> dict[str, Any]:
        context = self.host.active_publication_context()
        if not isinstance(context, Mapping):
            raise ValueError("no active player checkpoint is available")
        spec = context.get("spec")
        if getattr(spec, "kind", None) != "public_run":
            raise ValueError("only verified catalog public-run checkpoints are publishable")
        capture_status = _required_mapping(context.get("capture"), label="capture status")
        if capture_status.get("ready") is not True:
            raise ValueError(
                capture_status.get("error")
                or "finish the current episode before publishing"
            )
        capture = capture_status.get("latest")
        if not isinstance(capture, Mapping):
            raise ValueError(capture_status.get("error") or "complete an episode before publishing")
        return {**dict(context), "capture_document": deepcopy(dict(capture))}

    def current(self) -> dict[str, Any]:
        try:
            active = self._active()
        except ValueError as exc:
            return {"available": False, "message": str(exc)}
        capture = active["capture_document"]
        statuses = self.store.subject_statuses(
            subject_type="player-publication-capture",
            subject_ids=[str(capture["capture_id"])],
        )
        subject = statuses.get(str(capture["capture_id"]))
        return {
            "available": True,
            "capture": capture,
            "job": None if subject is None else self._public_subject(subject),
        }

    @staticmethod
    def _public_subject(subject: Mapping[str, Any]) -> dict[str, Any]:
        detail = subject.get("detail") if isinstance(subject.get("detail"), Mapping) else {}
        return {
            "job_id": str(subject.get("job_id") or ""),
            "state": str(subject.get("job_state") or subject.get("state") or ""),
            "cancel_requested": bool(subject.get("cancel_requested")),
            "message": detail.get("message") or subject.get("job_error"),
            "request_fingerprint": detail.get("request_fingerprint"),
            "urls": deepcopy(detail.get("urls") or {}),
            "progress": deepcopy(detail.get("progress") or {}),
            "settings": deepcopy(detail.get("settings") or {}),
        }

    def _load_exact_evidence(
        self,
        active: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        capture = _required_mapping(active.get("capture_document"), label="capture")
        bundle = active.get("bundle")
        if not isinstance(bundle, PolicyBundle):
            raise ValueError("active publication bundle is unavailable")
        source = active.get("source")
        run_config = getattr(source, "run_config", None)
        raw_manifest = (
            run_config.get("checkpoint_manifest")
            if isinstance(run_config, Mapping)
            else None
        )
        if not isinstance(raw_manifest, Mapping):
            raise ValueError("active public checkpoint has no verified manifest binding")
        checkpoint = CheckpointManifest.from_dict(raw_manifest)
        run_id = str(capture["run_id"])
        checkpoint_id = str(capture["checkpoint_id"])
        if checkpoint.run_id != run_id or checkpoint.checkpoint_id != checkpoint_id:
            raise ValueError("active public checkpoint manifest identity is inconsistent")
        if checkpoint.sha256 != bundle.checkpoint_sha256:
            raise ValueError("active public checkpoint manifest model hash is inconsistent")
        if checkpoint.recipe_document_sha256 != bundle.recipe_sha256:
            raise ValueError("active public checkpoint manifest recipe hash is inconsistent")

        load_repository_operator_environment(self.repo_root)
        authority = RunAuthority(RunStorageConfig.from_env())
        rejected: list[str] = []
        for key in authority.evaluation.iter_keys(f"runs/{run_id}/evals/"):
            if not key.endswith("/verified-result.json"):
                continue
            verified = authority.evaluation.get_json(key)
            if str(verified.get("checkpoint_id") or "") != checkpoint_id:
                continue
            idempotency_key = str(verified.get("idempotency_key") or "")
            intent = authority.eval_intent(
                run_id=run_id,
                idempotency_key=idempotency_key,
            )
            raw = authority.eval_result(
                run_id=run_id,
                idempotency_key=idempotency_key,
            )
            if intent is None or raw is None:
                rejected.append(f"{idempotency_key}: incomplete intent/raw join")
                continue
            evidence = {
                "checkpoint_manifest": checkpoint.to_dict(),
                "recipe": deepcopy(bundle.recipe),
                "intent": deepcopy(dict(intent)),
                "raw_result": deepcopy(dict(raw)),
                "verified_result": deepcopy(dict(verified)),
            }
            try:
                validate_publication_evidence(evidence)
            except ValueError as exc:
                rejected.append(f"{idempotency_key}: {exc}")
                continue
            return evidence
        detail = f"; rejected candidates: {' | '.join(rejected)}" if rejected else ""
        raise ValueError(
            "checkpoint does not have current exact verified evaluation evidence" + detail
        )

    def _load_evidence(self, active: Mapping[str, Any]) -> Mapping[str, Any]:
        capture = _required_mapping(active.get("capture_document"), label="capture")
        run_id = str(capture["run_id"])
        checkpoint_id = str(capture["checkpoint_id"])
        if self._evidence_loader is not None:
            return self._evidence_loader(run_id, checkpoint_id)
        return self._load_exact_evidence(active)

    def admit(
        self,
        settings: Mapping[str, Any],
        *,
        credential_result: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        active = self._active()
        capture = active["capture_document"]
        bundle = active["bundle"]
        assert isinstance(bundle, PolicyBundle)
        goal = _required_mapping(capture.get("goal"), label="capture goal")
        if goal_evaluation_mode(goal, label="capture goal") != "evaluated":
            raise ValueError("training-only goals cannot be published")
        release = goal.get("release")
        if not isinstance(release, Mapping) or not isinstance(release.get("huggingface"), Mapping):
            raise ValueError("the embedded goal does not enable Hugging Face release")
        evidence = validate_publication_evidence(
            self._load_evidence(active)
        )
        evaluation = _required_mapping(evidence.get("evaluation"), label="publication evaluation")
        normalized = normalize_publication_evaluation(
            evaluation,
            algorithm_id=str(bundle.model["policy"].get("algorithm_id") or ""),
        ).as_manifest_value()
        credentials = dict(credential_result or credential_preflight())
        if not credentials.get("ready"):
            raise ValueError("Hugging Face and YouTube credentials must both pass preflight")
        hf_principal = _required_mapping(credentials.get("huggingface"), label="Hugging Face")
        youtube_principal = _required_mapping(credentials.get("youtube"), label="YouTube")
        identity = publication_identity_from_policy_bundle(goal.get("goal_id"), bundle)
        repo_id = build_model_repo_id(identity)
        capture_subject = self.store.subject_statuses(
            subject_type="player-publication-capture",
            subject_ids=[str(capture["capture_id"])],
        ).get(str(capture["capture_id"]))
        repository_subject = self.store.subject_statuses(
            subject_type="player-publication-repo",
            subject_ids=[repo_id],
        ).get(repo_id)
        if (
            repository_subject is not None
            and str(repository_subject.get("job_state")) != "succeeded"
            and (
                capture_subject is None
                or repository_subject.get("job_id") != capture_subject.get("job_id")
            )
        ):
            raise PublicationConflict(
                "this Hugging Face repository already has an unresolved publication request",
                job=self._public_subject(repository_subject),
            )
        hf_credential = resolve_huggingface_credential()
        api = self.hf_api_factory(token=hf_credential.token)
        version, parent_commit = next_release_version(api, repo_id)
        metadata = generated_metadata(
            capture=capture,
            bundle=bundle,
            evaluation=normalized,
            repo_id=repo_id,
            settings=settings,
        )
        published_at = _utc_now()
        basis: dict[str, Any] = {
            "document_type": REQUEST_DOCUMENT_TYPE,
            "format_version": REQUEST_FORMAT_VERSION,
            "capture_id": capture["capture_id"],
            "capture_fence_sha256": capture["capture_fence_sha256"],
            "repo_id": repo_id,
            "release_version": version,
            "parent_commit": parent_commit,
            "published_at": published_at,
            "metadata": deepcopy(metadata),
            "principals": {
                "huggingface_username": hf_principal["username"],
                "huggingface_namespace": hf_principal["namespace"],
                "youtube_channel_id": youtube_principal["channel_id"],
                "youtube_channel_title": youtube_principal["channel_title"],
                "youtube_scopes": sorted(youtube_principal.get("scopes") or ()),
            },
            "evidence_sha256": canonical_json_sha256(evidence),
        }
        request_fingerprint = canonical_json_sha256(basis)
        marker = f"gradlab-publication-{request_fingerprint}"
        final_metadata = deepcopy(metadata)
        final_metadata["tags"] = _bounded_youtube_tags([marker, *final_metadata["tags"]])
        request = {
            **basis,
            "fingerprint_basis": basis,
            "request_fingerprint": request_fingerprint,
            "metadata_sha256": canonical_json_sha256(final_metadata),
            "marker": marker,
            "metadata": final_metadata,
        }
        existing = capture_subject
        if existing is not None:
            detail = existing.get("detail") if isinstance(existing.get("detail"), Mapping) else {}
            existing_settings = detail.get("settings")
            comparable_existing = (
                {**dict(existing_settings), "tags": [
                    tag
                    for tag in existing_settings.get("tags", ())
                    if not str(tag).startswith("gradlab-publication-")
                ]}
                if isinstance(existing_settings, Mapping)
                else None
            )
            if comparable_existing != metadata:
                raise PublicationConflict(
                    "this capture already has an immutable publication request",
                    job=self._public_subject(existing),
                )
            return {"created": False, "job": self._public_subject(existing)}

        snapshot = self._stage_snapshot(
            request=request,
            capture=capture,
            bundle=bundle,
            evidence=evidence,
        )
        try:
            result = self.store.enqueue(
                job_type=PLAYER_PUBLICATION_JOB_TYPE,
                handler_version=PLAYER_PUBLICATION_JOB_VERSION,
                payload={
                    "repo_root": str(self.repo_root),
                    "queue_root": str(self.store.root),
                    "request_root": str(snapshot),
                    "request_fingerprint": request_fingerprint,
                },
                idempotency_key=request_fingerprint,
                subjects=[
                    JobSubject(
                        subject_type="player-publication-capture",
                        subject_id=str(capture["capture_id"]),
                        exclusive_key=f"player-publication:capture:{capture['capture_id']}",
                        detail={
                            "request_fingerprint": request_fingerprint,
                            "settings": final_metadata,
                            "progress": {"phase": "queued"},
                            "urls": {},
                        },
                    ),
                    JobSubject(
                        subject_type="player-publication-repo",
                        subject_id=repo_id,
                        detail={"release_version": version},
                    ),
                ],
            )
        except sqlite3.IntegrityError as exc:
            existing = self.store.subject_statuses(
                subject_type="player-publication-capture",
                subject_ids=[str(capture["capture_id"])],
            ).get(str(capture["capture_id"]))
            raise PublicationConflict(
                "this capture was admitted concurrently",
                job=None if existing is None else self._public_subject(existing),
            ) from exc
        worker = ensure_flusher(self.store)
        if worker.state == "start_failed":
            raise RuntimeError(worker.message or "publication worker failed to start")
        assert result.job is not None
        subject = self.store.subject_statuses(
            subject_type="player-publication-capture",
            subject_ids=[str(capture["capture_id"])],
        )[str(capture["capture_id"])]
        return {"created": result.created, "job": self._public_subject(subject)}

    def preflight(self) -> dict[str, Any]:
        return credential_preflight()

    def job(self, job_id: str) -> dict[str, Any]:
        job = self.store.job(job_id)
        if job is None or job.get("job_type") != PLAYER_PUBLICATION_JOB_TYPE:
            raise ValueError(f"unknown player publication job: {job_id}")
        subjects = self.store.subjects(job_id)
        capture = next(
            (item for item in subjects if item["subject_type"] == "player-publication-capture"),
            None,
        )
        if capture is None:
            raise ValueError("publication job has no capture subject")
        return {
            **self._public_subject({**capture, "job_state": job["state"], "job_error": job.get("last_error"), "cancel_requested": job["cancel_requested"]}),
            "events": self.store.events(job_id),
        }

    def retry(self, job_id: str) -> dict[str, Any]:
        self.job(job_id)
        self.store.retry(job_id)
        worker = ensure_flusher(self.store)
        if worker.state == "start_failed":
            raise RuntimeError(worker.message or "publication worker failed to start")
        return self.job(job_id)

    def cancel(self, job_id: str) -> dict[str, Any]:
        self.job(job_id)
        self.store.request_cancel(job_id)
        return self.job(job_id)

    def resolve_youtube(self, job_id: str, video_id: str) -> dict[str, Any]:
        if not re.fullmatch(r"[A-Za-z0-9_-]{11}", str(video_id)):
            raise ValueError("YouTube video id must contain exactly 11 URL-safe characters")
        job = self.store.job(job_id)
        current = self.job(job_id)
        if job is None or job["state"] != "blocked":
            raise ValueError("only a blocked publication can be manually resolved")
        state_path = self.store.work_root / job_id / "publication_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(state, Mapping) or state.get("phase") != "youtube_uncertain":
            raise ValueError("publication is not awaiting YouTube uncertainty resolution")
        resolved = dict(state)
        resolved["resolved_video_id"] = str(video_id)
        resolved["message"] = "operator supplied a YouTube video id for reconciliation"
        atomic_write_json(state_path, resolved)
        os.chmod(state_path, 0o600)
        self.store.retry(job_id)
        worker = ensure_flusher(self.store)
        if worker.state == "start_failed":
            raise RuntimeError(worker.message or "publication worker failed to start")
        return {**current, **self.job(job_id)}

    def replay_path(self) -> Path:
        active = self._active()
        capture = active["capture_document"]
        path = self.root / "captures" / str(capture["capture_id"]) / "replay.mp4"
        replay = _required_mapping(capture.get("replay"), label="capture replay")
        if path.is_symlink() or not path.is_file() or file_sha256(path) != replay["sha256"]:
            raise ValueError("captured replay is unavailable or changed")
        return path

    def cleanup(self, job_id: str) -> dict[str, Any]:
        job = self.store.job(job_id)
        public = self.job(job_id)
        if job is None or job["state"] not in {"succeeded", "failed", "blocked", "canceled"}:
            raise ValueError("active publication staging cannot be cleaned")
        payload = job["payload"]
        request_root = Path(str(payload["request_root"])).resolve()
        work_root = (self.store.work_root / job_id).resolve()
        removed: list[str] = []
        for path, parent in (
            (request_root, self.requests_root.resolve()),
            (work_root, self.store.work_root.resolve()),
        ):
            if path.parent != parent or not path.exists():
                continue
            shutil.rmtree(path)
            removed.append(str(path))
        return {**public, "removed": removed}

    def _stage_snapshot(
        self,
        *,
        request: Mapping[str, Any],
        capture: Mapping[str, Any],
        bundle: PolicyBundle,
        evidence: Mapping[str, Any],
    ) -> Path:
        fingerprint = str(request["request_fingerprint"])
        destination = self.requests_root / fingerprint
        if destination.exists():
            existing = json.loads((destination / "request.json").read_text(encoding="utf-8"))
            if existing != request:
                raise ValueError("content-addressed publication request conflicts")
            return destination
        temporary = self.requests_root / f".{fingerprint}.{uuid.uuid4().hex}.tmp"
        temporary.mkdir(mode=0o700)
        try:
            source_root = temporary / "source"
            _copy_regular(bundle.checkpoint_path, source_root / "model.zip")
            _copy_regular(bundle.model_path, source_root / "model.json")
            _copy_regular(bundle.recipe_path, source_root / "recipe.json")
            capture_root = temporary / "capture"
            capture_source = self.root / "captures" / str(capture["capture_id"])
            _copy_regular(capture_source / "capture.json", capture_root / "capture.json")
            _copy_regular(capture_source / "replay.mp4", capture_root / "replay.mp4")
            evidence_root = temporary / "evidence"
            for name in ("intent", "raw_result", "verified_result", "checkpoint_manifest"):
                atomic_write_json(evidence_root / f"{name}.json", _required_mapping(evidence[name], label=name))
            atomic_write_json(evidence_root / "evaluation.json", _required_mapping(evidence["evaluation"], label="evaluation"))
            publication = {
                "request_fingerprint": request["request_fingerprint"],
                "huggingface_username": request["principals"]["huggingface_username"],
                "huggingface_namespace": request["principals"]["huggingface_namespace"],
                "youtube_channel_id": request["principals"]["youtube_channel_id"],
                "youtube_channel_title": request["principals"]["youtube_channel_title"],
                "youtube_privacy": request["metadata"]["privacy"],
            }
            atomic_write_json(temporary / "publication.json", publication)
            prepare_release_bundle(
                source_bundle=load_policy_bundle(source_root, source=str(source_root)),
                replay_path=capture_root / "replay.mp4",
                capture=capture,
                evaluation_document=evidence["evaluation"],
                publication=publication,
                release_version=str(request["release_version"]),
                published_at=str(request["published_at"]),
                youtube_url=YOUTUBE_PLACEHOLDER_URL,
                output=temporary / "provisional_release",
            )
            atomic_write_json(temporary / "request.json", dict(request))
            hashes = {
                path.relative_to(temporary).as_posix(): file_sha256(path)
                for path in sorted(temporary.rglob("*"))
                if path.is_file() and path.name != "snapshot_hashes.json"
            }
            atomic_write_json(temporary / "snapshot_hashes.json", hashes)
            fsync_tree(temporary)
            os.replace(temporary, destination)
            os.chmod(destination, 0o700)
            return destination
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise


__all__ = [
    "PLAYER_PUBLICATION_JOB_TYPE",
    "PLAYER_PUBLICATION_JOB_VERSION",
    "PlayerPublicationService",
    "PublicationConflict",
    "credential_preflight",
    "generated_metadata",
    "next_release_version",
    "prepare_release_bundle",
]
