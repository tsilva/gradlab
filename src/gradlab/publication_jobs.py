from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from huggingface_hub import (
    CommitOperationAdd,
    CommitOperationDelete,
    HfApi,
    hf_hub_download,
)
from huggingface_hub.errors import HfHubHTTPError

from gradlab.file_utils import atomic_write_json, file_sha256
from gradlab.job_queue import (
    HandlerResult,
    JobStore,
    SubjectUpdate,
    register_handler,
)
from gradlab.json_utils import canonical_json_sha256
from gradlab.play_capture import validate_capture_document
from gradlab.player_publication import (
    PLAYER_PUBLICATION_JOB_TYPE,
    PLAYER_PUBLICATION_JOB_VERSION,
    REQUEST_DOCUMENT_TYPE,
    REQUEST_FORMAT_VERSION,
    _youtube_access,
    prepare_release_bundle,
)
from gradlab.policy_bundle import load_policy_bundle
from gradlab.publication import HUGGINGFACE_RELEASE_FILES, validate_release_bundle, verify_replay
from gradlab.publication_credentials import resolve_huggingface_credential
from gradlab.youtube_publication import (
    YouTubeClient,
    YouTubePublicationError,
    YouTubeSubmissionUncertain,
    extract_thumbnail,
    validate_processed_video,
)


class PublicationCanceled(RuntimeError):
    pass


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return dict(value)


def _http_status(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)
    return int(response.status_code) if getattr(response, "status_code", None) else None


def _retry_delay(job: Mapping[str, Any]) -> int:
    return min(2 ** max(0, int(job.get("attempts") or 1) - 1), 300)


class PlayerPublicationJobHandler:
    job_type = PLAYER_PUBLICATION_JOB_TYPE
    version = PLAYER_PUBLICATION_JOB_VERSION

    def __init__(
        self,
        *,
        hf_api_factory: Callable[..., Any] = HfApi,
        youtube_client_factory: Callable[[str], YouTubeClient] = YouTubeClient,
        hub_download: Callable[..., str] = hf_hub_download,
    ) -> None:
        self.hf_api_factory = hf_api_factory
        self.youtube_client_factory = youtube_client_factory
        self.hub_download = hub_download

    @classmethod
    def validate_payload(cls, payload: Mapping[str, Any]) -> dict[str, Any]:
        if set(payload) != {
            "repo_root",
            "queue_root",
            "request_root",
            "request_fingerprint",
        }:
            raise ValueError("player publication job payload fields are malformed")
        repo_root = Path(str(payload.get("repo_root") or "")).expanduser().resolve()
        queue_root = Path(str(payload.get("queue_root") or "")).expanduser().resolve()
        request_root = Path(str(payload.get("request_root") or "")).expanduser().resolve()
        fingerprint = str(payload.get("request_fingerprint") or "")
        if not repo_root.is_dir() or not queue_root.is_absolute() or not request_root.is_dir():
            raise ValueError("player publication job paths are invalid")
        if request_root.is_symlink() or request_root.name != fingerprint:
            raise ValueError("player publication request root is not content-addressed")
        if len(fingerprint) != 64 or any(character not in "0123456789abcdef" for character in fingerprint):
            raise ValueError("player publication request fingerprint is invalid")
        return {
            "repo_root": str(repo_root),
            "queue_root": str(queue_root),
            "request_root": str(request_root),
            "request_fingerprint": fingerprint,
        }

    def _snapshot(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        root = Path(str(payload["request_root"]))
        expected_hashes = _json_object(root / "snapshot_hashes.json", label="snapshot hashes")
        actual_files: set[str] = set()
        for path in root.rglob("*"):
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                raise ValueError(f"publication snapshot contains a symlink: {relative}")
            if path.is_file():
                actual_files.add(relative)
        if actual_files != {*expected_hashes, "snapshot_hashes.json"}:
            raise ValueError("publication snapshot file set differs from its immutable hash fence")
        for relative, expected in expected_hashes.items():
            if not isinstance(relative, str) or not isinstance(expected, str):
                raise ValueError("publication snapshot hashes are malformed")
            if file_sha256(root / relative) != expected:
                raise ValueError(f"publication snapshot changed after admission: {relative}")

        request = _json_object(root / "request.json", label="publication request")
        basis = request.get("fingerprint_basis")
        if not isinstance(basis, Mapping):
            raise ValueError("publication request is missing its fingerprint basis")
        fingerprint = str(payload["request_fingerprint"])
        if canonical_json_sha256(basis) != fingerprint:
            raise ValueError("publication request fingerprint does not match its admitted basis")
        if request.get("request_fingerprint") != fingerprint:
            raise ValueError("publication request identity differs from its queue payload")
        if basis.get("document_type") != REQUEST_DOCUMENT_TYPE or basis.get(
            "format_version"
        ) != REQUEST_FORMAT_VERSION:
            raise ValueError("publication request contract is unsupported")
        marker = f"gradlab-publication-{fingerprint}"
        metadata = request.get("metadata")
        if not isinstance(metadata, Mapping) or request.get("marker") != marker:
            raise ValueError("publication request marker or metadata is malformed")
        if canonical_json_sha256(metadata) != request.get("metadata_sha256"):
            raise ValueError("publication request metadata hash does not match")
        tags = list(metadata.get("tags") or ())
        if tags.count(marker) != 1:
            raise ValueError("publication request must contain exactly one internal marker")
        comparable = deepcopy(dict(metadata))
        comparable["tags"] = [tag for tag in tags if tag != marker]
        if comparable != basis.get("metadata"):
            raise ValueError("publication metadata differs from its fingerprinted request")
        for key, expected in basis.items():
            if key != "metadata" and request.get(key) != expected:
                raise ValueError(f"publication request field changed after fingerprinting: {key}")

        capture = _json_object(root / "capture/capture.json", label="episode capture")
        validate_capture_document(capture)
        if capture.get("capture_id") != basis.get("capture_id") or capture.get(
            "capture_fence_sha256"
        ) != basis.get("capture_fence_sha256"):
            raise ValueError("episode capture differs from the admitted request")
        replay = capture.get("replay")
        if not isinstance(replay, Mapping) or file_sha256(
            root / "capture/replay.mp4"
        ) != replay.get("sha256"):
            raise ValueError("captured replay differs from its capture document")
        provisional = validate_release_bundle(root / "provisional_release")
        if provisional.get("publication", {}).get("request_fingerprint") != fingerprint:
            raise ValueError("provisional release differs from the admitted request")
        return {"root": root, "request": request, "capture": capture}

    def _state(self, store: JobStore, job_id: str) -> tuple[Path, dict[str, Any]]:
        root = store.work_root / job_id
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(root, 0o700)
        path = root / "publication_state.json"
        if not path.exists():
            return path, {"format_version": 1, "phase": "admitted", "urls": {}}
        return path, _json_object(path, label="publication state")

    @staticmethod
    def _save_state(path: Path, state: Mapping[str, Any]) -> None:
        atomic_write_json(path, dict(state))
        os.chmod(path, 0o600)

    @staticmethod
    def _details(request: Mapping[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "request_fingerprint": request["request_fingerprint"],
            "settings": deepcopy(dict(request["metadata"])),
            "progress": {
                "phase": state.get("phase"),
                "uploaded_bytes": state.get("uploaded_bytes"),
                "total_bytes": state.get("total_bytes"),
            },
            "urls": deepcopy(dict(state.get("urls") or {})),
            "message": state.get("message"),
        }

    def _subjects(
        self,
        request: Mapping[str, Any],
        state: Mapping[str, Any],
        *,
        subject_state: str,
    ) -> tuple[SubjectUpdate, ...]:
        detail = self._details(request, state)
        return (
            SubjectUpdate(
                "player-publication-capture",
                str(request["capture_id"]),
                subject_state,
                detail,
            ),
            SubjectUpdate(
                "player-publication-repo",
                str(request["repo_id"]),
                subject_state,
                detail,
            ),
        )

    def _checkpoint(
        self,
        store: JobStore,
        job_id: str,
        request: Mapping[str, Any],
        state: Mapping[str, Any],
        *,
        kind: str,
        subject_state: str = "running",
    ) -> None:
        canceled = store.checkpoint(
            job_id,
            kind=kind,
            subjects=self._subjects(request, state, subject_state=subject_state),
            detail={"phase": state.get("phase")},
        )
        if canceled:
            raise PublicationCanceled("publication canceled at a durable boundary")

    def _hf_access(self, request: Mapping[str, Any]) -> tuple[Any, str]:
        credential = resolve_huggingface_credential()
        api = self.hf_api_factory(token=credential.token)
        whoami = api.whoami()
        username = str(whoami.get("name") or "") if isinstance(whoami, Mapping) else ""
        principals = request["principals"]
        if username != str(principals["huggingface_username"]):
            raise ValueError("Hugging Face principal changed after publication admission")
        if str(principals["huggingface_namespace"]) != str(request["repo_id"]).split("/", 1)[0]:
            raise ValueError("Hugging Face namespace differs from the admitted repository")
        return api, credential.token

    def _youtube_access(self, repo_root: Path, request: Mapping[str, Any]) -> YouTubeClient:
        client, principal = _youtube_access(
            repo_root,
            client_factory=self.youtube_client_factory,
        )
        if principal.get("channel_id") != request["principals"]["youtube_channel_id"]:
            raise ValueError("YouTube channel changed after publication admission")
        return client

    def _retry(
        self,
        job: Mapping[str, Any],
        request: Mapping[str, Any],
        state: dict[str, Any],
        path: Path,
        message: str,
    ) -> HandlerResult:
        delay = _retry_delay(job)
        state["message"] = message
        self._save_state(path, state)
        return HandlerResult(
            state="retry_wait",
            available_at=time.time() + delay,
            message=message,
            subjects=self._subjects(request, state, subject_state="retry_wait"),
        )

    def _block(
        self,
        request: Mapping[str, Any],
        state: dict[str, Any],
        path: Path,
        message: str,
    ) -> HandlerResult:
        state["message"] = message
        self._save_state(path, state)
        return HandlerResult(
            state="blocked",
            message=message,
            subjects=self._subjects(request, state, subject_state="blocked"),
        )

    def _advance_youtube(
        self,
        *,
        job: Mapping[str, Any],
        store: JobStore,
        repo_root: Path,
        snapshot: Mapping[str, Any],
        state_path: Path,
        state: dict[str, Any],
    ) -> HandlerResult | None:
        request = snapshot["request"]
        root = snapshot["root"]
        job_id = str(job["job_id"])
        phase = str(state["phase"])
        client = self._youtube_access(repo_root, request)
        replay_path = Path(root) / "capture/replay.mp4"

        if phase == "validated":
            self._checkpoint(store, job_id, request, state, kind="before_youtube_upload")
            session = client.start_resumable_upload(
                video_path=replay_path,
                title=str(request["metadata"]["title"]),
                description=str(request["metadata"]["description"]),
                tags=list(request["metadata"]["tags"]),
                privacy=str(request["metadata"]["privacy"]),
            )
            state.update(
                phase="youtube_session",
                youtube_session=session,
                uploaded_bytes=0,
                total_bytes=replay_path.stat().st_size,
                message=None,
            )
            self._save_state(state_path, state)
            self._checkpoint(store, job_id, request, state, kind="youtube_session_started")
            phase = "youtube_session"

        if phase in {"youtube_session", "youtube_uncertain"}:
            if phase == "youtube_uncertain" and state.get("resolved_video_id"):
                state["youtube_video_id"] = str(state["resolved_video_id"])
                state["phase"] = "youtube_uploaded"
                self._save_state(state_path, state)
            else:
                try:
                    remote = client.query_resumable(
                        str(state["youtube_session"]),
                        total_bytes=int(state["total_bytes"]),
                    )
                except YouTubePublicationError as exc:
                    if exc.code == "session_expired" and phase == "youtube_uncertain":
                        return self._block(
                            request,
                            state,
                            state_path,
                            "YouTube final upload outcome is uncertain; provide the resulting "
                            "video id or explicitly retry after checking the admitted channel.",
                        )
                    raise
                if isinstance(remote, Mapping):
                    video_id = str(remote.get("id") or "")
                    if not video_id:
                        raise YouTubeSubmissionUncertain(
                            "YouTube completed the upload without a video id"
                        )
                    state["youtube_video_id"] = video_id
                    state["phase"] = "youtube_uploaded"
                    self._save_state(state_path, state)
                else:
                    state["uploaded_bytes"] = int(remote)

                    def progress(uploaded: int, total: int) -> None:
                        state.update(uploaded_bytes=uploaded, total_bytes=total)
                        self._save_state(state_path, state)
                        self._checkpoint(
                            store,
                            job_id,
                            request,
                            state,
                            kind="youtube_chunk",
                        )

                    try:
                        uploaded = client.upload_chunks(
                            str(state["youtube_session"]),
                            replay_path,
                            offset=int(remote),
                            progress=progress,
                        )
                    except YouTubeSubmissionUncertain:
                        state["phase"] = "youtube_uncertain"
                        self._save_state(state_path, state)
                        raise
                    state["youtube_video_id"] = str(uploaded["id"])
                    state["phase"] = "youtube_uploaded"
                    self._save_state(state_path, state)

        if state["phase"] in {"youtube_uploaded", "youtube_processing"}:
            video_id = str(state["youtube_video_id"])
            video = client.video_metadata(video_id)
            try:
                validate_processed_video(
                    video,
                    channel_id=str(request["principals"]["youtube_channel_id"]),
                    privacy=str(request["metadata"]["privacy"]),
                    marker=str(request["marker"]),
                    title=str(request["metadata"]["title"]),
                    description=str(request["metadata"]["description"]),
                )
            except YouTubePublicationError as exc:
                if exc.code == "processing_pending":
                    state["phase"] = "youtube_processing"
                    state["urls"] = {
                        **dict(state.get("urls") or {}),
                        "youtube": f"https://www.youtube.com/watch?v={video_id}",
                    }
                    self._save_state(state_path, state)
                    return HandlerResult(
                        state="retry_wait",
                        available_at=time.time() + 15,
                        message=str(exc),
                        subjects=self._subjects(
                            request, state, subject_state="retry_wait"
                        ),
                    )
                raise
            state["phase"] = "youtube_verified"
            state["urls"] = {
                **dict(state.get("urls") or {}),
                "youtube": f"https://www.youtube.com/watch?v={video_id}",
            }
            self._save_state(state_path, state)
            self._checkpoint(store, job_id, request, state, kind="youtube_verified")

        if state["phase"] == "youtube_verified":
            playlist_id = client.find_or_create_playlist(
                str(request["metadata"]["playlist"]),
                privacy=str(request["metadata"]["privacy"]),
            )
            client.add_video_to_playlist(
                playlist_id=playlist_id,
                video_id=str(state["youtube_video_id"]),
            )
            media = verify_replay(replay_path)
            requested = float(request["metadata"]["thumbnail_time"])
            duration = float(media["duration_seconds"])
            thumbnail = extract_thumbnail(
                replay_path,
                state_path.parent / "thumbnail.jpg",
                seconds=min(requested, max(0.0, duration - 0.25)),
            )
            client.upload_thumbnail(
                video_id=str(state["youtube_video_id"]),
                thumbnail_path=thumbnail,
            )
            state["phase"] = "youtube_complete"
            state["youtube_playlist_id"] = playlist_id
            state["urls"] = {
                **dict(state.get("urls") or {}),
                "playlist": f"https://www.youtube.com/playlist?list={playlist_id}",
            }
            self._save_state(state_path, state)
            self._checkpoint(store, job_id, request, state, kind="youtube_complete")
        return None

    def _build_release(
        self,
        *,
        snapshot: Mapping[str, Any],
        state_path: Path,
        state: dict[str, Any],
    ) -> None:
        request = snapshot["request"]
        root = Path(snapshot["root"])
        release_root = state_path.parent / "release"
        if release_root.exists():
            shutil.rmtree(release_root)
        publication = _json_object(root / "publication.json", label="publication provenance")
        manifest = prepare_release_bundle(
            source_bundle=load_policy_bundle(root / "source", source=str(root / "source")),
            replay_path=root / "capture/replay.mp4",
            capture=snapshot["capture"],
            evaluation_document=_json_object(
                root / "evidence/evaluation.json", label="publication evaluation"
            ),
            publication=publication,
            release_version=str(request["release_version"]),
            published_at=str(request["published_at"]),
            youtube_url=str(state["urls"]["youtube"]),
            output=release_root,
        )
        if manifest.get("publication", {}).get("request_fingerprint") != request[
            "request_fingerprint"
        ]:
            raise ValueError("final release manifest lost the publication request identity")
        state["phase"] = "release_built"
        state["release_hashes"] = {
            name: file_sha256(release_root / name)
            for name in sorted(HUGGINGFACE_RELEASE_FILES)
        }
        self._save_state(state_path, state)

    def _download_manifest(
        self,
        *,
        repo_id: str,
        revision: str,
        token: str,
    ) -> dict[str, Any] | None:
        try:
            path = self.hub_download(
                repo_id,
                "release_manifest.json",
                repo_type="model",
                revision=revision,
                token=token,
            )
        except Exception:
            return None
        return _json_object(Path(path), label="remote release manifest")

    def _advance_huggingface(
        self,
        *,
        job: Mapping[str, Any],
        store: JobStore,
        snapshot: Mapping[str, Any],
        state_path: Path,
        state: dict[str, Any],
    ) -> None:
        request = snapshot["request"]
        job_id = str(job["job_id"])
        api, token = self._hf_access(request)
        repo_id = str(request["repo_id"])
        release_root = state_path.parent / "release"
        phase = str(state["phase"])

        if phase == "release_built":
            self._checkpoint(store, job_id, request, state, kind="before_huggingface_commit")
            exists = True
            try:
                info = api.model_info(repo_id=repo_id)
            except HfHubHTTPError as exc:
                if _http_status(exc) != 404:
                    raise
                exists = False
                info = None
            if not exists:
                if request.get("parent_commit") is not None:
                    raise ValueError("admitted Hugging Face repository disappeared")
                try:
                    api.create_repo(
                        repo_id=repo_id,
                        repo_type="model",
                        private=False,
                        exist_ok=False,
                    )
                except Exception:
                    try:
                        api.model_info(repo_id=repo_id)
                    except Exception:
                        raise
                current_files = api.list_repo_files(repo_id, repo_type="model")
                if current_files:
                    raise ValueError("new Hugging Face repository is no longer empty")
                current_parent = None
            else:
                current_parent = str(getattr(info, "sha", "") or "") or None
                if current_parent != request.get("parent_commit"):
                    remote = self._download_manifest(
                        repo_id=repo_id,
                        revision="main",
                        token=token,
                    )
                    if remote and remote.get("publication", {}).get(
                        "request_fingerprint"
                    ) == request["request_fingerprint"]:
                        state["huggingface_commit"] = current_parent
                        state["phase"] = "huggingface_committed"
                        self._save_state(state_path, state)
                    else:
                        raise ValueError(
                            "Hugging Face main changed after publication admission"
                        )
            if state["phase"] == "release_built":
                existing_files = (
                    set(api.list_repo_files(repo_id, repo_type="model")) if exists else set()
                )
                operations: list[Any] = [
                    CommitOperationAdd(
                        path_in_repo=name,
                        path_or_fileobj=str(release_root / name),
                    )
                    for name in sorted(HUGGINGFACE_RELEASE_FILES)
                ]
                operations.extend(
                    CommitOperationDelete(path_in_repo=name)
                    for name in sorted(existing_files - HUGGINGFACE_RELEASE_FILES)
                )
                try:
                    result = api.create_commit(
                        repo_id=repo_id,
                        repo_type="model",
                        revision="main",
                        parent_commit=current_parent,
                        operations=operations,
                        commit_message=(
                            f"Publish GradLab checkpoint {request['release_version']}"
                        ),
                    )
                    commit = str(getattr(result, "oid", "") or "")
                except Exception:
                    try:
                        recovered = api.model_info(repo_id=repo_id)
                        recovered_commit = str(getattr(recovered, "sha", "") or "")
                    except Exception:
                        recovered_commit = ""
                    remote = (
                        self._download_manifest(
                            repo_id=repo_id,
                            revision=recovered_commit,
                            token=token,
                        )
                        if recovered_commit
                        else None
                    )
                    if not remote or remote.get("publication", {}).get(
                        "request_fingerprint"
                    ) != request["request_fingerprint"]:
                        raise
                    commit = recovered_commit
                if not commit:
                    raise RuntimeError("Hugging Face commit did not return an immutable id")
                state["huggingface_commit"] = commit
                state["phase"] = "huggingface_committed"
                state["urls"] = {
                    **dict(state.get("urls") or {}),
                    "huggingface": f"https://huggingface.co/{repo_id}",
                    "huggingface_commit": f"https://huggingface.co/{repo_id}/commit/{commit}",
                }
                self._save_state(state_path, state)
                self._checkpoint(store, job_id, request, state, kind="huggingface_committed")

        if state["phase"] == "huggingface_committed":
            commit = str(state["huggingface_commit"])
            refs = api.list_repo_refs(repo_id=repo_id, repo_type="model")
            tags = {
                str(tag.name): str(tag.target_commit)
                for tag in getattr(refs, "tags", ())
            }
            version = str(request["release_version"])
            if version in tags and tags[version] != commit:
                raise ValueError("Hugging Face release tag points to a different commit")
            if version not in tags:
                api.create_tag(
                    repo_id,
                    repo_type="model",
                    tag=version,
                    revision=commit,
                    exist_ok=False,
                )
            state["phase"] = "huggingface_tagged"
            self._save_state(state_path, state)
            self._checkpoint(store, job_id, request, state, kind="huggingface_tagged")

        if state["phase"] == "huggingface_tagged":
            manifest = _json_object(
                release_root / "release_manifest.json", label="release manifest"
            )
            title = f"{manifest['repository']['game_family']} Policies"
            matches = [
                item
                for item in api.list_collections(owner=request["principals"]["huggingface_namespace"], limit=100)
                if str(item.title) == title
            ]
            if len(matches) > 1:
                raise ValueError(f"multiple Hugging Face collections are titled {title!r}")
            collection = (
                matches[0]
                if matches
                else api.create_collection(
                    title,
                    namespace=str(request["principals"]["huggingface_namespace"]),
                    description="Published GradLab reinforcement-learning policies.",
                    private=False,
                    exists_ok=False,
                )
            )
            if bool(getattr(collection, "private", False)):
                raise ValueError("Hugging Face policy collection must be public")
            slug = str(collection.slug)
            api.add_collection_item(slug, repo_id, "model", exists_ok=True)
            state["huggingface_collection"] = slug
            state["phase"] = "huggingface_complete"
            state["urls"] = {
                **dict(state.get("urls") or {}),
                "huggingface_collection": f"https://huggingface.co/collections/{slug}",
            }
            self._save_state(state_path, state)
            self._checkpoint(store, job_id, request, state, kind="huggingface_complete")

    def _audit(
        self,
        *,
        repo_root: Path,
        snapshot: Mapping[str, Any],
        state: Mapping[str, Any],
    ) -> None:
        request = snapshot["request"]
        client = self._youtube_access(repo_root, request)
        video = client.video_metadata(str(state["youtube_video_id"]))
        validate_processed_video(
            video,
            channel_id=str(request["principals"]["youtube_channel_id"]),
            privacy=str(request["metadata"]["privacy"]),
            marker=str(request["marker"]),
            title=str(request["metadata"]["title"]),
            description=str(request["metadata"]["description"]),
        )
        api, token = self._hf_access(request)
        repo_id = str(request["repo_id"])
        version = str(request["release_version"])
        info = api.model_info(repo_id=repo_id, revision=version)
        if str(getattr(info, "sha", "")) != str(state["huggingface_commit"]):
            raise ValueError("Hugging Face tag changed before final audit")
        if set(api.list_repo_files(repo_id, revision=version, repo_type="model")) != set(
            HUGGINGFACE_RELEASE_FILES
        ):
            raise ValueError("Hugging Face release file set failed final audit")
        with tempfile.TemporaryDirectory(prefix="gradlab-player-publication-audit-") as name:
            root = Path(name)
            for filename in sorted(HUGGINGFACE_RELEASE_FILES):
                downloaded = self.hub_download(
                    repo_id,
                    filename,
                    repo_type="model",
                    revision=version,
                    token=token,
                )
                shutil.copyfile(downloaded, root / filename)
            manifest = validate_release_bundle(root)
        if manifest.get("publication", {}).get("request_fingerprint") != request[
            "request_fingerprint"
        ]:
            raise ValueError("remote release belongs to a different publication request")

    def advance(self, job: Mapping[str, Any]) -> HandlerResult:
        payload = self.validate_payload(job.get("payload") or {})
        snapshot = self._snapshot(payload)
        request = snapshot["request"]
        store = JobStore(payload["queue_root"])
        store.init()
        state_path, state = self._state(store, str(job["job_id"]))
        try:
            if job.get("cancel_requested"):
                raise PublicationCanceled("publication canceled before its next mutation")
            if state["phase"] == "admitted":
                state["phase"] = "validated"
                state["message"] = None
                self._save_state(state_path, state)
                self._checkpoint(
                    store,
                    str(job["job_id"]),
                    request,
                    state,
                    kind="snapshot_validated",
                )
            if str(state["phase"]).startswith("youtube") or state["phase"] == "validated":
                waiting = self._advance_youtube(
                    job=job,
                    store=store,
                    repo_root=Path(payload["repo_root"]),
                    snapshot=snapshot,
                    state_path=state_path,
                    state=state,
                )
                if waiting is not None:
                    return waiting
            if state["phase"] == "youtube_complete":
                self._build_release(snapshot=snapshot, state_path=state_path, state=state)
                self._checkpoint(
                    store,
                    str(job["job_id"]),
                    request,
                    state,
                    kind="release_built",
                )
            if state["phase"] in {
                "release_built",
                "huggingface_committed",
                "huggingface_tagged",
            }:
                self._advance_huggingface(
                    job=job,
                    store=store,
                    snapshot=snapshot,
                    state_path=state_path,
                    state=state,
                )
            if state["phase"] == "huggingface_complete":
                self._audit(
                    repo_root=Path(payload["repo_root"]),
                    snapshot=snapshot,
                    state=state,
                )
                state["phase"] = "succeeded"
                state["message"] = "YouTube and Hugging Face publication verified"
                self._save_state(state_path, state)
            if state["phase"] != "succeeded":
                raise RuntimeError(f"unsupported publication phase: {state['phase']}")
            return HandlerResult(
                state="succeeded",
                message=str(state["message"]),
                subjects=self._subjects(request, state, subject_state="succeeded"),
            )
        except PublicationCanceled as exc:
            state["phase"] = "canceled"
            state["message"] = str(exc)
            self._save_state(state_path, state)
            return HandlerResult(
                state="canceled",
                message=str(exc),
                subjects=self._subjects(request, state, subject_state="canceled"),
            )
        except YouTubeSubmissionUncertain as exc:
            state["phase"] = "youtube_uncertain"
            return self._block(request, state, state_path, str(exc))
        except YouTubePublicationError as exc:
            if exc.retryable:
                return self._retry(job, request, state, state_path, str(exc))
            return self._block(request, state, state_path, str(exc))
        except HfHubHTTPError as exc:
            status = _http_status(exc)
            if status == 429 or (status is not None and status >= 500):
                return self._retry(job, request, state, state_path, str(exc))
            return self._block(request, state, state_path, str(exc))
        except (OSError, TimeoutError) as exc:
            return self._retry(job, request, state, state_path, f"{type(exc).__name__}: {exc}")
        except (KeyError, TypeError, ValueError) as exc:
            return self._block(request, state, state_path, f"{type(exc).__name__}: {exc}")


def register_job_handler() -> None:
    register_handler(
        PLAYER_PUBLICATION_JOB_TYPE,
        PLAYER_PUBLICATION_JOB_VERSION,
        PlayerPublicationJobHandler,
        replace=True,
    )


__all__ = ["PlayerPublicationJobHandler", "PublicationCanceled", "register_job_handler"]
