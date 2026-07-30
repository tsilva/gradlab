from __future__ import annotations

import argparse
import json
import re
import shutil
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias
from urllib.parse import unquote, urlparse

from gradlab.file_utils import file_sha256
from gradlab.policy_bundle import (
    CHECKPOINT_FILENAME,
    MODEL_FILENAME,
    RECIPE_FILENAME,
    PolicyBundle,
    load_policy_bundle,
    load_policy_bundle_from_checkpoint,
)
from gradlab.local_paths import default_runs_dir
from gradlab.r2_store import public_object_request
from gradlab.run_contracts import CheckpointManifest, RUN_ID_PATTERN


HUGGINGFACE_MODEL_SCHEME = "hf://"
HUGGINGFACE_MODEL_URL_HOST = "huggingface.co"
_SHA256 = re.compile(r"[0-9a-f]{64}")
PUBLIC_CHECKPOINT_MANIFEST = re.compile(
    r"/runs/(gradlab-[0-9a-f]{32})/checkpoints/(\d+)-([0-9a-f]{64})/manifest\.json$"
)
DEFAULT_PUBLIC_MODELS_BASE_URL = (
    "https://pub-fc35c0b186ce4aad8eea5e93d38c99db.r2.dev"
)
ModelSourceKind: TypeAlias = Literal["manifest", "huggingface", "local", "public_run"]


def _safe_stem(value: str, fallback: str = "model") -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-._")
    return stem or fallback


@dataclass
class ResolvedModelSource:
    model_path: Path
    bundle: PolicyBundle
    artifact_ref: str | None = None
    artifact_name: str | None = None
    checkpoint_step: int | None = None
    run_config: dict[str, Any] = field(default_factory=dict)


def is_huggingface_model_ref(value: str) -> bool:
    text = str(value or "").strip()
    if text.startswith(HUGGINGFACE_MODEL_SCHEME):
        return True
    parsed = urlparse(text)
    return parsed.scheme in {"http", "https"} and parsed.netloc == HUGGINGFACE_MODEL_URL_HOST


def is_public_checkpoint_manifest_ref(value: str) -> bool:
    parsed = urlparse(str(value or "").strip())
    return bool(
        parsed.scheme in {"https", "file"}
        and not parsed.params
        and not parsed.query
        and not parsed.fragment
        and PUBLIC_CHECKPOINT_MANIFEST.search(parsed.path)
    )


def positional_model_source_arg(value: str) -> str:
    if is_huggingface_model_ref(value) or is_public_checkpoint_manifest_ref(value):
        return value
    raise argparse.ArgumentTypeError(
        "expected hf://owner/repo or an immutable public checkpoint manifest URL"
    )


def add_model_source_args(
    parser: argparse.ArgumentParser,
    *,
    positional_artifact: bool = False,
    allow_multiple_artifacts: bool = False,
    model_default: str | None = None,
    model_help: str | None = None,
) -> None:
    del allow_multiple_artifacts
    if positional_artifact:
        parser.add_argument(
            "model_ref",
            nargs="?",
            type=positional_model_source_arg,
            help="Immutable public checkpoint manifest or Hugging Face model ref.",
        )
    model_kwargs: dict[str, Any] = {}
    if model_default is not None:
        model_kwargs["default"] = model_default
    if model_help is not None:
        model_kwargs["help"] = model_help
    parser.add_argument("--model", **model_kwargs)
    parser.add_argument("--hf-revision", help="Hugging Face revision. Defaults to main.")
    parser.add_argument("--hf-model-root", default=str(default_runs_dir() / "hf_models"))
    parser.add_argument(
        "--public-model-root",
        default=str(default_runs_dir() / "public_models"),
    )


def model_source_ref(args: argparse.Namespace) -> str | None:
    for name in ("model_ref", "artifact_ref", "model"):
        value = str(getattr(args, name, "") or "").strip()
        if is_huggingface_model_ref(value) or is_public_checkpoint_manifest_ref(value):
            return value
    return None


def _public_json(url: str, *, max_bytes: int = 2 * 1024 * 1024) -> dict[str, Any]:
    with urllib.request.urlopen(
        public_object_request(url),
        timeout=30,
    ) as response:  # noqa: S310
        payload = response.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ValueError(f"public JSON document is larger than {max_bytes} bytes")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"public document must be a JSON object: {url}")
    return value


def _download_public_file(
    url: str,
    target: Path,
    *,
    expected_sha256: str,
    expected_size: int | None = None,
) -> Path:
    digest = str(expected_sha256).strip().lower()
    if _SHA256.fullmatch(digest) is None:
        raise ValueError("public file manifest has an invalid SHA-256")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.partial")
    try:
        with urllib.request.urlopen(
            public_object_request(url),
            timeout=60,
        ) as response:  # noqa: S310
            with temporary.open("wb") as destination:
                shutil.copyfileobj(response, destination, length=1024 * 1024)
        if expected_size is not None and temporary.stat().st_size != int(expected_size):
            raise ValueError(f"public file size mismatch: {url}")
        if file_sha256(temporary) != digest:
            raise ValueError(f"public file SHA-256 mismatch: {url}")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def download_public_checkpoint_manifest_source(
    manifest_url: str,
    *,
    root: Path,
) -> ResolvedModelSource:
    text = str(manifest_url).strip()
    if not is_public_checkpoint_manifest_ref(text):
        raise ValueError("public checkpoint source must be an immutable manifest URL")
    manifest = _public_json(text)
    match = PUBLIC_CHECKPOINT_MANIFEST.search(urlparse(text).path)
    assert match is not None
    run_id, step, path_sha256 = match.groups()
    if (
        str(manifest.get("run_id") or "") != run_id
        or int(manifest.get("step") or -1) != int(step)
        or str(manifest.get("sha256") or "") != path_sha256
    ):
        raise ValueError("public checkpoint manifest does not match its immutable URL")
    checkpoint_id = str(manifest.get("checkpoint_id") or "")
    target_dir = root / _safe_stem(f"{run_id}-{checkpoint_id}")
    model_path = _download_public_file(
        str(manifest["public_url"]),
        target_dir / CHECKPOINT_FILENAME,
        expected_sha256=path_sha256,
        expected_size=int(manifest["size_bytes"]),
    )
    _download_public_file(
        str(manifest["model_document_url"]),
        target_dir / MODEL_FILENAME,
        expected_sha256=str(manifest["model_document_sha256"]),
    )
    _download_public_file(
        str(manifest["recipe_document_url"]),
        target_dir / RECIPE_FILENAME,
        expected_sha256=str(manifest["recipe_document_sha256"]),
    )
    bundle = load_policy_bundle(
        target_dir,
        source=text,
        revision=path_sha256,
    )
    return ResolvedModelSource(
        model_path=model_path,
        artifact_name=text,
        checkpoint_step=int(step),
        bundle=bundle,
    )


def public_run_checkpoint_manifest_url(
    run_id: str,
    *,
    public_base_url: str = DEFAULT_PUBLIC_MODELS_BASE_URL,
) -> str:
    value = str(run_id).strip()
    if RUN_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("run id must match gradlab-<32 lowercase hex>")
    base = str(public_base_url).strip().rstrip("/")
    index = _public_json(f"{base}/runs/{value}/index.json")
    if str(index.get("run_id") or "") != value:
        raise ValueError("public run index identity mismatch")
    checkpoints: list[CheckpointManifest] = []
    for raw in index.get("checkpoints") or []:
        if not isinstance(raw, Mapping):
            raise ValueError("public run index contains an invalid checkpoint")
        manifest = CheckpointManifest.from_dict(raw)
        if manifest.run_id != value:
            raise ValueError("checkpoint does not belong to the public run index")
        checkpoints.append(manifest)
    promotion = index.get("promotion")
    if promotion is not None:
        if not isinstance(promotion, Mapping):
            raise ValueError("public run promotion is malformed")
        checkpoint_id = str(promotion.get("checkpoint_id") or "")
        selected = [row for row in checkpoints if row.checkpoint_id == checkpoint_id]
        if len(selected) != 1:
            raise ValueError("public promotion does not identify exactly one checkpoint")
        checkpoint = selected[0]
    else:
        finals = [row for row in checkpoints if row.purpose == "final"]
        if not finals:
            raise ValueError(f"run {value} has no promoted or final checkpoint")
        checkpoint = max(finals, key=lambda row: (row.step, row.sha256))
    model_url = checkpoint.public_url
    if not model_url.endswith("/model.zip"):
        raise ValueError("public checkpoint model URL is malformed")
    return f"{model_url.removesuffix('/model.zip')}/manifest.json"


def download_public_run_source(
    run_id: str,
    *,
    root: Path,
    public_base_url: str = DEFAULT_PUBLIC_MODELS_BASE_URL,
) -> ResolvedModelSource:
    return download_public_checkpoint_manifest_source(
        public_run_checkpoint_manifest_url(
            run_id,
            public_base_url=public_base_url,
        ),
        root=root,
    )


def parse_huggingface_model_ref(value: str) -> tuple[str, str | None]:
    text = str(value or "").strip()
    if text.startswith(HUGGINGFACE_MODEL_SCHEME):
        parts = [
            unquote(part)
            for part in text.removeprefix(HUGGINGFACE_MODEL_SCHEME).strip("/").split("/")
            if part
        ]
        if len(parts) != 2:
            raise ValueError(
                f"expected Hugging Face model ref like hf://owner/repo, got {value!r}"
            )
        repo_name, separator, revision = parts[1].partition("@")
        if separator and not revision:
            raise ValueError(f"Hugging Face model ref has an empty revision: {value!r}")
        return f"{parts[0]}/{repo_name}", revision or None
    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"} or parsed.netloc != HUGGINGFACE_MODEL_URL_HOST:
        raise ValueError(f"expected Hugging Face model ref, got {value!r}")
    parts = [unquote(part) for part in parsed.path.split("/") if part]
    if len(parts) != 2:
        raise ValueError(f"expected Hugging Face model URL with owner/repo, got {value!r}")
    return "/".join(parts), None


def _download_huggingface_release_closure(
    *,
    repo_id: str,
    revision: str,
    repo_files: set[str],
    target_dir: Path,
    hf_hub_download: Any,
) -> None:
    if "release_manifest.json" not in repo_files:
        return
    path = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            filename="release_manifest.json",
            local_dir=target_dir,
        )
    )
    if path.stat().st_size > 8 * 1024**2:
        raise ValueError("Hugging Face release manifest exceeds 8 MiB")
    document = json.loads(path.read_text(encoding="utf-8"))
    artifacts = document.get("artifacts") if isinstance(document, Mapping) else None
    if not isinstance(artifacts, Mapping):
        raise ValueError("Hugging Face release manifest has no artifact closure")
    for name in artifacts:
        filename = str(name)
        if Path(filename).name != filename or filename not in repo_files:
            raise ValueError(f"release manifest binds unavailable file {filename!r}")
        hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            filename=filename,
            local_dir=target_dir,
        )


def download_huggingface_model_source(
    ref: str,
    *,
    root: Path,
    revision: str | None = None,
) -> ResolvedModelSource:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise SystemExit("huggingface-hub is required for hf:// model refs") from exc
    try:
        repo_id, parsed_revision = parse_huggingface_model_ref(ref)
        requested_revision = revision or parsed_revision or "main"
        api = HfApi()
        immutable_revision = str(
            api.model_info(repo_id=repo_id, revision=requested_revision).sha or ""
        )
        if not immutable_revision:
            raise ValueError("model repository did not return an immutable commit SHA")
        repo_files = set(
            api.list_repo_files(
                repo_id=repo_id,
                repo_type="model",
                revision=immutable_revision,
            )
        )
    except Exception as exc:
        raise SystemExit(f"Could not inspect Hugging Face model {ref}: {exc}") from exc
    target_dir = root / _safe_stem(f"{repo_id}@{immutable_revision}")
    target_dir.mkdir(parents=True, exist_ok=True)
    for bundle_filename in (CHECKPOINT_FILENAME, MODEL_FILENAME, RECIPE_FILENAME):
        if bundle_filename not in repo_files:
            raise SystemExit(
                f"Hugging Face bundle {repo_id}@{immutable_revision} is missing "
                f"{bundle_filename}"
            )
        hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            revision=immutable_revision,
            filename=bundle_filename,
            local_dir=target_dir,
        )
    _download_huggingface_release_closure(
        repo_id=repo_id,
        revision=immutable_revision,
        repo_files=repo_files,
        target_dir=target_dir,
        hf_hub_download=hf_hub_download,
    )
    bundle = load_policy_bundle(
        target_dir,
        source=f"hf://{repo_id}",
        revision=immutable_revision,
    )
    return ResolvedModelSource(
        model_path=bundle.checkpoint_path,
        artifact_name=f"hf://{repo_id}@{immutable_revision}",
        checkpoint_step=bundle.model["checkpoint"].get("step"),
        bundle=bundle,
    )


def _download_model_ref(
    ref: str,
    *,
    public_root: Path,
    hf_root: Path,
    revision: str | None = None,
) -> ResolvedModelSource:
    text = str(ref).strip()
    if is_public_checkpoint_manifest_ref(text):
        return download_public_checkpoint_manifest_source(text, root=public_root)
    if is_huggingface_model_ref(text):
        return download_huggingface_model_source(
            text,
            root=hf_root,
            revision=revision,
        )
    raise ValueError(
        "remote model source must be an immutable public checkpoint manifest "
        "or Hugging Face model"
    )


def download_remote_model_source(
    ref: str,
    *,
    root: Path,
    require_pinned: bool = False,
) -> ResolvedModelSource:
    text = str(ref).strip()
    resolved = _download_model_ref(text, public_root=root, hf_root=root)
    pinned = str(resolved.artifact_name or "")
    if not pinned:
        raise ValueError("remote model source did not resolve an immutable locator")
    if require_pinned and pinned != text:
        raise ValueError(f"remote model locator is not immutable: expected {pinned!r}")
    return resolved


def resolve_model_source(
    kind: ModelSourceKind,
    value: str,
    *,
    public_root: Path,
    hf_root: Path,
    revision: str | None = None,
    public_base_url: str = DEFAULT_PUBLIC_MODELS_BASE_URL,
) -> ResolvedModelSource:
    text = str(value).strip()
    if kind == "public_run":
        return download_public_run_source(
            text,
            root=public_root,
            public_base_url=public_base_url,
        )
    if kind == "manifest":
        resolved = download_public_checkpoint_manifest_source(text, root=public_root)
        resolved.artifact_ref = text
        return resolved
    if kind == "huggingface":
        resolved = download_huggingface_model_source(
            text,
            root=hf_root,
            revision=revision,
        )
        resolved.artifact_ref = text
        return resolved
    if kind != "local":
        raise ValueError(f"unsupported model source kind: {kind}")
    model_path = Path(text).expanduser()
    if not model_path.is_file():
        raise FileNotFoundError(f"local model checkpoint not found: {model_path}")
    bundle = load_policy_bundle_from_checkpoint(model_path)
    if bundle is None:
        raise FileNotFoundError(
            f"{model_path} is missing its current policy bundle beginning at "
            f"{model_path.with_name(MODEL_FILENAME)}"
        )
    return ResolvedModelSource(
        model_path=model_path,
        bundle=bundle,
    )


def resolve_single_model_source(
    args: argparse.Namespace,
    *,
    resolved_ref: str | None = None,
) -> ResolvedModelSource:
    ref = resolved_ref if resolved_ref is not None else model_source_ref(args)
    if ref:
        kind: ModelSourceKind = (
            "manifest" if is_public_checkpoint_manifest_ref(ref) else "huggingface"
        )
        return resolve_model_source(
            kind,
            ref,
            public_root=Path(
                getattr(args, "public_model_root", default_runs_dir() / "public_models")
            ).expanduser(),
            hf_root=Path(
                getattr(args, "hf_model_root", default_runs_dir() / "hf_models")
            ).expanduser(),
            revision=getattr(args, "hf_revision", None),
        )
    return resolve_model_source(
        "local",
        str(args.model),
        public_root=Path(
            getattr(args, "public_model_root", default_runs_dir() / "public_models")
        ).expanduser(),
        hf_root=Path(
            getattr(args, "hf_model_root", default_runs_dir() / "hf_models")
        ).expanduser(),
    )
