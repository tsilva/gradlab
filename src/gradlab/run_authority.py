from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from gradlab.clock import (
    Clock,
    SystemClock,
    format_utc_datetime,
    parse_utc_datetime,
)
from gradlab.file_utils import atomic_write_bytes, atomic_write_json, file_sha256
from gradlab.json_utils import canonical_json_sha256
from gradlab.r2_store import (
    BucketConfig,
    ConditionalWriteConflict,
    R2Bucket,
    RunStorageConfig,
)
from gradlab.run_contracts import (
    CheckpointManifest,
    EarlyStopReceipt,
    EVAL_INVENTORY_SETTLED_STATUSES,
    EvalIntent,
    EvalResult,
    PromotionReceipt,
    RunManifest,
    TerminalReceipt,
    checkpoint_id,
)
from gradlab.policy_bundle import (
    RECIPE_FORMAT_VERSION,
    canonical_json_bytes,
    model_document_path,
    recipe_document_path,
    validate_recipe_document,
)
from gradlab.goal_variants import (
    GOAL_VARIANT_INDEX_SCHEMA_VERSION,
    GOAL_VARIANT_RUN_INDEX_SCHEMA_VERSION,
    goal_variant_scope_key,
    validate_goal_variant_descriptor,
)
from gradlab.recipe_variants import recipe_variant_id


LEASE_TTL_SECONDS = 60
LEASE_RENEW_SECONDS = 15
LEASE_MISSES_BEFORE_STOP = 2
MAX_RECIPE_DOCUMENT_BYTES = 2 * 1024 * 1024


class LeaseUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class Lease:
    run_id: str
    attempt_id: str
    holder_id: str
    generation: int
    acquired_at: str
    renewed_at: str
    expires_at: str
    etag: str

    @classmethod
    def from_document(cls, value: Mapping[str, Any], *, etag: str) -> Lease:
        return cls(
            run_id=str(value["run_id"]),
            attempt_id=str(value["attempt_id"]),
            holder_id=str(value["holder_id"]),
            generation=int(value["generation"]),
            acquired_at=str(value["acquired_at"]),
            renewed_at=str(value["renewed_at"]),
            expires_at=str(value["expires_at"]),
            etag=etag,
        )

    def document(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "holder_id": self.holder_id,
            "generation": self.generation,
            "acquired_at": self.acquired_at,
            "renewed_at": self.renewed_at,
            "expires_at": self.expires_at,
        }


class RunAuthority:
    def __init__(
        self,
        storage: RunStorageConfig,
        *,
        clock: Clock | None = None,
        bucket_factory: Callable[[BucketConfig], R2Bucket] = R2Bucket,
    ):
        self.storage = storage
        self.clock = clock or SystemClock()
        self.control = bucket_factory(storage.control)
        self.evaluation = bucket_factory(storage.evaluation)
        self.models = bucket_factory(storage.models)

    @staticmethod
    def run_prefix(run_id: str) -> str:
        return f"runs/{run_id}"

    def create_manifest(self, manifest: RunManifest) -> str:
        etag = self.control.put_json(
            f"{self.run_prefix(manifest.run_id)}/manifest.json",
            manifest.to_dict(),
            create_only=True,
        )
        self.create_attempt_manifest(manifest)
        self.project_goal_variant_best_effort(manifest)
        return etag

    def create_attempt_manifest(self, manifest: RunManifest) -> str:
        return self.control.put_json(
            f"{self.run_prefix(manifest.run_id)}/attempts/{manifest.attempt_id}/manifest.json",
            manifest.to_dict(),
            create_only=True,
        )

    def manifest(self, run_id: str) -> dict[str, Any] | None:
        return self.control.get_json_optional(f"{self.run_prefix(run_id)}/manifest.json")

    @staticmethod
    def recipe_document_key(recipe_sha256: str) -> str:
        digest = str(recipe_sha256 or "").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("recipe document key requires a lowercase SHA-256")
        return f"recipes/v2/sha256/{digest[:2]}/{digest}.json"

    def put_recipe_document(
        self,
        document: Mapping[str, Any],
        *,
        expected_sha256: str | None = None,
    ) -> str:
        validated = validate_recipe_document(document, source="control recipe document")
        if int(validated["format_version"]) != RECIPE_FORMAT_VERSION:
            raise ValueError("only recipe format v2 may be stored in the resolution catalog")
        payload = canonical_json_bytes(validated)
        if len(payload) > MAX_RECIPE_DOCUMENT_BYTES:
            raise ValueError(f"recipe document exceeds {MAX_RECIPE_DOCUMENT_BYTES} bytes")
        digest = hashlib.sha256(payload).hexdigest()
        if expected_sha256 is not None and digest != str(expected_sha256).strip().lower():
            raise ValueError("recipe document hash disagrees with the expected run hash")
        key = self.recipe_document_key(digest)
        try:
            self.control.put_bytes(
                key,
                payload,
                content_type="application/json",
                create_only=True,
                metadata={"sha256": digest},
            )
        except ConditionalWriteConflict:
            if self.control.get_bytes(key) != payload:
                raise ValueError("content-addressed recipe document conflicts with storage")
        if self.control.get_bytes(key) != payload:
            raise ValueError("recipe document read-back verification failed")
        return digest

    def recipe_document(self, recipe_sha256: str) -> dict[str, Any]:
        digest = str(recipe_sha256 or "").strip().lower()
        payload = self.control.get_bytes(self.recipe_document_key(digest))
        if len(payload) > MAX_RECIPE_DOCUMENT_BYTES:
            raise ValueError("stored recipe document exceeds the supported size")
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError("stored recipe document hash does not match its key")
        try:
            decoded = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("stored recipe document is not valid JSON") from exc
        if not isinstance(decoded, Mapping):
            raise ValueError("stored recipe document must be an object")
        validated = validate_recipe_document(decoded, source="stored recipe document")
        if int(validated["format_version"]) != RECIPE_FORMAT_VERSION:
            raise ValueError("stored resolution recipe must use format v2")
        return validated

    def recipe_document_optional(self, recipe_sha256: str) -> dict[str, Any] | None:
        key = self.recipe_document_key(recipe_sha256)
        if self.control.get_json_optional(key) is None:
            return None
        return self.recipe_document(recipe_sha256)

    def register_goal_variant(self, manifest: RunManifest) -> dict[str, Any]:
        if manifest.goal_variant is None:
            raise ValueError("run manifest has no goal variant descriptor")
        descriptor = validate_goal_variant_descriptor(manifest.goal_variant)
        scope_key = goal_variant_scope_key(goal_slug=manifest.goal_slug)
        descriptor_key = f"{scope_key}/descriptors/{descriptor['variant_id']}.json"
        try:
            self.control.put_json(descriptor_key, descriptor, create_only=True)
        except ConditionalWriteConflict:
            if self.control.get_json(descriptor_key) != descriptor:
                raise ValueError("immutable goal variant descriptor conflicts with storage")

        index_key = f"{scope_key}/index.json"
        for _attempt in range(8):
            current = self.control.get_json_optional(index_key)
            current_etag = (
                str(self.control.head(index_key)["etag"]) if current is not None else None
            )
            if current is None:
                entries: list[dict[str, Any]] = []
            else:
                if int(current.get("schema_version") or 0) != GOAL_VARIANT_INDEX_SCHEMA_VERSION:
                    raise ValueError("unsupported goal variant index schema")
                identity = current.get("scope")
                if not isinstance(identity, Mapping) or dict(identity) != {
                    "goal_slug": manifest.goal_slug,
                }:
                    raise ValueError("goal variant index scope mismatch")
                raw_entries = current.get("variants")
                if not isinstance(raw_entries, list):
                    raise ValueError("goal variant index variants must be a list")
                entries = [dict(item) for item in raw_entries if isinstance(item, Mapping)]
                if len(entries) != len(raw_entries):
                    raise ValueError("goal variant index contains an invalid entry")

            by_id = {str(item.get("variant_id") or ""): item for item in entries}
            existing = by_id.get(str(descriptor["variant_id"]))
            exact_resolution_run_id = (
                str(existing.get("exact_resolution_run_id") or "") if existing else ""
            )
            if not exact_resolution_run_id:
                try:
                    stored_recipe = self.recipe_document(manifest.recipe_sha256)
                except FileNotFoundError, KeyError, ValueError:
                    stored_recipe = None
                if stored_recipe is not None:
                    exact_resolution_run_id = manifest.run_id
            by_id[str(descriptor["variant_id"])] = {
                **descriptor,
                "descriptor_key": descriptor_key,
                "first_run_id": (
                    str(existing.get("first_run_id") or manifest.run_id)
                    if existing
                    else manifest.run_id
                ),
                **(
                    {"exact_resolution_run_id": exact_resolution_run_id}
                    if exact_resolution_run_id
                    else {}
                ),
            }
            document = {
                "schema_version": GOAL_VARIANT_INDEX_SCHEMA_VERSION,
                "scope": {"goal_slug": manifest.goal_slug},
                "variants": sorted(
                    by_id.values(),
                    key=lambda item: (
                        str(item.get("label") or "").casefold(),
                        str(item.get("variant_id") or ""),
                    ),
                ),
            }
            try:
                self.control.put_json(
                    index_key,
                    document,
                    create_only=current is None,
                    if_match=current_etag,
                )
                self.update_goal_variant_run(manifest, state="running")
                return document
            except ConditionalWriteConflict:
                continue
        raise ConditionalWriteConflict("goal variant index changed during every CAS attempt")

    def update_goal_variant_run(
        self,
        manifest: RunManifest,
        *,
        state: str,
        updated_at: str | None = None,
        metrics: Mapping[str, Any] | None = None,
        stop_reason: str | None = None,
        final_step: int | None = None,
        early_stop: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if manifest.goal_variant is None:
            raise ValueError("run manifest has no goal variant descriptor")
        descriptor = validate_goal_variant_descriptor(manifest.goal_variant)
        scope_key = goal_variant_scope_key(goal_slug=manifest.goal_slug)
        variant_id = str(descriptor["variant_id"])
        index_key = f"{scope_key}/runs/{variant_id}.json"
        for _attempt in range(8):
            current = self.control.get_json_optional(index_key)
            current_etag = (
                str(self.control.head(index_key)["etag"]) if current is not None else None
            )
            if current is None:
                entries: list[dict[str, Any]] = []
            else:
                if int(current.get("schema_version") or 0) != GOAL_VARIANT_RUN_INDEX_SCHEMA_VERSION:
                    raise ValueError("unsupported goal variant run index schema")
                if current.get("scope") != {
                    "goal_slug": manifest.goal_slug,
                    "variant_id": variant_id,
                }:
                    raise ValueError("goal variant run index scope mismatch")
                raw_entries = current.get("runs")
                if not isinstance(raw_entries, list):
                    raise ValueError("goal variant run index runs must be a list")
                entries = [dict(item) for item in raw_entries if isinstance(item, Mapping)]
                if len(entries) != len(raw_entries):
                    raise ValueError("goal variant run index contains an invalid entry")

            by_id = {str(item.get("run_id") or ""): item for item in entries}
            existing = by_id.get(manifest.run_id)
            normalized_metrics = {
                str(name): float(value)
                for name, value in dict(metrics or {}).items()
                if not isinstance(value, bool) and isinstance(value, int | float)
            }
            by_id[manifest.run_id] = {
                "run_id": manifest.run_id,
                "attempt_id": manifest.attempt_id,
                "name": str(manifest.wandb.get("display_name") or manifest.run_id),
                "state": (
                    str(existing.get("state") or state)
                    if (
                        existing
                        and existing.get("attempt_id") == manifest.attempt_id
                        and existing.get("stop_reason")
                        and state == "running"
                        and stop_reason is None
                    )
                    else str(state)
                ),
                "goal_slug": manifest.goal_slug,
                "recipe_slug": manifest.recipe_slug,
                "recipe_sha256": manifest.recipe_sha256,
                "recipe_overrides": list(manifest.recipe_overrides),
                "recipe_variant_id": recipe_variant_id(
                    recipe_slug=manifest.recipe_slug,
                    source_sha=manifest.source_sha,
                    recipe_overrides=manifest.recipe_overrides,
                ),
                "goal_contract_sha256": str(descriptor["goal_contract_sha256"]),
                "effective_goal_contract_sha256": str(descriptor["effective_goal_contract_sha256"]),
                "goal_variant_id": variant_id,
                "goal_variant_label": str(descriptor["label"]),
                "description": manifest.run_description,
                "seed": manifest.seed,
                "created_at": (
                    str(existing.get("created_at") or manifest.created_at)
                    if existing
                    else manifest.created_at
                ),
                "updated_at": (
                    str(existing.get("updated_at") or manifest.created_at)
                    if (
                        existing
                        and existing.get("attempt_id") == manifest.attempt_id
                        and existing.get("stop_reason")
                        and updated_at is None
                        and stop_reason is None
                    )
                    else str(updated_at or manifest.created_at)
                ),
                "url": str(manifest.wandb.get("url") or ""),
                "metrics": (
                    normalized_metrics
                    if metrics is not None
                    else dict(existing.get("metrics") or {})
                    if existing
                    else {}
                ),
                "stop_reason": (
                    str(stop_reason)
                    if stop_reason is not None
                    else str(existing.get("stop_reason") or "")
                    if existing and existing.get("attempt_id") == manifest.attempt_id
                    else ""
                ),
                "final_step": (
                    int(final_step)
                    if final_step is not None
                    else int(existing["final_step"])
                    if (
                        existing
                        and existing.get("attempt_id") == manifest.attempt_id
                        and existing.get("final_step") is not None
                    )
                    else None
                ),
                "early_stop": (
                    dict(early_stop)
                    if early_stop is not None
                    else dict(existing["early_stop"])
                    if (
                        existing
                        and existing.get("attempt_id") == manifest.attempt_id
                        and isinstance(existing.get("early_stop"), Mapping)
                    )
                    else None
                ),
            }
            document = {
                "schema_version": GOAL_VARIANT_RUN_INDEX_SCHEMA_VERSION,
                "scope": {
                    "goal_slug": manifest.goal_slug,
                    "variant_id": variant_id,
                },
                "runs": sorted(
                    by_id.values(),
                    key=lambda item: (
                        str(item.get("created_at") or ""),
                        str(item.get("run_id") or ""),
                    ),
                    reverse=True,
                ),
            }
            try:
                self.control.put_json(
                    index_key,
                    document,
                    create_only=current is None,
                    if_match=current_etag,
                )
                return document
            except ConditionalWriteConflict:
                continue
        raise ConditionalWriteConflict("goal variant run index changed during every CAS attempt")

    def update_goal_variant_run_best_effort(
        self,
        manifest: RunManifest,
        *,
        state: str,
        updated_at: str | None = None,
        metrics: Mapping[str, Any] | None = None,
        stop_reason: str | None = None,
        final_step: int | None = None,
        early_stop: Mapping[str, Any] | None = None,
    ) -> bool:
        if manifest.goal_variant is None:
            return False
        try:
            self.update_goal_variant_run(
                manifest,
                state=state,
                updated_at=updated_at,
                metrics=metrics,
                stop_reason=stop_reason,
                final_step=final_step,
                early_stop=early_stop,
            )
        except Exception:
            return False
        return True

    def record_goal_variant_projection(
        self,
        manifest: RunManifest,
        *,
        error: BaseException | None = None,
    ) -> None:
        self.control.put_json(
            f"{self.run_prefix(manifest.run_id)}/goal-variant-projection.json",
            {
                "schema_version": 1,
                "run_id": manifest.run_id,
                "variant_id": (
                    str(manifest.goal_variant.get("variant_id") or "")
                    if isinstance(manifest.goal_variant, Mapping)
                    else ""
                ),
                "status": "pending_rebuild" if error is not None else "indexed",
                "error_type": type(error).__name__ if error is not None else "",
                "updated_at": self.clock.utc_now(),
            },
            create_only=False,
        )

    def project_goal_variant_best_effort(self, manifest: RunManifest) -> bool:
        if manifest.goal_variant is None:
            return False
        try:
            self.register_goal_variant(manifest)
        except Exception as exc:
            try:
                self.record_goal_variant_projection(manifest, error=exc)
            except Exception:
                pass
            return False
        self.record_goal_variant_projection(manifest)
        return True

    def clear_goal_variant_catalog(self) -> dict[str, int]:
        catalog_keys = sorted(self.control.iter_keys("goal-variants/"))
        obsolete_projection_keys = sorted(
            key
            for key in self.control.iter_keys("runs/")
            if re.fullmatch(
                r"runs/gradlab-[0-9a-f]{32}/goal-variant-(?:projection|registration)\.json",
                key,
            )
        )
        for key in (*catalog_keys, *obsolete_projection_keys):
            self.control.delete(key, if_match=str(self.control.head(key)["etag"]))
        return {
            "catalog_objects": len(catalog_keys),
            "projection_receipts": len(obsolete_projection_keys),
        }

    def acquire_lease(
        self,
        *,
        run_id: str,
        attempt_id: str,
        holder_id: str,
        now: datetime | None = None,
    ) -> Lease:
        instant = (now or self.clock.utc_datetime()).astimezone(UTC)
        key = f"{self.run_prefix(run_id)}/writer-lease.json"
        current = self.control.get_json_optional(key)
        current_etag = str(self.control.head(key)["etag"]) if current is not None else None
        if current is not None and parse_utc_datetime(str(current["expires_at"])) > instant:
            if str(current["attempt_id"]) != attempt_id or str(current["holder_id"]) != holder_id:
                raise LeaseUnavailable(
                    f"writer lease is held by {current['attempt_id']}/{current['holder_id']}"
                )
        generation = int(current.get("generation") or 0) + 1 if current is not None else 1
        acquired = (
            str(current["acquired_at"])
            if current is not None
            and str(current["attempt_id"]) == attempt_id
            and str(current["holder_id"]) == holder_id
            else format_utc_datetime(instant)
        )
        renewed = format_utc_datetime(instant)
        document = {
            "schema_version": 1,
            "run_id": run_id,
            "attempt_id": attempt_id,
            "holder_id": holder_id,
            "generation": generation,
            "acquired_at": acquired,
            "renewed_at": renewed,
            "expires_at": (instant + timedelta(seconds=LEASE_TTL_SECONDS))
            .isoformat()
            .replace("+00:00", "Z"),
        }
        try:
            etag = self.control.put_json(
                key,
                document,
                create_only=current is None,
                if_match=current_etag,
            )
        except ConditionalWriteConflict as exc:
            raise LeaseUnavailable("writer lease changed while acquiring it") from exc
        return Lease.from_document(document, etag=etag)

    def renew_lease(self, lease: Lease, *, now: datetime | None = None) -> Lease:
        instant = (now or self.clock.utc_datetime()).astimezone(UTC)
        if parse_utc_datetime(lease.expires_at) <= instant:
            raise LeaseUnavailable("writer lease expired before renewal")
        key = f"{self.run_prefix(lease.run_id)}/writer-lease.json"
        document = {
            **lease.document(),
            "generation": lease.generation + 1,
            "renewed_at": format_utc_datetime(instant),
            "expires_at": (instant + timedelta(seconds=LEASE_TTL_SECONDS))
            .isoformat()
            .replace("+00:00", "Z"),
        }
        try:
            etag = self.control.put_json(
                key,
                document,
                create_only=False,
                if_match=lease.etag,
            )
        except ConditionalWriteConflict as exc:
            raise LeaseUnavailable("writer lease was lost during renewal") from exc
        return Lease.from_document(document, etag=etag)

    def release_lease(self, lease: Lease) -> None:
        key = f"{self.run_prefix(lease.run_id)}/writer-lease.json"
        try:
            self.control.delete(key, if_match=lease.etag)
        except ConditionalWriteConflict as exc:
            raise LeaseUnavailable("writer lease was lost before release") from exc

    def seal_metric_segment(
        self,
        *,
        run_id: str,
        attempt_id: str,
        events: Sequence[Mapping[str, Any]],
    ) -> tuple[str, str]:
        if not events:
            raise ValueError("cannot seal an empty metric segment")
        rows = [dict(event) for event in events]
        sequences = [int(row["event_seq"]) for row in rows]
        if sequences != sorted(set(sequences)):
            raise ValueError("metric segment event_seq values must be strictly increasing")
        payload = b"".join(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
            for row in rows
        )
        digest = hashlib.sha256(payload).hexdigest()
        key = (
            f"{self.run_prefix(run_id)}/attempts/{attempt_id}/metric-segments/"
            f"{sequences[0]:020d}-{sequences[-1]:020d}-{digest}.jsonl"
        )
        self.control.put_bytes(
            key,
            payload,
            content_type="application/x-ndjson",
            create_only=True,
            metadata={"sha256": digest},
        )
        return key, digest

    def archive_metric_journals(self, *, run_id: str) -> dict[str, Any]:
        active_prefix = f"{self.run_prefix(run_id)}/attempts"
        active_keys = sorted(
            key
            for key in self.control.iter_keys(active_prefix)
            if "/metric-segments/" in key and key.endswith(".jsonl")
        )
        archive_prefix = f"expiring-metric-journals/{run_id}/"
        archived_keys = sorted(
            key for key in self.control.iter_keys(archive_prefix) if key.endswith(".jsonl")
        )
        for source_key in active_keys:
            suffix = source_key.split(f"{active_prefix}/", 1)[1]
            attempt_id, remainder = suffix.split("/", 1)
            if not remainder.startswith("metric-segments/"):
                raise ValueError(f"invalid metric-journal key: {source_key}")
            destination_key = (
                f"expiring-metric-journals/{run_id}/{attempt_id}/"
                f"{remainder.removeprefix('metric-segments/')}"
            )
            source_etag = str(self.control.head(source_key)["etag"])
            self.control.copy_within(source_key, destination_key)
            self.control.delete(source_key, if_match=source_etag)
            if destination_key not in archived_keys:
                archived_keys.append(destination_key)
        archived_keys.sort()
        return {
            "prefix": archive_prefix,
            "segment_count": len(archived_keys),
            "keys": archived_keys,
        }

    @staticmethod
    def _archive_document_sha256(value: Mapping[str, Any]) -> str:
        return canonical_json_sha256(value)

    def state_archive_closure(self, *, run_id: str) -> dict[str, Any] | None:
        return self.control.get_json_optional(
            f"{self.run_prefix(run_id)}/state-archive/latest.json"
        )

    def publish_state_archive(
        self,
        *,
        run_id: str,
        attempt_id: str,
        archive_root: Path,
    ) -> dict[str, Any]:
        closure_path = archive_root / "closure.json"
        if not closure_path.is_file():
            raise FileNotFoundError(f"state archive has no closure: {closure_path}")
        closure = json.loads(closure_path.read_text(encoding="utf-8"))
        if (
            not isinstance(closure, Mapping)
            or closure.get("semantic_id") != "state-archive-v1"
            or int(closure.get("schema_version", 0)) != 1
        ):
            raise ValueError("state archive closure schema is unsupported")
        raw_files = closure.get("files")
        if isinstance(raw_files, str | bytes) or not isinstance(raw_files, Sequence):
            raise ValueError("state archive closure files must be a sequence")
        prior_objects: dict[str, str] = {}
        prior = self.state_archive_closure(run_id=run_id)
        if prior is not None:
            prior_generation = self.control.get_json(str(prior["generation_key"]))
            for raw_object in prior_generation.get("objects") or []:
                if isinstance(raw_object, Mapping):
                    prior_objects[str(raw_object["sha256"])] = str(raw_object["object_key"])
        objects: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        for raw_file in raw_files:
            if not isinstance(raw_file, Mapping):
                raise ValueError("state archive closure file entry must be an object")
            relative = Path(str(raw_file["path"]))
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or relative.as_posix() in seen_paths
            ):
                raise ValueError("state archive closure contains an unsafe or duplicate path")
            seen_paths.add(relative.as_posix())
            source = archive_root / relative
            payload = source.read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            if digest != str(raw_file["sha256"]) or len(payload) != int(raw_file["size_bytes"]):
                raise ValueError(f"state archive file failed closure verification: {relative}")
            object_key = prior_objects.get(digest) or (
                f"{self.run_prefix(run_id)}/state-archive/objects/{digest[:2]}/{digest[2:]}"
            )
            if digest not in prior_objects:
                self.control.put_bytes(
                    object_key,
                    payload,
                    create_only=True,
                    metadata={"sha256": digest},
                )
            objects.append(
                {
                    "path": relative.as_posix(),
                    "sha256": digest,
                    "size_bytes": len(payload),
                    "object_key": object_key,
                }
            )
        objects.sort(key=lambda row: str(row["path"]))
        generation = {
            "semantic_id": "state-archive-generation-v1",
            "schema_version": 1,
            "run_id": run_id,
            "attempt_id": attempt_id,
            "step": int(closure["step"]),
            "status": str(closure["status"]),
            "inventory_sha256": str(closure["inventory_sha256"]),
            "archive": dict(closure["archive"]),
            "closure": dict(closure),
            "objects": objects,
        }
        generation_sha256 = self._archive_document_sha256(generation)
        generation_key = (
            f"{self.run_prefix(run_id)}/state-archive/generations/"
            f"{int(closure['step']):020d}-{generation_sha256}.json"
        )
        self.control.put_json(generation_key, generation, create_only=True)
        latest = {
            "semantic_id": "state-archive-publication-v1",
            "schema_version": 1,
            "run_id": run_id,
            "attempt_id": attempt_id,
            "step": int(closure["step"]),
            "status": str(closure["status"]),
            "generation_key": generation_key,
            "generation_sha256": generation_sha256,
            "inventory_sha256": str(closure["inventory_sha256"]),
            "file_count": len(objects),
            "size_bytes": sum(int(row["size_bytes"]) for row in objects),
            "archive": dict(closure["archive"]),
        }
        self.control.put_json(
            f"{self.run_prefix(run_id)}/state-archive/latest.json",
            latest,
            create_only=False,
        )
        return latest

    def restore_state_archive(
        self,
        *,
        run_id: str,
        destination: Path,
    ) -> dict[str, Any] | None:
        latest = self.state_archive_closure(run_id=run_id)
        if latest is None:
            return None
        generation_key = str(latest["generation_key"])
        generation = self.control.get_json(generation_key)
        if self._archive_document_sha256(generation) != str(latest["generation_sha256"]):
            raise ValueError("state archive generation hash mismatch")
        objects = generation.get("objects")
        if isinstance(objects, str | bytes) or not isinstance(objects, Sequence):
            raise ValueError("state archive generation objects must be a sequence")
        destination.mkdir(parents=True, exist_ok=True)
        for raw_object in objects:
            if not isinstance(raw_object, Mapping):
                raise ValueError("state archive generation object must be a mapping")
            relative = Path(str(raw_object["path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("state archive generation contains an unsafe path")
            payload = self.control.get_bytes(str(raw_object["object_key"]))
            digest = hashlib.sha256(payload).hexdigest()
            if digest != str(raw_object["sha256"]) or len(payload) != int(raw_object["size_bytes"]):
                raise ValueError(f"state archive object failed verification: {relative}")
            target = destination / relative
            atomic_write_bytes(target, payload)
        closure = generation.get("closure")
        if not isinstance(closure, Mapping):
            raise ValueError("state archive generation has no closure")
        atomic_write_json(destination / "closure.json", dict(closure))
        return dict(latest)

    def publish_checkpoint(
        self,
        *,
        run_id: str,
        model_path: Path,
        step: int,
        purpose: str,
        contract_hashes: Mapping[str, str],
        recovery_sidecar: Mapping[str, Any],
        created_at: str | None = None,
    ) -> CheckpointManifest:
        digest = file_sha256(model_path)
        model_sidecar = model_document_path(model_path)
        recipe_sidecar = recipe_document_path(model_path)
        if not model_sidecar.is_file() or not recipe_sidecar.is_file():
            raise ValueError("checkpoint is missing its immutable model.json or recipe.json")
        model_sidecar_digest = file_sha256(model_sidecar)
        recipe_sidecar_digest = file_sha256(recipe_sidecar)
        identifier = checkpoint_id(step=step, sha256=digest)
        public_prefix = f"{self.run_prefix(run_id)}/checkpoints/{int(step)}-{digest}"
        model_key = f"{public_prefix}/model.zip"
        model_document_key = f"{public_prefix}/model.json"
        recipe_document_key = f"{public_prefix}/recipe.json"
        manifest_key = f"{public_prefix}/manifest.json"
        sidecar_key = f"{self.run_prefix(run_id)}/checkpoints/{identifier}/recovery-sidecar.json"
        self.control.put_json(sidecar_key, recovery_sidecar, create_only=True)
        self.models.put_file(
            model_key,
            model_path,
            sha256=digest,
            content_type="application/zip",
            cache_control="public, max-age=31536000, immutable",
        )
        self.models.put_file(
            model_document_key,
            model_sidecar,
            sha256=model_sidecar_digest,
            content_type="application/json",
            cache_control="public, max-age=31536000, immutable",
        )
        self.models.put_file(
            recipe_document_key,
            recipe_sidecar,
            sha256=recipe_sidecar_digest,
            content_type="application/json",
            cache_control="public, max-age=31536000, immutable",
        )
        manifest = CheckpointManifest(
            run_id=run_id,
            checkpoint_id=identifier,
            step=int(step),
            purpose=str(purpose),  # type: ignore[arg-type]
            sha256=digest,
            size_bytes=model_path.stat().st_size,
            public_url=self.models.public_url(model_key),
            model_document_url=self.models.public_url(model_document_key),
            model_document_sha256=model_sidecar_digest,
            recipe_document_url=self.models.public_url(recipe_document_key),
            recipe_document_sha256=recipe_sidecar_digest,
            goal_sha256=str(contract_hashes["goal_sha256"]),
            recipe_sha256=str(contract_hashes["recipe_sha256"]),
            environment_sha256=str(contract_hashes["environment_sha256"]),
            evaluation_contract_sha256=str(contract_hashes["evaluation_contract_sha256"]),
            recovery_sidecar_key=sidecar_key,
            created_at=str(created_at or self.clock.utc_now()),
        )
        self.models.put_json(
            manifest_key,
            manifest.to_dict(),
            create_only=True,
            cache_control="public, max-age=31536000, immutable",
        )
        self._upsert_public_index(manifest)
        return manifest

    def _update_public_index(
        self,
        run_id: str,
        *,
        checkpoint: CheckpointManifest | None = None,
        promotion: PromotionReceipt | None = None,
    ) -> None:
        key = f"{self.run_prefix(run_id)}/index.json"
        for _attempt in range(8):
            current = self.models.get_json_optional(key)
            etag = str(self.models.head(key)["etag"]) if current is not None else None
            rows = list((current or {}).get("checkpoints") or [])
            if checkpoint is not None:
                if checkpoint.run_id != run_id:
                    raise ValueError("checkpoint does not belong to the public run index")
                if not any(
                    str(row.get("checkpoint_id") or "") == checkpoint.checkpoint_id
                    for row in rows
                    if isinstance(row, Mapping)
                ):
                    rows.append(checkpoint.to_dict())
            rows.sort(key=lambda row: (int(row["step"]), str(row["sha256"])))
            promoted = (current or {}).get("promotion")
            if promotion is not None:
                if promotion.run_id != run_id:
                    raise ValueError("promotion does not belong to the public run index")
                if not any(
                    str(row.get("checkpoint_id") or "") == promotion.checkpoint_id
                    for row in rows
                    if isinstance(row, Mapping)
                ):
                    raise ValueError("promoted checkpoint is missing from the public run index")
                promoted = {
                    "checkpoint_id": promotion.checkpoint_id,
                    "checkpoint_step": promotion.checkpoint_step,
                    "eval_result_sha256": promotion.eval_result_sha256,
                    "accepted_episode_count": promotion.accepted_episode_count,
                    "promoted_at": promotion.promoted_at,
                }
            document = {
                "schema_version": 1,
                "run_id": run_id,
                "updated_at": self.clock.utc_now(),
                "checkpoints": rows,
                "promotion": promoted,
            }
            try:
                self.models.put_json(
                    key,
                    document,
                    create_only=current is None,
                    if_match=etag,
                    cache_control="no-store",
                )
                return
            except ConditionalWriteConflict:
                continue
        raise RuntimeError("public run index CAS did not converge")

    def _upsert_public_index(self, checkpoint: CheckpointManifest) -> None:
        self._update_public_index(checkpoint.run_id, checkpoint=checkpoint)

    def put_eval_intent(self, intent: EvalIntent) -> str:
        return self.evaluation.put_json(
            f"{self.run_prefix(intent.run_id)}/evals/{intent.idempotency_key}/intent.json",
            intent.to_dict(),
            create_only=True,
        )

    def eval_intent(
        self,
        *,
        run_id: str,
        idempotency_key: str,
    ) -> dict[str, Any] | None:
        return self.evaluation.get_json_optional(
            f"{self.run_prefix(run_id)}/evals/{idempotency_key}/intent.json"
        )

    def put_eval_dispatch(
        self,
        *,
        run_id: str,
        idempotency_key: str,
        attempt: int,
        modal_call_id: str,
    ) -> str:
        return self.evaluation.put_json(
            f"{self.run_prefix(run_id)}/evals/{idempotency_key}/dispatch-{int(attempt)}.json",
            {
                "schema_version": 1,
                "run_id": run_id,
                "idempotency_key": idempotency_key,
                "attempt": int(attempt),
                "modal_call_id": modal_call_id,
                "dispatched_at": self.clock.utc_now(),
            },
            create_only=True,
        )

    def prepare_eval_attempt(
        self,
        *,
        run_id: str,
        idempotency_key: str,
        attempt: int,
        expires_at: float,
    ) -> dict[str, Any]:
        key = f"{self.run_prefix(run_id)}/evals/{idempotency_key}/attempt-{int(attempt)}.json"
        document = {
            "schema_version": 1,
            "run_id": run_id,
            "idempotency_key": idempotency_key,
            "attempt": int(attempt),
            "expires_at": float(expires_at),
        }
        self.evaluation.put_json(key, document, create_only=True)
        return document

    def eval_attempt(
        self,
        *,
        run_id: str,
        idempotency_key: str,
        attempt: int,
    ) -> dict[str, Any] | None:
        return self.evaluation.get_json_optional(
            f"{self.run_prefix(run_id)}/evals/{idempotency_key}/attempt-{int(attempt)}.json"
        )

    def eval_dispatch(
        self,
        *,
        run_id: str,
        idempotency_key: str,
        attempt: int,
    ) -> dict[str, Any] | None:
        return self.evaluation.get_json_optional(
            f"{self.run_prefix(run_id)}/evals/{idempotency_key}/dispatch-{int(attempt)}.json"
        )

    def eval_result(
        self,
        *,
        run_id: str,
        idempotency_key: str,
    ) -> dict[str, Any] | None:
        return self.evaluation.get_json_optional(
            f"{self.run_prefix(run_id)}/evals/{idempotency_key}/result.json"
        )

    def put_verified_eval_result(self, result: EvalResult) -> str:
        return self.evaluation.put_json(
            f"{self.run_prefix(result.run_id)}/evals/{result.idempotency_key}/verified-result.json",
            result.to_dict(),
            create_only=True,
        )

    def create_promotion(self, receipt: PromotionReceipt) -> str:
        etag = self.control.put_json(
            f"{self.run_prefix(receipt.run_id)}/promotion.json",
            receipt.to_dict(),
            create_only=True,
        )
        self._update_public_index(receipt.run_id, promotion=receipt)
        return etag

    def early_stop_receipt(
        self,
        *,
        run_id: str,
        attempt_id: str,
    ) -> dict[str, Any] | None:
        return self.control.get_json_optional(
            f"{self.run_prefix(run_id)}/attempts/{attempt_id}/early-stop.json"
        )

    def create_early_stop(self, receipt: EarlyStopReceipt) -> str:
        key = f"{self.run_prefix(receipt.run_id)}/attempts/{receipt.attempt_id}/early-stop.json"
        document = receipt.to_dict()
        existing = self.control.get_json_optional(key)
        if existing is not None:
            validated = EarlyStopReceipt(**existing)
            validated.validate()
            if validated.to_dict() != document:
                raise ConditionalWriteConflict(
                    f"early-stop receipt already exists with different content: {key}"
                )
            return "existing"
        return self.control.put_json(key, document, create_only=True)

    def create_terminal(self, receipt: TerminalReceipt) -> str:
        receipt.validate()
        if receipt.state != "succeeded":
            raise ValueError("canonical terminal receipt is reserved for scientific success")
        if receipt.acceptance_required is not True:
            raise ValueError("canonical terminal receipt requires acceptance-backed evaluation")
        drain = dict(receipt.drain)
        if drain.get("complete") is not True:
            raise ValueError("successful terminal receipt requires a complete drain")
        if int(receipt.wandb_high_water_mark) <= 0:
            raise ValueError("successful terminal receipt requires W&B metric delivery")
        if int(drain.get("metric_segment_high_water") or 0) != int(receipt.wandb_high_water_mark):
            raise ValueError("R2 and W&B delivery high-water marks do not match")
        if int(drain.get("wandb_remote_high_water_mark") or 0) < int(receipt.wandb_high_water_mark):
            raise ValueError("W&B delivery is not remotely visible")
        capacity_ratio = drain.get("publication_capacity_ratio")
        if capacity_ratio is not None and float(capacity_ratio) < 2.0:
            raise ValueError("W&B publication capacity is below twice peak ingress")
        journal_archive = drain.get("journal_archive")
        if (
            not isinstance(journal_archive, Mapping)
            or int(journal_archive.get("segment_count") or 0) <= 0
            or not str(drain.get("journal_expires_at") or "")
        ):
            raise ValueError("delivered metric journals are not scheduled for expiry")

        checkpoints = [dict(row) for row in receipt.checkpoint_inventory]
        evals = [dict(row) for row in receipt.eval_inventory]
        if not checkpoints or not evals:
            raise ValueError("successful terminal receipt requires checkpoint and eval inventory")
        checkpoint_ids = {str(row.get("checkpoint_id") or "") for row in checkpoints}
        eval_checkpoint_ids = {str(row.get("checkpoint_id") or "") for row in evals}
        if "" in checkpoint_ids or len(checkpoints) != len(checkpoint_ids):
            raise ValueError("checkpoint inventory contains missing or duplicate identities")
        if "" in eval_checkpoint_ids or not eval_checkpoint_ids.issubset(checkpoint_ids):
            raise ValueError("eval inventory references a checkpoint outside the run inventory")
        if len(evals) != len(eval_checkpoint_ids):
            raise ValueError("eval inventory contains duplicate checkpoint entries")
        if any(
            str(row.get("status") or "") not in EVAL_INVENTORY_SETTLED_STATUSES for row in evals
        ):
            raise ValueError("eval inventory contains an unsettled evaluation")
        if not any(str(row.get("purpose") or "") == "final" for row in checkpoints):
            raise ValueError("successful terminal receipt requires a final checkpoint")
        maximum_step = max(int(row.get("step") or 0) for row in checkpoints)
        if int(receipt.final_step) != maximum_step:
            raise ValueError("terminal final_step does not match checkpoint inventory")
        accepted = [row for row in evals if str(row.get("status") or "") == "accepted"]
        if not accepted:
            raise ValueError("successful terminal receipt requires an accepted evaluation")
        selected = min(
            accepted,
            key=lambda row: (
                int(row.get("checkpoint_step") or 0),
                str(row.get("checkpoint_id") or ""),
            ),
        )
        promotion = self.control.get_json_optional(
            f"{self.run_prefix(receipt.run_id)}/promotion.json"
        )
        if (
            promotion is None
            or str(promotion.get("checkpoint_id") or "") != str(selected.get("checkpoint_id") or "")
            or int(promotion.get("checkpoint_step") or -1)
            != int(selected.get("checkpoint_step") or 0)
        ):
            raise ValueError("promotion is not the lowest-step accepted checkpoint")
        etag = self.control.put_json(
            f"{self.run_prefix(receipt.run_id)}/terminal.json",
            receipt.to_dict(),
            create_only=True,
        )
        self._update_run_index_from_terminal(receipt, metrics=None)
        return etag

    def create_attempt_terminal(
        self,
        receipt: TerminalReceipt,
        *,
        metrics: Mapping[str, Any] | None = None,
    ) -> str:
        etag = self.control.put_json(
            f"{self.run_prefix(receipt.run_id)}/attempts/{receipt.attempt_id}/terminal.json",
            receipt.to_dict(),
            create_only=True,
        )
        self._update_run_index_from_terminal(receipt, metrics=metrics)
        return etag

    def _update_run_index_from_terminal(
        self,
        receipt: TerminalReceipt,
        *,
        metrics: Mapping[str, Any] | None,
    ) -> None:
        document = self.control.get_json_optional(
            f"{self.run_prefix(receipt.run_id)}/attempts/{receipt.attempt_id}/manifest.json"
        )
        if document is None:
            document = self.manifest(receipt.run_id)
        if document is None:
            return
        try:
            manifest = RunManifest(**document)
            manifest.validate()
        except TypeError, ValueError:
            return
        if manifest.attempt_id != receipt.attempt_id:
            return
        self.update_goal_variant_run_best_effort(
            manifest,
            state=receipt.state,
            updated_at=receipt.completed_at,
            metrics=metrics,
            stop_reason=receipt.stop_reason,
            final_step=receipt.final_step,
            early_stop=receipt.early_stop,
        )

    def has_accepted_eval(self, run_id: str) -> bool:
        prefix = f"{self.run_prefix(run_id)}/evals"
        for key in self.evaluation.iter_keys(prefix):
            if not key.endswith("/verified-result.json"):
                continue
            if str(self.evaluation.get_json(key).get("status") or "") == "accepted":
                return True
        return False

    def semantic_state(self, run_id: str) -> dict[str, Any]:
        prefix = self.run_prefix(run_id)
        manifest = self.control.get_json_optional(f"{prefix}/manifest.json")
        terminal = self.control.get_json_optional(f"{prefix}/terminal.json")
        promotion = self.control.get_json_optional(f"{prefix}/promotion.json")
        public_index = self.models.get_json_optional(f"{prefix}/index.json")
        eval_keys = list(self.evaluation.iter_keys(f"{prefix}/evals"))
        control_keys = list(self.control.iter_keys(f"{prefix}/attempts"))
        attempt_manifests = [
            self.control.get_json(key) for key in control_keys if key.endswith("/manifest.json")
        ]
        attempt_terminals = [
            self.control.get_json(key) for key in control_keys if key.endswith("/terminal.json")
        ]
        attempt_manifests.sort(key=lambda row: str(row.get("created_at") or ""))
        attempt_terminals.sort(key=lambda row: str(row.get("completed_at") or ""))
        return {
            "run_id": run_id,
            "manifest": manifest,
            "terminal": terminal,
            "promotion": promotion,
            "public_index": public_index,
            "eval_intents": sum(key.endswith("/intent.json") for key in eval_keys),
            "eval_results": sum(key.endswith("/result.json") for key in eval_keys),
            "verified_eval_results": sum(
                key.endswith("/verified-result.json") for key in eval_keys
            ),
            "attempts": attempt_manifests,
            "attempt_terminals": attempt_terminals,
            "observed_at": self.clock.time(),
        }
