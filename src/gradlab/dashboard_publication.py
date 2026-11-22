"""Publish the shared game dashboard for a verified Hugging Face release."""

from __future__ import annotations

import csv
import io
import json
import re
from collections.abc import Callable, Mapping
from html import unescape
from pathlib import Path
from statistics import mean, median
from typing import Any

import yaml
from huggingface_hub import CommitOperationAdd, HfApi, SpaceInfo
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

from gradlab.eval_metrics import episode_is_complete
from gradlab.game_dashboard import dashboard_files
from gradlab.publication import (
    dashboard_repo_id,
    release_recipe_filename,
    validate_huggingface_environment_collection,
)
from gradlab.recipe_catalog import experiments_root


_MARIO = "SuperMarioBros-Nes-v0"
_MAX_FILE_BYTES = 2 * 1024 * 1024
_CLEAR_RATE = "stage_clear_rate"
_MARIO_LEVELS = tuple(f"{world}-{stage}" for world in range(1, 9) for stage in range(1, 5))


def _catalog_stages(environment: str) -> list[dict[str, str]]:
    if environment == _MARIO:
        return [{"id": f"Level{level}", "label": f"World {level}"} for level in _MARIO_LEVELS]
    root = experiments_root() / "goals"
    catalog = yaml.safe_load((root / "_catalog.yaml").read_text(encoding="utf-8"))
    stages = {}
    for name, entry in catalog["namespaces"].items():
        if entry["environment_id"] != environment:
            continue
        for path in sorted((root / name).rglob("_goal.yaml")):
            authored = yaml.safe_load(path.read_text(encoding="utf-8"))
            goal_id = str(authored.get("goal_id") or path.parent.name)
            label = str(authored.get("title") or goal_id)
            stages[goal_id] = {"id": goal_id, "label": label}
    return list(stages.values())


def _new_dashboard(
    environment: str,
    provider: str,
    collection_url: str,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    primary = evidence["acceptance"]["outcomes"][0]
    clear_rate = environment == _MARIO
    return {
        "version": 1,
        "environment": environment,
        "provider": provider,
        "provider_url": None,
        "collection_url": collection_url,
        "metric_key": _CLEAR_RATE if clear_rate else primary["metric"],
        "metric_name": "Stage clear rate" if clear_rate else primary["label"],
        "metric_column": "Clears" if clear_rate else primary["label"],
        "metric_direction": "ascending" if primary["operator"] in ("<", "<=") and not clear_rate else "descending",
        "methods": (
            "Results use verified evaluations of the linked checkpoints. "
            "Training steps are measured at the published checkpoint."
            + (" A stage is solved after one evaluation clear. Clear Steps averages successful episodes within each evaluation seed." if clear_rate else "")
        ),
        "clear_steps": clear_rate,
        "stages": _catalog_stages(environment),
    }


def _csv_table(source: str) -> tuple[list[str], list[dict[str, str]]]:
    reader = csv.DictReader(io.StringIO(source, newline=""))
    fields = reader.fieldnames or []
    rows = list(reader)
    if (
        not {"checkpoint_sha256", "model_url"}.issubset(fields)
        or len(set(fields)) != len(fields)
        or any(None in row or None in row.values() for row in rows)
    ):
        raise ValueError("dashboard CSV needs unique columns and checkpoint identity")
    return fields, rows


def _legacy_mario_dashboard(html: str, rows: list[dict[str, str]], collection_url: str) -> dict[str, Any]:
    # The first Mario Space stored metrics in a CSV and stage metadata in the results table.
    # Only migrate that known schema, checking the two files agree on order and model links.
    matches = re.findall(r'<tr\b([^<>]*>.*?)<th scope="row">World ([1-8]-[1-4])</th>(.*?)</tr>', html)
    if tuple(level for _, level, _ in matches) != _MARIO_LEVELS or tuple(row.get("level") for row in rows) != _MARIO_LEVELS:
        raise ValueError("legacy Mario dashboard stage order could not be verified")
    stages: list[dict[str, Any]] = []
    for (prefix, level, cells), row in zip(matches, rows, strict=True):
        solved = row["status"] == "Solved"
        status = "Solved" if solved else "Unsolved"
        if row["status"] != status or f'aria-label="{status}"' not in prefix:
            raise ValueError(f"legacy dashboard status differs for World {level}")
        stage: dict[str, Any] = {"id": f"Level{level}", "label": f"World {level}", "solved": solved}
        stages.append(stage)
        if row["model_url"]:
            algorithm = re.match(r"<td>([^<>]+)</td>", cells)
            model_card = row["model_url"].replace("/tree/", "/blob/") + "/README.md"
            if algorithm is None or model_card != row["model_card_url"] or f'href="{model_card}"' not in cells:
                raise ValueError(f"legacy dashboard model differs for World {level}")
            stage.update({
                "algorithm": unescape(algorithm[1]),
                "checkpoint_sha256": row["checkpoint_sha256"],
                "model_url": row["model_url"],
                "replay_url": row["replay_url"],
                "training_steps": int(row["training_steps"]),
                "evaluation_seeds": int(row["evaluation_seeds"]),
                "metric": {
                    "value": int(row["total_clears"]) / int(row["total_episodes"]),
                    "display": f'{row["total_clears"]} / {row["total_episodes"]}',
                    "detail": f'{float(row["clear_rate_percent"]):.1f}% ± {float(row["clear_rate_sample_stddev_pp"]):.1f} pp',
                },
            })
            recipe = re.search(
                r'href="(https://github\.com/tsilva/gradlab/blob/[a-f0-9]{40}/experiments/[^"<>]+)"[^>]*>Recipe</a>',
                cells,
            )
            if recipe:
                stage["current_recipe_url"] = unescape(recipe[1])
            if row["clear_steps_mean"]:
                steps = float(row["clear_steps_mean"])
                stage["clear_steps"] = {
                    "value": steps, "display": f"{steps:,.1f}",
                    "detail": f'± {float(row["clear_steps_sample_stddev"]):,.1f} steps',
                }
    methods = re.search(r'<p id="results-method"[^>]*>([^<>]+)</p>', html)
    if methods is None:
        raise ValueError("legacy Mario dashboard evaluation methods are missing")
    return {
        "version": 1,
        "environment": _MARIO,
        "provider": "env-supermariobrosnes-turbo-emu",
        "provider_url": "https://github.com/tsilva/env-SuperMarioBrosNes-turbo-emu",
        "collection_url": collection_url,
        "metric_name": "Stage clear rate",
        "metric_column": "Clears",
        "metric_key": _CLEAR_RATE,
        "metric_direction": "descending",
        "methods": unescape(methods[1]),
        "clear_steps": True,
        "stages": stages,
    }


def _model_repo(stage: Mapping[str, Any]) -> str:
    return str(stage.get("model_url") or "").removeprefix("https://huggingface.co/").split("/tree/", 1)[0]


def _select_stage(data: dict[str, Any], goal_id: str, repo_id: str) -> dict[str, Any]:
    stages = data["stages"]
    stage_ids = {goal_id}
    if data["environment"] == _MARIO and goal_id.removeprefix("Level") in _MARIO_LEVELS:
        stage_ids.add(goal_id.removeprefix("Level"))
    matches = [stage for stage in stages if stage["id"] in stage_ids or _model_repo(stage) == repo_id]
    if len(matches) > 1 or (matches and _model_repo(matches[0]) not in ("", repo_id)):
        raise ValueError("dashboard already selects a different policy for the release stage")
    if matches:
        return matches[0]
    catalog = _catalog_stages(data["environment"])
    candidate = next((stage for stage in catalog if stage["id"] == goal_id), None)
    if candidate is None:
        candidate = {"id": goal_id, "label": goal_id}
        stages.append(candidate)
        return candidate
    order = {stage["id"]: index for index, stage in enumerate(catalog)}
    index = next(
        (index for index, stage in enumerate(stages) if order.get(stage["id"], -1) > order[goal_id]),
        len(stages),
    )
    stages.insert(index, candidate)
    return candidate


def _release_row(
    data: Mapping[str, Any],
    stage: dict[str, Any],
    manifest: Mapping[str, Any],
    recipe: Mapping[str, Any],
    evidence: Mapping[str, Any],
    commit: str,
) -> tuple[bool, dict[str, Any]]:
    repo_id = manifest["repository"]["repo_id"]
    base = f"https://huggingface.co/{repo_id}"
    checkpoint = evidence["identity"]
    sha = checkpoint["checkpoint_sha256"]
    same_checkpoint = stage.get("checkpoint_sha256") == sha
    algorithm = manifest["lineage"]["algorithm"].upper()
    backend = recipe["recipe"]["train_config"].get("training_backend") or {}
    sil = (backend.get("config") or {}).get("sil") or {}
    if algorithm == "PPO" and sil.get("enabled") is True:
        algorithm = "PPO + SIL"
    stage.update({
        "algorithm": algorithm,
        "training_steps": checkpoint["checkpoint_step"],
        "checkpoint_sha256": sha,
        "model_url": f"{base}/tree/{commit}",
        "replay_url": f"{base}/resolve/{commit}/replay.mp4",
        "recipe_url": f"{base}/blob/{commit}/{release_recipe_filename(manifest.get('artifacts', {}))}",
    })
    episodes = evidence["episode_results"]
    clear_steps = [row["steps"] for row in episodes if episode_is_complete(row)]
    clears = len(clear_steps)
    total = len(episodes)
    if not total:
        raise ValueError("dashboard release evaluation has no verified episodes")
    seed = evidence["protocol"].get("seed")
    if not same_checkpoint:
        for name in ("evaluation_seeds", "metric", "clear_steps", "solved"):
            stage.pop(name, None)
        if type(seed) is int:
            stage["evaluation_seeds"] = 1
        metric_key = data.get("metric_key")
        if metric_key == _CLEAR_RATE or (
            metric_key is None and data["environment"] == _MARIO and data["metric_column"] == "Clears"
        ):
            stage["metric"] = {
                "value": clears / total,
                "display": f"{clears} / {total}",
                "detail": f"{clears / total:.1%}",
            }
            stage["solved"] = clears > 0
        else:
            metric = next(
                (value for value in evidence["acceptance"]["outcomes"] if value["metric"] == metric_key),
                None,
            )
            if metric:
                value = float(metric["value"])
                display = f"{value:.1%}" if metric["unit"] == "fraction" else f"{value:,.4g}"
                stage["metric"] = {"value": value, "display": display}
                stage["solved"] = evidence["acceptance"]["passed"] is True
        if clear_steps and data.get("clear_steps"):
            average = mean(clear_steps)
            stage["clear_steps"] = {"value": average, "display": f"{average:,.1f}"}
    failures = [row for row in episodes if not episode_is_complete(row)]
    life_loss = [row for row in failures if row.get("died") or "life_loss" in row.get("events", ())]
    stalled = [row for row in failures if "stalled" in row.get("events", ())]
    summary = {
        "total_clears": clears,
        "total_episodes": total,
        "clear_rate_percent": 100 * clears / total,
        "evaluation_seeds": 1 if type(seed) is int else "",
        "median_steps_to_clear": median(clear_steps) if clear_steps else "",
        "clear_steps_mean": mean(clear_steps) if clear_steps else "",
        "failed_episodes": len(failures),
        "failure_life_loss": len(life_loss),
        "failure_stalled": len(stalled),
        f"seed_{seed}_clears": clears,
        f"seed_{seed}_episodes": total,
    }
    return same_checkpoint, summary


def _refresh_csv(
    fields: list[str],
    rows: list[dict[str, str]],
    previous: Mapping[str, Any],
    stage: Mapping[str, Any],
    same_checkpoint: bool,
    evaluation: Mapping[str, Any],
    commit: str,
) -> str:
    level = stage["id"].removeprefix("Level")
    matches = [
        row for row in rows
        if (previous.get("model_url") and row["model_url"] == previous["model_url"])
        or row.get("stage") == stage["id"]
        or row.get("level") == level
    ]
    if len(matches) != 1 or matches[0]["checkpoint_sha256"] != (previous.get("checkpoint_sha256") or ""):
        raise ValueError("dashboard CSV cannot identify the release stage and prior checkpoint")
    row = matches[0]
    if not same_checkpoint:
        for key in fields:
            if key not in {"stage", "level", "label"}:
                row[key] = evaluation.get(key, "")
    values = {
        "status": {True: "Solved", False: "Unsolved"}.get(stage.get("solved"), ""),
        "checkpoint_status": "Published",
        "model_url": stage["model_url"],
        "model_card_url": stage["model_url"].replace("/tree/", "/blob/") + "/README.md",
        "replay_url": stage["replay_url"],
        "model_revision": commit,
        "checkpoint_sha256": stage["checkpoint_sha256"],
        "training_steps": stage["training_steps"],
        "algorithm": stage["algorithm"],
        "recipe_url": stage.get("current_recipe_url", stage["recipe_url"]),
    }
    row.update({key: values[key] for key in fields if key in values})
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _space_info(api: HfApi, repo_id: str, *, token: bool | None = None) -> SpaceInfo:
    space = api.space_info(repo_id, token=token, files_metadata=True)
    if space.id != repo_id or space.sdk != "static" or space.private or space.gated:
        raise ValueError(f"{repo_id} must be the exact public Static Space")
    if not re.fullmatch(r"[0-9a-f]{40}", space.sha or ""):
        raise ValueError(f"invalid dashboard commit identity for {repo_id}")
    return space


def _read_files(download: Callable[..., str], repo_id: str, space: SpaceInfo, names: set[str], token: str | bool) -> dict[str, str]:
    files = {}
    for sibling in space.siblings:
        name = sibling.rfilename
        if name not in names:
            continue
        if sibling.size is None or sibling.size > _MAX_FILE_BYTES:
            raise ValueError(f"dashboard file exceeds the allowed size: {name}")
        path = Path(download(repo_id, name, repo_type="space", revision=space.sha, token=token))
        with path.open("rb") as handle:
            content = handle.read(_MAX_FILE_BYTES + 1)
        if len(content) != sibling.size:
            raise ValueError(f"dashboard file size verification failed: {name}")
        files[name] = content.decode("utf-8")
    return files


def _pin_in_collection(api: HfApi, slug: str, owner: str, repo_id: str) -> None:
    def read():
        collection = api.get_collection(slug, token=False)
        validate_huggingface_environment_collection(collection, owner=owner)
        matches = [item for item in collection.items if item.item_type == "space" and item.item_id == repo_id]
        if len(matches) > 1:
            raise ValueError("environment collection contains duplicate dashboards")
        return collection.items, matches

    items, matches = read()
    if not matches:
        api.add_collection_item(slug, repo_id, "space", exists_ok=True)
        items, matches = read()
    if not matches:
        raise ValueError("dashboard was not added to the public collection")
    if items[0] != matches[0]:
        api.update_collection_item(slug, matches[0].item_object_id, position=0)
        items, matches = read()
    if len(matches) != 1 or items[0] != matches[0]:
        raise ValueError("dashboard is not first in the public environment collection")


def publish_game_dashboard(
    api: HfApi,
    download: Callable[..., str],
    *,
    token: str,
    collection_slug: str,
    manifest: Mapping[str, Any],
    evidence: Mapping[str, Any],
    recipe: Mapping[str, Any],
    commit: str,
) -> str:
    repository = manifest["repository"]
    owner = repository["repo_id"].split("/", 1)[0]
    environment = repository["canonical_environment_id"]
    repo_id = dashboard_repo_id(owner, environment)
    collection_url = f"https://huggingface.co/collections/{collection_slug}"
    provider = manifest["replay"]["execution"]["provider_id"]
    try:
        space = _space_info(api, repo_id, token=False)
    except HfHubHTTPError as exc:
        # The public endpoint also reports absent repositories as 401. Creation
        # uses the admitted principal and is followed by exact public identity checks.
        if exc.response.status_code != 404 and not (
            isinstance(exc, RepositoryNotFoundError) and exc.response.status_code == 401
        ):
            raise
        space = None
    current = (
        _read_files(download, repo_id, space, {"index.html", "dashboard.json", "results.csv", "README.md"}, token)
        if space else {}
    )
    csv_text = current.get("results.csv")
    fields, rows = _csv_table(csv_text) if csv_text is not None else ([], [])
    if "dashboard.json" in current:
        data = json.loads(current["dashboard.json"])
    elif "index.html" in current:
        if environment != _MARIO or not fields:
            raise ValueError("existing dashboard needs reviewed data before automated updates")
        data = _legacy_mario_dashboard(current["index.html"], rows, collection_url)
    else:
        filenames = {sibling.rfilename for sibling in space.siblings} if space else set()
        if filenames - {".gitattributes", "README.md"}:
            raise ValueError("existing Space has unrecognized dashboard files")
        data = _new_dashboard(environment, provider, collection_url, evidence)
    if data["environment"] != environment or data["provider"] != provider:
        raise ValueError("existing dashboard environment or provider differs from the accepted release")
    data["collection_url"] = collection_url
    stage = _select_stage(data, repository["goal_id"], repository["repo_id"])
    previous = dict(stage)
    same_checkpoint, evaluation = _release_row(data, stage, manifest, recipe, evidence, commit)
    if environment == _MARIO and stage.get("evaluation_seeds") == 1 and "base seed" not in data["methods"]:
        data["methods"] += " A single base seed does not provide a sample standard deviation across seeds."
    desired = dashboard_files(data)
    if fields:
        desired["results.csv"] = _refresh_csv(fields, rows, previous, stage, same_checkpoint, evaluation, commit)
    if space is None:
        api.create_repo(repo_id, repo_type="space", space_sdk="static", private=False, exist_ok=True)
        space = _space_info(api, repo_id)
        if {sibling.rfilename for sibling in space.siblings} - {".gitattributes", "README.md"}:
            raise ValueError("dashboard Space changed during creation; retry after reviewing it")
        current.update(_read_files(download, repo_id, space, {"README.md"}, token))
    if "README.md" in current:
        frontmatter = current["README.md"].split("---", 2)
        config = yaml.safe_load(frontmatter[1]) if len(frontmatter) == 3 else None
        if not isinstance(config, Mapping) or config.get("app_file", "index.html") != "index.html":
            raise ValueError("existing dashboard Space must serve index.html")
    else:
        desired["README.md"] = f"---\ntitle: {environment} Dashboard\nsdk: static\napp_file: index.html\n---\n"
    changes = [
        CommitOperationAdd(path_in_repo=name, path_or_fileobj=content.encode("utf-8"))
        for name, content in desired.items()
        if current.get(name) != content
    ]
    sha = space.sha
    if changes:
        receipt = api.create_commit(
            repo_id, repo_type="space", operations=changes, revision="main", parent_commit=sha,
            commit_message=f"Update dashboard for {repository['goal_id']} {manifest['release']['version']}",
        )
        sha = receipt.oid
    public = _space_info(api, repo_id, token=False)
    verified = _read_files(download, repo_id, public, set(desired), False)
    if public.sha != sha or verified != desired:
        raise ValueError("public dashboard changed before its files could be verified")
    _pin_in_collection(api, collection_slug, owner, repo_id)
    return f"https://huggingface.co/spaces/{repo_id}"
