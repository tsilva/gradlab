from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from typing import Any


GRAPHQL_URL = "https://api.github.com/graphql"
FEATURE_MARKER_PREFIX = "<!-- gradlab-featured:"


def _required_env(name: str) -> str:
    value = str(os.environ.get(name) or "").strip()
    if not value:
        raise ValueError(f"{name} is required")
    return value


def _json_request(
    url: str,
    *,
    token: str | None = None,
    payload: Mapping[str, Any] | None = None,
) -> Any:
    data = None if payload is None else json.dumps(dict(payload)).encode("utf-8")
    headers = {"Accept": "application/json", "User-Agent": "gradlab-research-flywheel"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def _graphql(token: str, query: str, variables: Mapping[str, Any]) -> dict[str, Any]:
    response = _json_request(
        GRAPHQL_URL,
        token=token,
        payload={"query": query, "variables": dict(variables)},
    )
    if not isinstance(response, Mapping) or response.get("errors"):
        raise RuntimeError(f"GitHub GraphQL failed: {(response or {}).get('errors')}")
    data = response.get("data")
    if not isinstance(data, Mapping):
        raise RuntimeError("GitHub GraphQL returned no data")
    return dict(data)


def featured_manifests(collection_slug: str) -> list[dict[str, Any]]:
    collection = _json_request(
        f"https://huggingface.co/api/collections/{urllib.parse.quote(collection_slug, safe='/')}"
    )
    if not isinstance(collection, Mapping):
        raise ValueError("Hugging Face collection response is malformed")
    manifests: list[dict[str, Any]] = []
    for item in collection.get("items") or ():
        if not isinstance(item, Mapping) or item.get("type") != "model":
            continue
        repo_id = str(item.get("id") or "")
        note_value = item.get("note")
        note = (
            str(note_value.get("text") or "")
            if isinstance(note_value, Mapping)
            else str(note_value or "")
        )
        marker = f"https://huggingface.co/{repo_id}/tree/"
        revision = note.split(marker, 1)[1].split()[0].rstrip(".,)") if marker in note else "main"
        manifest_url = (
            f"https://huggingface.co/{repo_id}/resolve/"
            f"{urllib.parse.quote(revision, safe='')}/release_manifest.json"
        )
        try:
            manifest = _json_request(manifest_url)
        except urllib.error.HTTPError:
            continue
        if (
            isinstance(manifest, Mapping)
            and manifest.get("format_version") == 3
            and manifest.get("repo_naming_schema") == 3
        ):
            manifests.append(dict(manifest))
    return sorted(
        manifests,
        key=lambda value: str((value.get("release") or {}).get("published_at") or ""),
        reverse=True,
    )


def discussion_body(manifests: list[dict[str, Any]]) -> str:
    lines = [
        "# GradLab Research Results",
        "",
        "This pinned index is generated from public immutable v3 research records in the Featured Research collection.",
        "",
        "| Task | Trainer / algorithm | Checkpoint | Evaluation | Model | Video |",
        "|---|---|---:|---|---|---|",
    ]
    for manifest in manifests:
        repository = manifest["repository"]
        release = manifest["release"]
        evaluation = manifest["evaluation"]
        repo_id = repository["repo_id"]
        version = release["version"]
        acceptance = evaluation.get("acceptance") or {}
        outcomes = acceptance.get("outcomes") or []
        result = "; ".join(
            f"{row.get('label')}: {row.get('value')} ({'pass' if row.get('passed') else 'fail'})"
            for row in outcomes
            if isinstance(row, Mapping)
        )
        lines.append(
            f"| `{repository['canonical_environment_id']}` / `{repository['goal_id']}` "
            f"| {repository['trainer']} {str(repository['algorithm']).upper()} "
            f"| {evaluation['checkpoint_step']} | {result} "
            f"| [model](https://huggingface.co/{repo_id}/tree/{version}) "
            f"| [video]({release.get('youtube_url')}) |"
        )
    lines.extend(
        [
            "",
            "Evaluation evidence and representative replay are distinct; each model link above resolves an immutable release tag.",
        ]
    )
    return "\n".join(lines) + "\n"


def update_github(manifests: list[dict[str, Any]]) -> None:
    token = _required_env("GITHUB_TOKEN")
    owner, repository = _required_env("GITHUB_REPOSITORY").split("/", 1)
    discussion_number = int(_required_env("RESEARCH_DISCUSSION_NUMBER"))
    update_category_id = _required_env("RESEARCH_UPDATE_CATEGORY_ID")
    query = """
      query($owner:String!,$name:String!,$number:Int!) {
        repository(owner:$owner,name:$name) {
          id
          discussion(number:$number) { id body }
          discussions(first:100) { nodes { body } }
        }
      }
    """
    data = _graphql(
        token,
        query,
        {"owner": owner, "name": repository, "number": discussion_number},
    )
    repo = data["repository"]
    if not repo or not repo.get("discussion"):
        raise ValueError("configured Research Results Discussion does not exist")
    _graphql(
        token,
        "mutation($id:ID!,$body:String!){updateDiscussion(input:{discussionId:$id,body:$body}){discussion{id}}}",
        {"id": repo["discussion"]["id"], "body": discussion_body(manifests)},
    )
    existing_bodies = [str(node.get("body") or "") for node in repo["discussions"]["nodes"]]
    for manifest in manifests:
        repo_id = manifest["repository"]["repo_id"]
        version = manifest["release"]["version"]
        marker = f"{FEATURE_MARKER_PREFIX}{repo_id}@{version} -->"
        if any(marker in body for body in existing_bodies):
            continue
        model_url = f"https://huggingface.co/{repo_id}/tree/{version}"
        body = (
            f"{marker}\n\nA newly featured GradLab research release is available.\n\n"
            f"- Model and evidence: {model_url}\n"
            f"- Representative replay: {manifest['release'].get('youtube_url')}\n"
        )
        title = (
            f"Research Update: {manifest['repository']['goal_id']} — "
            f"{manifest['repository']['trainer']} {str(manifest['repository']['algorithm']).upper()}"
        )
        _graphql(
            token,
            "mutation($repo:ID!,$category:ID!,$title:String!,$body:String!){createDiscussion(input:{repositoryId:$repo,categoryId:$category,title:$title,body:$body}){discussion{id}}}",
            {"repo": repo["id"], "category": update_category_id, "title": title, "body": body},
        )


def main() -> int:
    manifests = featured_manifests(_required_env("HF_FEATURED_COLLECTION_SLUG"))
    update_github(manifests)
    print(json.dumps({"featured_releases": len(manifests), "status": "updated"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
