from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

from gradlab.cli_parser import ExactArgumentParser
from gradlab.json_utils import canonical_json_sha256
from gradlab.metric_names import METRICS_SCHEMA_VERSION
from gradlab.wandb_utils import load_wandb_env, wandb_entity_from_env
from gradlab.wandb_workspace_declarations import (
    WandbWorkspaceSpec,
    WorkspacePanelSpec,
    compile_workspace_specs,
)


WorkspaceLoader = Callable[[str], Any]
WorkspaceSaver = Callable[[Any], Any]


def _metric_template_pattern(template: str) -> str:
    cursor = 0
    parts: list[str] = []
    for match in re.finditer(r"\{[a-z_]+\}", template):
        parts.append(re.escape(template[cursor : match.start()]))
        parts.append(r"[A-Za-z0-9_.-]+")
        cursor = match.end()
    parts.append(re.escape(template[cursor:]))
    return "".join(parts)


def _metric_selectors_regex(
    metrics: Sequence[str],
    templates: Sequence[str],
) -> str:
    patterns = [re.escape(metric) for metric in metrics]
    patterns.extend(_metric_template_pattern(template) for template in templates)
    if len(patterns) == 1:
        return f"^{patterns[0]}$"
    return "^(?:" + "|".join(patterns) + ")$"


_WORKSPACE_GRID_WIDTH = 24


def _line_panel(wr, panel: WorkspacePanelSpec, *, x: int, y: int):
    return wr.LinePlot(
        title=panel.title,
        x=panel.x,
        # Regex-backed selectors keep intentionally declared panels visible even
        # before a project's first matching run logs the metric.
        y=[],
        title_y=panel.y_title,
        metric_regex=_metric_selectors_regex(panel.y, panel.metric_templates),
        smoothing_type="none",
        layout=wr.Layout(x=x, y=y, w=panel.width, h=panel.height),
    )


def _section_panels(wr, section):
    panels = []
    slot_width = _WORKSPACE_GRID_WIDTH // section.columns
    row_y = 0
    for row_start in range(0, len(section.panels), section.columns):
        row = section.panels[row_start : row_start + section.columns]
        for column, panel in enumerate(row):
            if panel.kind == "line":
                panels.append(
                    _line_panel(
                        wr,
                        panel,
                        x=column * slot_width,
                        y=row_y,
                    )
                )
            else:
                raise AssertionError(f"unhandled workspace panel kind: {panel.kind}")
        row_y += max(panel.height for panel in row)
    return panels


def _run_scope_filter(run_scope: str) -> str:
    if run_scope == "current_metrics_schema":
        return f"Config('metrics_schema_version') = {METRICS_SCHEMA_VERSION}"
    raise AssertionError(f"unhandled workspace run scope: {run_scope}")


def build_wandb_workspace(spec: WandbWorkspaceSpec, *, entity: str):
    import wandb_workspaces.reports.v2 as wr
    import wandb_workspaces.workspaces as ws

    sections = []
    for section in spec.sections:
        panels = _section_panels(wr, section)
        sections.append(
            ws.Section(
                name=section.title,
                panels=panels,
                is_open=section.is_open,
                pinned=section.pinned,
                layout_settings=ws.SectionLayoutSettings(
                    columns=section.columns,
                    rows=math.ceil(len(panels) / section.columns),
                ),
                panel_settings=ws.SectionPanelSettings(
                    x_axis="train/global_step",
                    smoothing_type="none",
                    smoothing_weight=0,
                ),
            )
        )
    return ws.Workspace(
        entity=entity,
        project=spec.project,
        name=spec.display_name,
        sections=sections,
        settings=ws.WorkspaceSettings(
            x_axis="train/global_step",
            smoothing_type="none",
            smoothing_weight=0,
            sort_panels_alphabetically=False,
            group_by_prefix="first",
            max_runs=spec.max_runs,
        ),
        runset_settings=ws.RunsetSettings(
            filters=_run_scope_filter(spec.run_scope),
            pinned_columns=[
                "run:displayName",
                "run:state",
                "config:goal_slug",
                "config:recipe_slug",
                "config:seed",
                "config:algorithm_id",
            ],
        ),
        auto_generate_panels=False,
    )


def _normalized_workspace(value: Any) -> Any:
    if hasattr(value, "_to_model"):
        value = value._to_model().model_dump(by_alias=True, exclude_none=True)
    elif hasattr(value, "model_dump"):
        value = value.model_dump(by_alias=True, exclude_none=True)
    if isinstance(value, Mapping):
        return {
            str(key): _normalized_workspace(nested)
            for key, nested in sorted(value.items(), key=lambda item: str(item[0]))
            if key != "id" and not str(key).startswith("_")
        }
    if isinstance(value, tuple | list):
        return [_normalized_workspace(item) for item in value]
    return value


def workspace_structure_sha256(workspace: Any) -> str:
    return canonical_json_sha256(
        _normalized_workspace(workspace),
        default=str,
        ensure_ascii=True,
    )


def _managed_view_token(spec: WandbWorkspaceSpec, *, entity: str) -> str:
    digest = hashlib.sha256(
        f"{entity}\0{spec.project}\0{spec.identity}".encode("utf-8")
    ).hexdigest()
    return digest[:11]


def _managed_internal_name(spec: WandbWorkspaceSpec, *, entity: str) -> str:
    return f"nw-{_managed_view_token(spec, entity=entity)}-v"


def _managed_workspace_url(
    spec: WandbWorkspaceSpec,
    *,
    entity: str,
    app_url: str,
) -> str:
    token = _managed_view_token(spec, entity=entity)
    return (
        f"{app_url.rstrip('/')}/{quote(entity, safe='')}/{quote(spec.project, safe='')}?"
        + urlencode({"nw": token})
    )


def _adopt_managed_identity(
    workspace: Any,
    spec: WandbWorkspaceSpec,
    *,
    entity: str,
    existing: Any | None = None,
) -> None:
    workspace._internal_name = _managed_internal_name(spec, entity=entity)
    workspace._internal_id = str(getattr(existing, "_internal_id", "") or "")


def _app_url(api: Any) -> str:
    service_api = getattr(api, "__dict__", {}).get("_service_api")
    value = getattr(service_api, "app_url", "") if service_api is not None else ""
    if not value:
        value = getattr(getattr(api, "client", None), "app_url", "")
    value = str(value or "").strip()
    if not value:
        raise RuntimeError("W&B API did not expose an application URL")
    return value


def _load_managed_workspace(loader: WorkspaceLoader, url: str) -> tuple[str, Any | None]:
    try:
        return "existing", loader(url)
    except ValueError as exc:
        message = str(exc)
        if "Workspace `" in message and "not found" in message:
            return "missing", None
        if "Project `" in message and "not found" in message:
            return "pending_project", None
        raise


def _default_workspace_loader(url: str):
    from wandb_workspaces.workspaces import Workspace

    return Workspace.from_url(url)


def _default_workspace_saver(workspace: Any):
    return workspace.save()


def _prepared_workspaces(
    specs: Sequence[WandbWorkspaceSpec],
    *,
    entity: str,
) -> dict[str, tuple[Any, str]]:
    prepared: dict[str, tuple[Any, str]] = {}
    for spec in specs:
        workspace = build_wandb_workspace(spec, entity=entity)
        _adopt_managed_identity(workspace, spec, entity=entity)
        prepared[spec.project] = (workspace, workspace_structure_sha256(workspace))
    return prepared


def sync_workspaces(
    specs: Sequence[WandbWorkspaceSpec],
    *,
    api: Any | None = None,
    entity: str | None = None,
    workspace_loader: WorkspaceLoader | None = None,
    workspace_saver: WorkspaceSaver | None = None,
) -> list[dict[str, str]]:
    import wandb

    load_wandb_env()
    entity = entity or wandb_entity_from_env()
    api = api or wandb.Api(timeout=30)
    loader = workspace_loader or _default_workspace_loader
    saver = workspace_saver or _default_workspace_saver
    app_url = _app_url(api)
    prepared = _prepared_workspaces(specs, entity=entity)

    remote: dict[str, tuple[str, Any | None, str]] = {}
    for spec in specs:
        url = _managed_workspace_url(spec, entity=entity, app_url=app_url)
        state, existing = _load_managed_workspace(loader, url)
        remote[spec.project] = (state, existing, url)

    results: list[dict[str, str]] = []
    for spec in specs:
        desired, desired_sha = prepared[spec.project]
        state, existing, url = remote[spec.project]
        if state == "pending_project":
            results.append({"project": spec.project, "status": state, "url": url})
            continue
        if state == "existing":
            _adopt_managed_identity(desired, spec, entity=entity, existing=existing)
            if workspace_structure_sha256(existing) == desired_sha:
                results.append({"project": spec.project, "status": "unchanged", "url": url})
                continue
            status = "updated"
        else:
            status = "created"
        saver(desired)
        results.append({"project": spec.project, "status": status, "url": url})
    return results


def verify_workspaces(
    specs: Sequence[WandbWorkspaceSpec],
    *,
    api: Any | None = None,
    entity: str | None = None,
    workspace_loader: WorkspaceLoader | None = None,
) -> dict[str, Any]:
    import wandb

    load_wandb_env()
    entity = entity or wandb_entity_from_env()
    api = api or wandb.Api(timeout=30)
    loader = workspace_loader or _default_workspace_loader
    app_url = _app_url(api)
    prepared = _prepared_workspaces(specs, entity=entity)
    checked: list[dict[str, str]] = []
    issues: list[dict[str, str]] = []
    for spec in specs:
        desired, desired_sha = prepared[spec.project]
        url = _managed_workspace_url(spec, entity=entity, app_url=app_url)
        state, existing = _load_managed_workspace(loader, url)
        if state != "existing":
            issues.append({"project": spec.project, "issue": state, "url": url})
            continue
        actual_sha = workspace_structure_sha256(existing)
        checked.append({"project": spec.project, "url": url, "structure_sha256": actual_sha})
        if actual_sha != desired_sha:
            issues.append({"project": spec.project, "issue": "content_drift", "url": url})
    return {"ok": not issues, "checked": checked, "issues": issues}


def build_parser() -> argparse.ArgumentParser:
    parser = ExactArgumentParser(
        prog="gradlab workspaces",
        description="Plan, synchronize, and verify declarative W&B workspace views.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "sync", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--project", help="Limit work to one resolved W&B project.")
        command.add_argument("--repo-root", type=Path, default=Path("."))
        command.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    specs = compile_workspace_specs(args.repo_root, project=args.project)
    if args.command == "plan":
        payload = {"workspaces": [spec.to_json() for spec in specs]}
        if args.json:
            print(json.dumps(payload, sort_keys=True))
        else:
            for spec in specs:
                sections = ",".join(section.section_id for section in spec.sections)
                print(f"{spec.project}\t{spec.profile_id}\t{sections}")
        return 0
    if args.command == "sync":
        results = sync_workspaces(specs)
        if args.json:
            print(json.dumps({"workspaces": results}, sort_keys=True))
        else:
            for result in results:
                print(f"{result['status']}\t{result['project']}\t{result['url']}")
        return 0
    verification = verify_workspaces(specs)
    if args.json:
        print(json.dumps(verification, sort_keys=True))
    else:
        for row in verification["checked"]:
            print(f"ok\t{row['project']}\t{row['url']}")
        for row in verification["issues"]:
            print(f"error\t{row['project']}\t{row['issue']}\t{row['url']}")
    return 0 if verification["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
