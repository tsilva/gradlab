from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from gradlab.config_loader import load_mapping_document
from gradlab.metric_names import (
    METRIC_DEFINITIONS,
    TRAIN_GLOBAL_STEP,
    metric_definition,
    validate_metric_name,
)
from gradlab.recipe_documents import load_goal_contract
from gradlab.wandb_utils import resolve_wandb_project


WORKSPACE_SCHEMA_VERSION = 4
DEFAULT_WORKSPACE_MANIFEST = Path("experiments/goals/_workspaces.yaml")
_SAFE_ID = re.compile(r"^[a-z][a-z0-9_-]*$")
_RUN_SCOPES = frozenset({"all", "current_metrics_schema"})
_PANEL_KINDS = frozenset({"line"})
_WORKSPACE_GRID_WIDTH = 24


@dataclass(frozen=True)
class WorkspacePanelSpec:
    panel_id: str
    kind: str
    x: str
    y: tuple[str, ...]
    metric_templates: tuple[str, ...]
    width: int
    height: int
    y_title: str | None

    @property
    def title(self) -> str:
        """Use the canonical metric selectors as the visible panel title."""

        return " · ".join((*self.y, *self.metric_templates))

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.panel_id,
            "kind": self.kind,
            "title": self.title,
            "x": self.x,
            "y": list(self.y),
            "metric_templates": list(self.metric_templates),
            "width": self.width,
            "height": self.height,
            "y_title": self.y_title,
        }


@dataclass(frozen=True)
class WorkspaceSectionSpec:
    section_id: str
    title: str
    pinned: bool
    is_open: bool
    columns: int
    panels: tuple[WorkspacePanelSpec, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.section_id,
            "title": self.title,
            "pinned": self.pinned,
            "open": self.is_open,
            "columns": self.columns,
            "panels": [panel.to_json() for panel in self.panels],
        }


@dataclass(frozen=True)
class WorkspaceProfileSpec:
    profile_id: str
    display_name: str
    run_scope: str
    max_runs: int
    sections: tuple[WorkspaceSectionSpec, ...]


@dataclass(frozen=True)
class WandbWorkspaceSpec:
    identity: str
    view_id: str
    project: str
    profile_id: str
    display_name: str
    run_scope: str
    max_runs: int
    sections: tuple[WorkspaceSectionSpec, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "view_id": self.view_id,
            "project": self.project,
            "profile": self.profile_id,
            "display_name": self.display_name,
            "run_scope": self.run_scope,
            "max_runs": self.max_runs,
            "sections": [section.to_json() for section in self.sections],
        }


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], *, label: str) -> None:
    unknown = sorted(str(key) for key in value if key not in allowed)
    if unknown:
        raise ValueError(f"{label} has unknown field(s): {', '.join(unknown)}")


def _text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _identifier(value: Any, *, label: str) -> str:
    result = _text(value, label=label)
    if _SAFE_ID.fullmatch(result) is None:
        raise ValueError(f"{label} must match {_SAFE_ID.pattern}")
    return result


def _boolean(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _bounded_int(value: Any, *, minimum: int, maximum: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{label} must be an integer from {minimum} through {maximum}")
    return value


def _string_list(value: Any, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"{label} must be a list")
    result = tuple(_text(item, label=f"{label}[{index}]") for index, item in enumerate(value))
    if not result:
        raise ValueError(f"{label} must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must not contain duplicates")
    return result


def _validate_history_metric(name: str, *, label: str) -> str:
    validate_metric_name(name)
    definition = metric_definition(name)
    if definition is None or definition.placement != "history":
        raise ValueError(f"{label} must name a W&B history metric")
    return name


def _validate_metric_template(name: str, *, label: str) -> str:
    definitions = {definition.name: definition for definition in METRIC_DEFINITIONS}
    definition = definitions.get(name)
    if definition is None or "{" not in name:
        raise ValueError(f"{label} must exactly match a registered metric template")
    if definition.placement != "history":
        raise ValueError(f"{label} must name a W&B history metric template")
    return name


def _panel_spec(panel_id: str, value: Any, *, label: str) -> WorkspacePanelSpec:
    document = _mapping(value, label=label)
    _reject_unknown(
        document,
        {"kind", "x", "y", "metric_templates", "width", "height", "y_title"},
        label=label,
    )
    kind = _text(document.get("kind"), label=f"{label}.kind")
    if kind not in _PANEL_KINDS:
        raise ValueError(f"{label}.kind must be one of {sorted(_PANEL_KINDS)}")
    x = _validate_history_metric(
        _text(document.get("x"), label=f"{label}.x"),
        label=f"{label}.x",
    )
    if x != TRAIN_GLOBAL_STEP:
        raise ValueError(f"{label}.x must equal {TRAIN_GLOBAL_STEP}")
    raw_y = document.get("y", ())
    if not isinstance(raw_y, Sequence) or isinstance(raw_y, str | bytes):
        raise ValueError(f"{label}.y must be a list")
    y = tuple(
        _validate_history_metric(
            _text(item, label=f"{label}.y[{index}]"), label=f"{label}.y[{index}]"
        )
        for index, item in enumerate(raw_y)
    )
    raw_templates = document.get("metric_templates", ())
    if not isinstance(raw_templates, Sequence) or isinstance(raw_templates, str | bytes):
        raise ValueError(f"{label}.metric_templates must be a list")
    metric_templates = tuple(
        _validate_metric_template(
            _text(item, label=f"{label}.metric_templates[{index}]"),
            label=f"{label}.metric_templates[{index}]",
        )
        for index, item in enumerate(raw_templates)
    )
    if not y and not metric_templates:
        raise ValueError(f"{label} must declare y or metric_templates")
    if len(set((*y, *metric_templates))) != len((*y, *metric_templates)):
        raise ValueError(f"{label} metric declarations must be unique")
    y_title = document.get("y_title")
    if y_title is not None:
        y_title = _text(y_title, label=f"{label}.y_title")
    return WorkspacePanelSpec(
        panel_id=_identifier(panel_id, label=f"{label} id"),
        kind=kind,
        x=x,
        y=y,
        metric_templates=metric_templates,
        width=_bounded_int(
            document.get("width", 12), minimum=1, maximum=24, label=f"{label}.width"
        ),
        height=_bounded_int(
            document.get("height", 8), minimum=1, maximum=24, label=f"{label}.height"
        ),
        y_title=y_title,
    )


def _section_spec(section_id: str, value: Any, *, label: str) -> WorkspaceSectionSpec:
    document = _mapping(value, label=label)
    _reject_unknown(
        document,
        {"title", "pinned", "open", "columns", "panels"},
        label=label,
    )
    raw_panels = document.get("panels")
    if (
        not isinstance(raw_panels, Sequence)
        or isinstance(raw_panels, str | bytes)
        or not raw_panels
    ):
        raise ValueError(f"{label}.panels must be a non-empty list")
    panels: list[WorkspacePanelSpec] = []
    panel_ids: set[str] = set()
    for index, raw_panel in enumerate(raw_panels):
        panel_document = _mapping(raw_panel, label=f"{label}.panels[{index}]")
        panel_id = _identifier(
            panel_document.get("id"),
            label=f"{label}.panels[{index}].id",
        )
        if panel_id in panel_ids:
            raise ValueError(f"{label}.panels contains duplicate id {panel_id!r}")
        panel_ids.add(panel_id)
        panels.append(
            _panel_spec(
                panel_id,
                {key: nested for key, nested in panel_document.items() if key != "id"},
                label=f"{label}.panels[{index}]",
            )
        )
    columns = _bounded_int(
        document.get("columns"), minimum=1, maximum=4, label=f"{label}.columns"
    )
    slot_width = _WORKSPACE_GRID_WIDTH // columns
    too_wide = [panel.panel_id for panel in panels if panel.width > slot_width]
    if too_wide:
        raise ValueError(
            f"{label}.panels must be at most {slot_width} grid units wide "
            f"for {columns} columns: {', '.join(too_wide)}"
        )
    return WorkspaceSectionSpec(
        section_id=_identifier(section_id, label=f"{label} id"),
        title=_text(document.get("title"), label=f"{label}.title"),
        pinned=_boolean(document.get("pinned"), label=f"{label}.pinned"),
        is_open=_boolean(document.get("open"), label=f"{label}.open"),
        columns=columns,
        panels=tuple(panels),
    )


def _profile_spec(
    profile_id: str,
    value: Any,
    *,
    sections: Mapping[str, WorkspaceSectionSpec],
    label: str,
) -> WorkspaceProfileSpec:
    document = _mapping(value, label=label)
    _reject_unknown(
        document,
        {"display_name", "run_scope", "max_runs", "sections"},
        label=label,
    )
    run_scope = _text(document.get("run_scope"), label=f"{label}.run_scope")
    if run_scope not in _RUN_SCOPES:
        raise ValueError(f"{label}.run_scope must be one of {sorted(_RUN_SCOPES)}")
    section_ids = _string_list(document.get("sections"), label=f"{label}.sections")
    unknown = sorted(set(section_ids) - set(sections))
    if unknown:
        raise ValueError(f"{label}.sections references unknown section(s): {', '.join(unknown)}")
    return WorkspaceProfileSpec(
        profile_id=_identifier(profile_id, label=f"{label} id"),
        display_name=_text(document.get("display_name"), label=f"{label}.display_name"),
        run_scope=run_scope,
        max_runs=_bounded_int(
            document.get("max_runs"), minimum=1, maximum=100, label=f"{label}.max_runs"
        ),
        sections=tuple(sections[section_id] for section_id in section_ids),
    )


def _profile_sections_without_panels(
    profile: WorkspaceProfileSpec,
    excluded_panel_ids: Sequence[str],
    *,
    label: str,
) -> tuple[WorkspaceSectionSpec, ...]:
    excluded = set(excluded_panel_ids)
    available = {
        panel.panel_id
        for section in profile.sections
        for panel in section.panels
    }
    unknown = sorted(excluded - available)
    if unknown:
        raise ValueError(f"{label} references unknown panel(s): {', '.join(unknown)}")
    sections = tuple(
        replace(
            section,
            panels=tuple(
                panel for panel in section.panels if panel.panel_id not in excluded
            ),
        )
        for section in profile.sections
    )
    empty = [section.section_id for section in sections if not section.panels]
    if empty:
        raise ValueError(f"{label} leaves empty section(s): {', '.join(empty)}")
    return sections


def _sections_without_metrics(
    sections: Sequence[WorkspaceSectionSpec],
    excluded_metric_names: Sequence[str],
    *,
    label: str,
) -> tuple[WorkspaceSectionSpec, ...]:
    excluded = set(excluded_metric_names)
    available = {
        metric
        for section in sections
        for panel in section.panels
        for metric in (*panel.y, *panel.metric_templates)
    }
    unknown = sorted(excluded - available)
    if unknown:
        raise ValueError(f"{label} references unknown metric(s): {', '.join(unknown)}")
    filtered_sections: list[WorkspaceSectionSpec] = []
    for section in sections:
        filtered_panels: list[WorkspacePanelSpec] = []
        for panel in section.panels:
            filtered_panel = replace(
                panel,
                y=tuple(metric for metric in panel.y if metric not in excluded),
                metric_templates=tuple(
                    metric for metric in panel.metric_templates if metric not in excluded
                ),
            )
            if filtered_panel.y or filtered_panel.metric_templates:
                filtered_panels.append(filtered_panel)
        filtered_sections.append(replace(section, panels=tuple(filtered_panels)))
    result = tuple(filtered_sections)
    empty = [section.section_id for section in result if not section.panels]
    if empty:
        raise ValueError(f"{label} leaves empty section(s): {', '.join(empty)}")
    return result


def discover_wandb_projects(repo_root: Path | str = Path(".")) -> tuple[str, ...]:
    repo_root = Path(repo_root).resolve()
    projects: set[str] = set()
    for goal_path in sorted((repo_root / "experiments" / "goals").rglob("_goal.yaml")):
        goal = load_goal_contract(goal_path, repo_root)
        train = _mapping(goal.get("train"), label=f"{goal_path}.train")
        environment = _mapping(train.get("environment"), label=f"{goal_path}.train.environment")
        env_config = _mapping(
            environment.get("env_config"), label=f"{goal_path}.train.environment.env_config"
        )
        projects.add(
            resolve_wandb_project(
                None,
                str(env_config.get("game") or ""),
                env_provider=environment.get("env_provider"),
            )
        )
    if not projects:
        raise ValueError("workspace declarations require at least one checked-in active goal")
    return tuple(sorted(projects))


def load_workspace_declaration(
    path: Path,
    *,
    projects: Sequence[str],
) -> tuple[WandbWorkspaceSpec, ...]:
    document = load_mapping_document(path, label=f"workspace declaration {path}")
    _reject_unknown(
        document,
        {"schema_version", "view_id", "default_profile", "sections", "profiles", "projects"},
        label=str(path),
    )
    if document.get("schema_version") != WORKSPACE_SCHEMA_VERSION:
        raise ValueError(f"{path}.schema_version must equal {WORKSPACE_SCHEMA_VERSION}")
    view_id = _identifier(document.get("view_id"), label=f"{path}.view_id")

    raw_sections = _mapping(document.get("sections"), label=f"{path}.sections")
    if not raw_sections:
        raise ValueError(f"{path}.sections must not be empty")
    sections = {
        section_id: _section_spec(str(section_id), value, label=f"{path}.sections.{section_id}")
        for section_id, value in raw_sections.items()
    }

    raw_profiles = _mapping(document.get("profiles"), label=f"{path}.profiles")
    if not raw_profiles:
        raise ValueError(f"{path}.profiles must not be empty")
    profiles = {
        profile_id: _profile_spec(
            str(profile_id),
            value,
            sections=sections,
            label=f"{path}.profiles.{profile_id}",
        )
        for profile_id, value in raw_profiles.items()
    }
    default_profile = _identifier(document.get("default_profile"), label=f"{path}.default_profile")
    if default_profile not in profiles:
        raise ValueError(f"{path}.default_profile references unknown profile {default_profile!r}")

    discovered_projects = tuple(
        sorted({_text(project, label="W&B project") for project in projects})
    )
    raw_project_overrides = _mapping(document.get("projects"), label=f"{path}.projects")
    project_overrides = {
        _text(project, label=f"{path}.projects key"): value
        for project, value in raw_project_overrides.items()
    }
    unknown_projects = sorted(set(project_overrides) - set(discovered_projects))
    if unknown_projects:
        raise ValueError(
            f"{path}.projects contains project(s) not resolved from active goals: "
            + ", ".join(unknown_projects)
        )
    assignments = {project: default_profile for project in discovered_projects}
    panel_exclusions: dict[str, tuple[str, ...]] = {
        project: () for project in discovered_projects
    }
    metric_exclusions: dict[str, tuple[str, ...]] = {
        project: () for project in discovered_projects
    }
    for project, raw_override in project_overrides.items():
        override = _mapping(raw_override, label=f"{path}.projects.{project}")
        _reject_unknown(
            override,
            {"profile", "exclude_panels", "exclude_metrics"},
            label=f"{path}.projects.{project}",
        )
        profile_id = _identifier(
            override.get("profile"), label=f"{path}.projects.{project}.profile"
        )
        if profile_id not in profiles:
            raise ValueError(
                f"{path}.projects.{project}.profile references unknown profile {profile_id!r}"
            )
        assignments[project] = profile_id
        raw_exclusions = override.get("exclude_panels", ())
        if not isinstance(raw_exclusions, Sequence) or isinstance(
            raw_exclusions, str | bytes
        ):
            raise ValueError(f"{path}.projects.{project}.exclude_panels must be a list")
        panel_exclusions[project] = tuple(
            _identifier(
                panel_id,
                label=f"{path}.projects.{project}.exclude_panels[{index}]",
            )
            for index, panel_id in enumerate(raw_exclusions)
        )
        if len(set(panel_exclusions[project])) != len(panel_exclusions[project]):
            raise ValueError(
                f"{path}.projects.{project}.exclude_panels must not contain duplicates"
            )
        raw_metric_exclusions = override.get("exclude_metrics", ())
        if not isinstance(raw_metric_exclusions, Sequence) or isinstance(
            raw_metric_exclusions, str | bytes
        ):
            raise ValueError(f"{path}.projects.{project}.exclude_metrics must be a list")
        metric_exclusions[project] = tuple(
            _text(
                metric,
                label=f"{path}.projects.{project}.exclude_metrics[{index}]",
            )
            for index, metric in enumerate(raw_metric_exclusions)
        )
        if len(set(metric_exclusions[project])) != len(metric_exclusions[project]):
            raise ValueError(
                f"{path}.projects.{project}.exclude_metrics must not contain duplicates"
            )

    used_profiles = set(assignments.values())
    unused_profiles = sorted(set(profiles) - used_profiles)
    if unused_profiles:
        raise ValueError(
            f"{path}.profiles contains unused profile(s): {', '.join(unused_profiles)}"
        )
    used_sections = {
        section.section_id
        for profile_id in used_profiles
        for section in profiles[profile_id].sections
    }
    unused_sections = sorted(set(sections) - used_sections)
    if unused_sections:
        raise ValueError(
            f"{path}.sections contains unused section(s): {', '.join(unused_sections)}"
        )

    return tuple(
        WandbWorkspaceSpec(
            identity=f"gradlab/workspace/{view_id}",
            view_id=view_id,
            project=project,
            profile_id=assignments[project],
            display_name=profiles[assignments[project]].display_name,
            run_scope=profiles[assignments[project]].run_scope,
            max_runs=profiles[assignments[project]].max_runs,
            sections=_sections_without_metrics(
                _profile_sections_without_panels(
                    profiles[assignments[project]],
                    panel_exclusions[project],
                    label=f"{path}.projects.{project}.exclude_panels",
                ),
                metric_exclusions[project],
                label=f"{path}.projects.{project}.exclude_metrics",
            ),
        )
        for project in discovered_projects
    )


def compile_workspace_specs(
    repo_root: Path | str = Path("."),
    *,
    project: str | None = None,
) -> tuple[WandbWorkspaceSpec, ...]:
    repo_root = Path(repo_root).resolve()
    specs = load_workspace_declaration(
        repo_root / DEFAULT_WORKSPACE_MANIFEST,
        projects=discover_wandb_projects(repo_root),
    )
    if project is None:
        return specs
    selected = tuple(spec for spec in specs if spec.project == project)
    if not selected:
        raise ValueError(f"no declared W&B workspace found for project {project!r}")
    return selected


def validate_workspace_declarations(repo_root: Path | str = Path(".")) -> int:
    return len(compile_workspace_specs(repo_root))
