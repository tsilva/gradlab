from __future__ import annotations

import copy
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from gradlab.metric_names import METRIC_DEFINITIONS, metric_definition, validate_metric_name
from gradlab.wandb_workspace_declarations import (
    DEFAULT_WORKSPACE_MANIFEST,
    compile_workspace_specs,
    load_workspace_declaration,
)
from gradlab.wandb_workspaces import (
    _adopt_managed_identity,
    build_wandb_workspace,
    sync_workspaces,
    verify_workspaces,
    workspace_structure_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / DEFAULT_WORKSPACE_MANIFEST


def _fake_api():
    return SimpleNamespace(client=SimpleNamespace(app_url="https://wandb.example"))


class WandbWorkspaceDeclarationTests(unittest.TestCase):
    def test_default_profile_compiles_for_every_resolved_project(self) -> None:
        first = compile_workspace_specs(ROOT)
        second = compile_workspace_specs(ROOT)

        self.assertEqual(
            [spec.to_json() for spec in first],
            [spec.to_json() for spec in second],
        )
        self.assertEqual(len(first), 16)
        self.assertEqual({spec.profile_id for spec in first}, {"training"})
        self.assertIn("SuperMarioBros-Nes-v0", {spec.project for spec in first})
        self.assertIn("VizdoomDeathmatch-v1", {spec.project for spec in first})

    def test_every_declared_panel_metric_is_registered_history(self) -> None:
        spec = compile_workspace_specs(ROOT, project="Bandit-v0")[0]

        for section in spec.sections:
            for panel in section.panels:
                validate_metric_name(panel.x)
                self.assertEqual(metric_definition(panel.x).storage, "history")
                for metric in panel.y:
                    validate_metric_name(metric)
                    self.assertEqual(metric_definition(metric).storage, "history")
                for template in panel.metric_templates:
                    definition = next(item for item in METRIC_DEFINITIONS if item.name == template)
                    self.assertEqual(definition.storage, "history")

    def test_project_override_selects_a_complete_profile(self) -> None:
        document = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
        document["profiles"]["compact"] = {
            "display_name": "GradLab Compact",
            "run_scope": "current_metrics_schema",
            "max_runs": 5,
            "sections": ["training_essentials"],
        }
        document["projects"] = {"Bandit-v0": {"profile": "compact"}}

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "_workspaces.yaml"
            path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
            specs = load_workspace_declaration(
                path,
                projects=("Bandit-v0", "Breakout-Atari2600-v0"),
            )

        by_project = {spec.project: spec for spec in specs}
        self.assertEqual(by_project["Bandit-v0"].profile_id, "compact")
        self.assertEqual(by_project["Bandit-v0"].max_runs, 5)
        self.assertEqual(
            by_project["Bandit-v0"].identity,
            compile_workspace_specs(ROOT, project="Bandit-v0")[0].identity,
        )
        self.assertEqual(by_project["Breakout-Atari2600-v0"].profile_id, "training")

    def test_unknown_panel_metric_is_rejected(self) -> None:
        document = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
        document["sections"]["training_essentials"]["panels"][0]["y"][0] = "train/not_registered"

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "_workspaces.yaml"
            path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown metric name"):
                load_workspace_declaration(path, projects=("Bandit-v0",))


class WandbWorkspaceRenderingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = compile_workspace_specs(ROOT, project="Bandit-v0")[0]

    def test_workspace_is_manual_pinned_and_uses_scientific_axis(self) -> None:
        workspace = build_wandb_workspace(self.spec, entity="entity")

        self.assertFalse(workspace.auto_generate_panels)
        self.assertEqual(workspace.settings.x_axis, "train/global_step")
        self.assertEqual(workspace.settings.smoothing_type, "none")
        self.assertEqual(workspace.settings.max_runs, 25)
        self.assertEqual(
            workspace.runset_settings.filters,
            "Config('metrics_schema_version') = 16",
        )
        self.assertEqual(len(workspace.sections), 1)
        self.assertTrue(workspace.sections[0].pinned)
        self.assertTrue(workspace.sections[0].is_open)
        self.assertEqual(len(workspace.sections[0].panels), 10)
        for panel_spec, panel in zip(
            self.spec.sections[0].panels,
            workspace.sections[0].panels,
            strict=True,
        ):
            self.assertEqual(panel.y, [])
            for metric in panel_spec.y:
                self.assertIsNotNone(re.fullmatch(panel.metric_regex, metric))
            for template in panel_spec.metric_templates:
                concrete_metric = re.sub(r"\{[a-z_]+\}", "example", template)
                self.assertIsNotNone(re.fullmatch(panel.metric_regex, concrete_metric))
        self.assertEqual(
            [
                (panel.layout.x, panel.layout.y, panel.layout.w, panel.layout.h)
                for panel in workspace.sections[0].panels
            ],
            [
                (0, 0, 12, 8),
                (12, 0, 12, 8),
                (0, 8, 12, 8),
                (12, 8, 12, 8),
                (0, 16, 12, 8),
                (12, 16, 12, 8),
                (0, 24, 12, 8),
                (12, 24, 12, 8),
                (0, 32, 12, 8),
                (12, 32, 12, 8),
            ],
        )
        _adopt_managed_identity(workspace, self.spec, entity="entity")
        self.assertRegex(workspace._to_model().name, r"^nw-[a-f0-9]{11}-v$")

    def test_workspace_structure_digest_ignores_generated_panel_ids(self) -> None:
        first = build_wandb_workspace(self.spec, entity="entity")
        second = build_wandb_workspace(self.spec, entity="entity")
        _adopt_managed_identity(first, self.spec, entity="entity")
        _adopt_managed_identity(second, self.spec, entity="entity")

        self.assertEqual(
            workspace_structure_sha256(first),
            workspace_structure_sha256(second),
        )

    @patch("gradlab.wandb_workspaces.load_wandb_env")
    def test_sync_creates_updates_and_skips_unchanged_views(self, _load_env) -> None:
        saved: list[object] = []

        created = sync_workspaces(
            [self.spec],
            api=_fake_api(),
            entity="entity",
            workspace_loader=lambda _url: (_ for _ in ()).throw(
                ValueError("Workspace `managed` not found in project `Bandit-v0`")
            ),
            workspace_saver=lambda workspace: saved.append(workspace),
        )
        self.assertEqual(created[0]["status"], "created")
        self.assertEqual(len(saved), 1)

        existing = build_wandb_workspace(self.spec, entity="entity")
        _adopt_managed_identity(existing, self.spec, entity="entity")
        saved.clear()
        unchanged = sync_workspaces(
            [self.spec],
            api=_fake_api(),
            entity="entity",
            workspace_loader=lambda _url: existing,
            workspace_saver=lambda workspace: saved.append(workspace),
        )
        self.assertEqual(unchanged[0]["status"], "unchanged")
        self.assertEqual(saved, [])

        drifted = copy.deepcopy(existing)
        drifted.name = "Manual edit"
        updated = sync_workspaces(
            [self.spec],
            api=_fake_api(),
            entity="entity",
            workspace_loader=lambda _url: drifted,
            workspace_saver=lambda workspace: saved.append(workspace),
        )
        self.assertEqual(updated[0]["status"], "updated")
        self.assertEqual(len(saved), 1)

    @patch("gradlab.wandb_workspaces.load_wandb_env")
    def test_missing_project_is_pending_and_never_created(self, _load_env) -> None:
        saved: list[object] = []
        result = sync_workspaces(
            [self.spec],
            api=_fake_api(),
            entity="entity",
            workspace_loader=lambda _url: (_ for _ in ()).throw(
                ValueError("Project `entity/Bandit-v0` not found")
            ),
            workspace_saver=lambda workspace: saved.append(workspace),
        )

        self.assertEqual(result[0]["status"], "pending_project")
        self.assertEqual(saved, [])

    @patch("gradlab.wandb_workspaces.load_wandb_env")
    def test_verify_marks_missing_and_drifted_workspaces(self, _load_env) -> None:
        missing = verify_workspaces(
            [self.spec],
            api=_fake_api(),
            entity="entity",
            workspace_loader=lambda _url: (_ for _ in ()).throw(
                ValueError("Workspace `managed` not found in project `Bandit-v0`")
            ),
        )
        self.assertFalse(missing["ok"])
        self.assertEqual(missing["issues"][0]["issue"], "missing")

        drifted = build_wandb_workspace(self.spec, entity="entity")
        drifted.name = "Manual edit"
        verification = verify_workspaces(
            [self.spec],
            api=_fake_api(),
            entity="entity",
            workspace_loader=lambda _url: drifted,
        )
        self.assertFalse(verification["ok"])
        self.assertEqual(verification["issues"][0]["issue"], "content_drift")


if __name__ == "__main__":
    unittest.main()
