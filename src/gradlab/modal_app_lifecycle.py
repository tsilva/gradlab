from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


MODAL_WORKSPACE_APP_LIMIT = 200
MODAL_WORKSPACE_APP_RESERVE = 20
_OWNED_PREFIXES = {
    "main": ("rlab-eval-",),
    "rlab-eval": ("rlab-eval-v2-",),
    "gradlab-eval": ("gradlab-eval-v3-",),
}


@dataclass(frozen=True)
class ModalAppRetirement:
    environment_name: str
    app_id: str
    app_name: str
    created_at: str


def _task_count(value: object) -> int:
    try:
        return int(str(value or "0").strip())
    except ValueError:
        return 1


def plan_modal_app_retirements(
    apps_by_environment: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    protected_app_names: set[str],
    target_app_name: str,
    workspace_limit: int = MODAL_WORKSPACE_APP_LIMIT,
    reserve: int = MODAL_WORKSPACE_APP_RESERVE,
) -> tuple[ModalAppRetirement, ...]:
    if workspace_limit <= 0:
        raise ValueError("Modal workspace limit must be positive")
    if reserve < 0 or reserve >= workspace_limit:
        raise ValueError("Modal workspace reserve must be smaller than the limit")
    target = str(target_app_name).strip()
    if not target:
        raise ValueError("target Modal app name is required")
    protected = {str(name).strip() for name in protected_app_names if str(name).strip()}
    protected.add(target)

    deployed: list[tuple[str, Mapping[str, Any]]] = []
    for environment_name, apps in apps_by_environment.items():
        deployed.extend(
            (str(environment_name), app)
            for app in apps
            if str(app.get("state") or "") == "deployed"
        )
    target_exists = any(str(app.get("description") or "") == target for _, app in deployed)
    pending_deployments = 0 if target_exists else 1
    maximum_after_deploy = workspace_limit - reserve
    retirement_count = max(
        len(deployed) + pending_deployments - maximum_after_deploy,
        0,
    )
    if retirement_count == 0:
        return ()

    candidates: list[ModalAppRetirement] = []
    for environment_name, app in deployed:
        app_name = str(app.get("description") or "").strip()
        prefixes = _OWNED_PREFIXES.get(environment_name, ())
        if (
            not app_name.startswith(prefixes)
            or app_name in protected
            or _task_count(app.get("tasks")) != 0
        ):
            continue
        app_id = str(app.get("app_id") or "").strip()
        if not app_id:
            continue
        candidates.append(
            ModalAppRetirement(
                environment_name=environment_name,
                app_id=app_id,
                app_name=app_name,
                created_at=str(app.get("created_at") or ""),
            )
        )
    candidates.sort(
        key=lambda item: (
            item.created_at,
            item.environment_name,
            item.app_name,
            item.app_id,
        )
    )
    if len(candidates) < retirement_count:
        raise RuntimeError(
            "Modal evaluator cleanup cannot free the reserved workspace capacity "
            "without touching protected, busy, or unrelated apps"
        )
    return tuple(candidates[:retirement_count])


def _modal_command(*arguments: str) -> list[str]:
    executable = shutil.which("modal")
    if executable:
        return [executable, *arguments]
    return [sys.executable, "-m", "modal", *arguments]


def _modal_json(*arguments: str) -> Any:
    result = subprocess.run(
        _modal_command(*arguments),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"Modal command failed: {' '.join(arguments)}")
    return json.loads(result.stdout)


def retire_unreferenced_modal_apps(
    *,
    protected_app_names: set[str],
    target_app_name: str,
) -> tuple[ModalAppRetirement, ...]:
    environments = _modal_json("environment", "list", "--json")
    if not isinstance(environments, list):
        raise RuntimeError("Modal environment listing returned an invalid document")
    apps_by_environment: dict[str, list[dict[str, Any]]] = {}
    for environment in environments:
        if not isinstance(environment, Mapping):
            raise RuntimeError("Modal environment listing contains an invalid item")
        name = str(environment.get("name") or "").strip()
        if not name:
            raise RuntimeError("Modal environment listing contains an unnamed environment")
        apps = _modal_json("app", "list", "--env", name, "--json")
        if not isinstance(apps, list) or not all(isinstance(app, Mapping) for app in apps):
            raise RuntimeError(f"Modal app listing is invalid for environment {name}")
        apps_by_environment[name] = [dict(app) for app in apps]

    retirements = plan_modal_app_retirements(
        apps_by_environment,
        protected_app_names=protected_app_names,
        target_app_name=target_app_name,
    )
    for retirement in retirements:
        result = subprocess.run(
            _modal_command(
                "app",
                "stop",
                "--yes",
                "--env",
                retirement.environment_name,
                retirement.app_id,
            ),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode:
            raise RuntimeError(
                result.stderr.strip() or f"failed to stop stale Modal app {retirement.app_name}"
            )
        print(f"Retired stale Modal evaluator {retirement.environment_name}/{retirement.app_name}")
    print(f"Modal evaluator cleanup retired {len(retirements)} app(s)")
    return retirements


def parse_protected_app_names(value: str) -> set[str]:
    try:
        parsed = json.loads(str(value or "[]"))
    except json.JSONDecodeError as exc:
        raise ValueError("protected Modal apps must be a JSON array") from exc
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError("protected Modal apps must be a JSON string array")
    return {item.strip() for item in parsed if item.strip()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Retire idle, unreferenced GradLab Modal apps to preserve deploy capacity."
    )
    parser.add_argument("--protected-apps-json", required=True)
    parser.add_argument("--target-app-name", required=True)
    args = parser.parse_args(argv)
    retire_unreferenced_modal_apps(
        protected_app_names=parse_protected_app_names(args.protected_apps_json),
        target_app_name=args.target_app_name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
