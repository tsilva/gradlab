from __future__ import annotations

import subprocess
import tempfile
import tomllib
import unittest
from pathlib import Path

from containers.train.dependency_key import dependency_key
from containers.train.environment_contract import Distribution, validate_distribution_contract
from containers.train.gpu_key import gpu_key
from containers.train.lock_projection import projection_contents, train_plan_sha256
from containers.train.runtime_key import RUNTIME_INPUT_PATHS, overlay_key, runtime_key
from containers.train.vizdoom_smoke import _config as vizdoom_smoke_config


DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64


class TrainImageTests(unittest.TestCase):
    def test_vizdoom_smoke_declares_the_strict_constructor_contract(self) -> None:
        for game in ("VizdoomBasic-v1", "VizdoomDeathmatch-v1"):
            with self.subTest(game=game):
                config = vizdoom_smoke_config(game)
                for key in ("doom_map", "doom_skill", "game_args", "vizdoom_config"):
                    self.assertIn(key, config.env_args)
                    self.assertIsNone(config.env_args[key])

    def test_runtime_key_covers_overlay_inputs_not_dependency_contracts(self) -> None:
        root = Path(".").resolve()
        overlay = overlay_key(repo_root=root)
        key = runtime_key(repo_root=root, dependency_digest=DIGEST_A)

        self.assertEqual(len(overlay), 64)
        self.assertEqual(len(key), 64)
        self.assertNotEqual(key, runtime_key(repo_root=root, dependency_digest=DIGEST_B))
        self.assertIn("THIRD_PARTY_NOTICES.md", RUNTIME_INPUT_PATHS)
        self.assertIn("containers/train/vizdoom_smoke.py", RUNTIME_INPUT_PATHS)
        self.assertIn("experiments/goals", RUNTIME_INPUT_PATHS)
        self.assertIn("experiments/recipes", RUNTIME_INPUT_PATHS)
        self.assertIn("src", RUNTIME_INPUT_PATHS)
        self.assertIn("pyproject.toml", RUNTIME_INPUT_PATHS)
        self.assertNotIn("uv.lock", RUNTIME_INPUT_PATHS)
        self.assertNotIn("containers/train/gpu-linux-amd64.lock", RUNTIME_INPUT_PATHS)
        self.assertNotIn("tests", RUNTIME_INPUT_PATHS)
        self.assertNotIn("README.md", RUNTIME_INPUT_PATHS)

    def test_wheel_force_includes_are_runtime_image_inputs(self) -> None:
        project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
        force_includes = project["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
        dockerfile = Path("containers/train/Dockerfile").read_text(encoding="utf-8")
        app_package = dockerfile.split("FROM ${PYTHON_IMAGE} AS app-package", maxsplit=1)[1].split(
            "FROM scratch AS runtime-overlay", maxsplit=1
        )[0]

        for source in force_includes:
            with self.subTest(source=source):
                source_path = Path(source)
                self.assertTrue(source_path.exists())
                self.assertIn(source, RUNTIME_INPUT_PATHS)
                destination = "./" if source_path.is_file() else f"./{source}"
                self.assertIn(f"COPY {source} {destination}", app_package)

    def test_modal_deploy_group_covers_config_runtime(self) -> None:
        project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
        dependencies = project["dependency-groups"]["modal-deploy"]

        self.assertTrue(any(requirement.startswith("omegaconf") for requirement in dependencies))
        self.assertTrue(any(requirement.startswith("jinja2") for requirement in dependencies))
        self.assertTrue(any(requirement.startswith("modal") for requirement in dependencies))
        self.assertTrue(any(requirement.startswith("pydantic") for requirement in dependencies))

    def test_overlay_key_tracks_indexed_content_path_mode_and_runtime_docker_section(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            source = root / "src" / "module.py"
            source.parent.mkdir(parents=True)
            source.write_text("VALUE = 1\n", encoding="utf-8")
            dockerfile = root / "containers" / "train" / "Dockerfile"
            dockerfile.parent.mkdir(parents=True)
            dockerfile.write_text(
                "# syntax=docker/dockerfile:1.7\n"
                "ARG PYTHON_IMAGE=python@sha256:one\n"
                "ARG UV_IMAGE=uv@sha256:two\n"
                "# dependency-image-inputs-begin\n"
                "FROM base AS dependencies\n"
                "# dependency-image-inputs-end\n"
                "# runtime-image-inputs-begin\n"
                "FROM dependencies AS runtime\n"
                "COPY src /src\n"
                "# runtime-image-inputs-end\n",
                encoding="utf-8",
            )
            ignored = root / "uv.lock"
            ignored.write_text("version = 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            baseline = overlay_key(repo_root=root)

            ignored.write_text("version = 2\n", encoding="utf-8")
            subprocess.run(["git", "add", "uv.lock"], cwd=root, check=True)
            self.assertEqual(overlay_key(repo_root=root), baseline)

            source.write_text("VALUE = 2\n", encoding="utf-8")
            subprocess.run(["git", "add", "src/module.py"], cwd=root, check=True)
            content_key = overlay_key(repo_root=root)
            self.assertNotEqual(content_key, baseline)

            source.chmod(source.stat().st_mode | 0o111)
            subprocess.run(["git", "add", "src/module.py"], cwd=root, check=True)
            mode_key = overlay_key(repo_root=root)
            self.assertNotEqual(mode_key, content_key)

            dockerfile.write_text(
                dockerfile.read_text(encoding="utf-8").replace("COPY src /src", "COPY src /app"),
                encoding="utf-8",
            )
            self.assertNotEqual(overlay_key(repo_root=root), mode_key)

    def test_gpu_and_dependency_keys_follow_the_layer_dag(self) -> None:
        source = Path("containers/train/Dockerfile").read_text(encoding="utf-8")
        gpu_lock = Path("containers/train/gpu-linux-amd64.lock").read_text(encoding="utf-8")
        dependency_lock = Path("containers/train/train-dependencies-linux-amd64.lock").read_text(
            encoding="utf-8"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dockerfile = root / "Dockerfile"
            gpu_plan = root / "gpu.lock"
            dependency_plan = root / "dependencies.lock"
            dockerfile.write_text(source, encoding="utf-8")
            gpu_plan.write_text(gpu_lock, encoding="utf-8")
            dependency_plan.write_text(dependency_lock, encoding="utf-8")

            gpu_baseline = gpu_key(dockerfile=dockerfile, lockfile=gpu_plan)
            dependency_baseline = dependency_key(
                dockerfile=dockerfile,
                lockfile=dependency_plan,
                gpu_digest=DIGEST_A,
            )
            dockerfile.write_text(source + "\n# unrelated runtime edit\n", encoding="utf-8")
            self.assertEqual(gpu_key(dockerfile=dockerfile, lockfile=gpu_plan), gpu_baseline)
            self.assertEqual(
                dependency_key(
                    dockerfile=dockerfile,
                    lockfile=dependency_plan,
                    gpu_digest=DIGEST_A,
                ),
                dependency_baseline,
            )

            dependency_plan.write_text(dependency_lock + "# changed\n", encoding="utf-8")
            self.assertEqual(gpu_key(dockerfile=dockerfile, lockfile=gpu_plan), gpu_baseline)
            self.assertNotEqual(
                dependency_key(
                    dockerfile=dockerfile,
                    lockfile=dependency_plan,
                    gpu_digest=DIGEST_A,
                ),
                dependency_baseline,
            )
            self.assertNotEqual(
                dependency_key(
                    dockerfile=dockerfile,
                    lockfile=Path("containers/train/train-dependencies-linux-amd64.lock"),
                    gpu_digest=DIGEST_B,
                ),
                dependency_baseline,
            )

    def test_dockerfile_has_linked_three_layer_runtime(self) -> None:
        dockerfile = Path("containers/train/Dockerfile").read_text(encoding="utf-8")

        for section in ("gpu", "dependency", "runtime"):
            self.assertEqual(dockerfile.count(f"# {section}-image-inputs-begin"), 1)
            self.assertEqual(dockerfile.count(f"# {section}-image-inputs-end"), 1)
        self.assertIn("FROM ${PYTHON_IMAGE} AS gpu", dockerfile)
        self.assertIn("FROM ${PYTHON_IMAGE} AS dependency-overlay-build", dockerfile)
        self.assertIn("COPY THIRD_PARTY_NOTICES.md ./", dockerfile)
        self.assertIn("FROM scratch AS dependency-overlay", dockerfile)
        self.assertIn("FROM ${GPU_BASE} AS dependencies", dockerfile)
        self.assertIn("FROM scratch AS runtime-overlay", dockerfile)
        self.assertIn("COPY METRICS.md /root/gradlab/METRICS.md", dockerfile)
        self.assertIn(
            "COPY --chmod=755 containers/train/vizdoom_smoke.py "
            "/usr/local/bin/gradlab-vizdoom-smoke",
            dockerfile,
        )
        self.assertIn("FROM ${RUNTIME_BASE} AS runtime", dockerfile)
        runtime = dockerfile.split("FROM ${RUNTIME_BASE} AS runtime", maxsplit=1)[1]
        instructions = [line.strip() for line in runtime.splitlines() if line and not line.startswith(" ")]
        self.assertFalse(any(line.startswith("RUN ") for line in instructions))
        self.assertEqual(
            [line for line in instructions if line.startswith("COPY ")],
            ["COPY --link --from=runtime-overlay / /"],
        )
        dependencies = dockerfile.split("FROM ${GPU_BASE} AS dependencies", maxsplit=1)[1].split(
            "# dependency-image-inputs-end", maxsplit=1
        )[0]
        dependency_instructions = [
            line.strip()
            for line in dependencies.splitlines()
            if line and not line.startswith(" ")
        ]
        self.assertFalse(any(line.startswith("RUN ") for line in dependency_instructions))
        self.assertEqual(
            [line for line in dependency_instructions if line.startswith("COPY ")],
            ["COPY --link --from=dependency-overlay / /"],
        )
        dependency_build = dockerfile.split(
            "FROM ${PYTHON_IMAGE} AS dependency-overlay-build", maxsplit=1
        )[1].split("FROM scratch AS dependency-overlay", maxsplit=1)[0]
        self.assertIn("ENV UV_PROJECT_ENVIRONMENT=/opt/gradlab-dependencies", dependency_build)
        self.assertIn("gradlab-gpu.pth", dependency_build)
        self.assertNotIn("uv venv \"/root/gradlab/.venv\"", dependency_build)

        dependency_overlay = dockerfile.split(
            "FROM scratch AS dependency-overlay", maxsplit=1
        )[1].split("FROM ${GPU_BASE} AS dependencies", maxsplit=1)[0]
        self.assertIn("ARG BUILDKIT_SBOM_SCAN_STAGE=true", dependency_overlay)
        self.assertEqual(
            [line.strip() for line in dependency_overlay.splitlines() if line.startswith("COPY ")],
            [
                "COPY --from=dependency-overlay-build /opt/gradlab-dependencies "
                "/opt/gradlab-dependencies"
            ],
        )
        self.assertIn("ARG BUILDKIT_SBOM_SCAN_STAGE=false", dependencies)
        self.assertIn("ENV UV_PROJECT_ENVIRONMENT=/opt/gradlab-dependencies", dependencies)
        self.assertIn(
            'ENV PATH="/opt/gradlab-dependencies/bin:/root/gradlab/.venv/bin:${PATH}"',
            dependencies,
        )

    def test_image_receipt_requires_exact_vizdoom_smoke(self) -> None:
        workflow = Path(".github/workflows/gradlab-train-image.yml").read_text(
            encoding="utf-8"
        )

        self.assertIn("Smoke exact ViZDoom runtime image", workflow)
        self.assertIn("python /usr/local/bin/gradlab-vizdoom-smoke", workflow)
        self.assertIn('"smoke_contract_version": 2', workflow)
        self.assertIn('"VizdoomDeathmatch-v1"', workflow)
        self.assertIn('"schema_version": 7', workflow)
        self.assertIn('"provider_distribution": "vizdoom-turbo"', workflow)
        self.assertIn("train-dependencies-linux-amd64.lock", workflow)
        self.assertIn('"provider_version": expected_provider_version', workflow)
        self.assertNotIn('"provider_version": "1.3.0.post17"', workflow)
        self.assertIn(
            '"provider_version": os.environ["VIZDOOM_PROVIDER_VERSION"]',
            workflow,
        )
        self.assertIn(
            '"evidence_sha256": os.environ["VIZDOOM_SMOKE_EVIDENCE_SHA256"]',
            workflow,
        )

    def test_combined_environment_contract_accepts_cross_venv_requirements(self) -> None:
        gpu = {"torch": "2.12.0"}
        dependencies = {"stable-baselines3": "2.8.0"}
        installed = (
            Distribution("torch", "2.12.0", (), "gpu"),
            Distribution(
                "stable-baselines3",
                "2.8.0",
                ('torch>=2.11; python_version >= "3.10"',),
                "dependencies",
            ),
        )

        validate_distribution_contract(
            gpu=gpu,
            dependencies=dependencies,
            installed=installed,
        )

        with self.assertRaisesRegex(ValueError, "duplicate installed distribution torch"):
            validate_distribution_contract(
                gpu=gpu,
                dependencies=dependencies,
                installed=installed + (Distribution("torch", "2.12.0", (), "dependencies"),),
            )

        broken = installed[:-1] + (
            Distribution("stable-baselines3", "2.8.0", ("torch>=2.13",), "dependencies"),
        )
        with self.assertRaisesRegex(ValueError, "unsatisfied installed requirements"):
            validate_distribution_contract(
                gpu=gpu,
                dependencies=dependencies,
                installed=broken,
            )

    def test_projections_include_runtime_dependencies_and_isolate_gpu_packages(self) -> None:
        root = Path(".").resolve()
        first = projection_contents(root)
        second = projection_contents(root)
        self.assertEqual(first, second)
        self.assertEqual(
            set(first),
            {"gpu-linux-amd64.lock", "train-dependencies-linux-amd64.lock"},
        )
        for name, content in first.items():
            self.assertEqual(
                (root / "containers" / "train" / name).read_text(encoding="utf-8"),
                content,
            )

        gpu = Path("containers/train/gpu-linux-amd64.lock").read_text(encoding="utf-8")
        dependencies = Path("containers/train/train-dependencies-linux-amd64.lock").read_text(
            encoding="utf-8"
        )

        self.assertIn("vizdoom-turbo==1.3.0.post21", dependencies)
        self.assertNotIn("wandb-workspaces==", gpu + dependencies)
        self.assertIn("torch==2.12.0", gpu)
        gpu_lines = set(gpu.splitlines())
        dependency_lines = set(dependencies.splitlines())
        self.assertFalse(gpu_lines & dependency_lines)
        self.assertEqual(
            train_plan_sha256(root),
            "4648b35042eae4241f4bf5c0a66f852a606b76f994fd5b0935890074c5ab2674",
        )
        for line in gpu.splitlines():
            name = line.split("==", maxsplit=1)[0]
            self.assertTrue(name in {"torch", "triton"} or name.startswith(("cuda-", "nvidia-")))

    def test_workflows_publish_only_immutable_layer_tags(self) -> None:
        dependency = Path(".github/workflows/gradlab-train-dependencies.yml").read_text(
            encoding="utf-8"
        )
        foundations = Path(".github/workflows/gradlab-train-foundations.yml").read_text(
            encoding="utf-8"
        )
        runtime = Path(".github/workflows/gradlab-train-image.yml").read_text(encoding="utf-8")

        self.assertIn("branches-ignore: [main]", dependency)
        self.assertIn(
            "uses: ./.github/workflows/gradlab-train-foundations.yml",
            dependency,
        )
        self.assertIn("workflow_call:", foundations)
        self.assertEqual(foundations.count("runs-on: ubuntu-24.04"), 1)
        self.assertEqual(foundations.count("docker/setup-buildx-action@v3"), 1)
        self.assertIn("name: Build GPU foundation", foundations)
        self.assertIn("name: Build train dependencies", foundations)
        self.assertIn("uses: ./.github/workflows/gradlab-train-foundations.yml", runtime)
        self.assertIn("needs: foundations", runtime)
        self.assertNotIn("name: Build GPU foundation", runtime)
        self.assertNotIn("name: Build train dependencies", runtime)
        self.assertEqual(runtime.count("docker/setup-buildx-action@v3"), 1)
        self.assertIn("runtime-${{ steps.runtime_meta.outputs.runtime_input_sha256 }}", runtime)
        self.assertIn('"schema_version": 7', runtime)
        self.assertIn("--print-train-plan-sha256", foundations)
        self.assertNotIn("train-linux-amd64.lock", dependency + foundations + runtime)
        self.assertNotIn("buildcache", dependency + foundations + runtime)
        self.assertNotIn("cache-to:", dependency + foundations + runtime)

    def test_image_receipt_precedes_modal_readiness(self) -> None:
        workflow = Path(".github/workflows/gradlab-train-image.yml").read_text(encoding="utf-8")
        modal = Path(".github/workflows/gradlab-modal-eval.yml").read_text(encoding="utf-8")

        build = workflow.split("  build:", maxsplit=1)[1].split(
            "  deploy-modal-evaluator:", maxsplit=1
        )[0]
        self.assertIn("name: gradlab-train-image", build)
        self.assertIn('"schema_version": 7', build)
        self.assertIn('"gpu_plan_sha256"', build)
        self.assertIn("workflow_call:", modal)
        self.assertIn("name: gradlab-modal-eval-readiness", modal)
        self.assertIn("group: gradlab-modal-eval-${{ inputs.source_sha }}", modal)
        self.assertNotIn("group: ${{ github.workflow }}", modal)


if __name__ == "__main__":
    unittest.main()
