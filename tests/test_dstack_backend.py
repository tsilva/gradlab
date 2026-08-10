from __future__ import annotations

import json
import subprocess
import unittest
from unittest import mock

from gradlab.dstack_backend import (
    DSTACK_VERSION,
    ComputeRequest,
    DstackBackend,
    DstackResources,
    TaskRequest,
    render_fleet_config,
    render_task_config,
)
from gradlab.run_contracts import new_run_id


class DstackBackendTests(unittest.TestCase):
    def compute(self, **overrides) -> ComputeRequest:
        values = {
            "kind": "local",
            "target": "local-gpu",
            "max_price": None,
            "max_cost_usd": None,
            "allow_on_demand": False,
            "max_duration_seconds": 24 * 3600,
        }
        values.update(overrides)
        return ComputeRequest(**values)

    def task(self, **overrides) -> TaskRequest:
        run_id = new_run_id()
        values = {
            "run_id": run_id,
            "task_name": run_id,
            "image": "docker:registry.example/gradlab@sha256:" + "a" * 64,
            "manifest_uri": "s3://control/runs/manifest.json",
            "compute": self.compute(),
            "plain_env": {"MODAL_ENVIRONMENT": "gradlab-eval"},
            "secret_env": [
                "GRADLAB_CONTROL_R2_ACCESS_KEY_ID",
                "GRADLAB_CONTROL_R2_SECRET_ACCESS_KEY",
            ],
            "rom_mount": "/srv/gradlab/roms-ro:/roms",
        }
        values.update(overrides)
        return TaskRequest(**values)

    def test_local_config_reuses_one_named_fleet_machine(self) -> None:
        config = render_task_config(self.task())
        self.assertEqual(config["working_dir"], "/root/gradlab")
        self.assertEqual(config["fleets"], ["local-gpu"])
        self.assertEqual(config["creation_policy"], "reuse")
        self.assertEqual(config["resources"]["cpu"], "12..")
        self.assertEqual(config["resources"]["memory"], "40GB..")
        self.assertEqual(config["resources"]["gpu"], "1")
        self.assertEqual(config["resources"]["disk"], "50GB..")
        self.assertEqual(config["retry"]["on_events"], ["no-capacity", "interruption"])
        self.assertNotIn("error", config["retry"]["on_events"])
        self.assertEqual(config["max_duration"], "1d")
        self.assertNotIn("max_price", config)
        self.assertEqual(config["volumes"], ["/srv/gradlab/roms-ro:/roms"])
        self.assertIn("GRADLAB_ROM_CACHE_READ_ONLY=1", config["env"])
        self.assertIn(
            "GRADLAB_CONTROL_R2_ACCESS_KEY_ID=${{ secrets.GRADLAB_CONTROL_R2_ACCESS_KEY_ID }}",
            config["env"],
        )
        self.assertNotIn("GRADLAB_CONTROL_R2_ACCESS_KEY_ID", config["env"])
        self.assertIn("MODAL_ENVIRONMENT=gradlab-eval", config["env"])

    def test_local_config_uses_bound_fleet_resource_profile(self) -> None:
        config = render_task_config(
            self.task(
                resources=DstackResources(
                    cpu=12,
                    memory="28GB",
                    gpu="1",
                    disk="50GB",
                )
            )
        )

        self.assertEqual(
            config["resources"],
            {"cpu": "12..", "memory": "28GB..", "gpu": "1", "disk": "50GB.."},
        )

    def test_task_rejects_inline_secret_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "names only"):
            self.task(secret_env=["WANDB_API_KEY=inline-value"]).validate()
        with self.assertRaisesRegex(ValueError, "must use secret_env"):
            self.task(plain_env={"WANDB_API_KEY": "inline-value"}).validate()

    def test_task_rejects_names_longer_than_dstack_limit(self) -> None:
        self.task(task_name="r" * 41).validate()
        with self.assertRaisesRegex(ValueError, "DNS-style name"):
            self.task(task_name="r" * 42).validate()
        with self.assertRaisesRegex(ValueError, "DNS-style name"):
            self.task(task_name="1-invalid").validate()

    def test_spot_requires_both_price_and_total_cost(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires --max-price"):
            self.compute(kind="spot", target="aws").validate()
        compute = self.compute(
            kind="spot",
            target="aws",
            max_price=1.0,
            max_cost_usd=2.0,
            max_duration_seconds=10 * 3600,
        )
        config = render_task_config(self.task(compute=compute))
        self.assertEqual(config["spot_policy"], "spot")
        self.assertEqual(config["max_price"], 1.0)
        self.assertEqual(config["max_duration"], "2h")

    def test_on_demand_requires_explicit_permission(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires --allow-on-demand"):
            self.compute(
                kind="on-demand",
                max_price=1.0,
                max_cost_usd=10.0,
            ).validate()

    def test_auto_without_budget_waits_for_local_capacity(self) -> None:
        config = render_task_config(self.task(compute=self.compute(kind="auto")))
        self.assertEqual(config["fleets"], ["local-gpu"])
        self.assertEqual(config["creation_policy"], "reuse")

    def test_auto_with_budget_prefers_reuse_and_allows_bounded_spot(self) -> None:
        config = render_task_config(
            self.task(
                compute=self.compute(
                    kind="auto",
                    max_price=1.0,
                    max_cost_usd=3.0,
                ),
            )
        )
        self.assertEqual(config["creation_policy"], "reuse-or-create")
        self.assertEqual(config["spot_policy"], "spot")
        self.assertEqual(config["fleets"], ["local-gpu"])
        self.assertEqual(config["max_duration"], "3h")

    def test_auto_selection_uses_idle_local_fleet_before_spot(self) -> None:
        backend = DstackBackend(
            project="research",
            server_url="http://127.0.0.1:3000",
            token="token",
            environment={},
        )
        request = self.compute(
            kind="auto",
            max_price=1.0,
            max_cost_usd=3.0,
        )
        with (
            mock.patch.object(backend, "preflight"),
            mock.patch.object(
                backend,
                "_command",
                return_value=subprocess.CompletedProcess(
                    ["dstack", "offer"],
                    0,
                    json.dumps(
                        {
                            "offers": [
                                {
                                    "backend": "ssh",
                                    "availability": "idle",
                                    "price": 0.0,
                                }
                            ]
                        }
                    ),
                    "",
                ),
            ),
        ):
            selected, offer = backend.select_compute(request)
        self.assertEqual(selected.kind, "local")
        self.assertEqual(selected.target, "local-gpu")
        self.assertEqual(offer["backend"], "ssh")

    def test_auto_selection_falls_back_to_bounded_spot_when_local_fleet_is_busy(self) -> None:
        backend = DstackBackend(
            project="research",
            server_url="http://127.0.0.1:3000",
            token="token",
            environment={},
        )
        request = self.compute(
            kind="auto",
            max_price=1.0,
            max_cost_usd=3.0,
        )
        with (
            mock.patch.object(backend, "preflight"),
            mock.patch.object(
                backend,
                "_command",
                return_value=subprocess.CompletedProcess(
                    ["dstack", "offer"],
                    0,
                    json.dumps({"offers": [{"availability": "busy"}]}),
                    "",
                ),
            ),
        ):
            selected, offer = backend.select_compute(request)
        self.assertEqual(selected.kind, "spot")
        self.assertIsNone(selected.target)
        self.assertIsNone(offer)

    def test_fleet_is_one_unsplit_ssh_host(self) -> None:
        config = render_fleet_config(
            name="local-gpu",
            hostname="gpu-host.internal",
            user="operator",
            identity_file="~/.ssh/operator-key",
        )
        self.assertEqual(config["blocks"], 1)
        self.assertEqual(config["ssh_config"]["hosts"], ["gpu-host.internal"])

    def test_backend_requires_explicit_coordinator_project(self) -> None:
        with self.assertRaisesRegex(ValueError, "selected coordinator project"):
            DstackBackend(
                project="",
                server_url="http://127.0.0.1:3000",
                token="token",
                environment={"DSTACK_PROJECT": "ambient-project"},
            )

    def test_explicit_coordinator_metadata_overwrites_ambient_routing(self) -> None:
        backend = DstackBackend(
            project="bound-project",
            server_url="http://127.0.0.1:3002",
            token="bound-token",
            environment={
                "DSTACK_PROJECT": "ambient-project",
                "DSTACK_SERVER_URL": "http://127.0.0.1:3999",
                "DSTACK_TOKEN": "ambient-token",
            },
        )

        self.assertEqual(backend.project, "bound-project")
        self.assertEqual(backend.server_url, "http://127.0.0.1:3002")
        self.assertEqual(backend.environment["DSTACK_PROJECT"], "bound-project")
        self.assertEqual(backend.environment["DSTACK_SERVER_URL"], "http://127.0.0.1:3002")
        self.assertEqual(backend.environment["DSTACK_TOKEN"], "bound-token")

    @mock.patch("gradlab.dstack_backend.urllib.request.urlopen")
    @mock.patch("gradlab.dstack_backend.shutil.which", return_value="/bin/dstack")
    @mock.patch("gradlab.dstack_backend.subprocess.run")
    def test_submit_checks_version_and_sends_yaml_on_stdin(
        self,
        run,
        _which,
        urlopen,
    ) -> None:
        run.side_effect = [
            subprocess.CompletedProcess(["dstack", "-v"], 0, DSTACK_VERSION + "\n", ""),
            subprocess.CompletedProcess(["dstack", "ps"], 0, '{"runs": []}\n', ""),
            subprocess.CompletedProcess(["dstack", "apply"], 0, "submitted\n", ""),
        ]
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = b"{}"
        urlopen.return_value = response
        backend = DstackBackend(
            project="research",
            server_url="http://127.0.0.1:3000",
            token="secret",
            environment={
                "PATH": "/bin",
                "DSTACK_PROJECT": "research",
                "DSTACK_SERVER_URL": "http://127.0.0.1:3000",
                "DSTACK_TOKEN": "secret",
                "GRADLAB_CONTROL_R2_ACCESS_KEY_ID": "access-key",
                "GRADLAB_CONTROL_R2_SECRET_ACCESS_KEY": "secret-key",
            },
        )
        request = self.task()
        task = backend.submit(request)
        self.assertEqual(task.name, request.task_name)
        submitted = run.call_args_list[2]
        self.assertIn("on_events:", submitted.kwargs["input"])
        self.assertNotIn("DSTACK_TOKEN", submitted.kwargs["input"])

    @mock.patch("gradlab.dstack_backend.shutil.which", return_value="/bin/dstack")
    @mock.patch("gradlab.dstack_backend.subprocess.run")
    def test_preflight_authenticates_to_the_live_server(self, run, _which) -> None:
        run.side_effect = [
            subprocess.CompletedProcess(["dstack", "-v"], 0, DSTACK_VERSION + "\n", ""),
            subprocess.CompletedProcess(["dstack", "ps"], 0, '{"runs": []}\n', ""),
        ]
        backend = DstackBackend(
            project="research",
            server_url="http://127.0.0.1:3000",
            token="admin-token",
            environment={
                "PATH": "/bin",
                "DSTACK_PROJECT": "research",
                "DSTACK_SERVER_URL": "http://127.0.0.1:3000",
                "DSTACK_TOKEN": "admin-token",
            },
        )

        backend.preflight()

        self.assertEqual(
            run.call_args_list[1].args[0],
            ["dstack", "ps", "--project", "research", "--all", "--json"],
        )

    @mock.patch("gradlab.dstack_backend.urllib.request.urlopen")
    @mock.patch("gradlab.dstack_backend.shutil.which", return_value="/bin/dstack")
    @mock.patch("gradlab.dstack_backend.subprocess.run")
    def test_sync_project_secrets_uses_authenticated_server_api(
        self,
        run,
        _which,
        urlopen,
    ) -> None:
        run.return_value = subprocess.CompletedProcess(
            ["dstack", "-v"],
            0,
            DSTACK_VERSION + "\n",
            "",
        )
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = b"{}"
        urlopen.return_value = response
        backend = DstackBackend(
            project="research",
            server_url="http://127.0.0.1:3000",
            token="admin-token",
            environment={
                "PATH": "/bin",
                "DSTACK_PROJECT": "research",
                "DSTACK_SERVER_URL": "http://127.0.0.1:3000",
                "DSTACK_TOKEN": "admin-token",
                "WANDB_API_KEY": "credential-value",
            },
        )

        backend.sync_project_secrets(["WANDB_API_KEY"])

        request = urlopen.call_args.args[0]
        self.assertEqual(
            json.loads(request.data),
            {"name": "WANDB_API_KEY", "value": "credential-value"},
        )
        self.assertEqual(request.headers["Authorization"], "Bearer admin-token")
        self.assertEqual(request.headers["X-api-version"], DSTACK_VERSION)

    @mock.patch("gradlab.dstack_backend.subprocess.run")
    def test_status_reads_json(self, run) -> None:
        run.return_value = subprocess.CompletedProcess(
            ["dstack", "ps"],
            0,
            json.dumps(
                {
                    "runs": [
                        {
                            "run_name": None,
                            "submitted_at": "2026-07-24T12:00:00Z",
                            "status": "terminated",
                            "run_spec": {"configuration": {"name": "run-one"}},
                        },
                        {
                            "run_name": None,
                            "submitted_at": "2026-07-24T13:00:00Z",
                            "status": "running",
                            "run_spec": {"configuration": {"name": "run-one"}},
                        },
                    ]
                }
            ),
            "",
        )
        backend = DstackBackend(
            project="research",
            server_url="http://127.0.0.1:3000",
            token="token",
            environment={},
        )
        self.assertEqual(backend.status("run-one").status, "running")

    @mock.patch("gradlab.dstack_backend.subprocess.run")
    def test_status_rejects_noncurrent_name_and_state_aliases(self, run) -> None:
        run.return_value = subprocess.CompletedProcess(
            ["dstack", "ps"],
            0,
            json.dumps(
                {
                    "runs": [
                        {
                            "name": "run-one",
                            "state": "running",
                            "submitted_at": "2026-07-24T13:00:00Z",
                        }
                    ]
                }
            ),
            "",
        )

        with self.assertRaisesRegex(ValueError, "run_spec.configuration"):
            DstackBackend(
                project="research",
                server_url="http://127.0.0.1:3000",
                token="token",
                environment={},
            ).status("run-one")


if __name__ == "__main__":
    unittest.main()
