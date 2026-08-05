from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import unittest

from gradlab.main import main


class PublicCliHelpTests(unittest.TestCase):
    def test_bare_eval_prints_flat_help_and_returns_usage_error(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(main(["eval"]), 2)
        self.assertTrue(stdout.getvalue().startswith("usage: gradlab eval"))
        self.assertIn("--episodes", stdout.getvalue())

    def test_root_help_advertises_only_canonical_rom_command(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(main(["--help"]), 0)
        help_text = stdout.getvalue()
        self.assertIn("    rom ", help_text)
        self.assertNotIn("import-roms", help_text)

    def test_noncurrent_command_forms_are_rejected(self) -> None:
        for argv in (
            ["import-roms"],
            ["eval", "run", "--n-envs", "0"],
        ):
            with self.subTest(command=" ".join(argv)), self.assertRaises(SystemExit) as raised:
                main(argv)
            self.assertEqual(raised.exception.code, 2)

    def test_ordinary_help_does_not_import_optional_or_rom_import_stacks(self) -> None:
        script = """
import sys
from gradlab.main import main
try:
    main([\"--help\"])
except SystemExit:
    pass
for name in sorted(sys.modules):
    if (
        name == \"datasets\"
        or name == \"minari\"
        or name.startswith(\"gradlab.dataset_\")
        or name == \"stable_retro.scripts.import_path\"
    ):
        print(name)
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        imported = [
            line
            for line in completed.stdout.splitlines()
            if line == "datasets"
            or line == "minari"
            or line.startswith("gradlab.dataset_")
            or line == "stable_retro.scripts.import_path"
        ]
        self.assertEqual(imported, [])

    def test_delegated_help_uses_complete_public_command(self) -> None:
        cases = (
            (("train", "--help"), "usage: gradlab train"),
            (("experiment", "launch", "--help"), "usage: gradlab experiment launch"),
            (("experiment", "follow", "--help"), "usage: gradlab experiment follow"),
            (("eval", "--help"), "usage: gradlab eval"),
            (("play", "--help"), "usage: gradlab play"),
            (("rom", "import", "--help"), "usage: gradlab rom import"),
            (("benchmark", "run", "--help"), "usage: gradlab benchmark run"),
            (("validate", "--help"), "usage: gradlab validate"),
            (("env", "preflight", "--help"), "usage: gradlab env preflight"),
            (("dataset", "--help"), "usage: gradlab dataset"),
            (("dataset", "record", "--help"), "usage: gradlab dataset record"),
            (("dataset", "verify", "--help"), "usage: gradlab dataset verify"),
            (("leaders", "runs", "--help"), "usage: gradlab leaders runs"),
            (("reports", "plan", "--help"), "usage: gradlab reports plan"),
            (("workspaces", "plan", "--help"), "usage: gradlab workspaces plan"),
        )
        for argv, expected_usage in cases:
            with self.subTest(command=" ".join(argv)):
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout), self.assertRaises(SystemExit) as raised:
                    main(list(argv))
                self.assertEqual(raised.exception.code, 0)
                self.assertTrue(stdout.getvalue().startswith(expected_usage), stdout.getvalue())

    def test_launch_help_describes_exact_source_runtime_resolution(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), self.assertRaises(SystemExit) as raised:
            main(["experiment", "launch", "--help"])
        self.assertEqual(raised.exception.code, 0)
        help_text = stdout.getvalue()
        normalized_help = " ".join(help_text.split())
        self.assertIn("exact-source immutable runtime image", normalized_help)
        self.assertIn("never falls back to an older image", normalized_help)
        self.assertNotIn("defaults to latest", normalized_help)

    def test_eval_and_play_help_are_sb3_backend_neutral(self) -> None:
        for command in (("eval",), ("play",)):
            with self.subTest(command=command):
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout), self.assertRaises(SystemExit) as raised:
                    main([*command, "--help"])
                self.assertEqual(raised.exception.code, 0)
                help_text = stdout.getvalue()
                self.assertIn("gradlab", help_text)
                self.assertNotIn("SB3", help_text)
                self.assertNotIn("PPO", help_text)


if __name__ == "__main__":
    unittest.main()
