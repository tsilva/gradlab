"""Exercise the workflow's actual scanner wrapper without exposing real secrets."""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class SecretScanningTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repository = self.root / "repository"
        self.repository.mkdir()
        self.git("init", "--quiet")
        self.git("config", "user.name", "Scanner test")
        self.git("config", "user.email", "scanner@example.invalid")
        self.git("commit", "--quiet", "--allow-empty", "-m", "initial")
        self.base = self.git("rev-parse", "HEAD")
        self.git("commit", "--quiet", "--allow-empty", "-m", "next")
        self.head = self.git("rev-parse", "HEAD")
        workflow = (ROOT / ".github/workflows/secret-scanning.yml").read_text()
        script = workflow.split("python3 - <<'PY'\n", 1)[1].rsplit("          PY", 1)[0]
        self.wrapper = self.root / "scanner.py"
        self.wrapper.write_text(textwrap.dedent(script))
        binary = self.root / "infisical"
        binary.write_text(
            f"#!{sys.executable}\n"
            "import json, os, pathlib, sys\n"
            "arguments = sys.argv[1:]\n"
            "pathlib.Path(os.environ['SCAN_TEST_ARGS']).write_text(json.dumps(arguments))\n"
            "report = next(x.split('=', 1)[1] for x in arguments "
            "if x.startswith('--report-path='))\n"
            "mode = os.environ.get('SCAN_TEST_MODE', 'clean')\n"
            "print('PRIVATE_MATCH_SENTINEL')\n"
            "if mode == 'error': sys.exit(2)\n"
            "findings = [{'File': 'fixture.txt', 'StartLine': 1, 'RuleID': 'test-rule', "
            "'Secret': 'PRIVATE_MATCH_SENTINEL', 'Match': 'PRIVATE_MATCH_SENTINEL'}] "
            "if mode == 'finding' else []\n"
            "pathlib.Path(report).write_text(json.dumps(findings))\n"
            "sys.exit(1 if mode == 'finding' else 0)\n"
        )
        binary.chmod(0o700)

    def git(self, *arguments):
        result = subprocess.run(
            ["git", *arguments],
            cwd=self.repository,
            text=True,
            capture_output=True,
            check=True,
        )
        return result.stdout.strip()

    def scan(self, *, event="push", base=None, head=None, mode="clean", full=False):
        summary = self.root / "summary.txt"
        arguments = self.root / "arguments.json"
        environment = {
            **os.environ,
            "PATH": str(self.root) + os.pathsep + os.environ["PATH"],
            "RUNNER_TEMP": str(self.root),
            "GITHUB_STEP_SUMMARY": str(summary),
            "SCAN_HEAD": self.head if head is None else head,
            "SCAN_BASE": self.base if base is None else base,
            "SCAN_EVENT": event,
            "SCAN_FULL_HISTORY": "true" if full else "false",
            "SCAN_TEST_MODE": mode,
            "SCAN_TEST_ARGS": str(arguments),
        }
        result = subprocess.run(
            [sys.executable, str(self.wrapper)],
            cwd=self.repository,
            env=environment,
            text=True,
            capture_output=True,
        )
        published = (
            result.stdout
            + result.stderr
            + (summary.read_text() if summary.exists() else "")
        )
        self.assertNotIn("PRIVATE_MATCH_SENTINEL", published)
        self.assertNotIn("Traceback", published)
        invocation = json.loads(arguments.read_text()) if arguments.exists() else []
        return result, invocation, published

    def test_normal_push_scans_changed_commits(self):
        result, arguments, _ = self.scan()
        self.assertEqual(result.returncode, 0)
        self.assertIn(f"--log-opts={self.base}..{self.head}", arguments)

    def test_missing_before_commit_scans_entire_new_history(self):
        result, arguments, published = self.scan(base="1" * 40)
        self.assertEqual(result.returncode, 0)
        self.assertIn(f"--log-opts={self.head}", arguments)
        self.assertIn("Rewritten branch history", published)

    def test_diverged_push_scans_entire_new_history(self):
        self.git("checkout", "--quiet", "--orphan", "rewritten")
        self.git("commit", "--quiet", "--allow-empty", "-m", "rewritten")
        head = self.git("rev-parse", "HEAD")
        result, arguments, _ = self.scan(head=head)
        self.assertEqual(result.returncode, 0)
        self.assertIn(f"--log-opts={head}", arguments)

    def test_new_branch_scans_entire_history(self):
        result, arguments, _ = self.scan(base="0" * 40)
        self.assertEqual(result.returncode, 0)
        self.assertIn(f"--log-opts={self.head}", arguments)

    def test_missing_pull_request_base_fails_closed(self):
        result, arguments, _ = self.scan(event="pull_request", base="1" * 40)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(arguments, [])

    def test_invalid_revision_fails_closed(self):
        result, arguments, _ = self.scan(base="--all")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(arguments, [])

    def test_manual_scan_and_full_history_use_declared_scope(self):
        result, arguments, _ = self.scan(event="workflow_dispatch")
        self.assertEqual(result.returncode, 0)
        self.assertIn("--no-git", arguments)
        result, arguments, _ = self.scan(full=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("--log-opts=--all", arguments)

    def test_findings_fail_without_publishing_source(self):
        result, _, published = self.scan(mode="finding")
        self.assertEqual(result.returncode, 1)
        self.assertIn("fixture.txt", published)
        self.assertIn("test-rule", published)

    def test_scanner_failure_remains_a_failure(self):
        result, _, published = self.scan(mode="error")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Secret scanning failed to complete", published)


if __name__ == "__main__":
    unittest.main()
