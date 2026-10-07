"""Regression checks for metadata-only publication and remote Actions validation."""

import argparse
import importlib.util
import os
import tomllib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = load('release_operator', ROOT / 'scripts/release.py')
builder = load('release_builder', ROOT / 'scripts/release_build.py')


class ReleaseOperatorTests(unittest.TestCase):
    def test_release_config_preserves_project_supply_chain_policy(self):
        with (ROOT / 'pyproject.toml').open('rb') as handle:
            project = tomllib.load(handle)['tool']['uv']
        with (ROOT / 'uv-tool.toml').open('rb') as handle:
            config = tomllib.load(handle)
        for key in ('exclude-newer', 'exclude-newer-package', 'constraint-dependencies'):
            self.assertEqual(config[key], project[key], key)
        with (ROOT / 'uv.lock').open('rb') as handle:
            locked = tomllib.load(handle)['options']['exclude-newer-package']
        self.assertEqual(locked, config['exclude-newer-package'])

    def test_operator_lock_check_ignores_inherited_global_config(self):
        with patch.dict(os.environ, {'UV_CONFIG_FILE': '/unrelated/uv.toml'}), \
             patch.object(launcher.subprocess, 'run') as run:
            launcher.run(['uv', 'lock', '--check'])
        self.assertEqual(
            run.call_args.kwargs['env']['UV_CONFIG_FILE'], str(ROOT / 'uv-tool.toml')
        )

    def test_builder_uses_same_explicit_config_for_uv_commands(self):
        with patch.dict(os.environ, {'UV_CONFIG_FILE': '/unrelated/uv.toml'}), \
             patch.object(builder.subprocess, 'run') as run:
            builder.run(['/operator/bin/uv', 'build'])
        self.assertEqual(
            run.call_args.kwargs['env']['UV_CONFIG_FILE'], str(ROOT / 'uv-tool.toml')
        )

    def test_local_publication_gates_only_check_metadata(self):
        with patch.object(launcher, 'helper') as helper, patch.object(launcher, 'run') as run:
            launcher.run_release_gates('1.2.3')
        helper.assert_called_once_with('check-version', '--version', '1.2.3')
        self.assertEqual(run.call_args_list, [
            unittest.mock.call(['uv', 'lock', '--check']),
            unittest.mock.call(['git', 'diff', '--check']),
        ])

    def test_helper_uses_operator_python_without_package_environment(self):
        with patch.object(launcher, 'PYTHON', Path('/operator/bin/python')), \
             patch.object(launcher, 'run') as run:
            launcher.helper('check-version', '--version', '1.2.3')
        self.assertEqual(run.call_args.args[0][0], '/operator/bin/python')
        self.assertNotIn('.venv', run.call_args.args[0][0])

    def test_validation_dispatches_exact_remote_sha_without_preparing_local_state(self):
        with patch.object(launcher, 'upstream_ref', return_value='upstream/main'), \
             patch.object(launcher, 'require_gradlab_remote') as remote, \
             patch.object(launcher, 'capture', return_value='a' * 40), \
             patch.object(launcher, 'run') as run:
            launcher.validate_remote()
        remote.assert_called_once_with('upstream')
        self.assertEqual(run.call_args_list, [
            unittest.mock.call(['gh', 'auth', 'status']),
            unittest.mock.call(['git', 'fetch', 'upstream', 'main']),
            unittest.mock.call([
                'gh', 'workflow', 'run', 'release.yml', '--repo', 'tsilva/gradlab',
                '--ref', 'main', '-f', 'ref=' + 'a' * 40,
            ]),
        ])

    def test_validation_stops_on_wrong_upstream_branch(self):
        with patch.object(launcher, 'upstream_ref', return_value='origin/feature'), \
             patch.object(launcher, 'require_gradlab_remote'), patch.object(launcher, 'run') as run:
            with self.assertRaisesRegex(SystemExit, 'upstream main'):
                launcher.validate_remote()
        run.assert_not_called()

    def test_validation_stops_after_fetch_failure(self):
        with patch.object(launcher, 'upstream_ref', return_value='origin/main'), \
             patch.object(launcher, 'require_gradlab_remote'), \
             patch.object(launcher, 'run', side_effect=[None, RuntimeError('fetch failed')]) as run:
            with self.assertRaisesRegex(RuntimeError, 'fetch failed'):
                launcher.validate_remote()
        self.assertEqual(run.call_count, 2)

    def test_validation_rejects_version_mutation_flags(self):
        for flags in (['--to', '1.2.3'], ['--part', 'minor'], ['--dry-run-push']):
            with self.subTest(flags=flags), self.assertRaises(SystemExit):
                launcher.parse_args(['--validate', *flags])

    def test_validation_main_bypasses_clean_tree_and_version_preparation(self):
        with patch.object(launcher, 'parse_args', return_value=argparse.Namespace(validate=True)), \
             patch.object(launcher, 'validate_remote') as validate, \
             patch.object(launcher, 'ensure_clean', side_effect=AssertionError('dirty tree read')), \
             patch.object(launcher, 'prepare_version', side_effect=AssertionError('version mutation')):
            launcher.main()
        validate.assert_called_once_with()

    def test_atomic_push_keeps_branch_and_tag_together(self):
        with patch.object(launcher, 'run') as run:
            launcher.push_release('upstream', 'main', 'v1.2.3', dry_run=False)
        run.assert_called_once_with(['git', 'push', '--atomic', 'upstream', 'HEAD:main', 'v1.2.3'])

    def test_publication_requires_main(self):
        with patch.object(launcher, 'upstream_ref', return_value='origin/main'), \
             patch.object(launcher, 'require_gradlab_remote'), \
             patch.object(launcher, 'capture', return_value='feature'), \
             patch.object(launcher, 'run') as run:
            with self.assertRaisesRegex(SystemExit, 'publication requires main'):
                launcher.ensure_synced()
        run.assert_not_called()

    def test_publication_build_rejects_used_pypi_version_before_building(self):
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(builder, 'check_version'), \
             patch.object(builder, 'check_pypi', side_effect=builder.ReleaseError('used')), \
             patch.object(builder, 'run') as run:
            with self.assertRaisesRegex(builder.ReleaseError, 'used'):
                builder.build(ROOT, '1.2.3', Path(temporary) / 'dist')
        run.assert_not_called()

    def test_validation_build_still_runs_distribution_and_install_gates(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'dist'
            wheel, sdist = Path(temporary) / 'a.whl', Path(temporary) / 'a.tar.gz'
            wheel.touch()
            sdist.touch()
            with patch.object(builder, 'check_version') as version, \
                 patch.object(builder, 'check_pypi', side_effect=AssertionError('published check')), \
                 patch.object(builder, 'run') as run, \
                 patch.object(builder, 'audit', return_value=(wheel, sdist)) as audit, \
                 patch.object(builder, 'smoke_wheel') as smoke, \
                 patch.object(builder, 'digest', return_value='checksum'):
                builder.build(ROOT, '1.2.3', output, allow_published=True)
            version.assert_called_once_with(ROOT, '1.2.3')
            audit.assert_called_once_with(output, '1.2.3')
            smoke.assert_called_once_with(wheel)
            self.assertEqual(run.call_args_list[0].args[0][:2], ['uv', 'build'])
            self.assertIn('twine', run.call_args_list[1].args[0])


if __name__ == '__main__':
    unittest.main()
