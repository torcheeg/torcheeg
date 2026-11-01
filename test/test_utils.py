import hashlib
import io
import logging
import os
import random
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from torcheeg import utils
from torcheeg.utils import LazyLoader


class TestLazyLoader(unittest.TestCase):
    def test_import_is_deferred_and_cached(self):
        module = SimpleNamespace(value=42)
        with patch('torcheeg.utils._import_module', return_value=module) as load:
            lazy = LazyLoader('graph.backend')
            load.assert_not_called()
            self.assertEqual(lazy.value, 42)
            self.assertEqual(lazy.value, 42)
            load.assert_called_once_with('graph.backend')

    def test_missing_dependency_preserves_cause(self):
        error = ModuleNotFoundError('graph backend unavailable')
        with patch('torcheeg.utils._import_module', side_effect=error):
            lazy = LazyLoader('torch_scatter')
            with self.assertRaisesRegex(ImportError, 'Could not import optional dependency `torch_scatter`') as caught:
                lazy.scatter_add
            self.assertIs(caught.exception.__cause__, error)
            self.assertNotIn("pytorch-geometric", str(caught.exception))

    def test_custom_install_hint_and_retry(self):
        error = ImportError('backend binary incompatible')
        module = SimpleNamespace(value=42)
        with patch('torcheeg.utils._import_module', side_effect=[error, module]):
            lazy = LazyLoader(
                'backend', install_hint='See backend installation docs.')
            with self.assertRaisesRegex(ImportError, 'See backend installation docs') as caught:
                lazy.value
            self.assertIs(caught.exception.__cause__, error)
            self.assertEqual(lazy.value, 42)

    def test_class_attribute_can_be_used_as_a_base(self):
        lazy = LazyLoader('collections')

        class Derived(lazy.UserList):
            pass

        data = Derived([1, 2])
        data.append(3)
        self.assertEqual(data, [1, 2, 3])


class TestUtilsImports(unittest.TestCase):
    def _run_fresh(self, body):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, '-c', textwrap.dedent(body)], cwd=root,
            env=dict(os.environ, PYTHONPATH=str(root), MPLBACKEND='Agg'),
            capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_shared_utilities_do_not_load_visualizers(self):
        self._run_fresh('''
            import importlib.abc
            import sys
            import torcheeg

            class BlockVisualizers(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.startswith('torcheeg.visualizers'):
                        raise AssertionError('Unexpected visualization import: ' + fullname)

            sys.meta_path.insert(0, BlockVisualizers())
            import torcheeg.utils as utils
            from torcheeg.utils import download_url, get_random_dir_path, LazyLoader
            assert download_url is utils.download_url
            assert not any(name.startswith('torcheeg.visualizers') for name in sys.modules)
        ''')


class TestUtilityFunctions(unittest.TestCase):
    def setUp(self):
        logger = logging.getLogger('torcheeg')
        self.addCleanup(logger.setLevel, logger.level)
        logger.setLevel(logging.INFO)
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self._root = Path(self._directory.name)

    def test_file_checksums_match_sha256_across_read_blocks(self):
        content = b'0123456789' * 150000
        path = self._root / 'signal.bin'
        path.write_bytes(content)
        self.assertEqual(utils.compute_checksum(
            path), hashlib.sha256(content).hexdigest())
        self.assertEqual(utils.compute_fingerprint(content),
                         hashlib.sha256(content).hexdigest())
        path.write_bytes(content + b'changed')
        self.assertNotEqual(utils.compute_checksum(
            path), hashlib.sha256(content).hexdigest())

    def test_unique_paths_and_temporary_directories_are_independent(self):
        paths = {utils.get_random_dir_path(
            self._directory.name, 'eeg') for _ in range(20)}
        self.assertEqual(len(paths), 20)
        self.assertTrue(all(Path(path).parent == self._root for path in paths))
        self.assertTrue(all(Path(path).name.startswith('eeg_')
                        for path in paths))
        path = utils.get_temp_dir_path()
        self.addCleanup(shutil.rmtree, path, ignore_errors=True)
        self.assertTrue(Path(path).is_dir())
        with patch('torcheeg.utils.os.path.expanduser', return_value=self._directory.name):
            self.assertEqual(utils.get_package_dir_path(),
                             str(self._root / '.torcheeg'))

    def test_download_uses_requested_filename_and_reuses_existing_content(self):
        for verbose in (False, True):
            with self.subTest(verbose=verbose):
                directory = self._root / str(verbose) / 'download'
                with patch('urllib.request.urlopen', return_value=io.BytesIO(b'fake eeg')) as response:
                    path = utils.download_url('https://example.invalid/sample.bin?query=1',
                                              str(directory), verbose=verbose)
                response.assert_called_once()
                self.assertEqual(Path(path).name, 'sample.bin')
                self.assertEqual(Path(path).read_bytes(), b'fake eeg')
                with patch('urllib.request.urlopen') as response:
                    reused = utils.download_url('https://example.invalid/sample.bin',
                                                str(directory), verbose=verbose)
                self.assertEqual(reused, path)
                response.assert_not_called()
        with patch('urllib.request.urlopen', return_value=io.BytesIO(b'custom')):
            path = utils.download_url('https://example.invalid/original',
                                      str(self._root / 'custom'), filename='selected.bin', verbose=False)
        self.assertEqual(Path(path).name, 'selected.bin')
        self.assertEqual(Path(path).read_bytes(), b'custom')

    def test_seed_everything_repeats_python_numpy_and_torch_randomness(self):
        self.addCleanup(random.setstate, random.getstate())
        self.addCleanup(np.random.set_state, np.random.get_state())
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        previous = torch.backends.cudnn.deterministic
        self.addCleanup(setattr, torch.backends.cudnn,
                        'deterministic', previous)
        with patch.dict(os.environ):
            outcomes = []
            for _ in range(2):
                utils.seed_everything(42)
                outcomes.append(
                    (random.random(), np.random.rand(), torch.rand(3)))
            self.assertEqual(os.environ['PYTHONHASHSEED'], '42')
            self.assertEqual(outcomes[0][:2], outcomes[1][:2])
            torch.testing.assert_close(outcomes[0][2], outcomes[1][2])


class TestCacheValidation(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self._path = Path(self._directory.name) / 'cache'

    def _write(self, directory):
        (Path(directory) / 'samples.bin').write_bytes(b'original samples')

    def test_cache_reuse_skips_writer_and_validates_with_callback(self):
        validator = unittest.mock.Mock()
        writer = unittest.mock.Mock(side_effect=self._write)
        for _ in range(2):
            utils.ensure_cache(self._path, 'identity', writer,
                               files=['samples.bin'], validator=validator)
        writer.assert_called_once()
        self.assertEqual(validator.call_count, 2)
        utils.validate_cache(self._path, 'identity', files=['samples.bin'])
        self.assertEqual(
            (self._path / 'samples.bin').read_bytes(), b'original samples')

    def test_invalid_manifest_schemas_and_fingerprints_are_rejected(self):
        utils.ensure_cache(self._path, 'identity', self._write)
        manifest = self._path / 'manifest.json'
        for content in ('not-json', '[]', '{}', '{"fingerprint":"other","files":{}}',
                        '{"fingerprint":"identity","files":null}',
                        '{"fingerprint":"identity","files":{}}'):
            with self.subTest(manifest=content):
                manifest.write_text(content)
                with self.assertRaises(ValueError):
                    utils.validate_cache(self._path, 'identity')
        self.assertEqual(
            (self._path / 'samples.bin').read_bytes(), b'original samples')

    def test_output_corruption_extra_files_and_symlinks_are_rejected(self):
        utils.ensure_cache(self._path, 'identity', self._write)
        original_manifest = (self._path / 'manifest.json').read_bytes()
        sample = self._path / 'samples.bin'
        sample.write_bytes(b'corrupted')
        with self.assertRaisesRegex(ValueError, 'Corrupt'):
            utils.validate_cache(self._path, 'identity')
        sample.write_bytes(b'original samples')
        extra = self._path / 'unexpected'
        extra.touch()
        with self.assertRaises(ValueError):
            utils.validate_cache(self._path, 'identity')
        extra.unlink()
        with self.assertRaises(ValueError):
            utils.validate_cache(self._path, 'identity', files=['missing.bin'])
        outside = self._path.parent / 'outside'
        outside.write_bytes(sample.read_bytes())
        sample.unlink()
        sample.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'path'):
            utils.validate_cache(self._path, 'identity')
        self.assertEqual(
            (self._path / 'manifest.json').read_bytes(), original_manifest)

    def test_failed_or_incomplete_writer_can_retry_without_partial_results(self):
        def fail(directory):
            self._write(directory)
            raise RuntimeError('writer failed')
        with self.assertRaisesRegex(RuntimeError, 'writer failed'):
            utils.ensure_cache(self._path, 'identity', fail)
        self.assertFalse(self._path.exists())
        self.assertEqual(list(self._path.parent.glob('.torcheeg-cache-*')), [])
        with self.assertRaises(ValueError):
            utils.ensure_cache(self._path, 'identity', lambda directory: None)
        utils.ensure_cache(self._path, 'identity', self._write)
        utils.validate_cache(self._path, 'identity')


if __name__ == '__main__':
    unittest.main()
