import io
import os
import random
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from torcheeg import utils


class TestUtilsImports(unittest.TestCase):
    def run_fresh(self, body):
        root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [sys.executable, '-c', textwrap.dedent(body)], cwd=root,
            env=dict(os.environ, PYTHONPATH=str(root), MPLBACKEND='Agg'),
            capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_shared_utilities_do_not_load_visualizers(self):
        self.run_fresh('''
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


class TestUtilityRegressions(unittest.TestCase):
    def test_verbose_download_and_reuse_accept_standard_logging(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = os.path.join(directory, 'download')
            with patch('urllib.request.urlopen', return_value=io.BytesIO(b'fake EEG')):
                filename = utils.download_url(
                    'https://example.invalid/signal.bin', folder)
            with open(filename, 'rb') as file:
                self.assertEqual(file.read(), b'fake EEG')
            with patch('urllib.request.urlopen') as response:
                self.assertEqual(utils.download_url(
                    'https://example.invalid/signal.bin', folder), filename)
            response.assert_not_called()

    def test_seed_sets_the_standard_python_hash_variable(self):
        self.addCleanup(random.setstate, random.getstate())
        self.addCleanup(np.random.set_state, np.random.get_state())
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        self.addCleanup(setattr, torch.backends.cudnn,
                        'deterministic', torch.backends.cudnn.deterministic)
        with patch.dict(os.environ):
            utils.seed_everything(42)
            self.assertEqual(os.environ['PYTHONHASHSEED'], '42')


if __name__ == '__main__':
    unittest.main()
