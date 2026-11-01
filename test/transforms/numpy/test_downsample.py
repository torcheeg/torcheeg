import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

import numpy as np

from torcheeg.transforms import Downsample, SetSamplingRate


class TestOptionalModelDependencies(unittest.TestCase):
    def _run_without(self, modules, body):
        root = Path(__file__).resolve().parents[3]
        script = '''
import importlib.abc
import sys
import torch
import torcheeg

assert torcheeg.__file__.startswith(ROOT)

blocked = MODULES

class BlockOptionalDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise ModuleNotFoundError('Blocked optional dependency: ' + fullname)

assert not set(blocked).intersection(sys.modules)
sys.meta_path.insert(0, BlockOptionalDependencies())
'''.replace('MODULES', repr(modules)).replace('ROOT', repr(str(root))) + textwrap.dedent(body)
        env = dict(os.environ, PYTHONPATH=str(root), OMP_NUM_THREADS='1')
        result = subprocess.run([sys.executable, '-c', script], cwd=root,
                                env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_resampling_without_optional_backends(self):
        self._run_without(('samplerate', 'resampy', 'soxr'), """
            import numpy as np
            from torcheeg.transforms.numpy import downsample
            from torcheeg.utils import LazyLoader
            assert downsample.LazyLoader is LazyLoader
            eeg = np.ones((2, 32))
            transform = downsample.SetSamplingRate(32, 16, res_type='scipy')
            assert transform(eeg=eeg)['eeg'].shape == (2, 16)
            for backend, method in [('samplerate', 'linear'),
                                    ('resampy', 'kaiser_best'), ('soxr', 'soxr_hq')]:
                transform = downsample.SetSamplingRate(32, 16, res_type=method)
                try:
                    transform(eeg=eeg)
                except ImportError as error:
                    assert f'pip install {backend}' in str(error), str(error)
                    assert 'pytorch-geometric' not in str(error), str(error)
                else:
                    raise AssertionError('Selected backend must report its missing dependency')
        """)


class TestDownsampleTransforms(unittest.TestCase):

    def setUp(self):
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)

    def test_downsample(self):
        eeg = np.random.randn(32, 128)
        transformed_eeg = Downsample(num_points=32, axis=-1)(eeg=eeg)
        self.assertEqual(transformed_eeg['eeg'].shape, (32, 32))

    def test_set_sampling_rate(self):
        eeg = np.random.randn(32, 128)
        transformed_eeg = SetSamplingRate(
            origin_sampling_rate=128, target_sampling_rate=64)(eeg=eeg)
        self.assertEqual(transformed_eeg['eeg'].shape, (32, 64))

        eeg = np.random.randn(32, 128)
        transformed_eeg = SetSamplingRate(
            origin_sampling_rate=64, target_sampling_rate=128)(eeg=eeg)
        self.assertEqual(transformed_eeg['eeg'].shape, (32, 256))

        eeg = np.random.randn(32, 128, 4)
        transformed_eeg = SetSamplingRate(
            origin_sampling_rate=128, target_sampling_rate=256, axis=1)(eeg=eeg)
        self.assertEqual(transformed_eeg['eeg'].shape, (32, 256, 4))


if __name__ == '__main__':
    unittest.main()
