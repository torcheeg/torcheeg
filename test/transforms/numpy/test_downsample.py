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
        script = "\nimport importlib.abc\nimport sys\nimport torch\nimport torcheeg\n\nassert torcheeg.__file__.startswith(ROOT)\n\nblocked = MODULES\n\nclass BlockOptionalDependencies(importlib.abc.MetaPathFinder):\n    def find_spec(self, fullname, path=None, target=None):\n        if fullname.split('.')[0] in blocked:\n            raise ModuleNotFoundError('Blocked optional dependency: ' + fullname)\n\nassert not set(blocked).intersection(sys.modules)\nsys.meta_path.insert(0, BlockOptionalDependencies())\n".replace(
            'MODULES', repr(modules)).replace('ROOT', repr(str(root))) + textwrap.dedent(body)
        env = dict(os.environ, PYTHONPATH=str(root), OMP_NUM_THREADS='1')
        result = subprocess.run([sys.executable, '-c', script], cwd=root,
                                env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_resampling_without_optional_backends(self):
        self._run_without(('samplerate', 'resampy', 'soxr'),
                          "\n            import numpy as np\n            from torcheeg.transforms.numpy import downsample\n            from torcheeg.utils import LazyLoader\n            assert downsample.LazyLoader is LazyLoader\n            eeg = np.ones((2, 32))\n            transform = downsample.SetSamplingRate(32, 16, res_type='scipy')\n            assert transform(eeg).shape == (2, 16)\n            for backend, method in [('samplerate', 'linear'),\n                                    ('resampy', 'kaiser_best'), ('soxr', 'soxr_hq')]:\n                transform = downsample.SetSamplingRate(32, 16, res_type=method)\n                try:\n                    transform(eeg)\n                except ImportError as error:\n                    assert f'pip install {backend}' in str(error), str(error)\n                    assert 'pytorch-geometric' not in str(error), str(error)\n                else:\n                    raise AssertionError('Selected backend must report its missing dependency')\n        ")


class TestDownsampleTransforms(unittest.TestCase):

    def setUp(self):
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)

    def test_downsample(self):
        eeg = np.random.randn(32, 128)
        transformed_eeg = Downsample(num_points=32, axis=-1)(eeg)
        self.assertEqual(transformed_eeg.shape, (32, 32))

    def test_set_sampling_rate(self):
        eeg = np.random.randn(32, 128)
        transformed_eeg = SetSamplingRate(
            origin_sampling_rate=128, target_sampling_rate=64)(eeg)
        self.assertEqual(transformed_eeg.shape, (32, 64))
        eeg = np.random.randn(32, 128)
        transformed_eeg = SetSamplingRate(
            origin_sampling_rate=64, target_sampling_rate=128)(eeg)
        self.assertEqual(transformed_eeg.shape, (32, 256))
        eeg = np.random.randn(32, 128, 4)
        transformed_eeg = SetSamplingRate(
            origin_sampling_rate=128, target_sampling_rate=256, axis=1)(eeg)
        self.assertEqual(transformed_eeg.shape, (32, 256, 4))

    def test_repr_uses_current_configuration(self):
        self.assertIn('target_sampling_rate=64',
                      repr(SetSamplingRate(128, 64)))


if __name__ == '__main__':
    unittest.main()
