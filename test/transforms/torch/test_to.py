import unittest

import numpy as np
import torch

from torcheeg.transforms import ToTensor


class TestToTransforms(unittest.TestCase):

    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)

    def test_to_tensor(self):
        eeg = np.random.randn(32, 128)
        transformed_eeg = ToTensor()(eeg=eeg)
        self.assertEqual(eeg.shape, tuple(transformed_eeg['eeg'].shape))
        self.assertTrue(isinstance(transformed_eeg['eeg'], torch.Tensor))

    def test_conversion_preserves_values_and_cpu_float_dtype(self):
        eeg = np.arange(12, dtype=np.float64).reshape(3, 4)
        result = ToTensor()(eeg=eeg, y='rest')
        self.assertEqual(result['eeg'].device.type, 'cpu')
        self.assertEqual(result['eeg'].dtype, torch.float32)
        torch.testing.assert_close(
            result['eeg'], torch.tensor(eeg, dtype=torch.float32))
        self.assertEqual(result['y'], 'rest')
        np.testing.assert_array_equal(eeg, np.arange(12).reshape(3, 4))


if __name__ == '__main__':
    unittest.main()
