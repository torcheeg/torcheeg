import unittest

import numpy as np
import torch

from torcheeg.transforms import ToTensor


class TestToTransforms(unittest.TestCase):

    def test_to_tensor(self):
        eeg = np.random.randn(32, 128)
        transformed_eeg = ToTensor()(eeg=eeg)
        self.assertEqual(eeg.shape, tuple(transformed_eeg['eeg'].shape))
        self.assertTrue(isinstance(transformed_eeg['eeg'], torch.Tensor))


if __name__ == '__main__':
    unittest.main()
