import unittest

import numpy as np
import torch

from torcheeg.transforms import (BaselineCorrection, BaselineRemoval, Compose,
                                 ToTensor)


class TestBaselineTransforms(unittest.TestCase):
    def test_removal_uses_bound_baseline(self):
        eeg = np.arange(8, dtype=np.float32).reshape(2, 4)
        baseline = np.ones_like(eeg)
        transform = Compose([BaselineRemoval(baseline), ToTensor()])
        torch.testing.assert_close(
            transform(eeg), torch.from_numpy(eeg - baseline))
        np.testing.assert_array_equal(baseline, np.ones_like(eeg))
        np.testing.assert_array_equal(eeg, np.arange(8).reshape(2, 4))

    def test_correction_numpy_and_torch(self):
        baseline = np.arange(6, dtype=np.float32).reshape(2, 3)
        eeg = np.ones((2, 4), dtype=np.float32)
        expected = eeg - baseline.mean(axis=-1, keepdims=True)
        np.testing.assert_allclose(BaselineCorrection(baseline)(eeg), expected)
        torch.testing.assert_close(BaselineCorrection(torch.from_numpy(baseline))(torch.from_numpy(eeg)),
                                   torch.from_numpy(expected))

    def test_invalid_shapes_and_missing_baseline(self):
        with self.assertRaises(ValueError):
            BaselineRemoval(np.ones(3))(np.ones(4))
        with self.assertRaises(TypeError):
            BaselineRemoval()
