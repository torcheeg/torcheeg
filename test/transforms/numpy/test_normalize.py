import unittest

import numpy as np

from torcheeg.transforms import MeanStdNormalize, MinMaxNormalize


class TestNormalizeTransforms(unittest.TestCase):

    def test_mean_std_normalize(self):
        eeg = np.random.randn(32, 128)
        transformed_eeg = MeanStdNormalize(axis=None)(eeg=eeg)
        self.assertEqual(eeg.shape, transformed_eeg['eeg'].shape)

        eeg = np.random.randn(32, 128)
        transformed_eeg = MeanStdNormalize(axis=0)(eeg=eeg)
        self.assertEqual(eeg.shape, transformed_eeg['eeg'].shape)

        eeg = np.random.randn(32, 128)
        transformed_eeg = MeanStdNormalize(axis=1)(eeg=eeg)
        self.assertEqual(eeg.shape, transformed_eeg['eeg'].shape)

    def test_min_max_normalize(self):
        eeg = np.random.randn(32, 128)
        transformed_eeg = MinMaxNormalize(axis=None)(eeg=eeg)
        self.assertEqual(eeg.shape, transformed_eeg['eeg'].shape)

        eeg = np.random.randn(32, 128)
        transformed_eeg = MinMaxNormalize(axis=0)(eeg=eeg)
        self.assertEqual(eeg.shape, transformed_eeg['eeg'].shape)

        eeg = np.random.randn(32, 128)
        transformed_eeg = MinMaxNormalize(axis=1)(eeg=eeg)
        self.assertEqual(eeg.shape, transformed_eeg['eeg'].shape)


if __name__ == '__main__':
    unittest.main()
