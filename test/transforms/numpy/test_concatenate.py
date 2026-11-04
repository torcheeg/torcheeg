import unittest

import numpy as np

from torcheeg.transforms import (BandBinPower, BandDifferentialEntropy,
                                 BandPowerSpectralDensity, BandSkewness,
                                 Compose, Concatenate, MapChunk,
                                 MinMaxNormalize)


class TestConcatenateTransforms(unittest.TestCase):

    def setUp(self):
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)

    def test_concat(self):
        eeg = np.random.randn(32, 128)
        transformed_eeg = Concatenate([BandSkewness(), BandBinPower()])(eeg)
        self.assertEqual(transformed_eeg.shape, (32, 8))
        transformed_eeg = Concatenate(
            [Compose([MinMaxNormalize(), BandSkewness()]), BandBinPower()])(eeg)
        self.assertEqual(transformed_eeg.shape, (32, 8))

    def test_map_chunk(self):
        eeg = np.random.randn(64, 1000)
        transformed_eeg = MapChunk(
            BandDifferentialEntropy(), chunk_size=250, overlap=0)(eeg)
        self.assertEqual(transformed_eeg.shape, (64, 16))
        transformed_eeg = Concatenate([MapChunk(BandDifferentialEntropy(), chunk_size=250, overlap=0), MapChunk(
            BandDifferentialEntropy(), chunk_size=500, overlap=0), BandDifferentialEntropy()])(eeg)
        self.assertEqual(transformed_eeg.shape, (64, 28))
        transformed_eeg = Concatenate([MapChunk(BandDifferentialEntropy(), chunk_size=250, overlap=0), MapChunk(
            BandDifferentialEntropy(), chunk_size=500, overlap=0), BandDifferentialEntropy(), BandPowerSpectralDensity()])(eeg)
        self.assertEqual(transformed_eeg.shape, (64, 32))

    def test_map_chunk_uses_each_window(self):
        eeg = np.arange(8).reshape(1, 8)
        transform = MapChunk(lambda window: window.mean(axis=-1, keepdims=True),
                             chunk_size=4, overlap=2)
        np.testing.assert_array_equal(transform(eeg), [[1.5, 3.5, 5.5]])


if __name__ == '__main__':
    unittest.main()
