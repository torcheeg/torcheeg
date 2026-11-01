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
        transformed_eeg = Concatenate(
            [BandSkewness(), BandBinPower()])(eeg=eeg)
        self.assertEqual(transformed_eeg['eeg'].shape, (32, 8))

        transformed_eeg = Concatenate(
            [Compose([MinMaxNormalize(), BandSkewness()]),
             BandBinPower()])(eeg=eeg)
        self.assertEqual(transformed_eeg['eeg'].shape, (32, 8))

    def test_map_chunk(self):
        eeg = np.random.randn(64, 1000)
        transformed_eeg = MapChunk(BandDifferentialEntropy(),
                                   chunk_size=250,
                                   overlap=0)(eeg=eeg)
        self.assertEqual(transformed_eeg['eeg'].shape, (64, 16))

        transformed_eeg = Concatenate([
            MapChunk(BandDifferentialEntropy(), chunk_size=250, overlap=0),
            MapChunk(BandDifferentialEntropy(), chunk_size=500, overlap=0),
            BandDifferentialEntropy()
        ])(eeg=eeg)
        self.assertEqual(transformed_eeg['eeg'].shape, (64, 28))

        transformed_eeg = Concatenate([
            MapChunk(BandDifferentialEntropy(), chunk_size=250, overlap=0),
            MapChunk(BandDifferentialEntropy(), chunk_size=500, overlap=0),
            BandDifferentialEntropy(),
            BandPowerSpectralDensity()
        ])(eeg=eeg)
        self.assertEqual(transformed_eeg['eeg'].shape, (64, 32))


if __name__ == '__main__':
    unittest.main()
