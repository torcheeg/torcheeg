import unittest

import numpy as np
from torcheeg.transforms import ToGrid, ToInterpolatedGrid, To2d, MeanStdNormalize, MinMaxNormalize, BandDifferentialEntropy, BandPowerSpectralDensity, BandMeanAbsoluteDeviation, BandKurtosis, BandSkewness, Concatenate, ChunkConcatenate, PickElectrode, CWTSpectrum, ARRCoefficient
from torcheeg.datasets.constants import DEAP_CHANNEL_LOCATION_DICT, DEAP_CHANNEL_LIST


class TestNumpyTransforms(unittest.TestCase):
    def test_cwt_spectrum(self):
        eeg = np.random.randn(32, 1000)
        transformed_eeg = CWTSpectrum()(eeg=eeg)
        self.assertEqual(transformed_eeg['eeg'].shape, (32, 128, 1000))

        transformed_eeg = CWTSpectrum(contourf=True)(eeg=eeg)
        self.assertEqual(transformed_eeg['eeg'].shape, (32, 480, 640, 4))

    def test_pick_electrode(self):
        eeg = np.random.randn(32, 128)
        transformed_eeg = PickElectrode([1, 2])(eeg=eeg)
        self.assertEqual(transformed_eeg['eeg'].shape, (2, 128))

        pick = [
            'FP1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1',
            'P3', 'P7', 'PO3', 'O1', 'FP2', 'AF4', 'F4', 'F8', 'FC6', 'FC2',
            'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2'
        ]
        pick_list = PickElectrode.to_index_list(pick, DEAP_CHANNEL_LIST)
        self.assertEqual(len(pick_list), 28)

        transformed_eeg = PickElectrode(
            PickElectrode.to_index_list(pick, DEAP_CHANNEL_LIST))(eeg=eeg)
        self.assertEqual(transformed_eeg['eeg'].shape, (28, 128))

    def test_to_2d(self):
        eeg = np.random.randn(32, 128)
        transformed_eeg = To2d()(eeg=eeg)
        self.assertEqual(transformed_eeg['eeg'].shape, (1, 32, 128))

    def test_to_grid(self):
        eeg = np.random.randn(32, 128)
        transformed_eeg = ToGrid(DEAP_CHANNEL_LOCATION_DICT)(eeg=eeg)
        self.assertEqual(transformed_eeg['eeg'].shape, (128, 9, 9))

    def test_to_interpolated_grid(self):
        eeg = np.random.randn(32, 128)
        transformed_eeg = ToInterpolatedGrid(DEAP_CHANNEL_LOCATION_DICT)(
            eeg=eeg)
        self.assertEqual(transformed_eeg['eeg'].shape, (128, 9, 9))

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

    def test_differential_entropy(self):
        eeg = np.random.randn(32, 128)
        transformed_eeg = BandDifferentialEntropy()(eeg=eeg)
        self.assertEqual(transformed_eeg['eeg'].shape, (32, 4))

    def test_band_power_spectral_density(self):
        eeg = np.random.randn(32, 128)
        transformed_eeg = BandPowerSpectralDensity()(eeg=eeg)
        self.assertEqual(transformed_eeg['eeg'].shape, (32, 4))

    def test_band_mean_absolute_deviation(self):
        eeg = np.random.randn(32, 128)
        transformed_eeg = BandMeanAbsoluteDeviation()(eeg=eeg)
        self.assertEqual(transformed_eeg['eeg'].shape, (32, 4))

    def test_band_kurtosis(self):
        eeg = np.random.randn(32, 128)
        transformed_eeg = BandKurtosis()(eeg=eeg)
        self.assertEqual(transformed_eeg['eeg'].shape, (32, 4))

    def test_band_skewness(self):
        eeg = np.random.randn(32, 128)
        transformed_eeg = BandSkewness()(eeg=eeg)
        self.assertEqual(transformed_eeg['eeg'].shape, (32, 4))

    def test_arr_coefficient(self):
        eeg = np.random.randn(32, 128)
        transformed_eeg = ARRCoefficient(order=4)(eeg=eeg)
        self.assertEqual(transformed_eeg['eeg'].shape, (32, 4))

    def test_concat(self):
        eeg = np.random.randn(32, 128)
        transformed_eeg = Concatenate([BandSkewness(), BandSkewness()])(eeg=eeg)
        self.assertEqual(transformed_eeg['eeg'].shape, (32, 8))

    def test_chunk_concat(self):
        eeg = np.random.randn(64, 1000)
        transformed_eeg = ChunkConcatenate(
            [BandDifferentialEntropy(),
             BandMeanAbsoluteDeviation()],
            chunk_size=250,
            overlap=0)(eeg=eeg)
        self.assertEqual(transformed_eeg['eeg'].shape, (64, 32))

        transformed_eeg = Concatenate([
            ChunkConcatenate([BandDifferentialEntropy()],
                             chunk_size=250,
                             overlap=0),
            ChunkConcatenate([BandDifferentialEntropy()],
                             chunk_size=500,
                             overlap=0),
            BandDifferentialEntropy()
        ])(eeg=eeg)
        self.assertEqual(transformed_eeg['eeg'].shape, (64, 28))

        transformed_eeg = Concatenate([
            ChunkConcatenate(
                [BandDifferentialEntropy(),
                 BandPowerSpectralDensity()],
                chunk_size=250,
                overlap=0),
            ChunkConcatenate(
                [BandDifferentialEntropy(),
                 BandPowerSpectralDensity()],
                chunk_size=500,
                overlap=0),
            BandDifferentialEntropy(),
            BandPowerSpectralDensity()
        ])(eeg=eeg)
        self.assertEqual(transformed_eeg['eeg'].shape, (64, 56))


if __name__ == '__main__':
    unittest.main()