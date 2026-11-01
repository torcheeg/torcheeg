import unittest

import numpy as np

from torcheeg.transforms import (BandApproximateEntropy, BandBinPower,
                                 BandDetrendedFluctuationAnalysis,
                                 BandHiguchiFractalDimension, BandHjorth,
                                 BandHurst, BandPetrosianFractalDimension,
                                 BandSampleEntropy, BandSpectralEntropy,
                                 BandSVDEntropy)


class TestBandPyeegTransforms(unittest.TestCase):

    def setUp(self):
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)

    def test_ap_entropy(self):
        eeg = np.random.randn(32, 128)
        transformed_eeg = BandApproximateEntropy()(eeg)
        self.assertEqual(transformed_eeg.shape, (32, 4))

    def test_samp_entropy(self):
        eeg = np.random.randn(32, 128)
        transformed_eeg = BandSampleEntropy()(eeg)
        self.assertEqual(transformed_eeg.shape, (32, 4))

    def test_svd_entropy(self):
        eeg = np.random.randn(32, 128)
        transformed_eeg = BandSVDEntropy()(eeg)
        self.assertEqual(transformed_eeg.shape, (32, 4))

    def test_dfa(self):
        eeg = np.random.randn(32, 128)
        transformed_eeg = BandDetrendedFluctuationAnalysis()(eeg)
        self.assertEqual(transformed_eeg.shape, (32, 4))

    def test_hfd(self):
        eeg = np.random.randn(32, 128)
        transformed_eeg = BandHiguchiFractalDimension()(eeg)
        self.assertEqual(transformed_eeg.shape, (32, 4))

    def test_hjorth(self):
        eeg = np.random.randn(32, 128)
        transformed_eeg = BandHjorth()(eeg)
        self.assertEqual(transformed_eeg.shape, (32, 4))

    def test_hurst(self):
        eeg = np.random.randn(32, 128)
        transformed_eeg = BandHurst()(eeg)
        self.assertEqual(transformed_eeg.shape, (32, 4))

    def test_pdf(self):
        eeg = np.random.randn(32, 128)
        transformed_eeg = BandPetrosianFractalDimension()(eeg)
        self.assertEqual(transformed_eeg.shape, (32, 4))

    def test_bin_power(self):
        eeg = np.random.randn(32, 128)
        transformed_eeg = BandBinPower()(eeg)
        self.assertEqual(transformed_eeg.shape, (32, 4))

    def test_spectral_entropy(self):
        eeg = np.random.randn(32, 128)
        transformed_eeg = BandSpectralEntropy()(eeg)
        self.assertEqual(transformed_eeg.shape, (32, 1))


if __name__ == '__main__':
    unittest.main()
