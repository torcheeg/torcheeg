import unittest

import matplotlib.pyplot as plt
import numpy as np

from torcheeg.transforms import CWTSpectrum, DWTDecomposition


class TestSpectrumTransforms(unittest.TestCase):

    def setUp(self):
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)

    def tearDown(self):
        plt.close('all')

    def test_dwt_decomposition(self):
        eeg = np.random.randn(32, 1000)
        transformed_eeg = DWTDecomposition()(eeg)
        self.assertEqual(transformed_eeg.shape, (2, 32, 500))

    def test_cwt_spectrum(self):
        eeg = np.random.randn(32, 1000)
        transformed_eeg = CWTSpectrum()(eeg)
        self.assertEqual(transformed_eeg.shape, (32, 128, 1000))
        transformed_eeg = CWTSpectrum(contourf=True)(eeg)
        self.assertEqual(transformed_eeg.shape, (32, 480, 640, 4))


if __name__ == '__main__':
    unittest.main()
