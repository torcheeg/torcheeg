import unittest

import numpy as np

from torcheeg.transforms import ARRCoefficient


class TestCoefficientTransforms(unittest.TestCase):

    def setUp(self):
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)

    def test_arr_coefficient(self):
        eeg = np.random.randn(32, 128)
        transformed_eeg = ARRCoefficient(order=4)(eeg)
        self.assertEqual(transformed_eeg.shape, (32, 4))


if __name__ == '__main__':
    unittest.main()
