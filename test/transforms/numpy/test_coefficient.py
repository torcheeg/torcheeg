import unittest

import numpy as np

from torcheeg.transforms import ARRCoefficient


class TestCoefficientTransforms(unittest.TestCase):

    def test_arr_coefficient(self):
        eeg = np.random.randn(32, 128)
        transformed_eeg = ARRCoefficient(order=4)(eeg=eeg)
        self.assertEqual(transformed_eeg['eeg'].shape, (32, 4))


if __name__ == '__main__':
    unittest.main()
