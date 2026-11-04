import unittest

import numpy as np

from torcheeg.transforms import Flatten


class TestFlattenTransforms(unittest.TestCase):

    def setUp(self):
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)

    def test_flatten(self):
        eeg = np.random.randn(62, 5)
        transformed_eeg = Flatten()(eeg)
        self.assertEqual(transformed_eeg.shape, (310,))
        np.testing.assert_array_equal(transformed_eeg, eeg.ravel())


if __name__ == '__main__':
    unittest.main()
