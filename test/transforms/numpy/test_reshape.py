import unittest

import numpy as np

from torcheeg.transforms import Reshape


class TestReshape(unittest.TestCase):

    def test_reshape_preserves_values_and_input(self):
        eeg = np.arange(24).reshape(3, 8)
        result = Reshape((2, 3, 4))(eeg)
        self.assertEqual(result.shape, (2, 3, 4))
        np.testing.assert_array_equal(result.ravel(), np.arange(24))
        np.testing.assert_array_equal(eeg, np.arange(24).reshape(3, 8))

    def test_inferred_dimension_and_empty_arrays_follow_numpy(self):
        self.assertEqual(Reshape((-1, 2))(np.arange(6)).shape, (3, 2))
        self.assertEqual(Reshape((0, 2))(np.empty(0)).shape, (0, 2))

    def test_incompatible_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            Reshape((2, 4))(np.arange(6))


if __name__ == '__main__':
    unittest.main()
