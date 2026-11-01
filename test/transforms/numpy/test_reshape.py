import unittest

import numpy as np

from torcheeg.transforms import Reshape


class TestReshape(unittest.TestCase):
    def test_reshape_preserves_values_and_other_targets(self):
        eeg = np.arange(24).reshape(3, 8)
        baseline = eeg.copy()
        result = Reshape((2, 3, 4))(eeg=eeg, baseline=baseline, y='rest')
        self.assertEqual(result['eeg'].shape, (2, 3, 4))
        np.testing.assert_array_equal(result['eeg'].ravel(), np.arange(24))
        self.assertIs(result['baseline'], baseline)
        self.assertEqual(result['y'], 'rest')
        np.testing.assert_array_equal(eeg, np.arange(24).reshape(3, 8))

    def test_inferred_dimension_and_empty_arrays_follow_numpy(self):
        self.assertEqual(Reshape((-1, 2))(eeg=np.arange(6))['eeg'].shape,
                         (3, 2))
        self.assertEqual(Reshape((0, 2))(eeg=np.empty(0))['eeg'].shape,
                         (0, 2))

    def test_incompatible_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            Reshape((2, 4))(eeg=np.arange(6))


if __name__ == '__main__':
    unittest.main()
