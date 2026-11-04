import unittest

import numpy as np

from torcheeg.transforms import MeanStdNormalize, MinMaxNormalize


class TestNormalizeTransforms(unittest.TestCase):

    def setUp(self):
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)

    def test_mean_std_normalize(self):
        eeg = np.random.randn(32, 128)
        transformed_eeg = MeanStdNormalize(axis=None)(eeg)
        self.assertEqual(eeg.shape, transformed_eeg.shape)
        eeg = np.random.randn(32, 128)
        transformed_eeg = MeanStdNormalize(axis=0)(eeg)
        self.assertEqual(eeg.shape, transformed_eeg.shape)
        eeg = np.random.randn(32, 128)
        transformed_eeg = MeanStdNormalize(axis=1)(eeg)
        self.assertEqual(eeg.shape, transformed_eeg.shape)

    def test_min_max_normalize(self):
        eeg = np.random.randn(32, 128)
        transformed_eeg = MinMaxNormalize(axis=None)(eeg)
        self.assertEqual(eeg.shape, transformed_eeg.shape)
        eeg = np.random.randn(32, 128)
        transformed_eeg = MinMaxNormalize(axis=0)(eeg)
        self.assertEqual(eeg.shape, transformed_eeg.shape)
        eeg = np.random.randn(32, 128)
        transformed_eeg = MinMaxNormalize(axis=1)(eeg)
        self.assertEqual(eeg.shape, transformed_eeg.shape)

    def test_normalization_matches_independent_statistics(self):
        eeg = np.array([[1.0, 3.0, 8.0], [2.0, 6.0, 12.0]])
        original = eeg.copy()
        for axis in (None, 0, 1):
            with self.subTest(axis=axis):
                mean = eeg.mean(axis=axis, keepdims=True)
                std = eeg.std(axis=axis, keepdims=True)
                actual = MeanStdNormalize(axis=axis)(eeg)
                np.testing.assert_allclose(actual, (eeg - mean) / std)
                minimum = eeg.min(axis=axis, keepdims=True)
                maximum = eeg.max(axis=axis, keepdims=True)
                actual = MinMaxNormalize(axis=axis)(eeg)
                np.testing.assert_allclose(
                    actual, (eeg - minimum) / (maximum - minimum))
        np.testing.assert_array_equal(eeg, original)

    def test_normalization_of_constant_signals_is_finite(self):
        for transform_class in (MeanStdNormalize, MinMaxNormalize):
            for axis in (None, 0, 1):
                for dtype in (np.float32, np.float64, np.int64):
                    with self.subTest(transform=transform_class.__name__, axis=axis, dtype=dtype):
                        eeg = np.full((2, 3), 7, dtype=dtype)
                        result = transform_class(axis=axis)(eeg)
                        np.testing.assert_array_equal(result, np.zeros((2, 3)))
                        np.testing.assert_array_equal(eeg, np.full((2, 3), 7))
        eeg = np.array([[2.0, 2.0, 2.0], [1.0, 3.0, 5.0]])
        for transform_class in (MeanStdNormalize, MinMaxNormalize):
            result = transform_class(axis=1)(eeg)
            self.assertTrue(np.isfinite(result).all())
            np.testing.assert_array_equal(result[0], np.zeros(3))


if __name__ == '__main__':
    unittest.main()
