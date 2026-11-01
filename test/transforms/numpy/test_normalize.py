import unittest

import numpy as np

from torcheeg.transforms import MeanStdNormalize, MinMaxNormalize


class TestNormalizeTransforms(unittest.TestCase):

    def setUp(self):
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)

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

    def test_normalization_matches_independent_statistics(self):
        eeg = np.array([[1., 3., 8.], [2., 6., 12.]])
        original = eeg.copy()
        for axis in (None, 0, 1):
            with self.subTest(axis=axis):
                mean = eeg.mean(axis=axis, keepdims=True)
                std = eeg.std(axis=axis, keepdims=True)
                actual = MeanStdNormalize(axis=axis)(eeg=eeg)['eeg']
                np.testing.assert_allclose(actual, (eeg - mean) / std)
                minimum = eeg.min(axis=axis, keepdims=True)
                maximum = eeg.max(axis=axis, keepdims=True)
                actual = MinMaxNormalize(axis=axis)(eeg=eeg)['eeg']
                np.testing.assert_allclose(
                    actual, (eeg - minimum) / (maximum - minimum))
        np.testing.assert_array_equal(eeg, original)

    def test_constant_channel_is_finite_and_centers_at_zero(self):
        eeg = np.array([[2., 2., 2.], [1., 3., 5.]])
        for transform in (MeanStdNormalize(axis=1), MinMaxNormalize(axis=1)):
            with self.subTest(transform=type(transform).__name__):
                result = transform(eeg=eeg)['eeg']
                self.assertTrue(np.isfinite(result).all())
                np.testing.assert_array_equal(result[0], np.zeros(3))


if __name__ == '__main__':
    unittest.main()
