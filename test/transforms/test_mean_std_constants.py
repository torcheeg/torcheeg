import unittest
import warnings

import numpy as np

from torcheeg.transforms import MeanStdNormalize


def legacy_normalize(eeg, axis):
    mean = eeg.mean(axis=axis, keepdims=axis is not None)
    std = eeg.std(axis=axis, keepdims=axis is not None)
    return (eeg - mean) / np.where(std == 0, 1, std)


class TestMeanStdConstantStatistics(unittest.TestCase):
    def test_computed_float_constants_are_exact_zero_for_each_axis(self):
        for dtype in [np.float16, np.float32, np.float64]:
            for value in [.113596, .1, -17.1234]:
                for axis in [None, 0, 1, -1]:
                    with self.subTest(dtype=dtype, value=value, axis=axis):
                        eeg = np.full((19, 250), value, dtype=dtype)
                        actual = MeanStdNormalize(axis=axis)(eeg=eeg)['eeg']
                        self.assertTrue(np.isfinite(actual).all())
                        np.testing.assert_array_equal(actual, np.zeros_like(actual))
                        self.assertEqual(actual.dtype, legacy_normalize(eeg, axis).dtype)

    def test_only_constant_channels_change_in_mixed_input(self):
        eeg = np.stack([np.full(250, .113596), np.linspace(-2, 3, 250),
                        np.cos(np.arange(250))]).astype(np.float32)
        original = eeg.copy()
        actual = MeanStdNormalize(axis=-1)(eeg=eeg)['eeg']
        np.testing.assert_array_equal(actual[0], 0)
        np.testing.assert_array_equal(actual[1:], legacy_normalize(eeg, -1)[1:])
        np.testing.assert_array_equal(eeg, original)

    def test_variable_and_near_constant_values_keep_existing_numerical_behavior(self):
        generator = np.random.default_rng(42)
        for dtype in [np.float16, np.float32, np.float64]:
            for axis in [None, 0, 1, -1]:
                eeg = generator.standard_normal((19, 250)).astype(dtype)
                np.testing.assert_array_equal(MeanStdNormalize(axis=axis)(eeg=eeg)['eeg'], legacy_normalize(eeg, axis))
        values = np.tile(np.array([1, np.nextafter(np.float32(1), np.float32(2))]), (19, 125))
        actual = MeanStdNormalize(axis=-1)(eeg=values)['eeg']
        np.testing.assert_array_equal(actual, legacy_normalize(values, -1))
        self.assertTrue(np.any(actual != 0))

    def test_supplied_statistics_keep_their_mean_and_remain_unchanged(self):
        eeg = np.full((2, 250), .113596, dtype=np.float32)
        mean = np.array([0, .25], dtype=np.float32)
        std = np.array([0, 2], dtype=np.float32)
        old_mean, old_std = mean.copy(), std.copy()
        actual = MeanStdNormalize(mean=mean, std=std, axis=0)(eeg=eeg)['eeg']
        expected = (eeg - old_mean[:, None]) / np.where(old_std[:, None] == 0, 1, old_std[:, None])
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue(np.any(actual != 0))
        np.testing.assert_array_equal(mean, old_mean)
        np.testing.assert_array_equal(std, old_std)

    def test_baseline_and_noncontiguous_inputs_use_independent_statistics(self):
        eeg = np.full((250, 19), .113596, dtype=np.float32).T
        baseline = np.full((250, 19), -.113596, dtype=np.float32).T
        source, old_baseline = eeg.copy(), baseline.copy()
        output = MeanStdNormalize(axis=-1, apply_to_baseline=True)(eeg=eeg, baseline=baseline)
        np.testing.assert_array_equal(output['eeg'], 0)
        np.testing.assert_array_equal(output['baseline'], 0)
        np.testing.assert_array_equal(eeg, source)
        np.testing.assert_array_equal(baseline, old_baseline)

    def test_empty_arrays_keep_the_existing_empty_result(self):
        for axis in [None, 0, 1]:
            eeg = np.empty((0, 3), dtype=np.float32)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                actual = MeanStdNormalize(axis=axis)(eeg=eeg)['eeg']
                expected = legacy_normalize(eeg, axis)
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.dtype, expected.dtype)
            np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
