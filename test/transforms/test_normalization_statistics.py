import unittest

import numpy as np

from torcheeg import transforms


class TestNormalizationStatistics(unittest.TestCase):

    def test_constant_global_sample_has_finite_zero_output(self):
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype):
                eeg = np.full((3, 8), 7, dtype=dtype)
                actual = transforms.MeanStdNormalize()(eeg=eeg)['eeg']
                np.testing.assert_array_equal(actual, np.zeros_like(eeg))
                self.assertEqual(actual.dtype, eeg.dtype)
                np.testing.assert_array_equal(eeg, np.full((3, 8), 7))

    def test_per_axis_constants_and_nonconstants(self):
        sample = np.array([[2., 2., 2.], [1., 2., 3.]])
        expected = np.array([[0., 0., 0.], [-np.sqrt(1.5), 0., np.sqrt(1.5)]])
        for axis, eeg, reference in ((1, sample, expected),
                                     (-1, sample, expected),
                                     (0, sample.T, expected.T)):
            with self.subTest(axis=axis):
                actual = transforms.MeanStdNormalize(axis=axis)(eeg=eeg)['eeg']
                np.testing.assert_allclose(actual, reference)

    def test_nonconstant_global_normalization_is_unchanged(self):
        eeg = np.arange(24, dtype=np.float64).reshape(3, 8)
        actual = transforms.MeanStdNormalize()(eeg=eeg)['eeg']
        np.testing.assert_allclose(actual, (eeg - eeg.mean()) / eeg.std())
        self.assertAlmostEqual(actual.mean(), 0.)
        self.assertAlmostEqual(actual.std(), 1.)

    def test_supplied_statistics_are_not_modified(self):
        for readonly in (False, True):
            with self.subTest(readonly=readonly):
                mean = np.array([2., 4.])
                std = np.array([0., 2.])
                if readonly:
                    mean.flags.writeable = False
                    std.flags.writeable = False
                transform = transforms.MeanStdNormalize(mean=mean, std=std, axis=0)
                eeg = np.array([[1., 2., 3.], [2., 4., 6.]])
                expected = np.array([[-1., 0., 1.], [-1., 0., 1.]])
                for _ in range(2):
                    np.testing.assert_array_equal(transform(eeg=eeg)['eeg'], expected)
                np.testing.assert_array_equal(mean, [2., 4.])
                np.testing.assert_array_equal(std, [0., 2.])
                self.assertIs(transform.mean, mean)
                self.assertIs(transform.std, std)

    def test_supplied_default_axis_statistics_are_not_modified(self):
        mean = np.array([1., 2., 3.])
        std = np.array([0., 2., 4.])
        eeg = np.array([[2., 4., 7.], [3., 6., 11.]])
        actual = transforms.MeanStdNormalize(mean=mean, std=std)(eeg=eeg)['eeg']
        np.testing.assert_array_equal(actual, [[1., 1., 1.], [2., 2., 2.]])
        np.testing.assert_array_equal(std, [0., 2., 4.])

    def test_baseline_and_compose_keep_the_key_contract(self):
        eeg = np.full((2, 4), 7.)
        baseline = np.full((2, 4), 3.)
        transform = transforms.Compose([
            transforms.MeanStdNormalize(apply_to_baseline=True),
            transforms.ToTensor(apply_to_baseline=True)
        ])
        actual = transform(eeg=eeg, baseline=baseline, sample_id='kept')
        np.testing.assert_array_equal(actual['eeg'].numpy(), np.zeros_like(eeg))
        np.testing.assert_array_equal(actual['baseline'].numpy(), np.zeros_like(baseline))
        self.assertEqual(actual['sample_id'], 'kept')
        untouched = transforms.MeanStdNormalize()(eeg=eeg, baseline=baseline)
        self.assertIs(untouched['baseline'], baseline)

    def test_training_statistics_are_reused_on_held_out_samples(self):
        train = np.array([[[1., 3.], [4., 4.]],
                          [[3., 5.], [4., 4.]]])
        mean = train.mean(axis=(0, 2))
        std = train.std(axis=(0, 2))
        transform = transforms.MeanStdNormalize(mean=mean, std=std, axis=0)
        sample = np.array([[5., 7.], [6., 8.]])
        actual = transform(eeg=sample)['eeg']
        expected = (sample - mean[:, None]) / np.array([[np.sqrt(2.)], [1.]])
        np.testing.assert_allclose(actual, expected)
        np.testing.assert_array_equal(std, [np.sqrt(2.), 0.])
        self.assertFalse(np.allclose(actual.mean(axis=1), 0.))


if __name__ == '__main__':
    unittest.main()
