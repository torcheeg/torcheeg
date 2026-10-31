"""Independent population-skewness checks for the full frequency-band transform."""

import unittest

import numpy as np
from scipy.signal import butter, lfilter
from scipy.stats import skew

from torcheeg.transforms import BandSkewness

BANDS = {
    'theta': (4, 8),
    'alpha': (8, 14),
    'beta': (14, 31),
    'gamma': (31, 45),
}


def reference_features(eeg):
    features = []
    for low, high in BANDS.values():
        b, a = butter(5, [low / 64, high / 64], btype='band')
        filtered = lfilter(b, a, eeg, axis=-1)
        # SciPy uses centered moments rather than the optimized raw moments.
        features.append(skew(filtered, axis=-1, bias=True))
    return np.stack(features, axis=-1)


class TestBandSkewness(unittest.TestCase):
    def test_filtered_values_against_independent_centered_moments(self):
        rng = np.random.default_rng(322)
        transform = BandSkewness(band_dict=BANDS)
        for dtype in (np.float32, np.float64, np.int16):
            for points in (128, 512, 2048):
                for scale, offset in ((1.0, 0.0), (1e-6, 0.0), (10.0, 100.0)):
                    if dtype is np.int16 and scale == 1e-6:
                        continue
                    with self.subTest(
                        dtype=dtype, points=points, scale=scale, offset=offset
                    ):
                        eeg = (
                            rng.standard_normal((4, points)) * scale + offset
                        ).astype(dtype)
                        before = eeg.copy()
                        actual = transform(eeg=eeg)['eeg']
                        np.testing.assert_allclose(
                            actual, reference_features(eeg), rtol=1e-10, atol=1e-12
                        )
                        np.testing.assert_array_equal(eeg, before)

    def test_strided_read_only_inputs_and_baseline_are_independent(self):
        rng = np.random.default_rng(323)
        eeg = rng.normal(size=(4, 1024))[:, ::2]
        baseline = rng.normal(size=(4, 1024))[:, ::-2] * 1e-6
        eeg.setflags(write=False)
        baseline.setflags(write=False)
        before_eeg, before_baseline = eeg.copy(), baseline.copy()
        actual = BandSkewness(band_dict=BANDS, apply_to_baseline=True)(
            eeg=eeg, baseline=baseline
        )
        np.testing.assert_allclose(
            actual['eeg'], reference_features(eeg), rtol=1e-10, atol=1e-12
        )
        np.testing.assert_allclose(
            actual['baseline'], reference_features(baseline), rtol=1e-10, atol=1e-12
        )
        np.testing.assert_array_equal(eeg, before_eeg)
        np.testing.assert_array_equal(baseline, before_baseline)

    def test_zero_energy_retains_undefined_population_skewness(self):
        with np.errstate(invalid='ignore', divide='ignore'):
            actual = BandSkewness(band_dict=BANDS)(
                eeg=np.zeros((2, 128)))['eeg']
        self.assertEqual(actual.shape, (2, 4))
        self.assertTrue(np.isnan(actual).all())


if __name__ == '__main__':
    unittest.main()
