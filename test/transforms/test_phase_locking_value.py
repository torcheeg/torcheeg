import unittest

import numpy as np
from scipy.signal import hilbert

from torcheeg.transforms import (Compose, PhaseLockingCorrelation,
                                 PhaseLockingValue)


def periodic_cosine(cycles, phase=0.0, n_samples=256):
    """Use integer cycles to avoid Hilbert-transform boundary artifacts."""
    theta = 2.0 * np.pi * np.arange(n_samples) / n_samples
    return np.cos(cycles * theta + phase)


def legacy_reference(eeg):
    """The original normalized analytic correlation, expressed as a Gram matrix."""
    analytic = hilbert(eeg, axis=-1)
    norms = np.linalg.norm(analytic, axis=-1)
    correlation = (analytic @ analytic.conj().T) / np.outer(norms, norms)
    return np.angle(correlation)


class TestPhaseLockingValue(unittest.TestCase):

    def test_constant_phase_differences_are_fully_locked(self):
        reference = periodic_cosine(16)
        for phase in (0.0, np.pi / 2.0, np.pi):
            with self.subTest(phase=phase):
                eeg = np.stack([reference, periodic_cosine(16, phase)])
                result = PhaseLockingValue()(eeg=eeg)['eeg']
                np.testing.assert_allclose(result, np.ones((1, 2, 2)),
                                           rtol=0.0, atol=1e-12)

    def test_different_integer_cycle_carriers_are_not_locked(self):
        eeg = np.stack([periodic_cosine(16), periodic_cosine(19)])
        result = PhaseLockingValue()(eeg=eeg)['eeg']
        np.testing.assert_allclose(result[0], np.eye(2),
                                   rtol=0.0, atol=1e-12)

    def test_amplitude_modulation_separates_plv_from_correlation_magnitude(self):
        theta = 2.0 * np.pi * np.arange(256) / 256
        carrier = np.cos(16 * theta)
        modulation = 0.5 * np.cos(2 * theta)
        eeg = np.stack([(1.0 + modulation) * carrier,
                        (1.0 - modulation) * carrier])

        # Positive, slowly varying envelopes preserve the same analytic phase.
        result = PhaseLockingValue()(eeg=eeg)['eeg']
        np.testing.assert_allclose(result, np.ones((1, 2, 2)),
                                   rtol=0.0, atol=1e-12)

        # E[A_x A_y] / sqrt(E[A_x**2] E[A_y**2]) = 7/9, not PLV.
        analytic = hilbert(eeg, axis=-1)
        magnitude = abs(np.vdot(analytic[1], analytic[0])) / (
            np.linalg.norm(analytic[0]) * np.linalg.norm(analytic[1]))
        self.assertAlmostEqual(magnitude, 7.0 / 9.0, places=12)
        self.assertGreater(abs(result[0, 0, 1] - magnitude), 0.2)

    def test_range_symmetry_and_unit_diagonal(self):
        eeg = np.random.RandomState(42).normal(size=(4, 128))
        result = PhaseLockingValue()(eeg=eeg)['eeg'][0]
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue((result >= 0.0).all())
        self.assertTrue((result <= 1.0 + 1e-12).all())
        np.testing.assert_allclose(result, result.T, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(np.diag(result), np.ones(4),
                                   rtol=0.0, atol=1e-12)

    def test_partial_locking_matches_known_analytic_signal(self):
        theta = 2.0 * np.pi * np.arange(256) / 256
        # Both harmonics are below Nyquist. Their analytic signals are known
        # without using Hilbert or the implementation's phase extraction.
        eeg = np.stack([np.cos(16 * theta),
                        np.cos(16 * theta) + 0.5 * np.cos(19 * theta)])
        relative = 1.0 + 0.5 * np.exp(3j * theta)
        expected = abs(np.mean(relative / np.abs(relative)))
        actual = PhaseLockingValue()(eeg=eeg)['eeg'][0, 0, 1]
        self.assertGreater(expected, 0.0)
        self.assertLess(expected, 1.0)
        self.assertAlmostEqual(actual, expected, places=12)

    def test_signed_constant_scaling_does_not_change_plv(self):
        eeg = np.stack([periodic_cosine(16),
                        periodic_cosine(16, np.pi / 3.0),
                        periodic_cosine(19)])
        expected = PhaseLockingValue()(eeg=eeg)['eeg']
        scaled = eeg * np.array([-3.0, 0.2, -7.0])[:, np.newaxis]
        actual = PhaseLockingValue()(eeg=scaled)['eeg']
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)

    def test_small_nonzero_analytic_amplitudes_are_valid(self):
        eeg = np.stack([periodic_cosine(16),
                        periodic_cosine(16, np.pi / 2.0)]) * 1e-200
        result = PhaseLockingValue()(eeg=eeg)['eeg']
        np.testing.assert_allclose(result, np.ones((1, 2, 2)),
                                   rtol=0.0, atol=1e-12)

    def test_float32_and_float64_inputs(self):
        eeg = np.stack([periodic_cosine(16),
                        periodic_cosine(16, np.pi / 2.0),
                        periodic_cosine(19)])
        for dtype, tolerance in ((np.float32, 2e-6), (np.float64, 1e-12)):
            with self.subTest(dtype=dtype):
                result = PhaseLockingValue()(eeg=eeg.astype(dtype))['eeg']
                np.testing.assert_allclose(
                    result[0], [[1.0, 1.0, 0.0],
                                [1.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0]],
                    rtol=0.0, atol=tolerance)

    def test_two_and_three_dimensional_output_shapes(self):
        locked = np.stack([periodic_cosine(16),
                           periodic_cosine(16, np.pi / 2.0)])
        unlocked = np.stack([periodic_cosine(16), periodic_cosine(19)])
        transform = PhaseLockingValue()
        self.assertEqual(transform(eeg=locked)['eeg'].shape, (1, 2, 2))
        result = transform(eeg=np.stack([locked, unlocked]))['eeg']
        self.assertEqual(result.shape, (2, 2, 2))
        np.testing.assert_allclose(result[0], np.ones((2, 2)),
                                   rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(result[1], np.eye(2),
                                   rtol=0.0, atol=1e-12)

    def test_baseline_is_only_transformed_when_enabled(self):
        eeg = np.stack([periodic_cosine(16),
                        periodic_cosine(16, np.pi / 2.0)])
        baseline = np.stack([periodic_cosine(16), periodic_cosine(19)])
        unchanged = PhaseLockingValue()(eeg=eeg, baseline=baseline)
        np.testing.assert_array_equal(unchanged['baseline'], baseline)
        np.testing.assert_allclose(unchanged['eeg'], np.ones((1, 2, 2)),
                                   rtol=0.0, atol=1e-12)
        transformed = PhaseLockingValue(apply_to_baseline=True)(
            eeg=eeg, baseline=baseline)
        np.testing.assert_allclose(transformed['baseline'][0], np.eye(2),
                                   rtol=0.0, atol=1e-12)
        self.assertEqual(PhaseLockingValue()(eeg=eeg)['eeg'].shape, (1, 2, 2))

        bands = np.stack([eeg, baseline])
        transformed = PhaseLockingValue(apply_to_baseline=True)(
            eeg=bands, baseline=bands)
        self.assertEqual(transformed['baseline'].shape, (2, 2, 2))
        np.testing.assert_array_equal(transformed['eeg'], transformed['baseline'])

        # An unused baseline is passed through even if its phase is undefined.
        invalid_baseline = np.zeros_like(baseline)
        unchanged = PhaseLockingValue()(eeg=eeg, baseline=invalid_baseline)
        self.assertIs(unchanged['baseline'], invalid_baseline)
        with self.assertRaises(ValueError):
            PhaseLockingValue(apply_to_baseline=True)(
                eeg=eeg, baseline=invalid_baseline)

    def test_compose_preserves_other_targets(self):
        eeg = np.stack([periodic_cosine(16),
                        periodic_cosine(16, np.pi / 2.0)])
        result = Compose([PhaseLockingValue(apply_to_baseline=True)])(
            eeg=eeg, baseline=eeg, y=3)
        np.testing.assert_allclose(result['eeg'], np.ones((1, 2, 2)),
                                   rtol=0.0, atol=1e-12)
        np.testing.assert_array_equal(result['baseline'], result['eeg'])
        self.assertEqual(result['y'], 3)

    def test_empty_or_unsupported_shapes_raise_value_error(self):
        for shape in ((0, 32), (2, 0), (0, 2, 32), (2, 0, 32),
                      (2, 2, 0), (32,), (1, 1, 2, 32)):
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    PhaseLockingValue()(eeg=np.empty(shape))

    def test_complex_or_nonfinite_inputs_raise_value_error(self):
        eeg = np.stack([periodic_cosine(16), periodic_cosine(19)])
        with self.assertRaises(ValueError):
            PhaseLockingValue()(eeg=eeg.astype(np.complex128))
        for value in (np.nan, np.inf, -np.inf):
            invalid = eeg.copy()
            invalid[1, 3] = value
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    PhaseLockingValue()(eeg=invalid)

    def test_any_exact_zero_analytic_amplitude_is_rejected(self):
        # The impulse has analytic value exactly zero at sample 2; it is
        # neither an all-zero channel nor merely a real-valued zero crossing.
        valid = np.array([1.0, 0.0, -1.0, 0.0])
        for invalid in (np.zeros(4), np.array([1.0, 0.0, 0.0, 0.0])):
            with self.subTest(invalid=invalid):
                eeg = np.stack([valid, invalid])
                with self.assertRaises(ValueError):
                    PhaseLockingValue()(eeg=eeg)
                with self.assertRaises(ValueError):
                    PhaseLockingValue()(eeg=eeg[np.newaxis, ...])

    def test_real_zero_crossings_have_defined_analytic_phase(self):
        eeg = np.array([[1.0, 0.0, -1.0, 0.0],
                        [0.0, 1.0, 0.0, -1.0]])
        result = PhaseLockingValue()(eeg=eeg)['eeg']
        np.testing.assert_allclose(result, np.ones((1, 2, 2)),
                                   rtol=0.0, atol=1e-12)


class TestPhaseLockingCorrelationCompatibility(unittest.TestCase):

    def test_legacy_formula_and_output_shapes_are_unchanged(self):
        eeg = np.random.RandomState(21).normal(size=(3, 64))
        transform = PhaseLockingCorrelation()
        result = transform(eeg=eeg)['eeg']
        self.assertEqual(result.shape, (1, 3, 3))
        np.testing.assert_allclose(result[0], legacy_reference(eeg),
                                   rtol=0.0, atol=1e-12)
        bands = np.stack([eeg, np.roll(eeg, 3, axis=-1)])
        result = transform(eeg=bands)['eeg']
        self.assertEqual(result.shape, (2, 3, 3))
        expected = np.stack([legacy_reference(band) for band in bands])
        np.testing.assert_allclose(result, expected, rtol=0.0, atol=1e-12)

    def test_legacy_baseline_constructor_behavior_is_unchanged(self):
        eeg = np.stack([periodic_cosine(16),
                        periodic_cosine(16, np.pi / 2.0)])
        baseline = np.roll(eeg, 3, axis=-1)
        default = PhaseLockingCorrelation()(eeg=eeg, baseline=baseline)
        np.testing.assert_array_equal(default['baseline'], baseline)
        transform = PhaseLockingCorrelation(apply_to_baseline=True)
        transformed = transform(eeg=eeg, baseline=baseline)
        np.testing.assert_allclose(transformed['baseline'][0],
                                   legacy_reference(baseline),
                                   rtol=0.0, atol=1e-12)
        self.assertEqual(repr(transform),
                         'PhaseLockingCorrelation(apply_to_baseline=True)')


if __name__ == '__main__':
    unittest.main()
