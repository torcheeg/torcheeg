import unittest

import numpy as np
from scipy.signal import hilbert

from torcheeg.transforms import Compose, PearsonCorrelation, PhaseLockingValue


def _periodic_cosine(cycles, phase=0.0, n_samples=256):
    """Use integer cycles to avoid Hilbert-transform boundary artifacts."""
    theta = 2.0 * np.pi * np.arange(n_samples) / n_samples
    return np.cos(cycles * theta + phase)


class TestCorrelationTransforms(unittest.TestCase):

    def setUp(self):
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)

    def test_pearson_correlation(self):
        eeg = np.random.randn(32, 4)
        transformed_eeg = PearsonCorrelation()(eeg)
        self.assertEqual(transformed_eeg.shape, (1, 32, 32))

    def test_phase_locking_value(self):
        eeg = np.random.randn(32, 128)
        transformed_eeg = PhaseLockingValue()(eeg)
        self.assertEqual(transformed_eeg.shape, (1, 32, 32))

    def test_phase_locking_value_constant_phase_differences_are_fully_locked(self):
        reference = _periodic_cosine(16)
        for phase in (0.0, np.pi / 2.0, np.pi):
            with self.subTest(phase=phase):
                eeg = np.stack([reference, _periodic_cosine(16, phase)])
                result = PhaseLockingValue()(eeg)
                np.testing.assert_allclose(
                    result, np.ones((1, 2, 2)), rtol=0.0, atol=1e-12)

    def test_phase_locking_value_different_integer_cycle_carriers_are_not_locked(self):
        eeg = np.stack([_periodic_cosine(16), _periodic_cosine(19)])
        result = PhaseLockingValue()(eeg)
        np.testing.assert_allclose(result[0], np.eye(2), rtol=0.0, atol=1e-12)

    def test_phase_locking_value_amplitude_modulation_separates_plv_from_correlation_magnitude(self):
        theta = 2.0 * np.pi * np.arange(256) / 256
        carrier = np.cos(16 * theta)
        modulation = 0.5 * np.cos(2 * theta)
        eeg = np.stack([(1.0 + modulation) * carrier,
                       (1.0 - modulation) * carrier])
        result = PhaseLockingValue()(eeg)
        np.testing.assert_allclose(
            result, np.ones((1, 2, 2)), rtol=0.0, atol=1e-12)
        analytic = hilbert(eeg, axis=-1)
        magnitude = abs(np.vdot(analytic[1], analytic[0])) / \
            (np.linalg.norm(analytic[0]) * np.linalg.norm(analytic[1]))
        self.assertAlmostEqual(magnitude, 7.0 / 9.0, places=12)
        self.assertGreater(abs(result[0, 0, 1] - magnitude), 0.2)

    def test_phase_locking_value_range_symmetry_and_unit_diagonal(self):
        eeg = np.random.RandomState(42).normal(size=(4, 128))
        result = PhaseLockingValue()(eeg)[0]
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue((result >= 0.0).all())
        self.assertTrue((result <= 1.0 + 1e-12).all())
        np.testing.assert_allclose(result, result.T, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(
            np.diag(result), np.ones(4), rtol=0.0, atol=1e-12)

    def test_phase_locking_value_partial_locking_matches_known_analytic_signal(self):
        theta = 2.0 * np.pi * np.arange(256) / 256
        eeg = np.stack(
            [np.cos(16 * theta), np.cos(16 * theta) + 0.5 * np.cos(19 * theta)])
        relative = 1.0 + 0.5 * np.exp(3j * theta)
        expected = abs(np.mean(relative / np.abs(relative)))
        actual = PhaseLockingValue()(eeg)[0, 0, 1]
        self.assertGreater(expected, 0.0)
        self.assertLess(expected, 1.0)
        self.assertAlmostEqual(actual, expected, places=12)

    def test_phase_locking_value_signed_constant_scaling_does_not_change_plv(self):
        eeg = np.stack([_periodic_cosine(16), _periodic_cosine(
            16, np.pi / 3.0), _periodic_cosine(19)])
        expected = PhaseLockingValue()(eeg)
        scaled = eeg * np.array([-3.0, 0.2, -7.0])[:, np.newaxis]
        actual = PhaseLockingValue()(scaled)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)

    def test_phase_locking_value_small_nonzero_analytic_amplitudes_are_valid(self):
        eeg = np.stack(
            [_periodic_cosine(16), _periodic_cosine(16, np.pi / 2.0)]) * 1e-200
        result = PhaseLockingValue()(eeg)
        np.testing.assert_allclose(
            result, np.ones((1, 2, 2)), rtol=0.0, atol=1e-12)

    def test_phase_locking_value_float32_and_float64_inputs(self):
        eeg = np.stack([_periodic_cosine(16), _periodic_cosine(
            16, np.pi / 2.0), _periodic_cosine(19)])
        for dtype, tolerance in ((np.float32, 2e-06), (np.float64, 1e-12)):
            with self.subTest(dtype=dtype):
                result = PhaseLockingValue()(eeg.astype(dtype))
                np.testing.assert_allclose(result[0], [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [
                                           0.0, 0.0, 1.0]], rtol=0.0, atol=tolerance)

    def test_phase_locking_value_two_and_three_dimensional_output_shapes(self):
        locked = np.stack(
            [_periodic_cosine(16), _periodic_cosine(16, np.pi / 2.0)])
        unlocked = np.stack([_periodic_cosine(16), _periodic_cosine(19)])
        transform = PhaseLockingValue()
        self.assertEqual(transform(locked).shape, (1, 2, 2))
        result = transform(np.stack([locked, unlocked]))
        self.assertEqual(result.shape, (2, 2, 2))
        np.testing.assert_allclose(
            result[0], np.ones((2, 2)), rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(result[1], np.eye(2), rtol=0.0, atol=1e-12)

    def test_phase_locking_value_explicit_baseline_transform(self):
        eeg = np.stack(
            [_periodic_cosine(16), _periodic_cosine(16, np.pi / 2.0)])
        baseline = np.stack([_periodic_cosine(16), _periodic_cosine(19)])
        transform = Compose([PhaseLockingValue()])
        np.testing.assert_allclose(
            transform(eeg), np.ones((1, 2, 2)), atol=1e-12)
        np.testing.assert_allclose(
            transform(baseline)[0], np.eye(2), atol=1e-12)
        with self.assertRaises(ValueError):
            transform(np.zeros_like(baseline))

    def test_phase_locking_value_empty_or_unsupported_shapes_raise_value_error(self):
        for shape in ((2, 0), (0, 2, 32), (2, 2, 0), (32,), (1, 1, 2, 32)):
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    PhaseLockingValue()(np.empty(shape))

    def test_phase_locking_value_complex_or_nonfinite_inputs_raise_value_error(self):
        eeg = np.stack([_periodic_cosine(16), _periodic_cosine(19)])
        with self.assertRaises(ValueError):
            PhaseLockingValue()(eeg.astype(np.complex128))
        for value in (np.nan, np.inf, -np.inf):
            invalid = eeg.copy()
            invalid[1, 3] = value
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    PhaseLockingValue()(invalid)

    def test_phase_locking_value_any_exact_zero_analytic_amplitude_is_rejected(self):
        valid = np.array([1.0, 0.0, -1.0, 0.0])
        for invalid in (np.zeros(4), np.array([1.0, 0.0, 0.0, 0.0])):
            with self.subTest(invalid=invalid):
                eeg = np.stack([valid, invalid])
                with self.assertRaises(ValueError):
                    PhaseLockingValue()(eeg)
                with self.assertRaises(ValueError):
                    PhaseLockingValue()(eeg[np.newaxis, ...])

    def test_phase_locking_value_real_zero_crossings_have_defined_analytic_phase(self):
        eeg = np.array([[1.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, -1.0]])
        result = PhaseLockingValue()(eeg)
        np.testing.assert_allclose(
            result, np.ones((1, 2, 2)), rtol=0.0, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
