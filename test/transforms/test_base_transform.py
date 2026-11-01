import unittest

import numpy as np

from torcheeg.transforms.base_transform import (BaseTransform, EEGTransform,
                                                LabelTransform)


class _OffsetTransform(EEGTransform):
    def apply(self, eeg, offset=2, **kwargs):
        return eeg + offset


class _DependentTransform(_OffsetTransform):
    @property
    def targets_as_params(self):
        return ['reference']

    def get_params(self):
        return {'offset': 99}

    def get_params_dependent_on_targets(self, params):
        return {'offset': params['reference']}


class TestBaseTransform(unittest.TestCase):
    def setUp(self):
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)

    def test_required_hooks_identify_the_concrete_class(self):
        for transform, method in ((BaseTransform(), 'targets'),
                                  (BaseTransform(), 'apply'),
                                  (EEGTransform(), 'apply'),
                                  (LabelTransform(), 'apply')):
            with self.subTest(transform=type(transform).__name__, method=method):
                with self.assertRaisesRegex(
                        NotImplementedError, type(transform).__name__):
                    value = getattr(transform, method)
                    if callable(value):
                        value(None)

    def test_only_named_inputs_are_accepted(self):
        with self.assertRaises(KeyError):
            _OffsetTransform()(np.ones(2))

    def test_baseline_and_additional_targets_share_the_eeg_rule(self):
        eeg = np.array([1, 2])
        baseline = np.array([3, 4])
        for enabled in (False, True):
            with self.subTest(apply_to_baseline=enabled):
                result = _OffsetTransform(enabled)(
                    eeg=eeg, baseline=baseline, y='rest', ignored=None)
                np.testing.assert_array_equal(result['eeg'], [3, 4])
                np.testing.assert_array_equal(
                    result['baseline'], baseline + (2 if enabled else 0))
                self.assertEqual(result['y'], 'rest')
                self.assertNotIn('ignored', result)
        transform = _OffsetTransform()
        transform.add_targets({'other': 'eeg'})
        np.testing.assert_array_equal(transform(other=eeg)['other'], [3, 4])
        np.testing.assert_array_equal(eeg, [1, 2])

    def test_target_dependent_parameters_override_defaults(self):
        transform = _DependentTransform()
        with self.assertRaisesRegex(ValueError, 'reference'):
            transform(eeg=np.zeros(2))
        np.testing.assert_array_equal(
            transform(eeg=np.zeros(2), reference=5)['eeg'], [5, 5])


if __name__ == '__main__':
    unittest.main()
