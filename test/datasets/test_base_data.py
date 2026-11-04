import unittest
from test.runners.test_base import _WindowData

import numpy as np

from torcheeg.datasets import BaseData
from torcheeg.transforms import Compose


class TestBaseData(unittest.TestCase):

    def test_required_record_steps_identify_the_class(self):
        data = BaseData()
        for method, args in (('set_records', ()), ('read_record', ('record',)), ('fake_record', ()), ('process_record', ('record',))):
            with self.subTest(method=method), self.assertRaisesRegex(NotImplementedError, 'Method ' + method + ' is not implemented in class BaseData'):
                getattr(data, method)(*args)

    def test_record_configuration_is_independent_of_io(self):
        data = _WindowData(num_records=3)
        self.assertEqual(data.get_records(), [0, 1, 2])
        self.assertEqual(len(list(data.iter_record(0))), 8)
        self.assertFalse(hasattr(data, 'io_path'))

    def test_removed_hooks_are_rejected(self):
        for name in ('before_trial', 'after_trial', 'after_session', 'after_subject'):
            with self.subTest(name=name), self.assertRaises(TypeError):
                BaseData(**{name: lambda x: x})

    def test_static_record_steps_are_supported(self):

        class StaticData(BaseData):

            @staticmethod
            def read_record(record, **kwargs):
                return {'samples': record}

            @staticmethod
            def process_record(record, samples, **kwargs):
                yield {'eeg': samples, 'info': {'record': record}}
        self.assertEqual(list(StaticData().iter_record('record')), [
                         {'eeg': 'record', 'info': {'record': 'record'}}])

    def test_transform_composition_preserves_source_configuration(self):
        def increment(sample):
            return {**sample, 'eeg': sample['eeg'] + 1}

        def double(sample):
            return {**sample, 'eeg': sample['eeg'] * 2}

        data = _WindowData()
        first = _WindowData(transform=increment)
        chained = _WindowData(transform=Compose([increment, double]))
        composed = _WindowData(transform=Compose(
            [increment, Compose([double])]))
        np.testing.assert_array_equal(next(data.iter_record(0))['eeg'], 0)
        np.testing.assert_array_equal(next(first.iter_record(0))['eeg'], 1)
        np.testing.assert_array_equal(next(chained.iter_record(0))['eeg'], 2)
        np.testing.assert_array_equal(next(composed.iter_record(0))['eeg'], 2)
        with self.assertRaises(TypeError):
            _WindowData(transform=1)
