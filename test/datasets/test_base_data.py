import unittest
from test.runners.test_base import _WindowData

from torcheeg.datasets import BaseData


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
