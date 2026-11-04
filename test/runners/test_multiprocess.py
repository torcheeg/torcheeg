import multiprocessing
import time
import unittest
from test.runners.test_base import _WindowData

from torcheeg.runners import MultiprocessRunner


class _FailingData(_WindowData):
    def process_record(self, record, **kwargs):
        yield from super().process_record(record)
        raise ValueError('deliberate worker failure')


class _LargeRecordData(_WindowData):
    def set_records(self, **kwargs):
        return [0]

    def process_record(self, record, **kwargs):
        yield {'eeg': [1], 'info': {'index': 0}}
        # The first sample must be delivered before the record finishes.
        raise ValueError('failure after first sample')


def _increment_sample(eeg, **kwargs):
    return {'eeg': eeg + 1}


class TestMultiprocessRunner(unittest.TestCase):
    def test_start_methods_preserve_order_and_values(self):
        for method in multiprocessing.get_all_start_methods():
            with self.subTest(method=method):
                runner = MultiprocessRunner(
                    2, multiprocessing_context=method, prefetch=1)
                samples = list(
                    runner(_WindowData(transform=_increment_sample)))
                self.assertEqual([(s['info']['subject_id'], s['info']['index']) for s in samples],
                                 [(f'{record:03}', index) for record in range(3) for index in range(8)])
                self.assertEqual(samples[0]['eeg'].sum(), 8)

    def test_first_sample_is_not_delayed_until_record_completion(self):
        samples = MultiprocessRunner(1)(_LargeRecordData())
        self.assertEqual(next(samples)['eeg'], [1])
        with self.assertRaisesRegex(RuntimeError, 'failure after first sample'):
            next(samples)

    def test_early_close_cleans_children(self):
        before = {p.pid for p in multiprocessing.active_children()}
        samples = MultiprocessRunner(2, prefetch=1)(_WindowData())
        next(samples)
        samples.close()
        self.assertEqual(
            {p.pid for p in multiprocessing.active_children()} - before, set())

    def test_serial_mode_and_invalid_buffer(self):
        self.assertEqual(len(list(MultiprocessRunner(0)(_WindowData()))), 24)
        with self.assertRaises(ValueError):
            MultiprocessRunner(1, prefetch=0)
