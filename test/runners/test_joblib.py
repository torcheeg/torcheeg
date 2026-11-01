import unittest
from test.runners.test_base import _WindowData
from test.runners.test_multiprocess import _LargeRecordData

from torcheeg.runners import JoblibRunner


class TestJoblibRunner(unittest.TestCase):
    def test_stream_order_and_early_close(self):
        runner = JoblibRunner(2, prefetch=1)
        samples = list(runner(_WindowData()))
        self.assertEqual([(s['info']['subject_id'], s['info']['index']) for s in samples],
                         [(f'{record:03}', index) for record in range(3) for index in range(8)])
        samples = runner(_WindowData())
        next(samples)
        samples.close()

    def test_error_after_first_sample(self):
        samples = JoblibRunner(1)(_LargeRecordData())
        self.assertEqual(next(samples)['eeg'], [1])
        with self.assertRaisesRegex(RuntimeError, 'failure after first sample'):
            next(samples)

    def test_serial_and_invalid_prefetch(self):
        self.assertEqual(len(list(JoblibRunner(0)(_WindowData()))), 24)
        with self.assertRaises(ValueError):
            JoblibRunner(1, prefetch=0)
