import unittest
from test.runners.test_base import _WindowData

from torcheeg.runners import SequentialRunner


class TestSequentialRunner(unittest.TestCase):
    def test_order(self):
        samples = list(SequentialRunner()(_WindowData()))
        self.assertEqual([(s['info']['subject_id'], s['info']['index']) for s in samples],
                         [(f'{record:03}', index) for record in range(3) for index in range(8)])
