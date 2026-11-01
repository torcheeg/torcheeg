import unittest

import numpy as np

from torcheeg.transforms import before_hook_normalize


class TestBeforeHookTransforms(unittest.TestCase):

    def setUp(self):
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)

    def test_before_hook(self):
        fake_data_np = np.random.rand(32, 128)
        normalized_data = before_hook_normalize(fake_data_np)
        self.assertEqual(normalized_data.shape, (32, 128))


if __name__ == '__main__':
    unittest.main()
