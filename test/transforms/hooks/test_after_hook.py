import unittest

import numpy as np
import torch

from torcheeg.transforms import (after_hook_linear_dynamical_system,
                                 after_hook_normalize, after_hook_running_norm)


class TestAfterHookTransforms(unittest.TestCase):

    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)

    def test_after_hook(self):
        fake_data_np = [np.random.rand(32, 128) for _ in range(32)]
        fake_data_torch = [torch.rand(32, 128) for _ in range(32)]
        normalized_data = after_hook_normalize(fake_data_np)
        self.assertEqual(len(normalized_data), 32)
        self.assertEqual(normalized_data[0].shape, (32, 128))
        normalized_data = after_hook_normalize(fake_data_torch)
        self.assertEqual(len(normalized_data), 32)
        self.assertEqual(normalized_data[0].shape, (32, 128))
        normalized_data = after_hook_running_norm(fake_data_np)
        self.assertEqual(len(normalized_data), 32)
        self.assertEqual(normalized_data[0].shape, (32, 128))
        normalized_data = after_hook_running_norm(fake_data_torch)
        self.assertEqual(len(normalized_data), 32)
        self.assertEqual(normalized_data[0].shape, (32, 128))
        normalized_data = after_hook_linear_dynamical_system(fake_data_np)
        self.assertEqual(len(normalized_data), 32)
        self.assertEqual(normalized_data[0].shape, (32, 128))
        normalized_data = after_hook_linear_dynamical_system(fake_data_torch)
        self.assertEqual(len(normalized_data), 32)
        self.assertEqual(normalized_data[0].shape, (32, 128))


if __name__ == '__main__':
    unittest.main()
