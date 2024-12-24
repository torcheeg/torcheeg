import unittest

import numpy as np
import torch

from torcheeg.transforms import (after_hook_normalize, after_hook_running_norm,
                                 after_hook_linear_dynamical_system,
                                 before_hook_normalize)


class TestHooks(unittest.TestCase):

    def test_before_hook(self):
        fake_data_np = np.random.rand(32, 128)

        # Test before_hook_normalize
        normalized_data = before_hook_normalize(fake_data_np)
        self.assertEqual(normalized_data.shape, (32, 128))

    def test_after_hook(self):
        fake_data_np = [np.random.rand(32, 128) for _ in range(32)]
        fake_data_torch = [torch.rand(32, 128) for _ in range(32)]

        # Test after_hook_normalize
        normalized_data = after_hook_normalize(fake_data_np)
        self.assertEqual(len(normalized_data), 32)
        self.assertEqual(normalized_data[0].shape, (32, 128))

        normalized_data = after_hook_normalize(fake_data_torch)
        self.assertEqual(len(normalized_data), 32)
        self.assertEqual(normalized_data[0].shape, (32, 128))

        # Test after_hook_running_norm
        normalized_data = after_hook_running_norm(fake_data_np)
        self.assertEqual(len(normalized_data), 32)
        self.assertEqual(normalized_data[0].shape, (32, 128))

        normalized_data = after_hook_running_norm(fake_data_torch)
        self.assertEqual(len(normalized_data), 32)
        self.assertEqual(normalized_data[0].shape, (32, 128))

        # Test after_hook_linear_dynamical_system
        normalized_data = after_hook_linear_dynamical_system(fake_data_np)
        self.assertEqual(len(normalized_data), 32)
        self.assertEqual(normalized_data[0].shape, (32, 128))

        normalized_data = after_hook_linear_dynamical_system(fake_data_torch)
        self.assertEqual(len(normalized_data), 32)
        self.assertEqual(normalized_data[0].shape, (32, 128))


if __name__ == '__main__':
    unittest.main()
