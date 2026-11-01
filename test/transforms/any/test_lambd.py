import unittest

import numpy as np
import torch

from torcheeg.transforms import Lambda


class TestLambdTransforms(unittest.TestCase):

    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)

    def test_lambda(self):
        transform = Lambda(lambd=lambda x: x + 1)
        self.assertEqual(transform(1), 2)
        transform = Lambda(lambd=lambda x: x + 1)
        self.assertEqual(
            tuple(transform(torch.randn(32, 128)).shape), (32, 128))
        self.assertEqual(transform(np.random.randn(32, 128)).shape, (32, 128))


if __name__ == '__main__':
    unittest.main()
