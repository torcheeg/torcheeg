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
        # label
        transform = Lambda(lambd=lambda x: x + 1)
        self.assertEqual(transform(y=1)['y'], 2)
        # torch
        transform = Lambda(targets=['eeg'], lambd=lambda x: x + 1)
        self.assertEqual(
            tuple(transform(eeg=torch.randn(32, 128))['eeg'].shape), (32, 128))
        # numpy
        self.assertEqual(
            transform(eeg=np.random.randn(32, 128))['eeg'].shape, (32, 128))


if __name__ == '__main__':
    unittest.main()
