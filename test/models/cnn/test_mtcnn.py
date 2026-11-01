import unittest

import torch

from torcheeg.models import MTCNN


class TestMTCNN(unittest.TestCase):

    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)

    def test_forward_returns_expected_shape(self):
        eeg = torch.randn(1, 8, 8, 9)
        model = MTCNN(num_classes=2,
                      in_channels=8,
                      grid_size=(8, 9),
                      dropout=0.2)
        pred = model(eeg)
        for output in pred if isinstance(pred, (tuple, list)) else (pred,):
            self.assertEqual(output.device.type, 'cpu')
            self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(tuple(pred[0].shape), (1, 2))
        self.assertEqual(tuple(pred[1].shape), (1, 2))


if __name__ == '__main__':
    unittest.main()
