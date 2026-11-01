import unittest

import torch

from torcheeg.models import DGCNN


class TestDGCNN(unittest.TestCase):

    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)

    def test_forward_returns_expected_shape(self):
        eeg = torch.randn(1, 62, 200)
        model = DGCNN(in_channels=200, num_electrodes=62,
                      hid_channels=32, num_layers=2, num_classes=2)
        pred = model(eeg)
        for output in pred if isinstance(pred, (tuple, list)) else (pred,):
            self.assertEqual(output.device.type, 'cpu')
            self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(tuple(pred.shape), (1, 2))


if __name__ == '__main__':
    unittest.main()
