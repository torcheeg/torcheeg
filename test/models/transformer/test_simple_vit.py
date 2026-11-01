import unittest

import torch

from torcheeg.models import SimpleViT


class TestSimpleViT(unittest.TestCase):

    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)

    def test_forward_returns_expected_shape(self):
        eeg = torch.randn(1, 128, 9, 9)
        model = SimpleViT(chunk_size=128, t_patch_size=32,
                          s_patch_size=(3, 3), num_classes=2)
        pred = model(eeg)
        for output in pred if isinstance(pred, (tuple, list)) else (pred,):
            self.assertEqual(output.device.type, 'cpu')
            self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(tuple(pred.shape), (1, 2))


if __name__ == '__main__':
    unittest.main()
