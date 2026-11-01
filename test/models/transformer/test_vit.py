import unittest

import torch

from torcheeg.models import ViT


class TestViT(unittest.TestCase):

    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)

    def test_forward_returns_expected_shape(self):
        eeg = torch.randn(1, 128, 9, 9)
        model = ViT(chunk_size=128, t_patch_size=32,
                    s_patch_size=(3, 3), num_classes=2)
        pred = model(eeg)
        for output in pred if isinstance(pred, (tuple, list)) else (pred,):
            self.assertEqual(output.device.type, 'cpu')
            self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(tuple(pred.shape), (1, 2))


class TestViTValidation(unittest.TestCase):
    def test_temporal_patch_size_must_divide_chunk_size(self):
        with self.assertRaises(ValueError):
            ViT(chunk_size=129, t_patch_size=32)


if __name__ == '__main__':
    unittest.main()
