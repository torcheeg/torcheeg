import unittest

import torch

from torcheeg.models import SimpleViT


class TestSimpleVit(unittest.TestCase):

    def test_simple_vit(self):
        eeg = torch.randn(1, 128, 9, 9)
        model = SimpleViT(chunk_size=128, t_patch_size=32,
                          s_patch_size=(3, 3), num_classes=2)
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (1, 2))


if __name__ == '__main__':
    unittest.main()
