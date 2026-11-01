import unittest

import torch

from torcheeg.models import ArjunViT


class TestArjunVit(unittest.TestCase):

    def test_arjun_vit(self):
        eeg = torch.randn(1, 32, 128)
        model = ArjunViT(chunk_size=128, t_patch_size=32, num_classes=2)
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (1, 2))


if __name__ == '__main__':
    unittest.main()
