import unittest

import torch

from torcheeg.models import VanillaTransformer


class TestVanillaTransformer(unittest.TestCase):

    def test_vanilla_transformer(self):
        eeg = torch.randn(1, 32, 128)
        model = VanillaTransformer(chunk_size=128,
                                   t_patch_size=32,
                                   hid_channels=32,
                                   depth=3,
                                   heads=4,
                                   head_channels=64,
                                   mlp_channels=64,
                                   num_classes=2)
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (1, 2))


if __name__ == '__main__':
    unittest.main()
