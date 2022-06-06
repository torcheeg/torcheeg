import unittest

import torch

from torch_geometric.data import Data, Batch
from torcheeg.models import SimpleViT


class TestTransformer(unittest.TestCase):
    def test_simple_vit(self):
        eeg = torch.randn(1, 5, 9, 9)
        model = SimpleViT(hid_channels=32,
                          depth=3,
                          heads=4,
                          mlp_channels=64,
                          grid_size=(9, 9),
                          patch_size=3,
                          num_classes=2,
                          in_channels=5,
                          head_channels=8)
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (1, 2))

        eeg = eeg.cuda()
        model = model.cuda()
        pred = model(eeg).cpu()
        self.assertEqual(tuple(pred.shape), (1, 2))


if __name__ == '__main__':
    unittest.main()
