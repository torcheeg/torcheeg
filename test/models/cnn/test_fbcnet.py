import unittest

import torch

from torcheeg.models import FBCNet


class TestFbcnet(unittest.TestCase):

    def test_fbcnet(self):
        eeg = torch.randn(1, 4, 32, 512)
        model = FBCNet(num_classes=2,
                       num_electrodes=32,
                       chunk_size=512,
                       in_channels=4,
                       num_S=32)
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (1, 2))


if __name__ == '__main__':
    unittest.main()
