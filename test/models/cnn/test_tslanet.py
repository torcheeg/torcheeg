import unittest

import torch

from torcheeg.models import TSLANet


class TestTslanet(unittest.TestCase):

    def test_tslanet(self):
        eeg = torch.randn(2, 1, 3000)
        model = TSLANet(num_classes=5,
                        chunk_size=3000,
                        patch_size=200,
                        num_electrodes=1)
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (2, 5))


if __name__ == '__main__':
    unittest.main()
