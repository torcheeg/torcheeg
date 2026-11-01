import unittest

import torch

from torcheeg.models import STNet


class TestStnet(unittest.TestCase):

    def test_stnet(self):
        eeg = torch.randn(1, 128, 9, 9)
        model = STNet(num_classes=2,
                      chunk_size=128,
                      grid_size=(9, 9),
                      dropout=0.2)
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (1, 2))


if __name__ == '__main__':
    unittest.main()
