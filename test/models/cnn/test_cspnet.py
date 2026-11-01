import unittest

import torch

from torcheeg.models import CSPNet


class TestCspnet(unittest.TestCase):
    def test_cspnet(self):
        eeg = torch.randn(2, 1, 22, 1750)
        model = CSPNet(chunk_size=1750,
                       num_electrodes=22,
                       num_classes=5,
                       num_filters_t=20,
                       filter_size_t=25)
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (2, 5))


if __name__ == '__main__':
    unittest.main()
