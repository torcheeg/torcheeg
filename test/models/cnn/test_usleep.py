import unittest

import torch

from torcheeg.models import USleep


class TestUsleep(unittest.TestCase):

    def test_usleep(self):
        eeg = torch.randn(2, 1, 3000)
        model = USleep(num_electrodes=1,
                       patch_size=100,
                       num_patchs=30,
                       num_classes=5)
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (2, 5))


if __name__ == '__main__':
    unittest.main()
