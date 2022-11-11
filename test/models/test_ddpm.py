import unittest

import torch
from torcheeg.models import BUNet, BCUNet


class TestDDPM(unittest.TestCase):
    def test_bddpm(self):
        unet = BUNet()
        mock_eeg = torch.randn(2, 4, 9, 9)
        t = torch.randint(low=1, high=1000, size=(2, ))
        fake_X = unet(mock_eeg, t)
        self.assertEqual(tuple(fake_X.shape), (2, 4, 9, 9))

        unet = unet.cuda()
        mock_eeg = mock_eeg.cuda()
        t = t.cuda()
        fake_X = unet(mock_eeg, t)
        self.assertEqual(tuple(fake_X.shape), (2, 4, 9, 9))

    def test_bcddpm(self):
        unet = BCUNet(num_classes=2)
        mock_eeg = torch.randn(2, 4, 9, 9)
        t = torch.randint(low=1, high=1000, size=(2, ))
        y = torch.randint(low=0, high=2, size=(1, ))
        fake_X = unet(mock_eeg, t, y)
        self.assertEqual(tuple(fake_X.shape), (2, 4, 9, 9))

        unet = unet.cuda()
        mock_eeg = mock_eeg.cuda()
        t = t.cuda()
        y = y.cuda()
        fake_X = unet(mock_eeg, t, y)
        self.assertEqual(tuple(fake_X.shape), (2, 4, 9, 9))


if __name__ == '__main__':
    unittest.main()
