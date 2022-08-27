import unittest

import torch
from torcheeg.models import GRU, LSTM


class TestRNN(unittest.TestCase):
    def test_gru(self):
        model = GRU(num_electrodes=32, hid_channels=64, num_classes=2)

        eeg = torch.randn(2, 32, 128)
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (2, 2))

        eeg = eeg.cuda()
        model = model.cuda()
        pred = model(eeg).cpu()
        self.assertEqual(tuple(pred.shape), (2, 2))

    def test_lstm(self):
        model = LSTM(num_electrodes=32, hid_channels=64, num_classes=2)

        eeg = torch.randn(2, 32, 128)
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (2, 2))

        eeg = eeg.cuda()
        model = model.cuda()
        pred = model(eeg).cpu()
        self.assertEqual(tuple(pred.shape), (2, 2))


if __name__ == '__main__':
    unittest.main()
