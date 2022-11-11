import unittest

import torch
from torcheeg.models import (CCNN, FBCCNN, MTCNN, EEGNet, FBCNet, STNet, TSCeption, SSTEmotionNet)


class TestCNN(unittest.TestCase):

    def test_tsception(self):
        eeg = torch.randn(1, 1, 28, 512)
        model = TSCeption(num_classes=2,
                          num_electrodes=28,
                          sampling_rate=128,
                          num_T=15,
                          num_S=15,
                          hid_channels=32,
                          dropout=0.5)
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (1, 2))

        eeg = eeg.cuda()
        model = model.cuda()
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (1, 2))

    def test_eegnet(self):
        eeg = torch.randn(1, 1, 32, 128)
        model = EEGNet(chunk_size=128,
                       num_electrodes=32,
                       dropout=0.5,
                       kernel_1=64,
                       kernel_2=16,
                       F1=8,
                       D=2,
                       F2=16,
                       num_classes=2)
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (1, 2))

        eeg = eeg.cuda()
        model = model.cuda()
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (1, 2))

    def test_stnet(self):
        eeg = torch.randn(1, 128, 9, 9)
        model = STNet(num_classes=2, chunk_size=128, grid_size=(9, 9), dropout=0.2)
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (1, 2))

        eeg = eeg.cuda()
        model = model.cuda()
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (1, 2))

    def test_mtcnn(self):
        eeg = torch.randn(1, 8, 8, 9)
        model = MTCNN(num_classes=2, in_channels=8, grid_size=(8, 9), dropout=0.2)
        pred = model(eeg)
        self.assertEqual(tuple(pred[0].shape), (1, 2))
        self.assertEqual(tuple(pred[1].shape), (1, 2))

        eeg = eeg.cuda()
        model = model.cuda()
        pred = model(eeg)
        self.assertEqual(tuple(pred[0].shape), (1, 2))
        self.assertEqual(tuple(pred[1].shape), (1, 2))

    def test_fbccnn(self):
        eeg = torch.randn(1, 4, 9, 9)
        model = FBCCNN(num_classes=2, in_channels=4, grid_size=(9, 9))
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (1, 2))

        eeg = eeg.cuda()
        model = model.cuda()
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (1, 2))

    def test_fbcnet(self):
        eeg = torch.randn(1, 4, 32, 512)
        model = FBCNet(num_classes=2, num_electrodes=32, chunk_size=512, in_channels=4, num_S=32)
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (1, 2))

        eeg = eeg.cuda()
        model = model.cuda()
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (1, 2))

    def test_ccnn(self):
        eeg = torch.randn(1, 4, 9, 9)
        model = CCNN(num_classes=2, in_channels=4, grid_size=(9, 9))
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (1, 2))

        eeg = eeg.cuda()
        model = model.cuda()
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (1, 2))

    def test_sst_emotion_net(self):
        eeg = torch.randn(2, 32 + 4, 16, 16)
        model = SSTEmotionNet(temporal_in_channels=32, spectral_in_channels=4, grid_size=(16, 16), num_classes=2)
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (2, 2))

        eeg = eeg.cuda()
        model = model.cuda()
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (2, 2))


if __name__ == '__main__':
    unittest.main()
