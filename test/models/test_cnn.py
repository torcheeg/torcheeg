import unittest

import torch

from torcheeg.models import (CCNN, FBCCNN, LMDA, MTCNN, CSPNet, DeepSleepNet,
                             EEGNet, FBCNet, FBMSNet, SSTEmotionNet, STNet,
                             TinySleepNet, TSCeption, TSLANet, USleep)


class TestCNN(unittest.TestCase):
    def test_cspnet(self):
        eeg = torch.randn(2, 1, 22, 1750)
        model = CSPNet(chunk_size=1750,
                       num_electrodes=22,
                       num_classes=5,
                       num_filters_t=20,
                       filter_size_t=25)
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (2, 5))

    def test_deep_sleepnet(self):
        eeg = torch.randn(2, 1, 3000, 1)
        model = DeepSleepNet(num_classes=5,
                             chunk_size=3000,
                             num_electrodes=1)
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (2, 5))

    def test_tiny_sleepnet(self):
        eeg = torch.randn(2, 1, 3000, 1)
        model = TinySleepNet(num_classes=5,
                             chunk_size=3000,
                             num_electrodes=1)
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (2, 5))

    def test_lmda(self):
        eeg = torch.randn(2, 1, 22, 1750)
        model = LMDA(num_electrodes=22,
                     chunk_size=1750,
                     num_classes=4,
                     depth=9,
                     kernel=75)
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (2, 4))

    def test_usleep(self):
        eeg = torch.randn(2, 1, 3000)
        model = USleep(num_electrodes=1,
                       patch_size=100,
                       num_patchs=30,
                       num_classes=5)
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (2, 5))

    def test_tslanet(self):
        eeg = torch.randn(2, 1, 3000)
        model = TSLANet(num_classes=5,
                        chunk_size=3000,
                        patch_size=200,
                        num_electrodes=1)
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (2, 5))

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

    def test_stnet(self):
        eeg = torch.randn(1, 128, 9, 9)
        model = STNet(num_classes=2,
                      chunk_size=128,
                      grid_size=(9, 9),
                      dropout=0.2)
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (1, 2))

    def test_mtcnn(self):
        eeg = torch.randn(1, 8, 8, 9)
        model = MTCNN(num_classes=2,
                      in_channels=8,
                      grid_size=(8, 9),
                      dropout=0.2)
        pred = model(eeg)
        self.assertEqual(tuple(pred[0].shape), (1, 2))
        self.assertEqual(tuple(pred[1].shape), (1, 2))

    def test_fbccnn(self):
        eeg = torch.randn(1, 4, 9, 9)
        model = FBCCNN(num_classes=2, in_channels=4, grid_size=(9, 9))
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (1, 2))

    def test_fbcnet(self):
        eeg = torch.randn(1, 4, 32, 512)
        model = FBCNet(num_classes=2,
                       num_electrodes=32,
                       chunk_size=512,
                       in_channels=4,
                       num_S=32)
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (1, 2))

    def test_ccnn(self):
        eeg = torch.randn(1, 4, 9, 9)
        model = CCNN(num_classes=2, in_channels=4, grid_size=(9, 9))
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (1, 2))

    def test_sst_emotion_net(self):
        eeg = torch.randn(2, 32 + 4, 16, 16)
        model = SSTEmotionNet(temporal_in_channels=32,
                              spectral_in_channels=4,
                              grid_size=(16, 16),
                              num_classes=2)
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (2, 2))

    def test_fbmsnet(self):
        eeg = torch.randn(2, 9, 22, 512)
        model = FBMSNet(in_channels=9,
                        num_electrodes=22,
                        chunk_size=512,
                        num_classes=4)
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (2, 4))


if __name__ == '__main__':
    unittest.main()
