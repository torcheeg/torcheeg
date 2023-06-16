import unittest

import torch

from torcheeg.models import SimpleViT, ArjunViT, VanillaTransformer, ViT, Conformer


class TestTransformer(unittest.TestCase):

    def test_simple_vit(self):
        eeg = torch.randn(1, 128, 9, 9)
        model = SimpleViT(chunk_size=128, t_patch_size=32, s_patch_size=(3, 3), num_classes=2)
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (1, 2))

        model = SimpleViT(chunk_size=128, t_patch_size=32, s_patch_size=(3, 3), num_classes=2)
        eeg = eeg.cuda()
        model = model.cuda()
        pred = model(eeg).cpu()
        self.assertEqual(tuple(pred.shape), (1, 2))

    def test_arjun_vit(self):
        eeg = torch.randn(1, 32, 128)
        model = ArjunViT(chunk_size=128, t_patch_size=32, num_classes=2)
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (1, 2))

        eeg = eeg.cuda()
        model = model.cuda()
        pred = model(eeg).cpu()
        self.assertEqual(tuple(pred.shape), (1, 2))

    def test_vanilla_transformer(self):
        eeg = torch.randn(1, 32, 128)
        model = VanillaTransformer(chunk_size=128,
                                   t_patch_size=32,
                                   hid_channels=32,
                                   depth=3,
                                   heads=4,
                                   head_channels=64,
                                   mlp_channels=64,
                                   num_classes=2)
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (1, 2))

        eeg = eeg.cuda()
        model = model.cuda()
        pred = model(eeg).cpu()
        self.assertEqual(tuple(pred.shape), (1, 2))

    def test_vit(self):
        eeg = torch.randn(1, 128, 9, 9)
        model = ViT(chunk_size=128, t_patch_size=32, s_patch_size=(3, 3), num_classes=2)
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (1, 2))

        eeg = eeg.cuda()
        model = model.cuda()
        pred = model(eeg).cpu()
        self.assertEqual(tuple(pred.shape), (1, 2))

    def test_conformer(self):
        eeg = torch.randn(2, 1, 32, 128)
        model = Conformer(num_electrodes=32, sampling_rate=128, num_classes=2)
        pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (2, 2))

        eeg = eeg.cuda()
        model = model.cuda()
        pred = model(eeg).cpu()
        self.assertEqual(tuple(pred.shape), (2, 2))


if __name__ == '__main__':
    unittest.main()
