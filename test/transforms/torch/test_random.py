import unittest

import torch

from torcheeg.transforms import (RandomChannelShuffle, RandomFlip,
                                 RandomFrequencyShift, RandomMask,
                                 RandomMask2D, RandomMaskGrid, RandomNoise,
                                 RandomPCANoise, RandomShift, RandomSignFlip,
                                 RandomWindowSlice, RandomWindowWarp)


class TestRandomTransforms(unittest.TestCase):

    def test_random_noise(self):
        eeg = torch.randn(32, 128)
        transformed_eeg = RandomNoise(p=1.0)(eeg=eeg)
        self.assertEqual(tuple(transformed_eeg['eeg'].shape), (32, 128))

    def test_random_mask(self):
        eeg = torch.randn(32, 128)
        transformed_eeg = RandomMask(p=1.0)(eeg=eeg)
        self.assertEqual(tuple(transformed_eeg['eeg'].shape), (32, 128))

    def test_random_window_slice(self):
        eeg = torch.randn(32, 128)
        transformed_eeg = RandomWindowSlice(p=1.0)(eeg=eeg)
        self.assertEqual(tuple(transformed_eeg['eeg'].shape), (32, 128))

        eeg = torch.randn(1, 32, 128)
        transformed_eeg = RandomWindowSlice(p=1.0, window_size=100)(eeg=eeg)
        self.assertEqual(tuple(transformed_eeg['eeg'].shape), (1, 32, 128))

        eeg = torch.randn(128, 9, 9)
        transformed_eeg = RandomWindowSlice(p=1.0, series_dim=0)(eeg=eeg)
        self.assertEqual(tuple(transformed_eeg['eeg'].shape), (128, 9, 9))

    def test_random_window_warp(self):
        eeg = torch.randn(32, 128)
        transformed_eeg = RandomWindowWarp(p=1.0)(eeg=eeg)
        self.assertEqual(tuple(transformed_eeg['eeg'].shape), (32, 128))

        eeg = torch.randn(1, 32, 128)
        transformed_eeg = RandomWindowWarp(
            p=1.0, window_size=24, warp_size=48)(eeg=eeg)
        self.assertEqual(tuple(transformed_eeg['eeg'].shape), (1, 32, 128))

        eeg = torch.randn(128, 9, 9)
        transformed_eeg = RandomWindowWarp(p=1.0, series_dim=0)(eeg=eeg)
        self.assertEqual(tuple(transformed_eeg['eeg'].shape), (128, 9, 9))

    def test_random_pca_noise(self):
        eeg = torch.randn(32, 128)
        transformed_eeg = RandomPCANoise(p=1.0)(eeg=eeg)
        self.assertEqual(tuple(transformed_eeg['eeg'].shape), (32, 128))

        eeg = torch.randn(1, 32, 128)
        transformed_eeg = RandomPCANoise(
            p=1.0, mean=0.5, std=2.0, n_components=4)(eeg=eeg)
        self.assertEqual(tuple(transformed_eeg['eeg'].shape), (1, 32, 128))

        eeg = torch.randn(128, 9, 9)
        transformed_eeg = RandomPCANoise(p=1.0, series_dim=0)(eeg=eeg)
        self.assertEqual(tuple(transformed_eeg['eeg'].shape), (128, 9, 9))

    def test_random_flip(self):
        eeg = torch.randn(32, 128)
        transformed_eeg = RandomFlip(p=1.0, dim=-1)(eeg=eeg)
        self.assertEqual(tuple(transformed_eeg['eeg'].shape), (32, 128))

    def test_random_sign_flip(self):
        eeg = torch.randn(32, 128)
        transformed_eeg = RandomSignFlip(p=1.0)(eeg=eeg)
        self.assertEqual(tuple(transformed_eeg['eeg'].shape), (32, 128))

    def test_random_shift(self):
        eeg = torch.randn(32, 128)
        transformed_eeg = RandomShift(
            p=1.0, dim=-1, shift_min=8, shift_max=24)(eeg=eeg)
        self.assertEqual(tuple(transformed_eeg['eeg'].shape), (32, 128))

    def test_random_channel_shuffle(self):
        eeg = torch.randn(32, 128)
        transformed_eeg = RandomChannelShuffle(p=1.0)(eeg=eeg)
        self.assertEqual(tuple(transformed_eeg['eeg'].shape), (32, 128))

    def test_random_sampling_rate_shift(self):
        eeg = torch.randn(32, 128)
        transformed_eeg = RandomFrequencyShift(p=1.0)(eeg=eeg)
        self.assertEqual(tuple(transformed_eeg['eeg'].shape), (32, 128))

        eeg = torch.randn(1, 32, 128)
        transformed_eeg = RandomFrequencyShift(
            p=1.0, sampling_rate=128, shift_min=-1.0, shift_max=1.0)(eeg=eeg)
        self.assertEqual(tuple(transformed_eeg['eeg'].shape), (1, 32, 128))

        eeg = torch.randn(128, 9, 9)
        transformed_eeg = RandomFrequencyShift(p=1.0, series_dim=0)(eeg=eeg)
        self.assertEqual(tuple(transformed_eeg['eeg'].shape), (128, 9, 9))

    def test_random_mask_2d(self):
        eeg = torch.randn(1, 32, 128)
        transformed_eeg = RandomMask2D()(eeg=eeg)
        self.assertEqual(transformed_eeg['eeg'].shape, (1, 32, 128))

    def test_random_mask_grid(self):
        eeg = torch.randn(32, 9, 9)
        transformed_eeg = RandomMaskGrid()(eeg=eeg)
        self.assertEqual(transformed_eeg['eeg'].shape, (32, 9, 9))


if __name__ == '__main__':
    unittest.main()
