import unittest
import torch
import numpy as np
from torcheeg.transforms import ToTensor, Resize, RandomNoise, RandomMask


class TestTorchTransforms(unittest.TestCase):
    def test_to_tensor(self):
        eeg = np.random.randn(32, 128)
        transformed_eeg = ToTensor()(eeg)
        self.assertEqual(eeg.shape, tuple(transformed_eeg.shape))
        self.assertTrue(isinstance(transformed_eeg, torch.Tensor))

    def test_resize(self):
        eeg = torch.randn(128, 9, 9)
        transformed_eeg = Resize(size=(64, 64))(eeg)
        self.assertEqual(tuple(transformed_eeg.shape), (128, 64, 64))

    def test_random_noise(self):
        eeg = torch.randn(32, 128)
        transformed_eeg = RandomNoise()(eeg)
        self.assertEqual(tuple(transformed_eeg.shape), (32, 128))

    def test_random_mask(self):
        eeg = torch.randn(32, 128)
        transformed_eeg = RandomMask()(eeg)
        self.assertEqual(tuple(transformed_eeg.shape), (32, 128))


if __name__ == '__main__':
    unittest.main()