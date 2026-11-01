import unittest

import numpy as np

from torcheeg.transforms import (BaselineCorrection, BaselineRemoval, Compose,
                                 RandomMask, RandomNoise, Resize, ToTensor)


class TestBaselineTransforms(unittest.TestCase):
    def test_to_baseline_removal(self):
        transform = Compose([
            ToTensor(apply_to_baseline=True),
            Resize(size=(64, 64), apply_to_baseline=True),
            RandomNoise(p=0.1),
            RandomMask(p=0.1),
            BaselineRemoval()
        ])
        transformed_eeg = transform(eeg=np.random.randn(128, 9, 9),
                                    baseline=np.random.randn(128, 9, 9))['eeg']
        self.assertEqual(tuple(transformed_eeg.shape), (128, 64, 64))

        eeg = np.random.randn(128, 9, 9)
        transform = BaselineRemoval()
        transformed_eeg = transform(eeg=eeg, baseline=np.ones(
            (128, 9, 9)))['eeg']
        self.assertTrue(np.abs(eeg - (transformed_eeg + 1.0)).sum() < 1e-6)

        transform = BaselineRemoval()
        transformed_eeg = transform(eeg=eeg)['eeg']
        self.assertTrue(np.abs(eeg - transformed_eeg).sum() < 1e-6)

    def test_baseline_correction(self):
        # torch
        transform = Compose([
            ToTensor(apply_to_baseline=True),
            Resize(size=(64, 64), apply_to_baseline=True),
            RandomNoise(p=0.1),
            RandomMask(p=0.1),
            BaselineCorrection()
        ])
        transformed_eeg, transformed_baseline = transform(eeg=np.random.randn(128, 9, 9),
                                                          baseline=np.random.randn(128, 9, 9)).values()
        self.assertEqual(tuple(transformed_eeg.shape), (128, 64, 64))
        self.assertEqual(tuple(transformed_baseline.shape), (128, 64, 64))

        # numpy
        eeg = np.random.randn(128, 9, 9)
        transform = BaselineCorrection()
        transformed_eeg = transform(eeg=eeg, baseline=np.ones(
            (128, 9, 9)))['eeg']
        self.assertTrue(np.abs(eeg - (transformed_eeg + 1.0)).sum() < 1e-6)

        transform = BaselineCorrection()
        transformed_eeg = transform(eeg=eeg)['eeg']
        self.assertTrue(np.abs(eeg - transformed_eeg).sum() < 1e-6)


if __name__ == '__main__':
    unittest.main()
