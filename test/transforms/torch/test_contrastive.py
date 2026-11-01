import unittest

import torch

from torcheeg.transforms import Contrastive, RandomNoise


class TestContrastiveTransforms(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)

    def test_contrastive(self):
        eeg = torch.randn(32, 128)
        transformed_eeg = Contrastive(RandomNoise(p=1.0), num_views=2)(eeg=eeg)
        self.assertEqual(tuple(transformed_eeg['eeg'][0].shape), (32, 128))
        self.assertEqual(tuple(transformed_eeg['eeg'][1].shape), (32, 128))
        # transformed_eeg['eeg'][0] should be different from transformed_eeg['eeg'][1]
        self.assertFalse(torch.abs(
            transformed_eeg['eeg'][0] - transformed_eeg['eeg'][1]).sum().item() == 0)


if __name__ == '__main__':
    unittest.main()
