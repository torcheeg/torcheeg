import unittest

import torch

from torcheeg import transforms
from torcheeg.transforms import Resize


class TestInputValidation(unittest.TestCase):

    def test_tensor_shape_is_checked(self):
        with self.assertRaises(ValueError):
            transforms.Resize(size=(4, 4))(torch.zeros(2, 3))


class TestResizeTransforms(unittest.TestCase):

    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)

    def test_resize(self):
        eeg = torch.randn(128, 9, 9)
        transformed_eeg = Resize(size=(64, 64))(eeg)
        self.assertEqual(tuple(transformed_eeg.shape), (128, 64, 64))


if __name__ == '__main__':
    unittest.main()
