import unittest

import torch

from torcheeg.models import BDiscriminator, BGenerator


class TestBgan(unittest.TestCase):
    def test_bgan(self):
        g_model = BGenerator(in_channels=128)
        d_model = BDiscriminator(in_channels=4)
        z = torch.normal(mean=0, std=1, size=(1, 128))
        fake_X = g_model(z)
        disc_X = d_model(fake_X)

        self.assertEqual(tuple(disc_X.shape), (1, 1))


if __name__ == '__main__':
    unittest.main()
