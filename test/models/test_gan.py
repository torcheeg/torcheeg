import unittest

import torch
from torcheeg.models import BCGenerator, BCDiscriminator, BGenerator, BDiscriminator


class TestGAN(unittest.TestCase):
    def test_bgan(self):
        g_model = BGenerator(in_channels=128)
        d_model = BDiscriminator(in_channels=4)
        z = torch.normal(mean=0, std=1, size=(1, 128))
        fake_X = g_model(z)
        disc_X = d_model(fake_X)

        self.assertEqual(tuple(disc_X.shape), (1, 1))

        g_model = g_model.cuda()
        d_model = d_model.cuda()
        z = z.cuda()
        fake_X = g_model(z)
        disc_X = d_model(fake_X)
        self.assertEqual(tuple(disc_X.shape), (1, 1))

    def test_bcgan(self):
        g_model = BCGenerator(in_channels=128, num_classes=3)
        d_model = BCDiscriminator(in_channels=4, num_classes=3)
        z = torch.normal(mean=0, std=1, size=(1, 128))
        y = torch.randint(low=0, high=3, size=(1, ))
        fake_X = g_model(z, y)
        disc_X = d_model(fake_X, y)

        self.assertEqual(tuple(disc_X.shape), (1, 1))

        g_model = g_model.cuda()
        d_model = d_model.cuda()
        z = z.cuda()
        y = y.cuda()
        fake_X = g_model(z, y)
        disc_X = d_model(fake_X, y)
        self.assertEqual(tuple(disc_X.shape), (1, 1))


if __name__ == '__main__':
    unittest.main()
