import unittest

import torch
from torcheeg.models import BCGenerator, BCDiscriminator, BGenerator, BDiscriminator,EEGfuseNet,EFDiscriminator

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

    def test_eegfusenet(self):

        g_model = EEGfuseNet(num_electrodes=20,hid_channels_gru=16,num_layers_gru=1,hid_channels_cnn=1,chunk_size=128)
        d_model = EFDiscriminator(num_electrodes=20,hid_channels_cnn=1,chunk_size=128)
        X = torch.rand(2,1,20,128)
        fake_X, deep_feature = g_model(X)
        p_real,p_fake = d_model(X),d_model(fake_X)

        self.assertEqual(tuple(fake_X.shape), tuple(X.shape))
        self.assertEqual(tuple(deep_feature.shape), (2,64))
        self.assertEqual(tuple(p_real.shape),(2,1))
        self.assertEqual(tuple(p_fake.shape),(2,1))
        
        # to gpu
        g_model = EEGfuseNet(num_electrodes=20,hid_channels_gru=16,num_layers_gru=1,hid_channels_cnn=1,chunk_size=128).cuda(0)
        d_model = EFDiscriminator(num_electrodes=20,hid_channels_cnn=1,chunk_size=128).cuda(0)
        X = torch.rand(2,1,20,128).cuda(0)
        fake_X, deep_feature = g_model(X)
        p_real,p_fake = d_model(X),d_model(fake_X)

        self.assertEqual(tuple(fake_X.shape), tuple(X.shape))
        self.assertEqual(tuple(deep_feature.shape), (2,64))
        self.assertEqual(tuple(p_real.shape),(2,1))
        self.assertEqual(tuple(p_fake.shape),(2,1))
        

        # other shapes 1 
        g_model = EEGfuseNet(num_electrodes=64,hid_channels_gru=16,num_layers_gru=1,hid_channels_cnn=1,chunk_size=512)
        d_model = EFDiscriminator(num_electrodes=64,hid_channels_cnn=1,chunk_size=512)
        X = torch.rand(2,1,64,512)
        fake_X, deep_feature = g_model(X)
        p_real,p_fake = d_model(X),d_model(fake_X)

        self.assertEqual(tuple(fake_X.shape), tuple(X.shape))
        self.assertEqual(tuple(deep_feature.shape), (2,256))
        self.assertEqual(tuple(p_real.shape),(2,1))
        self.assertEqual(tuple(p_fake.shape),(2,1))

        

        # other shapes 2
        g_model = EEGfuseNet(num_electrodes=32,hid_channels_gru=16,num_layers_gru=1,hid_channels_cnn=1,chunk_size=384)
        d_model = EFDiscriminator(num_electrodes=32,hid_channels_cnn=1,chunk_size=384)
        X = torch.randn(2,1,32,384)
        fake_X, deep_feature = g_model(X)
        p_real,p_fake = d_model(X),d_model(fake_X)

        self.assertEqual(tuple(fake_X.shape), tuple(X.shape))
        self.assertEqual(tuple(deep_feature.shape), (2,192))
        self.assertEqual(tuple(p_real.shape),(2,1))
        self.assertEqual(tuple(p_fake.shape),(2,1))


if __name__ == '__main__':
    unittest.main()
