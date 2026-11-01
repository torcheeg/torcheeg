import unittest

import torch

from torcheeg.models import FBMSNet


class TestInputValidation(unittest.TestCase):

    def test_model_configuration_is_checked(self):
        with self.assertRaises(ValueError):
            FBMSNet(in_channels=9, num_electrodes=22,
                    chunk_size=129, stride_factor=4)
        with self.assertRaisesRegex(ValueError, 'temporal'):
            FBMSNet(in_channels=9, num_electrodes=22,
                    chunk_size=512, temporal='unknown')


class TestFBMSNet(unittest.TestCase):

    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)

    def test_forward_returns_expected_shape(self):
        eeg = torch.randn(2, 9, 22, 512)
        model = FBMSNet(in_channels=9,
                        num_electrodes=22,
                        chunk_size=512,
                        num_classes=4)
        pred = model(eeg)
        for output in pred if isinstance(pred, (tuple, list)) else (pred,):
            self.assertEqual(output.device.type, 'cpu')
            self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(tuple(pred.shape), (2, 4))


if __name__ == '__main__':
    unittest.main()
