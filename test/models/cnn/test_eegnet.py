import unittest

import torch

from torcheeg.models import EEGNet


class TestEEGNet(unittest.TestCase):

    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)

    def test_forward_returns_expected_shape(self):
        eeg = torch.randn(1, 1, 32, 128)
        model = EEGNet(chunk_size=128,
                       num_electrodes=32,
                       dropout=0.5,
                       kernel_1=64,
                       kernel_2=16,
                       F1=8,
                       D=2,
                       F2=16,
                       num_classes=2)
        pred = model(eeg)
        for output in pred if isinstance(pred, (tuple, list)) else (pred,):
            self.assertEqual(output.device.type, 'cpu')
            self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(tuple(pred.shape), (1, 2))

    def test_eegnet_same_padding(self):
        # block1 downsamples the time axis by 4 and block2 by a further 8, so
        # both convolutions must leave the time axis untouched.
        for kernel_1, kernel_2 in [(64, 16), (63, 15)]:
            for chunk_size in [127, 128, 151]:
                eeg = torch.randn(1, 1, 32, chunk_size)
                model = EEGNet(chunk_size=chunk_size,
                               num_electrodes=32,
                               dropout=0.5,
                               kernel_1=kernel_1,
                               kernel_2=kernel_2,
                               F1=8,
                               D=2,
                               F2=16,
                               num_classes=2)
                with torch.no_grad():
                    block1_out = model.block1(eeg)
                    block2_out = model.block2(block1_out)
                self.assertEqual(block1_out.shape[3], chunk_size // 4)
                self.assertEqual(block2_out.shape[3], chunk_size // 4 // 8)
                self.assertEqual(tuple(model(eeg).shape), (1, 2))

    def test_eegnet_checkpoint_parameter_names(self):
        batch_norm_fields = ['weight', 'bias', 'running_mean',
                             'running_var', 'num_batches_tracked']
        expected_keys = ['block1.0.weight']
        expected_keys.extend(
            'block1.1.' + field for field in batch_norm_fields)
        expected_keys.append('block1.2.weight')
        expected_keys.extend(
            'block1.3.' + field for field in batch_norm_fields)
        expected_keys.extend(['block2.0.weight', 'block2.1.weight'])
        expected_keys.extend(
            'block2.2.' + field for field in batch_norm_fields)
        expected_keys.append('lin.weight')
        for chunk_size in (128, 151):
            with self.subTest(chunk_size=chunk_size):
                model = EEGNet(chunk_size=chunk_size, num_electrodes=32)
                self.assertEqual(list(model.state_dict()), expected_keys)
                self.assertEqual(tuple(model.lin.weight.shape), (2, 64))

    def test_eegnet_odd_kernel_padding_matches_symmetric_convolution(self):
        model = EEGNet(chunk_size=128, num_electrodes=32,
                       kernel_1=63, kernel_2=15)
        for conv in (model.block1[0], model.block2[0]):
            reference = torch.nn.Conv2d(conv.in_channels, conv.out_channels,
                                        conv.kernel_size, groups=conv.groups,
                                        padding=(0, conv.kernel_size[1] // 2), bias=False)
            reference.load_state_dict(conv.state_dict())
            eeg = torch.randn(2, conv.in_channels, 32, 128)
            torch.testing.assert_close(conv(eeg), reference(eeg))

    def test_backward_produces_finite_parameter_and_input_gradients(self):
        model = EEGNet(chunk_size=128, num_electrodes=8, num_classes=2)
        eeg = torch.randn(2, 1, 8, 128, requires_grad=True)
        model(eeg).square().mean().backward()
        self.assertIsNotNone(eeg.grad)
        self.assertTrue(torch.isfinite(eeg.grad).all())
        gradients = [parameter.grad for parameter in model.parameters()
                     if parameter.requires_grad]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all()
                        for gradient in gradients))
        self.assertTrue(any(gradient.abs().sum() >
                        0 for gradient in gradients))


if __name__ == '__main__':
    unittest.main()
