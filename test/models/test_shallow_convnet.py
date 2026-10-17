import json
from pathlib import Path
import unittest

import torch
from torch import nn

from torcheeg.models import ShallowConvNet


def reference_state(name, value):
    count = value.numel()
    if name == 'norm.weight':
        values = torch.linspace(0.8, 1.2, count, dtype=torch.float64)
    elif name == 'norm.running_var':
        values = torch.linspace(0.7, 1.3, count, dtype=torch.float64)
    elif name == 'norm.running_mean':
        values = torch.linspace(-0.1, 0.1, count, dtype=torch.float64)
    elif name.endswith('num_batches_tracked'):
        return torch.tensor(4, dtype=torch.int64)
    elif name.endswith('bias'):
        values = torch.linspace(-0.02, 0.03, count, dtype=torch.float64)
    else:
        offset = (sum(map(ord, name)) % 11 - 5) * 0.002
        values = torch.linspace(-0.08, 0.08, count, dtype=torch.float64) + offset
    return values.reshape(value.shape)


class TestShallowConvNet(unittest.TestCase):
    def test_default_and_custom_window_shapes(self):
        for parameters in [dict(), dict(num_electrodes=3, chunk_size=512, num_classes=4),
                           dict(num_electrodes=2, chunk_size=99, temporal_filters=3,
                                spatial_filters=5, batch_norm=False)]:
            with self.subTest(parameters=parameters):
                model = ShallowConvNet(**parameters).eval()
                result = model(torch.randn(2, 1, model.num_electrodes, model.chunk_size))
                self.assertEqual(result.shape, (2, model.num_classes))
                expected_size = (model.chunk_size - model.temporal_kernel + 1 - model.pool_kernel) // model.pool_stride + 1
                self.assertEqual(model.feature_dim, model.spatial_filters * expected_size)
                self.assertEqual(model.classifier.kernel_size, (1, expected_size))

    def assert_gradient_summary(self, value, expected):
        flat = value.detach().reshape(-1)
        torch.testing.assert_close(flat[:8], torch.tensor(expected['first'], dtype=torch.float64),
                                   rtol=1e-7, atol=1e-9)
        for actual, target in [(flat.sum(), expected['sum']), (flat.norm(), expected['norm'])]:
            torch.testing.assert_close(actual, torch.tensor(target, dtype=torch.float64),
                                       rtol=1e-7, atol=1e-9)

    def test_independent_braindecode_forward_and_backward_reference(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures' /
                              'shallow_convnet_reference.json').read_text())
        self.assertEqual(fixture['version'], '0.8.1')
        self.assertEqual(fixture['state_algorithm'], 'canonical_linspace_v1')
        self.assertEqual(fixture['wheel_sha256'],
                         '6c2a571095a6ca366ade93a1a12bafce3225ce7d395eca6f7e2eb66edb5dd7da')
        for case in fixture['cases']:
            with self.subTest(parameters=case['parameters']):
                model = ShallowConvNet(**case['parameters']).double().eval()
                with torch.no_grad():
                    for name, value in model.state_dict().items():
                        value.copy_(reference_state(name, value))
                values = torch.arange(2 * model.num_electrodes * model.chunk_size,
                                      dtype=torch.float64)
                eeg = (torch.sin(values / 17) + torch.cos(values / 31)).reshape(
                    2, 1, model.num_electrodes, model.chunk_size).requires_grad_(True)
                result = model(eeg)
                torch.testing.assert_close(result, torch.tensor(case['output'], dtype=torch.float64),
                                           rtol=1e-9, atol=1e-9)
                result.square().mean().backward()
                self.assert_gradient_summary(eeg.grad, case['input_gradient'])
                self.assertEqual(set(dict(model.named_parameters())), set(case['parameter_gradients']))
                for name, parameter in model.named_parameters():
                    self.assert_gradient_summary(parameter.grad, case['parameter_gradients'][name])

    def test_initialization_keeps_batchnorm_statistics_clean(self):
        model = ShallowConvNet()
        self.assertEqual(model.norm.num_batches_tracked.item(), 0)
        torch.testing.assert_close(model.norm.running_mean, torch.zeros(40))
        torch.testing.assert_close(model.norm.running_var, torch.ones(40))
        self.assertEqual(model.norm.eps, 1e-5)
        self.assertEqual(model.drop.p, 0.5)
        self.assertTrue(model.training)

    def test_independent_eeg_arena_constrained_reference(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures' /
                              'shallow_convnet_arena_reference.json').read_text())
        self.assertEqual(fixture['commit'], 'c1aa1292aed21c0e170dd0da3cd0a4a60581f6ad')
        for case in fixture['cases']:
            with self.subTest(parameters=case['parameters']):
                model = ShallowConvNet(**case['parameters']).double().eval()
                with torch.no_grad():
                    for name, value in model.state_dict().items():
                        expected = reference_state(name, value)
                        if name.endswith('weight') and name != 'norm.weight':
                            expected = expected * 80
                        value.copy_(expected)
                values = torch.arange(2 * model.num_electrodes * model.chunk_size,
                                      dtype=torch.float64)
                eeg = (torch.sin(values / 17) + torch.cos(values / 31)).reshape(
                    2, 1, model.num_electrodes, model.chunk_size).requires_grad_(True)
                result = model(eeg)
                torch.testing.assert_close(result, torch.tensor(case['output'], dtype=torch.float64),
                                           rtol=1e-9, atol=1e-9)
                result.square().mean().backward()
                self.assert_gradient_summary(eeg.grad, case['input_gradient'])
                self.assertEqual(set(dict(model.named_parameters())), set(case['parameter_gradients']))
                for name, parameter in model.named_parameters():
                    self.assert_gradient_summary(parameter.grad, case['parameter_gradients'][name])

    def arena_model(self):
        return ShallowConvNet(num_electrodes=3, chunk_size=128, temporal_filters=4,
            spatial_filters=6, dropout=0, temporal_bias=False, conv_max_norm=2,
            classifier_max_norm=0.5, initialization='pytorch')

    def test_constraints_keep_parameters_and_accumulated_backward_valid(self):
        model = self.arena_model()
        layers = [(model.temporal_conv, 2), (model.spatial_conv, 2),
                  (model.classifier, 0.5)]
        pointers = [layer.weight.data_ptr() for layer, _ in layers]
        with torch.no_grad():
            for layer, _ in layers:
                layer.weight.fill_(20)
        eeg = torch.randn(2, 1, 3, 128, requires_grad=True)
        result = model(eeg) + model(eeg * 0.9)
        result.square().mean().backward()
        self.assertTrue(torch.isfinite(eeg.grad).all())
        for index, (layer, maximum) in enumerate(layers):
            self.assertEqual(layer.weight.data_ptr(), pointers[index])
            self.assertTrue(torch.all(layer.weight.flatten(1).norm(dim=1) <= maximum + 1e-6))
            self.assertTrue(torch.isfinite(layer.weight.grad).all())

    def test_frozen_constraints_preserve_weights_and_input_gradients(self):
        model = self.arena_model().eval().requires_grad_(False)
        with torch.no_grad():
            for layer in [model.temporal_conv, model.spatial_conv, model.classifier]:
                layer.weight.fill_(20)
        before = {name: parameter.clone() for name, parameter in model.named_parameters()}
        eeg = torch.randn(2, 1, 3, 128, requires_grad=True)
        result = model(eeg)
        self.assertTrue(torch.isfinite(result).all())
        result.square().mean().backward()
        self.assertTrue(torch.isfinite(eeg.grad).all())
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)

    def test_native_initialization_and_arena_bias_configuration(self):
        model = self.arena_model()
        self.assertIsNone(model.temporal_conv.bias)
        self.assertIsNone(model.spatial_conv.bias)
        self.assertTrue(torch.any(model.classifier.bias != 0))
        self.assertEqual(model.norm.num_batches_tracked.item(), 0)
        classic = ShallowConvNet(num_electrodes=3, chunk_size=128)
        torch.testing.assert_close(classic.temporal_conv.bias, torch.zeros(40))
        torch.testing.assert_close(classic.classifier.bias, torch.zeros(2))

    def test_training_step_updates_temporal_spatial_and_classifier_weights(self):
        torch.manual_seed(42)
        model = ShallowConvNet(num_electrodes=3, chunk_size=256,
                               temporal_filters=4, spatial_filters=6, dropout=0)
        eeg = torch.randn(4, 1, 3, 256, requires_grad=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        loss = nn.functional.cross_entropy(model(eeg), torch.tensor([0, 1, 1, 0]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(eeg.grad).all())
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        optimizer.step()
        for name in ['temporal_conv.weight', 'spatial_conv.weight', 'classifier.weight']:
            self.assertFalse(torch.equal(dict(model.named_parameters())[name], before[name]))

    def test_zero_windows_have_finite_forward_and_gradients(self):
        model = ShallowConvNet(num_electrodes=3, chunk_size=128, dropout=0)
        eeg = torch.zeros(2, 1, 3, 128, requires_grad=True)
        result = model(eeg)
        self.assertTrue(torch.isfinite(result).all())
        result.square().mean().backward()
        self.assertTrue(torch.isfinite(eeg.grad).all())
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()))

    def test_strict_state_dictionary_restores_predictions(self):
        parameters = dict(num_electrodes=3, chunk_size=256, spatial_filters=6,
                          temporal_filters=4)
        model = ShallowConvNet(**parameters).eval()
        clone = ShallowConvNet(**parameters).eval()
        clone.load_state_dict(model.state_dict(), strict=True)
        eeg = torch.randn(2, 1, 3, 256)
        torch.testing.assert_close(clone(eeg), model(eeg), rtol=0, atol=0)

    def test_invalid_parameters_and_input_shapes(self):
        cases = [dict(num_electrodes=0), dict(chunk_size=98), dict(num_classes=True),
                 dict(temporal_filters=1.5), dict(spatial_filters=-1),
                 dict(pool_stride=0), dict(log_eps=0), dict(log_eps=float('nan')),
                 dict(batch_norm_momentum=2), dict(dropout=-0.1),
                 dict(conv_max_norm=0), dict(classifier_max_norm=float('inf')),
                 dict(temporal_bias=3), dict(initialization='none')]
        for parameters in cases:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                ShallowConvNet(**parameters)
        model = ShallowConvNet(num_electrodes=3, chunk_size=128)
        for shape in [(2, 3, 128), (2, 2, 3, 128), (2, 1, 4, 128), (2, 1, 3, 129)]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                model(torch.zeros(*shape))


if __name__ == '__main__':
    unittest.main()
