from contextlib import nullcontext
import json
from pathlib import Path
import unittest

import torch
from torch import nn

from torcheeg.models import DeepConvNet


def reference_state(name, value):
    count = value.numel()
    if name.endswith('norm.weight'):
        result = torch.linspace(0.8, 1.2, count, dtype=torch.float64)
    elif name.endswith('norm.running_var'):
        result = torch.linspace(0.7, 1.3, count, dtype=torch.float64)
    elif name.endswith('norm.running_mean'):
        result = torch.linspace(-0.1, 0.1, count, dtype=torch.float64)
    elif name.endswith('num_batches_tracked'):
        return torch.tensor(4, dtype=torch.int64)
    elif name.endswith('bias'):
        result = torch.linspace(-0.02, 0.03, count, dtype=torch.float64)
    else:
        offset = (sum(map(ord, name)) % 11 - 5) * 0.002
        result = torch.linspace(-0.08, 0.08, count, dtype=torch.float64) + offset
    return result.reshape(value.shape)


class TestDeepConvNet(unittest.TestCase):
    def model(self, **overrides):
        parameters = dict(num_electrodes=3, chunk_size=512, temporal_filters=3,
                          spatial_filters=3, hidden_channels=(6, 12, 24), dropout=0)
        parameters.update(overrides)
        return DeepConvNet(**parameters)

    def arena_model(self, **overrides):
        parameters = dict(temporal_bias=False, conv_max_norm=2,
                          classifier_max_norm=0.5, initialization='pytorch',
                          auto_kernel=True)
        parameters.update(overrides)
        return self.model(**parameters)

    def test_default_custom_and_no_batchnorm_shapes(self):
        cases = [dict(), dict(num_electrodes=3, chunk_size=512, num_classes=4),
                 dict(num_electrodes=2, chunk_size=200, temporal_kernel=3,
                      temporal_filters=2, spatial_filters=3,
                      hidden_channels=(6, 12, 24), batch_norm=False)]
        for parameters in cases:
            with self.subTest(parameters=parameters):
                model = DeepConvNet(**parameters).eval()
                eeg = torch.randn(2, 1, model.num_electrodes, model.chunk_size)
                result = model(eeg)
                self.assertEqual(result.shape, (2, model.num_classes))
                time = model.chunk_size
                for _ in range(4):
                    time = (time - model.temporal_kernel + 1 - model.pool_kernel) // model.pool_stride + 1
                self.assertEqual(model.feature_dim, model.hidden_channels[-1] * time)
        self.assertEqual(DeepConvNet().feature_dim, 1400)

    def assert_summary(self, value, expected):
        flat = value.detach().reshape(-1)
        torch.testing.assert_close(flat[:8], torch.tensor(expected['first'], dtype=torch.float64),
                                   rtol=1e-7, atol=1e-9)
        for actual, target in [(flat.sum(), expected['sum']), (flat.norm(), expected['norm'])]:
            torch.testing.assert_close(actual, torch.tensor(target, dtype=torch.float64),
                                       rtol=1e-7, atol=1e-9)

    def check_reference(self, filename, constrained=False):
        # Match the reference generator's reduction order and restore caller state.
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(2)
            self._check_reference(filename, constrained)
        finally:
            torch.set_num_threads(threads)

    def _check_reference(self, filename, constrained=False):
        fixture = json.loads((Path(__file__).parent / 'fixtures' / filename).read_text())
        if constrained:
            self.assertEqual(fixture['commit'], 'c1aa1292aed21c0e170dd0da3cd0a4a60581f6ad')
        else:
            self.assertEqual(fixture['version'], '0.8.1')
            self.assertEqual(fixture['wheel_sha256'],
                             '6c2a571095a6ca366ade93a1a12bafce3225ce7d395eca6f7e2eb66edb5dd7da')
        for case in fixture['cases']:
            with self.subTest(parameters=case['parameters']):
                adapted = constrained and case['actual_temporal_kernel'] != case['parameters'].get('temporal_kernel', 10)
                context = self.assertWarnsRegex(UserWarning, 'temporal_kernel reduced') if adapted else nullcontext()
                with context:
                    model = DeepConvNet(**case['parameters']).double().eval()
                if constrained:
                    self.assertEqual(model.temporal_kernel, case['actual_temporal_kernel'])
                    self.assertEqual(model.feature_dim, case['feature_dim'])
                with torch.no_grad():
                    for name, value in model.state_dict().items():
                        state = reference_state(name, value)
                        if constrained and name.endswith('weight') and '.norm.' not in name and not name.startswith('norm.'):
                            state = state * 80
                        value.copy_(state)
                values = torch.arange(2 * model.num_electrodes * model.chunk_size, dtype=torch.float64)
                eeg = (torch.sin(values / 17) + torch.cos(values / 31)).reshape(
                    2, 1, model.num_electrodes, model.chunk_size).requires_grad_(True)
                result = model(eeg)
                torch.testing.assert_close(result, torch.tensor(case['output'], dtype=torch.float64),
                                           rtol=1e-9, atol=1e-9)
                result.square().mean().backward()
                self.assert_summary(eeg.grad, case['input_gradient'])
                self.assertEqual(set(dict(model.named_parameters())), set(case['parameter_gradients']))
                for name, parameter in model.named_parameters():
                    self.assert_summary(parameter.grad, case['parameter_gradients'][name])

    def test_independent_braindecode_forward_and_backward_reference(self):
        self.check_reference('deep_convnet_reference.json')

    def test_independent_arena_adaptation_and_constrained_reference(self):
        self.check_reference('deep_convnet_arena_reference.json', constrained=True)

    def test_kernel_adaptation_is_explicit_and_reports_actual_geometry(self):
        for samples, kernel in [(1000, 10), (512, 10), (441, 10), (440, 9),
                                (250, 5), (200, 3), (128, 2)]:
            with self.subTest(samples=samples):
                context = self.assertWarnsRegex(UserWarning, f'to {kernel} ') if kernel != 10 else nullcontext()
                with context:
                    model = self.arena_model(chunk_size=samples)
                self.assertEqual(model.temporal_kernel, kernel)
                self.assertEqual(model.requested_temporal_kernel, 10)
                self.assertEqual(model(torch.randn(2, 1, 3, samples)).shape, (2, 2))
        with self.assertRaises(ValueError):
            self.model(chunk_size=128)
        for samples in [120, 20]:
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                self.arena_model(chunk_size=samples)

    def test_initialization_preserves_four_batchnorm_statistics(self):
        model = self.model()
        norms = [module for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
        self.assertEqual(len(norms), 4)
        for norm in norms:
            self.assertEqual(norm.num_batches_tracked.item(), 0)
            self.assertEqual(norm.eps, 1e-5)
            torch.testing.assert_close(norm.running_mean, torch.zeros_like(norm.running_mean))
            torch.testing.assert_close(norm.running_var, torch.ones_like(norm.running_var))
        self.assertTrue(model.training)
        torch.testing.assert_close(model.temporal_conv.bias, torch.zeros_like(model.temporal_conv.bias))
        arena = self.arena_model()
        self.assertTrue(torch.any(arena.classifier.bias != 0))
        self.assertTrue(all(module.bias is None for module in
                            [arena.temporal_conv, arena.spatial_conv] + [block.conv for block in arena.blocks]))

    def test_optimizer_updates_all_convolutional_stages(self):
        for batch_norm in [True, False]:
            with self.subTest(batch_norm=batch_norm):
                torch.manual_seed(42)
                model = self.model(batch_norm=batch_norm)
                eeg = torch.randn(4, 1, 3, 512, requires_grad=True)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
                loss = nn.functional.cross_entropy(model(eeg), torch.tensor([0, 1, 0, 1]))
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(torch.isfinite(eeg.grad).all())
                for parameter in model.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                optimizer.step()
                for name in ['temporal_conv.weight', 'spatial_conv.weight', 'blocks.0.conv.weight',
                             'blocks.2.conv.weight', 'classifier.weight']:
                    self.assertFalse(torch.equal(dict(model.named_parameters())[name], before[name]))


    def test_zero_windows_have_finite_outputs_and_gradients(self):
        model = self.model()
        eeg = torch.zeros(2, 1, 3, 512, requires_grad=True)
        result = model(eeg)
        self.assertTrue(torch.isfinite(result).all())
        result.square().mean().backward()
        self.assertTrue(torch.isfinite(eeg.grad).all())
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()))

    def test_constraints_preserve_parameters_and_accumulated_backward(self):
        model = self.arena_model()
        layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
        pointers = [module.weight.data_ptr() for module in layers]
        with torch.no_grad():
            for layer in layers:
                layer.weight.fill_(20)
        eeg = torch.randn(2, 1, 3, 512, requires_grad=True)
        result = model(eeg) + model(eeg * 0.9)
        result.square().mean().backward()
        self.assertTrue(torch.isfinite(eeg.grad).all())
        for index, layer in enumerate(layers):
            self.assertEqual(layer.weight.data_ptr(), pointers[index])
            self.assertTrue(torch.all(layer.weight.flatten(1).norm(dim=1) <= layer.max_norm + 1e-6))
            self.assertTrue(torch.isfinite(layer.weight.grad).all())

    def test_frozen_projection_preserves_stored_state_and_input_gradients(self):
        model = self.arena_model().eval().requires_grad_(False)
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, nn.Conv2d):
                    module.weight.fill_(20)
        before = {name: value.clone() for name, value in model.state_dict().items()}
        eeg = torch.randn(2, 1, 3, 512, requires_grad=True)
        result = model(eeg)
        result.square().mean().backward()
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(torch.isfinite(eeg.grad).all())
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_classifier_dropout_operates_after_last_feature_block(self):
        model = self.model(classifier_dropout=1)
        captured = []
        handle = model.blocks[-1].register_forward_hook(lambda module, inputs, output: captured.append(output.detach()))
        try:
            result = model(torch.randn(2, 1, 3, 512))
        finally:
            handle.remove()
        self.assertTrue(torch.any(captured[0] != 0))
        torch.testing.assert_close(result, model.classifier.bias.expand_as(result), rtol=0, atol=0)

    def test_strict_state_dictionary_restores_predictions(self):
        model = self.model()
        eeg = torch.randn(2, 1, 3, 512)
        model(eeg)
        clone = self.model().eval()
        clone.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(clone(eeg), model.eval()(eeg), rtol=0, atol=0)

    def test_invalid_parameters_and_input_layouts(self):
        cases = [dict(num_electrodes=0), dict(chunk_size=440), dict(temporal_kernel=True),
                 dict(hidden_channels=(1, 2)), dict(hidden_channels=3), dict(pool_stride=0),
                 dict(dropout=float('nan')), dict(classifier_dropout=1.1), dict(conv_max_norm=0),
                 dict(classifier_max_norm=float('inf')), dict(initialization='bad'),
                 dict(batch_norm_momentum=-0.1), dict(auto_kernel=3)]
        for parameters in cases:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                DeepConvNet(**parameters)
        model = self.model()
        for shape in [(2, 3, 512), (2, 2, 3, 512), (2, 1, 4, 512), (2, 1, 3, 513)]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                model(torch.zeros(*shape))


if __name__ == '__main__':
    unittest.main()
