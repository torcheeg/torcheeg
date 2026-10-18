import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

from scipy.signal import resample
import torch

from torcheeg.models import STEEGFormer, STEEGFormerClassifier, STEEGFormerRegression, STEEGFormerEmbedding


class TestSTEEGFormerHeads(unittest.TestCase):
    def backbone(self):
        return STEEGFormer(hid_channels=16, mlp_channels=32, depth=2, num_heads=4)

    def classifier(self, **kwargs):
        return STEEGFormerClassifier(self.backbone(), ['Fp1', 'Fp2'], 100, sampling_rate=200, **kwargs)

    def test_official_classifier_full_signal_and_parameter_gradients(self):
        root = Path(__file__).parent / 'fixtures'
        base = json.loads((root / 'steegformer_reference.json').read_text())
        ref = json.loads((root / 'steegformer_head_reference.json').read_text())

        def tensor(record):
            return torch.tensor(record['values'], dtype=torch.float64).reshape(record['shape'])

        model = self.classifier(num_classes=3, dropout=0., normalize_signal=True).double().eval()
        model.backbone.load_state_dict({k: tensor(v) for k, v in base['state_dict'].items()}, strict=True)
        with torch.no_grad():
            model.task_head.output.weight.copy_(tensor(ref['head_weight']))
            model.task_head.output.bias.copy_(tensor(ref['head_bias']))
        expected_gradient_names = {name for name, _ in model.backbone.named_parameters() if not name.startswith('norm.')} | {'head.weight', 'head.bias'}
        self.assertEqual(set(ref['parameter_gradients']), expected_gradient_names)
        x = tensor(ref['input']).requires_grad_()
        torch.testing.assert_close(model.preprocess(x), tensor(ref['preprocessed']), rtol=1e-10, atol=1e-11)
        torch.testing.assert_close(model.feature(x), tensor(ref['features']), rtol=1e-10, atol=1e-11)
        output = model(x)
        torch.testing.assert_close(output, tensor(ref['logits']), rtol=1e-10, atol=1e-11)
        (output * tensor(ref['coefficient'])).sum().backward()
        torch.testing.assert_close(x.grad, tensor(ref['input_gradient']), rtol=1e-9, atol=1e-10)
        for name, parameter in model.backbone.named_parameters():
            expected = ref['parameter_gradients'].get(name)
            if expected is None:
                self.assertIsNone(parameter.grad, name)
            else:
                torch.testing.assert_close(parameter.grad, tensor(expected), rtol=1e-9, atol=1e-10, msg=name)
        for field in ('weight', 'bias'):
            torch.testing.assert_close(getattr(model.task_head.output, field).grad,
                                       tensor(ref['parameter_gradients']['head.' + field]), rtol=1e-10, atol=1e-11)

    def test_population_and_sample_std_precede_complete_window_resampling(self):
        x = torch.linspace(-3, 7, 2 * 2 * 100, dtype=torch.float64).reshape(2, 2, 100)
        x = x + torch.sin(x * 5.)
        for correction in (0, 1):
            model = self.classifier(normalize_signal=True, std_correction=correction).double()
            normalized = (x.numpy() - x.numpy().mean(-1, keepdims=True)) / x.numpy().std(-1, keepdims=True, ddof=correction)
            expected = resample(normalized, 64, axis=-1)
            torch.testing.assert_close(model.preprocess(x), torch.from_numpy(expected), rtol=1e-12, atol=1e-12)
            self.assertEqual((model.native_samples, model.num_patches), (64, 4))
        raw = self.classifier(normalize_signal=False).double()
        torch.testing.assert_close(raw.preprocess(x), torch.from_numpy(resample(x.numpy(), 64, axis=-1)), rtol=1e-12, atol=1e-12)

    def test_embedding_layout_pooling_and_explicit_final_norm(self):
        backbone = self.backbone().eval()
        x = torch.randn(2, 2, 64)
        grid = STEEGFormerEmbedding(backbone, ['Fp1', 'Fp2'], 64, pooling='none').eval()
        expected = backbone.forward_features(x, grid.channel_indices, normalize=False).permute(0, 2, 1, 3)
        torch.testing.assert_close(grid(x), expected, rtol=0, atol=0)
        self.assertEqual(grid(x).shape, (2, 2, 4, 16))
        for pooling in ('mean', 'flatten'):
            model = STEEGFormerEmbedding(backbone, ['Fp1', 'Fp2'], 64, pooling=pooling).eval()
            target = expected.mean((1, 2)) if pooling == 'mean' else expected.flatten(1)
            torch.testing.assert_close(model(x), target, rtol=0, atol=0)
        normalized = STEEGFormerEmbedding(backbone, ['Fp1', 'Fp2'], 64, normalize_features=True).eval()
        torch.testing.assert_close(normalized(x), backbone(x, grid.channel_indices), rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(grid.feature(x, flatten=True), expected.flatten(1), rtol=0, atol=0)

    def test_regression_scalar_axis_and_unconstrained_outputs(self):
        for count in (1, 3):
            model = STEEGFormerRegression(self.backbone(), ['Fp1', 'Fp2'], 64, num_outputs=count, dropout=0.)
            with torch.no_grad():
                model.task_head.output.weight.zero_()
                model.task_head.output.bias.fill_(-2.)
            output = model(torch.randn(2, 2, 64))
            torch.testing.assert_close(output, torch.full((2, count), -2.), rtol=0, atol=0)

    def test_frozen_encoder_mode_input_gradients_and_head_updates(self):
        model = self.classifier(freeze_encoder=True, dropout=.5).double()
        before = {k: v.clone() for k, v in model.backbone.state_dict().items()}
        head = model.task_head.output.weight.detach().clone()
        x = torch.randn(2, 2, 100, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        model.train()
        self.assertFalse(model.backbone.training)
        self.assertTrue(model.task_head.dropout.training)
        for _ in range(2):
            optimizer.zero_grad()
            model(x).square().sum().backward()
            optimizer.step()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.abs().sum().item(), 0.)
        self.assertFalse(torch.equal(head, model.task_head.output.weight))
        for name, parameter in model.backbone.named_parameters():
            self.assertFalse(parameter.requires_grad)
            self.assertIsNone(parameter.grad)
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
        model.eval()
        torch.testing.assert_close(model(x), model(x), rtol=0, atol=0)

    def test_zero_variance_signals_and_low_precision_statistics(self):
        for correction in (0, 1):
            model = self.classifier(normalize_signal=True, std_correction=correction).double().eval()
            x = torch.full((2, 2, 100), 7., dtype=torch.float64, requires_grad=True)
            torch.testing.assert_close(model.preprocess(x), torch.zeros(2, 2, 64, dtype=x.dtype), rtol=0, atol=0)
            model(x).square().sum().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
        model = self.classifier(normalize_signal=True).half()
        x = torch.randn(2, 2, 100).half()
        expected = self.classifier(normalize_signal=True).preprocess(x.float())
        actual = model.preprocess(x)
        self.assertEqual(actual.dtype, torch.float16)
        torch.testing.assert_close(actual.float(), expected, rtol=.01, atol=.002)

    def test_validation_does_not_mutate_caller_backbone(self):
        cases = [dict(chunk_size=101), dict(chunk_size=15, sampling_rate=128), dict(sampling_rate=0),
                 dict(channel_names=['unknown']), dict(channel_names=[]), dict(std_floor=0.),
                 dict(std_correction=True), dict(std_correction=2), dict(normalize_signal=1),
                 dict(normalize_features=1), dict(freeze_encoder=1), dict(num_classes=0), dict(dropout=1.)]
        for update in cases:
            backbone = self.backbone().eval()
            before = [(id(p), p.requires_grad) for p in backbone.parameters()]
            options = dict(channel_names=['Fp1', 'Fp2'], chunk_size=100, sampling_rate=200, freeze_encoder=True)
            options.update(update)
            with self.subTest(options=update), self.assertRaises(ValueError):
                STEEGFormerClassifier(backbone, **options)
            self.assertEqual(before, [(id(p), p.requires_grad) for p in backbone.parameters()])
            self.assertFalse(backbone.training)

    def test_source_shape_validation_and_preserved_flags(self):
        backbone = self.backbone()
        backbone.patch_embed.requires_grad_(False)
        flags = [p.requires_grad for p in backbone.parameters()]
        model = STEEGFormerClassifier(backbone, ['Fp1', 'Fp2'], 64)
        self.assertEqual(flags, [p.requires_grad for p in backbone.parameters()])
        for x in (torch.randn(2, 3, 64), torch.randn(0, 2, 64), torch.randn(2, 2, 65),
                  torch.ones(2, 2, 64, dtype=torch.long), torch.randn(2, 64), None):
            with self.assertRaisesRegex(ValueError, 'configured source shape'):
                model(x)
        with self.assertRaisesRegex(ValueError, 'flatten'):
            model.feature(torch.randn(2, 2, 64), flatten=1)

    def test_safe_backbone_factory_and_full_task_state_restoration(self):
        backbone = self.backbone()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'backbone.pt'
            torch.save({'state_dict': {'encoder.' + k: v for k, v in backbone.state_dict().items()}}, path)
            kwargs = dict(hid_channels=16, mlp_channels=32, depth=2, num_heads=4)
            model = STEEGFormerClassifier.from_pretrained(path, kwargs, prefix='encoder.',
                channel_names=['Fp1', 'Fp2'], chunk_size=100, sampling_rate=200,
                normalize_signal=True, normalize_features=True, freeze_encoder=True).double().eval()
            restored = STEEGFormerClassifier(self.backbone().double(), ['Fp1', 'Fp2'], 100,
                sampling_rate=200, normalize_signal=True, normalize_features=True, freeze_encoder=True).eval()
            restored.load_state_dict(model.state_dict(), strict=True)
            copied = copy.deepcopy(model)
            x = torch.randn(2, 2, 100, dtype=torch.float64)
            torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)
            torch.testing.assert_close(copied(x), model(x), rtol=0, atol=0)
            for name, value in backbone.state_dict().items():
                torch.testing.assert_close(model.backbone.state_dict()[name], value.double(), rtol=0, atol=0)

    @unittest.skipUnless(os.environ.get('TORCHEEG_TEST_COMPILE') == '1' or os.environ.get('GITHUB_ACTIONS') == 'true',
                         'Enable TORCHEEG_TEST_COMPILE=1 with a C++ compiler; GitHub Actions runs this regression.')
    def test_compiled_normalization_resampling_training_and_constant_inputs(self):
        previous = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous)
        torch.set_num_threads(2)
        model = self.classifier(dropout=0., normalize_signal=True).eval()
        reference = copy.deepcopy(model)
        x = torch.randn(2, 2, 100, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        expected = reference(x)
        coefficient = torch.tensor([[.2, -.3], [.7, .1]])
        (expected * coefficient).sum().backward()
        compiled = torch.compile(model, backend='inductor', fullgraph=True)
        actual = compiled(y)
        (actual * coefficient).sum().backward()
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(y.grad, x.grad, rtol=1e-3, atol=1e-5)
        for (name, left), (_, right) in zip(model.named_parameters(), reference.named_parameters()):
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-3, atol=1e-5, msg=name)
        model.zero_grad(set_to_none=True)
        constant = torch.zeros(2, 2, 100, requires_grad=True)
        output = compiled(constant)
        output.sum().backward()
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(constant.grad).all())
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))


if __name__ == '__main__':
    unittest.main()
