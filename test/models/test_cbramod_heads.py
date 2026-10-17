import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scipy.signal import resample
import torch
from torch import nn

from torcheeg.models import CBraMod, CBraModClassifier, CBraModRegression, CBraModEmbedding
from torcheeg.models.transformer import CBraModClassifier as TransformerImport


CONFIG = dict(patch_size=25, hid_channels=16, mlp_channels=32, depth=1,
              num_heads=4, dropout=0., position_kernel=(3, 3))


def backbone():
    return CBraMod(**CONFIG)


def task(kind=CBraModClassifier, **kwargs):
    return kind(backbone(), num_electrodes=3, chunk_size=128, num_patches=4, **kwargs)


def set_reference_weights(head):
    with torch.no_grad():
        for index, layer in enumerate(m for m in head.modules() if isinstance(m, nn.Linear)):
            row = torch.arange(layer.out_features, dtype=torch.float64)[:, None]
            col = torch.arange(layer.in_features, dtype=torch.float64)[None, :]
            layer.weight.copy_(((row * 7 + col * 3 + index) % 19 - 9) / (19 * layer.in_features ** .5))
            layer.bias.copy_((torch.arange(layer.out_features, dtype=torch.float64) % 7 - 3) / 20)


class TestCBraModHeads(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_public_import_and_independent_official_four_head_reference(self):
        self.assertIs(CBraModClassifier, TransformerImport)
        reference = json.loads((Path(__file__).parent / 'fixtures/cbramod_head_reference.json').read_text())
        self.assertEqual(reference['commit'], 'b9e961003214326972c567eff390e75b0287e32a')
        self.assertEqual(reference['sha256'], 'a7d720d2c1603cffdca4485d33d8f3d1db66876c9b9e48dd18f063af5e65b297')
        configurations = [('mean', ()), ('flatten', ()), ('flatten', (200,)), ('flatten', None)]
        for row, (pooling, hidden) in zip(reference['cases'], configurations):
            with self.subTest(kind=row['kind']):
                encoder = CBraMod(depth=1, dropout=0.).double()
                model = CBraModClassifier(encoder, 22, 800, num_classes=3, num_patches=4,
                    pooling=pooling, head_channels=hidden, dropout=0.).double().eval()
                set_reference_weights(model.task_head)
                features = ((torch.arange(17600, dtype=torch.float64).reshape(1, 22, 4, 200) % 71 - 35) / 35).requires_grad_()
                with patch.object(encoder, 'forward_features', return_value=features):
                    output = model(torch.zeros(1, 22, 800, dtype=torch.float64))
                torch.testing.assert_close(output, torch.tensor(row['output'], dtype=torch.float64), rtol=1e-11, atol=1e-12)
                (output * torch.tensor([.2, -.3, .7], dtype=torch.float64)).sum().backward()
                expected = torch.tensor(row['input_gradients'], dtype=torch.float64)
                torch.testing.assert_close(features.grad.flatten()[row['input_gradient_indices']], expected, rtol=1e-10, atol=1e-12)
                layers = [m for m in model.task_head.modules() if isinstance(m, nn.Linear)]
                self.assertEqual(len(layers), len(row['weight_gradients']))
                for layer, gradients in zip(layers, row['weight_gradients']):
                    for key, actual in [('sum', layer.weight.grad.sum()), ('norm', layer.weight.grad.norm())]:
                        torch.testing.assert_close(actual, torch.tensor(gradients[key], dtype=torch.float64), rtol=1e-10, atol=1e-12)
                    torch.testing.assert_close(layer.weight.grad.flatten()[:5], torch.tensor(gradients['first'], dtype=torch.float64), rtol=1e-10, atol=1e-12)
                    torch.testing.assert_close(layer.bias.grad, torch.tensor(gradients['bias'], dtype=torch.float64), rtol=1e-10, atol=1e-12)

    def test_features_match_scipy_patchwise_preprocessing_and_backbone(self):
        model = task().double().eval()
        x = torch.randn(2, 3, 128, dtype=torch.float64)
        source = x.numpy().reshape(2, 3, 4, 32)
        expected_input = torch.tensor(resample(source, 25, axis=-1))
        expected = model.backbone.forward_features(expected_input)
        actual = model.feature(x)
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
        torch.testing.assert_close(model.feature(x, flatten=True), expected.flatten(1))
        torch.testing.assert_close(model(x), model.task_head(expected.flatten(1)))
        self.assertEqual(model.feature_dim, 3 * 4 * 16)

    def test_patch_resampling_preserves_segment_boundaries_and_no_resize_identity(self):
        encoder = backbone()
        model = CBraModEmbedding(encoder, 3, 100, num_patches=4, flatten=False)
        x = torch.randn(2, 3, 100)
        with patch.object(encoder, 'forward_features', side_effect=lambda values, mask=None: values) as observe:
            actual = model(x)
        torch.testing.assert_close(actual, x.reshape(2, 3, 4, 25), rtol=0, atol=0)
        self.assertEqual(observe.call_count, 1)
        self.assertEqual(model.state_dict().keys(), {'backbone.' + name for name in encoder.state_dict()})

    def test_frozen_backbone_stays_eval_and_keeps_input_gradients(self):
        encoder = CBraMod(**dict(CONFIG, dropout=.7))
        model = CBraModClassifier(encoder, 3, 128, num_patches=4, dropout=0., freeze_encoder=True)
        model.train()
        self.assertTrue(model.task_head.training)
        self.assertTrue(all(not m.training for m in encoder.modules()))
        x = torch.randn(2, 3, 128, requires_grad=True)
        first = model(x)
        torch.testing.assert_close(first, model(x), rtol=0, atol=0)
        first.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.norm().item(), 0)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in encoder.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.task_head.parameters()))
        model.eval().train()
        self.assertFalse(encoder.training)

    def test_optimizer_updates_head_and_preserves_frozen_complete_backbone(self):
        model = task(freeze_encoder=True, dropout=0.)
        before = {name: value.clone() for name, value in model.backbone.state_dict().items()}
        head = model.task_head.output.weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        optimizer.zero_grad()
        nn.functional.cross_entropy(model(torch.randn(2, 3, 128)), torch.tensor([0, 1])).backward()
        optimizer.step()
        self.assertFalse(torch.equal(head, model.task_head.output.weight))
        for name, value in model.backbone.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_unfrozen_gradients_and_unused_native_projection(self):
        model = task(dropout=0.)
        model(torch.randn(2, 3, 128)).square().mean().backward()
        for name, parameter in model.backbone.named_parameters():
            if name.startswith('proj_out.') or name.endswith('mask_encoding'):
                self.assertIsNone(parameter.grad, name)
            else:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertTrue(all(key in model.state_dict() for key in ['backbone.proj_out.0.weight', 'backbone.proj_out.0.bias']))

    def test_existing_parameter_flags_dtype_device_and_dropout_configuration(self):
        encoder = backbone().double()
        encoder.encoder.layers[0].linear1.weight.requires_grad_(False)
        flags = {name: p.requires_grad for name, p in encoder.named_parameters()}
        model = CBraModClassifier(encoder, 3, 128, num_patches=4, dropout=.37)
        self.assertEqual(flags, {name: p.requires_grad for name, p in encoder.named_parameters()})
        self.assertTrue(all(p.dtype == torch.float64 for p in model.parameters()))
        self.assertEqual([m.p for m in model.task_head.modules() if isinstance(m, nn.Dropout)], [.37, .37])
        x = torch.randn(2, 3, 128, dtype=torch.float64)
        model.train()
        self.assertFalse(torch.equal(model(x), model(x)))
        model.eval()
        torch.testing.assert_close(model(x), model(x), rtol=0, atol=0)

    def test_regression_scalar_vector_and_embedding_contracts(self):
        x = torch.randn(2, 3, 128, requires_grad=True)
        for outputs in (1, 3):
            model = task(CBraModRegression, num_outputs=outputs, dropout=0.)
            prediction = model(x)
            self.assertEqual(prediction.shape, (2, outputs))
            nn.functional.mse_loss(prediction, torch.ones_like(prediction)).backward()
            self.assertGreater(model.task_head.output.weight.grad.norm().item(), 0)
        for flatten in (True, False):
            model = task(CBraModEmbedding, flatten=flatten).eval()
            torch.testing.assert_close(model(x), model.feature(x, flatten=flatten))
            self.assertEqual(model(x).shape, (2, 192) if flatten else (2, 3, 4, 16))

    def test_mask_forwarding_and_validation(self):
        model = task(dropout=0.).eval()
        x = torch.randn(2, 3, 128)
        mask = torch.zeros(2, 3, 4, dtype=torch.bool)
        mask[:, :, 1] = True
        expected = model.backbone.forward_features(model.resample(x.reshape(2, 3, 4, 32)), mask=mask)
        torch.testing.assert_close(model.feature(x, mask=mask), expected)
        torch.testing.assert_close(model(x, mask=mask), model.task_head(expected.flatten(1)))
        with self.assertRaises(ValueError):
            model(x, mask=torch.full((2, 3, 4), .5))

    def test_safe_local_backbone_factory_and_complete_task_roundtrip(self):
        encoder = backbone()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'backbone.pt'
            for nested in (False, True):
                state = encoder.state_dict()
                checkpoint = {'state_dict': {'model.' + k: v for k, v in state.items()}, 'epoch': 2} if nested else state
                torch.save(checkpoint, path)
                for kind in (CBraModClassifier, CBraModRegression, CBraModEmbedding):
                    model = kind.from_pretrained(path, CONFIG, prefix='model.' if nested else '',
                        num_electrodes=3, chunk_size=128, num_patches=4)
                    for name, value in model.backbone.state_dict().items():
                        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
                    saved = Path(directory) / 'task.pt'
                    torch.save(model.state_dict(), saved)
                    restored = kind(backbone(), 3, 128, num_patches=4)
                    restored.load_state_dict(torch.load(saved, weights_only=True), strict=True)
                    x = torch.randn(2, 3, 128)
                    torch.testing.assert_close(restored.eval()(x), model.eval()(x), rtol=0, atol=0)

    def test_factory_rejects_incomplete_or_invalid_backbone_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'backbone.pt'
            state = backbone().state_dict()
            state.pop('proj_out.0.weight')
            torch.save(state, path)
            with self.assertRaises(ValueError):
                CBraModClassifier.from_pretrained(path, CONFIG, num_electrodes=3, chunk_size=128, num_patches=4)
            with self.assertRaises(ValueError):
                CBraModEmbedding.from_pretrained(path, [], num_electrodes=3, chunk_size=128)

    def test_explicit_legacy_hidden_widths_and_linear_heads(self):
        legacy = CBraModClassifier(backbone(), 3, 125, num_patches=5, head_channels=(5 * 16, 16))
        self.assertEqual(legacy.task_head.linear0.out_features, 80)
        for pooling, width in [('flatten', 192), ('mean', 16)]:
            model = task(pooling=pooling, head_channels=())
            self.assertEqual(len(list(model.task_head.children())), 1)
            self.assertEqual(model.task_head.output.in_features, width)
        self.assertEqual(task(head_channels=[7]).head_channels, (7,))

    def test_invalid_geometry_head_configuration_and_inputs(self):
        for settings in [dict(num_electrodes=0), dict(chunk_size=True), dict(chunk_size=127),
                         dict(num_patches=0), dict(freeze_encoder=1), dict(pooling='max'),
                         dict(head_channels=4), dict(head_channels=(False,)), dict(head_channels=(0,)),
                         dict(dropout=float('nan')), dict(num_classes=0)]:
            config = dict(num_electrodes=3, chunk_size=128, num_patches=4)
            config.update(settings)
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                CBraModClassifier(backbone(), **config)
        with self.assertRaises(ValueError):
            CBraModClassifier(nn.Identity(), 3, 128)
        with self.assertRaises(ValueError):
            task(CBraModEmbedding, flatten=1)
        model = task()
        for x in [None, torch.ones(2, 3, 128, dtype=torch.long), torch.randn(2, 4, 128),
                  torch.randn(2, 3, 127), torch.empty(0, 3, 128), torch.randn(2, 3, 4, 32)]:
            with self.assertRaises(ValueError):
                model(x)
        with self.assertRaises(ValueError):
            model.feature(torch.randn(2, 3, 128), flatten=1)


if __name__ == '__main__':
    unittest.main()
