import io
import json
import unittest
from pathlib import Path

import torch
from torch import nn

from torcheeg.models import CTNet


class TestCTNet(unittest.TestCase):
    def make_model(self, **overrides):
        parameters = dict(num_electrodes=3, chunk_size=128, f1=4, d=2,
                          depth=2, heads=2, pool_size1=4, pool_size2=4,
                          dropout=0., transformer_dropout=0., forward_dropout=0.,
                          pos_dropout=0., cls_dropout=0.)
        parameters.update(overrides)
        return CTNet(**parameters)

    def test_default_and_custom_shapes_and_feature_dimensions(self):
        cases = [dict(), dict(num_electrodes=22, chunk_size=1000),
                 dict(chunk_size=129, pool_size1=3, pool_size2=5),
                 dict(f1=6, d=3, heads=3, temporal_kernel=63, spatial_kernel=15)]
        for parameters in cases:
            with self.subTest(parameters=parameters):
                model = self.make_model(**parameters).eval()
                eeg = torch.randn(2, model.num_electrodes, model.chunk_size)
                features = model.feature(eeg)
                self.assertEqual(features.shape, (2, model.feature_dim()))
                self.assertEqual(model(eeg).shape, (2, 4))
                self.assertEqual(model.feature_dim(),
                                 model.f1 * model.d * (model.chunk_size // model.pool_size1 // model.pool_size2))
        default = CTNet().eval()
        self.assertEqual(default(torch.randn(1, 22, 1000)).shape, (1, 4))

    def test_initialization_does_not_update_batchnorm_running_statistics(self):
        for module in self.make_model().modules():
            if isinstance(module, nn.BatchNorm2d):
                self.assertEqual(module.num_batches_tracked.item(), 0)
                torch.testing.assert_close(module.running_mean, torch.zeros_like(module.running_mean))
                torch.testing.assert_close(module.running_var, torch.ones_like(module.running_var))

    def test_optimizer_step_reaches_cnn_transformer_position_and_classifier(self):
        torch.manual_seed(42)
        model = self.make_model()
        eeg = torch.randn(4, 3, 128, requires_grad=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        before = model.classification_head[1].weight.detach().clone()
        logits = model(eeg)
        loss = nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 3]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(eeg.grad).all())
        for name, parameter in model.named_parameters():
            with self.subTest(name=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
        for module in (model.patch_embedding, model.transformer_encoder,
                       model.positional_encoding, model.classification_head):
            self.assertGreater(sum(parameter.grad.abs().sum().item() for parameter in module.parameters()), 0.)
        optimizer.step()
        self.assertFalse(torch.equal(before, model.classification_head[1].weight))

    def test_eval_is_deterministic_and_batch_members_are_independent(self):
        model = self.make_model(dropout=0.5, transformer_dropout=0.5,
                                pos_dropout=0.5, cls_dropout=0.5).eval()
        eeg = torch.randn(3, 3, 128)
        with torch.no_grad():
            expected = model(eeg)
            torch.testing.assert_close(model(eeg), expected)
            torch.testing.assert_close(model(eeg[:1]), expected[:1], rtol=1e-5, atol=1e-6)

    def test_state_dict_roundtrip_preserves_outputs(self):
        model = self.make_model().eval()
        eeg = torch.randn(2, 3, 128)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = self.make_model().eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
        torch.testing.assert_close(restored(eeg), model(eeg))

    def test_input_dimensions_must_match_constructor(self):
        model = self.make_model()
        for shape in ((3, 128), (2, 1, 3, 128), (2, 4, 128), (2, 3, 127)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, 'CTNet expects'):
                model(torch.randn(shape))

    def test_invalid_constructor_parameters_raise_clear_errors(self):
        cases = [{'chunk_size': 0}, {'depth': -1}, {'f1': True}, {'heads': 3},
                 {'pool_size1': 129}, {'pool_size2': 0}, {'max_length': 1},
                 {'dropout': -0.1}, {'pos_dropout': float('nan')}, {'num_classes': 0},
                 {'forward_expansion': 1.5}, {'cls_dropout': 1.1}]
        for parameters in cases:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                self.make_model(**parameters)

    def test_positional_capacity_can_be_selected_explicitly(self):
        model = self.make_model(max_length=8)
        self.assertEqual(tuple(model.positional_encoding.encoding.shape), (1, 8, 8))
        self.assertEqual(tuple(model(torch.randn(2, 3, 128)).shape), (2, 4))

    def test_torch_compile_can_capture_the_full_forward_graph(self):
        if not hasattr(torch, 'compile'):
            self.skipTest('torch.compile is unavailable on this runtime')
        model = self.make_model(depth=1).eval()
        eeg = torch.randn(2, 3, 128)
        compiled = torch.compile(model, backend='eager', fullgraph=True)
        torch.testing.assert_close(compiled(eeg), model(eeg))

    def test_matches_original_author_reference_fixture(self):
        fixture = json.loads(Path(__file__).with_name('fixtures').joinpath('ctnet_reference.json').read_text())
        model = CTNet(**fixture['parameters']).eval()
        with torch.no_grad():
            for index, parameter in enumerate(model.parameters()):
                values = torch.arange(parameter.numel(), dtype=parameter.dtype).reshape(parameter.shape)
                parameter.copy_(torch.sin(values * 0.03 + index + 1) * 0.1)
            eeg = torch.sin(torch.arange(2 * 22 * 128, dtype=torch.float32) * 0.02).reshape(2, 22, 128)
            features, logits = model.feature(eeg), model(eeg)
        torch.testing.assert_close(features, torch.tensor(fixture['features']), rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(logits, torch.tensor(fixture['logits']), rtol=1e-4, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
