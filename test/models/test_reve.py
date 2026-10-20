import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from torcheeg.models import REVE, REVEClassifier


class TestREVE(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.config = dict(hid_channels=16, depth=2, num_heads=2, head_channels=8,
                           num_frequencies=2, patch_size=200, patch_overlap=20)
        self.eeg = torch.randn(2, 3, 420)
        self.positions = torch.randn(2, 3, 3) * .05

    def test_official_reference_outputs_and_all_gradients(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures/reve_reference.json').read_text())
        model = REVE(**self.config).eval()
        expected = model.state_dict()
        model.load_state_dict({name: torch.tensor(value, dtype=expected[name].dtype)
                               for name, value in fixture['state'].items()}, strict=True)
        x = torch.tensor(fixture['eeg'], requires_grad=True)
        positions = torch.tensor(fixture['positions'], requires_grad=True)
        features = model(x, positions)
        pooled = model.attention_pooling(features)
        torch.testing.assert_close(features, torch.tensor(fixture['output']), rtol=1e-4, atol=3e-6)
        torch.testing.assert_close(pooled, torch.tensor(fixture['pooled']), rtol=1e-4, atol=3e-6)
        pooled.square().mean().backward()
        torch.testing.assert_close(x.grad, torch.tensor(fixture['eeg_gradient']), rtol=3e-4, atol=2e-6)
        torch.testing.assert_close(positions.grad, torch.tensor(fixture['coordinate_gradient']), rtol=3e-4, atol=2e-5)
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                torch.testing.assert_close(parameter.grad, torch.tensor(fixture['parameter_gradients'][name]),
                                           rtol=3e-4, atol=3e-6)

    def test_heterogeneous_batches_match_individuals(self):
        model = REVE(**self.config).eval()
        x = self.eeg.clone().requires_grad_()
        p = self.positions.clone().requires_grad_()
        original = p.detach().clone()
        batched = model(x, p)
        individual = torch.cat([model(x[i:i+1], p[i:i+1]) for i in range(2)])
        torch.testing.assert_close(batched, individual)
        batched[0].square().sum().backward()
        torch.testing.assert_close(x.grad[1], torch.zeros_like(x.grad[1]))
        torch.testing.assert_close(p.grad[1], torch.zeros_like(p.grad[1]))
        torch.testing.assert_close(p, original)

    def test_shared_positions_layers_and_tail(self):
        model = REVE(**self.config).eval()
        p = self.positions[0]
        first = model(self.eeg, p)
        torch.testing.assert_close(first, model(self.eeg, p.expand(2, -1, -1)))
        self.assertEqual(first.shape, (2, 3, 2, 16))
        layers = model.forward_features(self.eeg, p, return_layers=True)
        self.assertEqual(len(layers), 3)
        torch.testing.assert_close(layers[-1], first.flatten(1, 2))
        changed = self.eeg.clone()
        changed[:, :, 380:] += 100
        torch.testing.assert_close(model(changed, p), first)

    def test_channel_permutation_equivariance(self):
        model = REVE(**self.config).eval()
        order = [2, 0, 1]
        features = model(self.eeg, self.positions)
        permuted = model(self.eeg[:, order], self.positions[:, order])
        torch.testing.assert_close(permuted, features[:, order], rtol=2e-5, atol=3e-6)
        torch.testing.assert_close(model.attention_pooling(permuted), model.attention_pooling(features))

    def test_strict_safe_loader_and_corruption(self):
        model = REVE(**self.config)
        state = model.state_dict()
        with tempfile.TemporaryDirectory() as temp, patch.object(REVE, 'variants', {'tiny': self.config}):
            path = Path(temp) / 'model.safetensors'
            save_file(state, str(path))
            loaded = REVE.from_pretrained(path, 'tiny')
            for name, value in state.items():
                torch.testing.assert_close(loaded.state_dict()[name], value)
            save_file({'encoder.'+name: value for name, value in state.items()}, str(path))
            REVE.from_pretrained(path, 'tiny', prefix='encoder.')
            name = 'cls_query_token'
            for replacement in [None, state[name][..., :-1], state[name].double(),
                                torch.full_like(state[name], float('nan'))]:
                bad = dict(state)
                if replacement is None:
                    del bad[name]
                else:
                    bad[name] = replacement
                save_file(bad, str(path))
                with self.assertRaises(ValueError):
                    REVE.from_pretrained(path, 'tiny')
            save_file(dict(state, unexpected=torch.zeros(1)), str(path))
            with self.assertRaises(ValueError):
                REVE.from_pretrained(path, 'tiny')
            for variant, prefix in [('unknown', ''), (False, ''), ('tiny', 1)]:
                with self.assertRaises(ValueError):
                    REVE.from_pretrained(path, variant, prefix)

    def test_classifier_freezing_optimizer_restore_and_override(self):
        model = REVEClassifier(REVE(**self.config), num_classes=3, dropout=0,
                               freeze_backbone=True, positions=self.positions[0])
        model.train()
        self.assertFalse(model.backbone.training)
        original = copy.deepcopy(model.state_dict())
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        logits = model(self.eeg)
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 2])).backward()
        optimizer.step()
        self.assertFalse(torch.equal(original['classifier.2.weight'], model.classifier[2].weight))
        for name, value in model.backbone.state_dict().items():
            torch.testing.assert_close(value, original['backbone.'+name])
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        restored = REVEClassifier(REVE(**self.config), 3, dropout=0, freeze_backbone=True,
                                  positions=self.positions[0])
        restored.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(model(self.eeg), restored(self.eeg))
        expected = model.classifier(model.backbone.attention_pooling(model.backbone(self.eeg, self.positions)))
        torch.testing.assert_close(model(self.eeg, self.positions), expected)

    def test_classifier_finetunes_pretrained_query(self):
        model = REVEClassifier(REVE(**self.config), dropout=0)
        model(self.eeg, self.positions).square().mean().backward()
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(model.backbone.cls_query_token.grad.abs().sum().item(), 0)

    def test_invalid_configuration(self):
        for kwargs in [dict(depth=0), dict(num_heads=True), dict(hid_channels=15), dict(num_frequencies=1),
                       dict(mlp_ratio=float('nan')), dict(patch_overlap=200), dict(patch_overlap=-1),
                       dict(patch_size=0), dict(use_geglu=1)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                REVE(**dict(self.config, **kwargs))
        self.assertEqual(REVE.variants['base']['depth'], 22)
        self.assertEqual(REVE.variants['large']['hid_channels'], 1216)

    def test_invalid_inputs(self):
        model = REVE(**self.config)
        for x, p in [(self.eeg.long(), self.positions), (self.eeg[:, :, :199], self.positions),
                     (self.eeg[:0], self.positions[:0]), (self.eeg, None),
                     (self.eeg, self.positions[:, :, :2]), (self.eeg, self.positions.long())]:
            with self.assertRaises(ValueError):
                model(x, p)
        with self.assertRaises(ValueError):
            model.forward_features(self.eeg, self.positions, return_layers=1)
        for values in [torch.empty(0, 3, 2, 16), torch.randn(2, 3, 16)]:
            with self.assertRaises(ValueError):
                model.attention_pooling(values)
        for kwargs in [dict(num_classes=0), dict(dropout=1), dict(freeze_backbone=1),
                       dict(positions=torch.ones(2, 2)), dict(positions=torch.full((3, 3), float('nan')))]:
            with self.assertRaises(ValueError):
                REVEClassifier(model, **kwargs)

    def test_gelu_alternative(self):
        model = REVE(**dict(self.config, use_geglu=False))
        model(self.eeg, self.positions).square().mean().backward()
        self.assertTrue(all(p.grad is not None for name, p in model.named_parameters() if name != 'cls_query_token'))

    def test_attention_fallback_without_sdpa(self):
        model = REVE(**self.config).eval()
        reference = copy.deepcopy(model)
        expected = reference(self.eeg, self.positions)
        expected.square().mean().backward()
        with patch.object(torch.nn.functional, 'scaled_dot_product_attention', create=True):
            del torch.nn.functional.scaled_dot_product_attention
            actual = model(self.eeg, self.positions)
            actual.square().mean().backward()
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-6)
        for name, parameter in model.named_parameters():
            if name != 'cls_query_token':
                torch.testing.assert_close(parameter.grad, dict(reference.named_parameters())[name].grad,
                                           rtol=2e-4, atol=3e-6)


if __name__ == '__main__':
    unittest.main()
