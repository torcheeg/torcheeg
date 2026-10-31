"""MSCFormer numerical contracts, complete training state and old-path checks."""

import copy
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from torcheeg.models import MSCFormer
from torcheeg.models.transformer.mscformer import _MSCAttention


class TestMSCFormer(unittest.TestCase):
    def make_model(self, **overrides):
        config = dict(
            num_electrodes=3,
            chunk_size=113,
            num_classes=4,
            num_filters=4,
            temporal_kernels=(15, 32, 65),
            pool_size=13,
            heads=3,
            depth=2,
        )
        config.update(overrides)
        return MSCFormer(**config)

    def test_complete_default_geometry_states_and_features(self):
        model = MSCFormer().eval()
        self.assertEqual(len(model.cnn.branches), 3)
        self.assertEqual(len(model.trans.layers), 5)
        self.assertEqual(model.position.encoding.shape, (1, 100, 48))
        self.assertEqual(model.final_layer[1].in_features, 48)
        self.assertEqual(model.cnn.branches[0][1].groups, 16)
        x = torch.randn(2, 22, 1000)
        features, logits = model(x, return_features=True)
        self.assertEqual(features.shape, (2, 48))
        torch.testing.assert_close(model(x.unsqueeze(1)), logits, rtol=0, atol=0)
        for branch in model.cnn.branches:
            self.assertEqual(branch[2].num_batches_tracked.item(), 0)
        changed = x.clone()
        changed[1] += 3
        torch.testing.assert_close(model(changed)[0], logits[0], rtol=0, atol=0)

    def test_attention_against_independent_numpy_score_scaling(self):
        module = _MSCAttention(12, 3, 0.0, 12**-0.5).eval()
        x = torch.randn(2, 5, 12, requires_grad=True)

        def linear(layer, value):
            return value @ layer.weight.detach().numpy().T + layer.bias.detach().numpy()

        values = []
        for layer in (module.queries, module.keys, module.values):
            values.append(
                linear(layer, x.detach().numpy())
                .reshape(2, 5, 3, 4)
                .transpose(0, 2, 1, 3)
            )
        q, k, v = values
        scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(12)
        weights = np.exp(scores - scores.max(-1, keepdims=True))
        weights /= weights.sum(-1, keepdims=True)
        expected = linear(
            module.projection, (weights @ v).transpose(0, 2, 1, 3).reshape(2, 5, 12)
        )
        actual = module(x)
        np.testing.assert_allclose(
            actual.detach().numpy(), expected, rtol=1e-5, atol=1e-6
        )
        actual.square().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_old_padding_and_attention_paths_preserve_outputs_and_all_gradients(self):
        model = self.make_model(
            dropout=0, attn_dropout=0, pos_dropout=0, final_dropout=0
        ).eval()
        legacy = copy.deepcopy(model)
        legacy.cnn._native_same = False
        for layer in legacy.trans.layers:
            layer.attention.module._native_attention = False
        left = torch.randn(2, 3, 113, requires_grad=True)
        right = left.detach().clone().requires_grad_()
        actual, expected = model(left), legacy(right)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        actual.square().mean().backward()
        expected.square().mean().backward()
        torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-6)
        for key, parameter in model.named_parameters():
            torch.testing.assert_close(
                parameter.grad,
                dict(legacy.named_parameters())[key].grad,
                rtol=1e-5,
                atol=1e-6,
            )
        legacy.load_state_dict(model.state_dict(), strict=True)

    def test_complete_checkpoint_optimizer_rng_next_step_resume(self):
        torch.manual_seed(42)
        model = self.make_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        x, target = torch.randn(3, 3, 113), torch.tensor([0, 1, 3])

        def step(net, optim):
            optim.zero_grad()
            loss = F.cross_entropy(net(x), target)
            loss.backward()
            self.assertTrue(
                all(
                    p.grad is not None and torch.isfinite(p.grad).all()
                    for p in net.parameters()
                )
            )
            optim.step()
            return loss.detach()

        step(model, optimizer)
        state, opt_state = (
            copy.deepcopy(model.state_dict()),
            copy.deepcopy(optimizer.state_dict()),
        )
        rng = torch.get_rng_state().clone()
        expected = step(model, optimizer)
        restored = self.make_model()
        restored.load_state_dict(state, strict=True)
        other = torch.optim.AdamW(restored.parameters(), lr=0.001)
        other.load_state_dict(opt_state)
        torch.set_rng_state(rng)
        torch.testing.assert_close(step(restored, other), expected, rtol=0, atol=0)
        for key, value in restored.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
        self.assertEqual(len(other.state), len(list(restored.parameters())))

    def test_invalid_config_input_and_long_position_table(self):
        for config in (
            dict(num_electrodes=0),
            dict(chunk_size=True),
            dict(num_classes=0),
            dict(temporal_kernels=(3, 5)),
            dict(temporal_kernels=(3, 5, 0)),
            dict(pool_size=114),
            dict(heads=5),
            dict(depth=0),
            dict(mlp_ratio=0),
            dict(dropout=float('nan')),
            dict(attention_scale=0),
            dict(attention_scale=float('inf')),
        ):
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    self.make_model(**config)
        model = self.make_model()
        for bad in (torch.ones(2, 3, 113, dtype=torch.long), 'bad'):
            with self.assertRaises(TypeError):
                model(bad)
        for bad in (
            torch.randn(2, 2, 113),
            torch.randn(2, 3, 112),
            torch.randn(2, 2, 3, 113),
        ):
            with self.assertRaises(ValueError):
                model(bad)
        long = self.make_model(
            num_filters=2, heads=2, depth=1, chunk_size=1000, pool_size=8
        ).eval()
        self.assertEqual(long.position.encoding.shape, (1, 126, 6))
        self.assertEqual(long(torch.randn(2, 3, 1000)).shape, (2, 4))

    @unittest.skipUnless(hasattr(torch, 'compile'), 'torch.compile requires PyTorch 2.')
    def test_eager_full_graph_compile(self):
        model = self.make_model().eval()
        x = torch.randn(2, 3, 113)
        torch.testing.assert_close(
            torch.compile(model, backend='eager', fullgraph=True)(x),
            model(x),
            rtol=0,
            atol=0,
        )


if __name__ == '__main__':
    unittest.main()
