"""Complete hierarchical model, independent equations and checkpoint boundaries."""

import copy
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from torcheeg.models import SleepTransformer
from torcheeg.models.transformer.sleeptransformer import (
    _position_table,
    _SleepContextAttention,
    _SleepTransformerBlock,
)


class TestSleepTransformer(unittest.TestCase):
    def small(self, **kwargs):
        config = dict(
            num_freqs=8,
            num_frames=5,
            sequence_length=3,
            heads=2,
            frame_ffn_size=13,
            sequence_ffn_size=17,
            attention_hidden_size=7,
            hid_channels=11,
        )
        config.update(kwargs)
        return SleepTransformer(**config)

    def test_full_default_trainable_architecture_and_all_gradients(self):
        torch.manual_seed(13)
        model = SleepTransformer()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 2448517)
        self.assertEqual(len(model.state_dict()), 65)
        self.assertEqual(len(model.frame_encoder.blocks), 2)
        self.assertEqual(len(model.sequence_encoder.blocks), 2)
        logits = model(torch.randn(2, 20, 29, 128, 1))
        model.compute_loss(logits, torch.randint(5, (2, 20))).backward()
        self.assertEqual(tuple(logits.shape), (2, 20, 5))
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_native_batch_sum_epoch_mean_and_all_parameter_l2(self):
        model = self.small(dropout=0, attention_dropout=0)
        x = torch.randn(2, 3, 5, 8, 1)
        labels = torch.tensor([[0, 1, 2], [2, 3, 4]])
        scores = model(x)
        ce = (
            torch.stack(
                [
                    F.cross_entropy(scores[:, i], labels[:, i], reduction='sum')
                    for i in range(3)
                ]
            ).sum()
            / 3
        )
        torch.testing.assert_close(model.compute_loss(scores, labels, 0), ce)
        penalty = sum(p.square().sum() for p in model.parameters()) * 0.5
        torch.testing.assert_close(
            model.compute_loss(scores, labels), ce + 1e-4 * penalty
        )
        repeated = model.compute_loss(scores.repeat(2, 1, 1), labels.repeat(2, 1), 0)
        torch.testing.assert_close(repeated, 2 * ce)

    def test_numpy_sinusoidal_positions_and_no_checkpoint_buffer(self):
        expected = np.empty((5, 8), dtype=np.float32)
        for t in range(5):
            for d in range(8):
                angle = t / 10000 ** (2 * (d // 2) / 8)
                expected[t, d] = np.sin(angle) if d % 2 == 0 else np.cos(angle)
        np.testing.assert_array_equal(_position_table(5, 8).numpy(), expected)
        self.assertFalse(any('positions' in key for key in self.small().state_dict()))

    def test_independent_attention_equations_without_output_projection(self):
        torch.manual_seed(19)
        block = _SleepTransformerBlock(8, 2, 13, 0, 0).double().eval()
        x = torch.randn(2, 4, 8, dtype=torch.float64)
        q, k, v = [
            F.linear(x, layer.weight, layer.bias)
            .reshape(2, 4, 2, 4)
            .permute(0, 2, 1, 3)
            for layer in [block.query, block.key, block.value]
        ]
        weights = torch.softmax(torch.einsum('bhqd,bhkd->bhqk', q, k) / 2, -1)
        mixed = (
            torch.einsum('bhqk,bhkd->bhqd', weights, v)
            .permute(0, 2, 1, 3)
            .reshape(2, 4, 8)
        )

        def norm(value, layer):
            return F.layer_norm(value, (8,), layer.weight, layer.bias, eps=1e-8)

        hidden = norm(mixed + x, block.attention_norm)
        expected = norm(
            F.linear(
                F.relu(F.linear(hidden, block.ffn_in.weight, block.ffn_in.bias)),
                block.ffn_out.weight,
                block.ffn_out.bias,
            )
            + hidden,
            block.ffn_norm,
        )
        actual, attention = block(x)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(attention, weights, rtol=1e-12, atol=1e-12)
        self.assertEqual(len(block.state_dict()), 14)

    def test_independent_context_pooling_and_input_finite_differences(self):
        torch.manual_seed(23)
        pool = _SleepContextAttention(8, 7).double()
        x = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
        array = x.detach().numpy()
        score = (
            np.tanh(array @ pool.weight.detach().numpy() + pool.bias.detach().numpy())
            @ pool.vector.detach().numpy()
        )
        weights = np.exp(score) / np.exp(score).sum(-1, keepdims=True)
        output, probabilities = pool(x)
        np.testing.assert_allclose(
            probabilities.detach().numpy(), weights, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            output.detach().numpy(),
            (array * weights[..., None]).sum(1),
            rtol=1e-12,
            atol=1e-12,
        )
        model = self.small(dropout=0, attention_dropout=0).double().eval()
        x = torch.randn(1, 3, 5, 8, 1, dtype=torch.float64, requires_grad=True)
        target = torch.tensor([[0, 2, 4]])
        model.compute_loss(model(x), target, 0).backward()
        for index in [(0, 0, 0, 0, 0), (0, 2, 4, 7, 0)]:
            plus = x.detach().clone()
            minus = x.detach().clone()
            plus[index] += 1e-5
            minus[index] -= 1e-5
            with torch.no_grad():
                expected = (
                    model.compute_loss(model(plus), target, 0)
                    - model.compute_loss(model(minus), target, 0)
                ) / 2e-5
            torch.testing.assert_close(x.grad[index], expected, rtol=1e-5, atol=1e-7)

    def test_attention_hierarchy_variable_context_and_batch_isolation(self):
        model = self.small(num_electrodes=2, heads=4).eval()
        x = torch.randn(3, 2, 4, 8, 2)
        scores, attention = model(x, return_attention=True)
        self.assertEqual(tuple(scores.shape), (3, 2, 5))
        self.assertEqual(tuple(attention['frame_context'].shape), (3, 2, 4))
        self.assertEqual(len(attention['frame_self_attention']), 2)
        self.assertEqual(len(attention['sequence_self_attention']), 2)
        for weights in (
            [attention['frame_context']]
            + attention['frame_self_attention']
            + attention['sequence_self_attention']
        ):
            torch.testing.assert_close(
                weights.sum(-1), torch.ones_like(weights.sum(-1))
            )
        changed = x.clone()
        changed[1] += 10
        torch.testing.assert_close(model(changed)[0], scores[0], rtol=0, atol=0)

    def test_complete_native_checkpoint_roundtrip_and_strict_boundaries(self):
        model = self.small()
        native = {
            key + ':0': value.clone()
            for key, value in model.native_state_dict().items()
        }
        clone = self.small()
        clone.load_native_state_dict(native)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(clone.state_dict()[name], value, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            clone.load_native_state_dict({})
        duplicate = dict(native)
        key = next(iter(native))
        duplicate[key[:-2]] = native[key]
        with self.assertRaises(ValueError):
            clone.load_native_state_dict(duplicate)
        corrupt = dict(native)
        corrupt[key] = torch.zeros(1)
        with self.assertRaises(RuntimeError):
            clone.load_native_state_dict(corrupt)
        with self.assertRaises(RuntimeError):
            clone.load_state_dict({})

    def test_dropout_eval_and_optimizer_rng_next_step_restore(self):
        model = self.small().train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.randn(2, 3, 5, 8, 1)
        labels = torch.randint(5, (2, 3))

        def step(net, opt):
            opt.zero_grad()
            result = net(x)
            net.compute_loss(result, labels).backward()
            opt.step()
            return result.detach()

        step(model, optimizer)
        clone = self.small()
        clone.load_state_dict(copy.deepcopy(model.state_dict()))
        restored = torch.optim.Adam(clone.parameters(), lr=1e-3)
        restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        rng = torch.get_rng_state().clone()
        expected = step(model, optimizer)
        torch.set_rng_state(rng)
        actual = step(clone, restored)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(clone.state_dict()[name], value, rtol=0, atol=0)
        model.eval()
        torch.testing.assert_close(model(x), model(x), rtol=0, atol=0)

    def test_configuration_input_and_loss_boundaries(self):
        for config in [
            dict(heads=3),
            dict(frame_depth=0),
            dict(sequence_length=True),
            dict(dropout=1),
            dict(attention_dropout=float('nan')),
        ]:
            with self.assertRaises(ValueError):
                self.small(**config)
        model = self.small()
        with self.assertRaises(TypeError):
            model(torch.ones(2, 3, 5, 8, 1, dtype=torch.int64))
        for shape in [
            (2, 4, 5, 8, 1),
            (2, 3, 6, 8, 1),
            (2, 3, 5, 8, 2),
            (0, 3, 5, 8, 1),
        ]:
            with self.assertRaises(ValueError):
                model(torch.randn(*shape))
        with self.assertRaises(TypeError):
            model(torch.randn(2, 3, 5, 8, 1), return_attention=1)
        with self.assertRaises(ValueError):
            model.compute_loss(
                torch.randn(2, 3, 5), torch.zeros(2, 2, dtype=torch.long)
            )
        with self.assertRaises(ValueError):
            model.compute_loss(
                torch.randn(2, 3, 5), torch.zeros(2, 3, dtype=torch.long), -1
            )


if __name__ == '__main__':
    unittest.main()
