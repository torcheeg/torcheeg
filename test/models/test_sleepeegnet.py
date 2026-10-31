"""Full decoder behavior, recurrent equations, gradients and strict restoration."""

import copy
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from torcheeg.models import SleepEEGNet
from torcheeg.models.rnn.sleepeegnet import _SleepSequenceCell


class TestSleepEEGNet(unittest.TestCase):
    def small(self, **kwargs):
        config = dict(
            chunk_size=600,
            sequence_length=3,
            hid_channels=8,
            embedding_size=4,
            attention_size=7,
        )
        config.update(kwargs)
        return SleepEEGNet(**config)

    def test_both_complete_default_geometries_and_every_gradient(self):
        for bidirectional, count, states in [(False, 2642438, 28), (True, 5107206, 32)]:
            with self.subTest(bidirectional=bidirectional):
                model = SleepEEGNet(bidirectional=bidirectional)
                self.assertEqual(sum(v.numel() for v in model.parameters()), count)
                self.assertEqual(len(model.state_dict()), states)
                self.assertEqual(model.feature_size, 3072)
                x = torch.randn(2, 10, 1, 3000)
                decoder, targets = model.prepare_targets(
                    torch.arange(20).reshape(2, 10) % 5
                )
                logits = model(x, decoder)
                self.assertEqual(tuple(logits.shape), (2, 11, 7))
                model.compute_loss(logits, targets).backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                if bidirectional:
                    self.assertEqual(model.encoder_backward[1].kernel.shape, (256, 512))
                    self.assertEqual(model.decoder[0].kernel.shape, (330, 1024))

    def test_independent_numpy_single_kernel_gate_equations_and_forget_bias(self):
        torch.manual_seed(13)
        model = _SleepSequenceCell(3, 2).double()
        x = torch.randn(2, 3, dtype=torch.float64)
        cell = torch.randn(2, 2, dtype=torch.float64)
        hidden = torch.randn(2, 2, dtype=torch.float64)
        incoming, candidate, forget, output = np.split(
            np.concatenate((x.numpy(), hidden.numpy()), -1)
            @ model.kernel.detach().numpy()
            + model.bias.detach().numpy(),
            4,
            -1,
        )

        def sigmoid(value):
            return 1 / (1 + np.exp(-value))

        expected_cell = cell.numpy() * sigmoid(forget + 1) + sigmoid(
            incoming
        ) * np.tanh(candidate)
        expected_hidden = np.tanh(expected_cell) * sigmoid(output)
        actual, state = model(x, (cell, hidden))
        np.testing.assert_allclose(
            actual.detach().numpy(), expected_hidden, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            state[0].detach().numpy(), expected_cell, rtol=1e-12, atol=1e-12
        )

    def test_independent_per_sequence_class_average_ce_and_non_bias_l2(self):
        model = self.small()
        logits = torch.randn(2, 4, 7, requires_grad=True)
        targets = torch.tensor([[0, 0, 2, 6], [4, 1, 1, 6]])
        log_probabilities = logits.log_softmax(-1)
        per_sequence = []
        for b in range(2):
            class_losses = []
            for stage in targets[b].unique():
                positions = torch.nonzero(targets[b] == stage).flatten()
                class_losses.append(-log_probabilities[b, positions, stage].mean())
            per_sequence.append(torch.stack(class_losses).sum())
        expected = torch.stack(per_sequence).mean()
        torch.testing.assert_close(model.compute_loss(logits, targets, 0), expected)
        penalty = (
            sum(
                p.square().sum()
                for name, p in model.named_parameters()
                if not name.endswith('bias')
            )
            * 0.5
        )
        torch.testing.assert_close(
            model.compute_loss(logits, targets), expected + 0.001 * penalty
        )
        ordinary = F.cross_entropy(logits.transpose(1, 2), targets)
        self.assertGreater(abs(float(expected.detach() - ordinary.detach())), 0.1)

    def test_teacher_tokens_attention_and_batch_isolation(self):
        model = self.small(bidirectional=True).eval()
        labels = torch.tensor([[0, 1, 2], [3, 4, 0]])
        decoder, targets = model.prepare_targets(labels)
        self.assertEqual(decoder.tolist(), [[5, 0, 1, 2], [5, 3, 4, 0]])
        self.assertEqual(targets.tolist(), [[0, 1, 2, 6], [3, 4, 0, 6]])
        x = torch.randn(2, 3, 1, 600)
        output, attention = model(x, decoder, return_attention=True)
        torch.testing.assert_close(attention.sum(-1), torch.ones(2, 4))
        changed = x.clone()
        changed[1] *= 10
        torch.testing.assert_close(
            model(changed, decoder)[0], output[0], rtol=0, atol=0
        )

    def test_autoregression_stops_at_eod_and_bounds_without_ground_truth(self):
        model = self.small(dropout=0).eval()
        x = torch.randn(2, 3, 1, 600)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        generated = model.generate(x)
        self.assertEqual(tuple(generated['tokens'].shape), (2, 5))
        self.assertEqual(generated['lengths'].tolist(), [5, 5])
        self.assertFalse(generated['logits'].requires_grad)
        with torch.no_grad():
            for layer in model.decoder:
                layer.bias[layer.hidden_size : 2 * layer.hidden_size].fill_(1)
            model.attention_projection.weight.fill_(0.1)
            model.output.weight[model.end_token].fill_(1)
        generated = model.generate(x)
        self.assertEqual(generated['tokens'].tolist(), [[6], [6]])
        self.assertEqual(generated['lengths'].tolist(), [1, 1])
        with self.assertRaises(ValueError):
            model.generate(x, max_steps=0)

    def test_independent_input_finite_differences(self):
        torch.manual_seed(29)
        model = self.small(dropout=0).double().eval()
        x = torch.randn(1, 3, 1, 600, dtype=torch.float64, requires_grad=True)
        decoder, targets = model.prepare_targets(torch.tensor([[0, 2, 4]]))
        model.compute_loss(model(x, decoder), targets, 0).backward()
        for index in [(0, 0, 0, 20), (0, 2, 0, 399)]:
            plus = x.detach().clone()
            minus = x.detach().clone()
            plus[index] += 1e-5
            minus[index] -= 1e-5
            with torch.no_grad():
                expected = (
                    model.compute_loss(model(plus, decoder), targets, 0)
                    - model.compute_loss(model(minus, decoder), targets, 0)
                ) / 2e-5
            torch.testing.assert_close(x.grad[index], expected, rtol=1e-4, atol=1e-7)

    def test_all_native_states_and_shapes_round_trip_strictly(self):
        for bidirectional in [False, True]:
            model = self.small(bidirectional=bidirectional)
            native = {
                key + ':0': value.clone()
                for key, value in model.native_state_dict().items()
            }
            clone = self.small(bidirectional=bidirectional)
            clone.load_native_state_dict(native)
            for name, tensor in model.state_dict().items():
                torch.testing.assert_close(
                    tensor, clone.state_dict()[name], rtol=0, atol=0
                )
            with self.assertRaises(ValueError):
                clone.load_native_state_dict({})
            key = next(iter(native))
            corrupted = dict(native)
            corrupted[key] = torch.zeros(1)
            with self.assertRaises(RuntimeError):
                clone.load_native_state_dict(corrupted)
            duplicate = dict(native)
            duplicate[key[:-2]] = native[key]
            with self.assertRaises(ValueError):
                clone.load_native_state_dict(duplicate)

    def test_dropout_and_optimizer_rng_next_training_step_restoration(self):
        model = self.small()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.randn(2, 3, 1, 600)
        decoder, targets = model.prepare_targets(torch.tensor([[0, 1, 2], [3, 4, 0]]))

        def step(net, opt):
            opt.zero_grad()
            scores = net(x, decoder)
            net.compute_loss(scores, targets).backward()
            opt.step()
            return scores.detach()

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
        torch.testing.assert_close(model(x, decoder), model(x, decoder), rtol=0, atol=0)

    def test_explicit_configuration_and_token_boundaries(self):
        for config in [
            dict(chunk_size=0),
            dict(num_layers=True),
            dict(hid_channels=-1),
            dict(dropout=float('inf')),
        ]:
            with self.assertRaises(ValueError):
                self.small(**config)
        model = self.small()
        x = torch.randn(2, 3, 1, 600)
        for shape in [(2, 4, 1, 600), (2, 3, 2, 600), (0, 3, 1, 600)]:
            with self.assertRaises(ValueError):
                model.encode(torch.randn(*shape))
        with self.assertRaises(TypeError):
            model.encode(torch.ones(2, 3, 1, 600, dtype=torch.long))
        with self.assertRaises(TypeError):
            model(x, torch.ones(2, 4))
        with self.assertRaises(ValueError):
            model(x, torch.full((2, 4), 7))
        with self.assertRaises(ValueError):
            model.prepare_targets(torch.tensor([[5, 0, 1]]))
        with self.assertRaises(ValueError):
            model.compute_loss(
                torch.randn(2, 4, 7), torch.zeros(2, 3, dtype=torch.long)
            )
        with self.assertRaises(ValueError):
            model.compute_loss(
                torch.randn(2, 4, 7), torch.zeros(2, 4, dtype=torch.long), -1
            )


if __name__ == '__main__':
    unittest.main()
