import copy
import io
import unittest

import torch
from torch import nn

from torcheeg.models import EEGNeX


class TestEEGNeX(unittest.TestCase):
    def test_complete_structure_and_public_inputs(self):
        model = EEGNeX().eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 56452)
        self.assertEqual(len(model.state_dict()), 32)
        self.assertIn('block_3.0.parametrizations.weight.original', model.state_dict())
        self.assertIn(
            'final_layer.parametrizations.weight.original', model.state_dict()
        )
        self.assertEqual(model.block_3[0].groups, 32)
        self.assertEqual(model.block_3[0].out_channels, 64)
        self.assertEqual(model.block_4[0].dilation, (1, 2))
        self.assertEqual(model.block_5[0].dilation, (1, 4))
        self.assertEqual(model.final_layer.in_features, 248)
        x = torch.randn(3, 22, 1000)
        logits = model(x)
        self.assertEqual(logits.shape, (3, 4))
        torch.testing.assert_close(model(x.unsqueeze(1)), logits, rtol=0, atol=0)
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.assertEqual(module.num_batches_tracked.item(), 0)
                self.assertEqual(module.momentum, 0.1)
                self.assertEqual(module.eps, 1e-5)

    def test_active_constraints_are_differentiable_and_do_not_mutate_originals(self):
        model = EEGNeX(
            num_electrodes=3,
            chunk_size=137,
            num_classes=3,
            num_filters=(3, 5),
            depth_multiplier=3,
            temporal_kernels=(7, 6, 4),
            dilations=(3, 2),
            pool_sizes=(3, 5),
            max_norm_conv=0.15,
            max_norm_linear=0.1,
        )
        constrained = [model.block_3[0], model.final_layer]
        with torch.no_grad():
            for module in constrained:
                module.parametrizations.weight.original.mul_(20)
        originals = [
            m.parametrizations.weight.original.detach().clone() for m in constrained
        ]
        x = torch.randn(4, 3, 137, requires_grad=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        loss = nn.functional.cross_entropy(model(x), torch.tensor([0, 1, 2, 0]))
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        for name, value in model.named_parameters():
            self.assertIsNotNone(value.grad, name)
            self.assertTrue(torch.isfinite(value.grad).all(), name)
        for module, original, limit in zip(constrained, originals, [0.15, 0.1]):
            torch.testing.assert_close(
                module.parametrizations.weight.original, original, rtol=0, atol=0
            )
            self.assertGreater(
                module.parametrizations.weight.original.grad.abs().sum().item(), 0
            )
            self.assertLessEqual(
                module.weight.flatten(1).norm(dim=1).max().item(), limit + 1e-6
            )
        optimizer.step()
        for module, original in zip(constrained, originals):
            self.assertFalse(
                torch.equal(module.parametrizations.weight.original, original)
            )

    def test_complete_checkpoint_restores_next_training_step(self):
        model = EEGNeX(num_electrodes=3, chunk_size=137, num_classes=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        x, y = torch.randn(4, 3, 137), torch.tensor([0, 1, 0, 1])

        def step(network, optim):
            optim.zero_grad(set_to_none=True)
            logits = network(x)
            loss = nn.functional.cross_entropy(logits, y)
            loss.backward()
            optim.step()
            return logits.detach(), loss.detach()

        step(model, optimizer)
        stream = io.BytesIO()
        torch.save(
            {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng': torch.get_rng_state(),
            },
            stream,
        )
        expected = step(model, optimizer)
        stream.seek(0)
        payload = torch.load(stream, weights_only=True)
        restored = EEGNeX(num_electrodes=3, chunk_size=137, num_classes=2)
        restored.load_state_dict(payload['model'], strict=True)
        resumed = torch.optim.AdamW(restored.parameters(), lr=0.001)
        resumed.load_state_dict(payload['optimizer'])
        torch.set_rng_state(payload['rng'])
        actual = step(restored, resumed)
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(
                value, restored.state_dict()[name], rtol=0, atol=0
            )
        for key, state in optimizer.state_dict()['state'].items():
            for name, value in state.items():
                torch.testing.assert_close(
                    value, resumed.state_dict()['state'][key][name], rtol=0, atol=0
                )

    def test_legacy_padding_logits_and_gradients(self):
        model = EEGNeX(
            num_electrodes=3,
            chunk_size=137,
            temporal_kernels=(8, 4, 6),
            dilations=(3, 1),
        ).eval()
        legacy = copy.deepcopy(model)
        for module in legacy.modules():
            if hasattr(module, '_native_same'):
                module._native_same = False
        left = torch.randn(3, 3, 137, requires_grad=True)
        right = left.detach().clone().requires_grad_()
        actual, expected = legacy(left), model(right)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-6)
        for name, value in legacy.named_parameters():
            torch.testing.assert_close(
                value.grad,
                dict(model.named_parameters())[name].grad,
                rtol=1e-5,
                atol=1e-6,
                msg=name,
            )

    def test_invalid_settings_and_geometry(self):
        for kwargs in [
            {'num_electrodes': True},
            {'chunk_size': 1},
            {'num_classes': 0},
            {'num_filters': (8,)},
            {'num_filters': (8, 0)},
            {'depth_multiplier': 1.5},
            {'temporal_kernels': (4, 2)},
            {'dilations': (2, False)},
            {'pool_sizes': (1, 8)},
            {'dropout': float('nan')},
            {'max_norm_conv': 0},
            {'max_norm_linear': float('inf')},
            {'activation': nn.ELU()},
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                EEGNeX(**kwargs)
        model = EEGNeX()
        for shape in [
            (3, 22, 999),
            (3, 21, 1000),
            (3, 2, 22, 1000),
            (22, 1000),
            (3, 22, 1000, 1),
        ]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                model(torch.randn(shape))

    @unittest.skipUnless(
        hasattr(torch, 'compile'), 'torch.compile requires PyTorch 2.x'
    )
    def test_fullgraph_eager_compilation(self):
        model = EEGNeX(num_electrodes=3, chunk_size=137).eval()
        x = torch.randn(3, 3, 137)
        compiled = torch.compile(copy.deepcopy(model), backend='eager', fullgraph=True)
        torch.testing.assert_close(compiled(x), model(x), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
