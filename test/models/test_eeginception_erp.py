import copy
import io
import unittest

import torch
from torch import nn

from torcheeg.models import EEGInceptionERP


class TestEEGInceptionERP(unittest.TestCase):
    def test_complete_structure_public_inputs_and_strict_state(self):
        model = EEGInceptionERP()
        self.assertEqual(len(model.inception_block_1.branches), 3)
        self.assertEqual(len(model.inception_block_2.branches), 3)
        self.assertEqual(
            [b[0].kernel_size for b in model.inception_block_1.branches],
            [(1, 64), (1, 32), (1, 16)],
        )
        for branch in model.inception_block_1.branches:
            self.assertEqual(branch[4].groups, 8)
            self.assertEqual(branch[4].kernel_size, (8, 1))
            self.assertEqual(branch[4].out_channels, 16)
        model.eval()
        x = torch.randn(3, 8, 128)
        logits = model(x)
        self.assertEqual(logits.shape, (3, 2))
        torch.testing.assert_close(logits, model(x.unsqueeze(1)), rtol=0, atol=0)
        restored = EEGInceptionERP()
        restored.load_state_dict(model.state_dict(), strict=True)
        restored.eval()
        torch.testing.assert_close(restored(x), logits, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(logits).all())
        self.assertFalse(torch.allclose(logits.sum(-1), torch.ones(3)))

    def test_custom_kernels_pooling_activation_and_all_gradients(self):
        model = EEGInceptionERP(
            num_electrodes=3,
            chunk_size=137,
            num_classes=4,
            sampling_rate=32,
            num_filters=5,
            scales=(7 / 32, 5 / 32, 9 / 32),
            pool_sizes=(3, 2, 2, 2),
            activation=nn.ReLU,
        )
        x = torch.randn(3, 3, 137, requires_grad=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        loss = nn.functional.cross_entropy(model(x), torch.tensor([0, 2, 3]))
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
        optimizer.step()
        self.assertTrue(
            any(not torch.equal(p, before[n]) for n, p in model.named_parameters())
        )

    def test_initialization_without_probe_forward(self):
        model = EEGInceptionERP()
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.assertEqual(module.eps, 1e-5)
                self.assertEqual(module.momentum, 0.01)
                self.assertEqual(module.num_batches_tracked.item(), 0)
                self.assertTrue(
                    torch.equal(
                        module.running_mean, torch.zeros_like(module.running_mean)
                    )
                )
                self.assertTrue(
                    torch.equal(module.running_var, torch.ones_like(module.running_var))
                )
                self.assertTrue(
                    torch.equal(module.weight, torch.ones_like(module.weight))
                )
            if getattr(module, 'bias', None) is not None:
                self.assertTrue(torch.equal(module.bias, torch.zeros_like(module.bias)))

    def test_safe_checkpoint_restores_next_training_update(self):
        model = EEGInceptionERP()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        x, y = torch.randn(4, 8, 128), torch.tensor([0, 1, 0, 1])

        def step(network, optim):
            optim.zero_grad(set_to_none=True)
            output = network(x)
            loss = nn.functional.cross_entropy(output, y)
            loss.backward()
            optim.step()
            return output.detach(), loss.detach()

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
        restored = EEGInceptionERP()
        restored.load_state_dict(payload['model'], strict=True)
        resumed = torch.optim.Adam(restored.parameters(), lr=0.001)
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

    def test_invalid_configuration_and_geometry(self):
        for kwargs in [
            {'num_electrodes': True},
            {'chunk_size': 31},
            {'num_classes': 0},
            {'num_filters': 1},
            {'depth_multiplier': 1.5},
            {'sampling_rate': float('inf')},
            {'sampling_rate': False},
            {'dropout': float('nan')},
            {'batch_norm_momentum': -0.1},
            {'scales': (0.5, 0.25)},
            {'scales': (0.5, 0.25, 0.001)},
            {'scales': (float('inf'), 0.25, 0.125)},
            {'pool_sizes': (4, 2, 0, 2)},
            {'pool_sizes': (4, 2, 2)},
            {'activation': nn.ELU()},
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                EEGInceptionERP(**kwargs)
        model = EEGInceptionERP()
        for shape in [
            (3, 8, 129),
            (3, 7, 128),
            (3, 2, 8, 128),
            (8, 128),
            (3, 8, 128, 1),
        ]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                model(torch.randn(shape))

    @unittest.skipUnless(
        hasattr(torch, 'compile'), 'torch.compile requires PyTorch 2.x'
    )
    def test_fullgraph_eager_compilation(self):
        model = EEGInceptionERP().eval()
        x = torch.randn(3, 8, 128)
        compiled = torch.compile(copy.deepcopy(model), backend='eager', fullgraph=True)
        torch.testing.assert_close(compiled(x), model(x), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
