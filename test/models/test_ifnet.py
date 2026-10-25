import copy
import io
import unittest

import torch
from torch import nn

from torcheeg.models import IFNet, IFNetV2


class TestIFNet(unittest.TestCase):
    def test_complete_defaults_initialization_and_public_layouts(self):
        for cls in [IFNet, IFNetV2]:
            model = cls().eval()
            self.assertEqual(sum(p.numel() for p in model.parameters()), 10884)
            self.assertEqual(len(model.state_dict()), 20)
            self.assertEqual(model.stem.sconv.conv.groups, 2)
            self.assertEqual(
                [m.conv.kernel_size for m in model.stem.tconv], [(63,), (31,)]
            )
            self.assertEqual(model.fc[0].in_features, 384)
            self.assertEqual(model.stem.dp.p, 0.5)
            for module in model.modules():
                if isinstance(module, nn.BatchNorm1d):
                    self.assertEqual(module.num_batches_tracked.item(), 0)
                    self.assertTrue(
                        torch.equal(
                            module.running_mean, torch.zeros_like(module.running_mean)
                        )
                    )
                    self.assertTrue(
                        torch.equal(
                            module.running_var, torch.ones_like(module.running_var)
                        )
                    )
            x = torch.randn(3, 2, 22, 750)
            actual = model(x)
            self.assertEqual(actual.shape, (3, 4))
            torch.testing.assert_close(actual, model(x.flatten(1, 2)), rtol=0, atol=0)
            self.assertTrue(torch.isfinite(actual).all())

    def test_all_gradients_optimizer_and_v2_projection(self):
        for cls in [IFNet, IFNetV2]:
            model = cls(
                num_electrodes=3,
                chunk_size=128,
                num_filters=4,
                num_bands=3,
                temporal_kernel=31,
                patch_size=16,
            )
            with torch.no_grad():
                model.fc[0].weight.mul_(30)
            before = model.fc[0].weight.detach().clone()
            x = torch.randn(3, 3, 3, 128, requires_grad=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            # Accumulating two forward graphs must retain the author behavior.
            logits = model(x) + model(x)
            loss = nn.functional.cross_entropy(logits, torch.tensor([0, 1, 3]))
            loss.backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            for name, value in model.named_parameters():
                self.assertIsNotNone(value.grad, name)
                self.assertTrue(torch.isfinite(value.grad).all(), name)
            if cls is IFNetV2:
                self.assertLessEqual(
                    model.fc[0].weight.norm(dim=1).max().item(), 0.5 + 1e-6
                )
                self.assertFalse(torch.equal(model.fc[0].weight, before))
            else:
                torch.testing.assert_close(model.fc[0].weight, before, rtol=0, atol=0)
            snapshot = {n: p.detach().clone() for n, p in model.named_parameters()}
            optimizer.step()
            self.assertTrue(
                any(
                    not torch.equal(p, snapshot[n]) for n, p in model.named_parameters()
                )
            )

    def test_full_safe_checkpoint_restores_next_training_update(self):
        for cls in [IFNet, IFNetV2]:
            config = dict(
                num_electrodes=3,
                chunk_size=128,
                num_filters=4,
                temporal_kernel=7,
                patch_size=16,
            )
            model = cls(**config)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            x, y = torch.randn(4, 2, 3, 128), torch.tensor([0, 1, 2, 3])

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
            restored = cls(**config)
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

    def test_average_pool_remainder_and_log_power_geometry(self):
        model = IFNet(
            num_electrodes=3,
            chunk_size=135,
            num_filters=4,
            temporal_kernel=7,
            patch_size=16,
        )
        self.assertEqual(model(torch.randn(3, 2, 3, 135)).shape, (3, 4))
        self.assertEqual(model.fc[0].in_features, 32)
        with self.assertRaisesRegex(ValueError, 'exact multiple'):
            IFNetV2(chunk_size=751)
        for cls in [IFNet, IFNetV2]:
            model = cls(
                num_electrodes=3,
                chunk_size=128,
                num_filters=4,
                temporal_kernel=7,
                patch_size=16,
            ).eval()
            self.assertTrue(torch.isfinite(model(torch.zeros(3, 2, 3, 128))).all())

    def test_invalid_configuration_and_input(self):
        for cls in [IFNet, IFNetV2]:
            for kwargs in [
                {'num_electrodes': True},
                {'num_classes': 0},
                {'chunk_size': 124},
                {'num_filters': 0},
                {'num_bands': 0},
                {'temporal_kernel': 62},
                {'num_bands': 7},
                {'patch_size': 0},
                {'dropout': float('nan')},
            ]:
                with (
                    self.subTest(cls=cls, kwargs=kwargs),
                    self.assertRaises(ValueError),
                ):
                    cls(**kwargs)
            model = cls()
            for shape in [
                (3, 22, 750),
                (3, 44, 749),
                (3, 22, 2, 750),
                (3, 2, 21, 750),
                (3, 1, 44, 750),
                (44, 750),
            ]:
                with self.subTest(shape=shape), self.assertRaises(ValueError):
                    model(torch.randn(shape))
        with self.assertRaises(ValueError):
            IFNetV2(max_norm=0)

    @unittest.skipUnless(
        hasattr(torch, 'compile'), 'torch.compile requires PyTorch 2.x'
    )
    def test_fullgraph_eager_compilation(self):
        for cls in [IFNet, IFNetV2]:
            model = cls(
                num_electrodes=3,
                chunk_size=128,
                num_filters=4,
                temporal_kernel=7,
                patch_size=16,
            ).eval()
            x = torch.randn(3, 2, 3, 128)
            compiled = torch.compile(
                copy.deepcopy(model), backend='eager', fullgraph=True
            )
            torch.testing.assert_close(compiled(x), model(x), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
