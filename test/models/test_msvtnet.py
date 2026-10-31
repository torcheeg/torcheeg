import copy
import io
import unittest

import torch
from torch import nn

from torcheeg.models import MSVTNet, MSVTNetLoss
from torcheeg.models.transformer.msvtnet import _SamePadConv2d


class TestMSVTNet(unittest.TestCase):
    def test_native_and_legacy_same_padding_outputs_and_all_gradients(self):
        for kernel in (15, 16):
            with self.subTest(kernel=kernel):
                native = _SamePadConv2d(4, 4, (1, kernel), groups=4, bias=False)
                legacy = copy.deepcopy(native)
                legacy._native_same = False
                left = torch.randn(2, 4, 3, 113, requires_grad=True)
                right = left.detach().clone().requires_grad_()
                actual, expected = native(left), legacy(right)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                actual.square().mean().backward()
                expected.square().mean().backward()
                torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(
                    native.weight.grad, legacy.weight.grad, rtol=1e-5, atol=1e-6
                )
                legacy.load_state_dict(native.state_dict(), strict=True)

    def make_model(self, **overrides):
        config = dict(
            num_electrodes=3,
            chunk_size=128,
            num_classes=4,
            num_filters=(2, 2, 2, 2),
            heads=2,
            depth=1,
            dropout=0.0,
            attn_dropout=0.0,
            return_branch_logits=True,
        )
        config.update(overrides)
        return MSVTNet(**config)

    def test_complete_default_geometry_and_input_layouts(self):
        model = MSVTNet(return_branch_logits=True).eval()
        eeg = torch.randn(2, 22, 1000)
        main, branches = model(eeg)
        self.assertEqual(main.shape, (2, 4))
        self.assertEqual(len(branches), 4)
        self.assertEqual(len(model.transformer.trans.layers), 2)
        self.assertEqual(model.transformer.pos_embedding.pe.shape, (1, 18, 72))
        self.assertEqual(model.branch_head[0][1].in_features, 17 * 18)
        actual = model(eeg.unsqueeze(1))
        for left, right in zip((main,) + branches, (actual[0],) + actual[1]):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        model.return_branch_logits = False
        torch.testing.assert_close(model(eeg), main, rtol=0, atol=0)

    def test_batchnorm_starts_pristine_and_shapes_are_analytic(self):
        model = self.make_model(chunk_size=113, temporal_kernels=(16, 32, 64, 126))
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                self.assertEqual(layer.num_batches_tracked.item(), 0)
                self.assertTrue(
                    torch.equal(
                        layer.running_mean, torch.zeros_like(layer.running_mean)
                    )
                )
                self.assertTrue(
                    torch.equal(layer.running_var, torch.ones_like(layer.running_var))
                )
        model.eval()
        main, branches = model(torch.randn(2, 3, 113))
        self.assertEqual(main.shape, (2, 4))
        self.assertEqual(len(branches), 4)

    def test_joint_loss_is_weighted_main_plus_sum_not_mean(self):
        outputs = self.make_model()(torch.randn(3, 3, 128))
        labels = torch.tensor([0, 1, 2])
        terms = [
            nn.functional.cross_entropy(value, labels)
            for value in (outputs[0],) + outputs[1]
        ]
        expected = 0.6 * terms[0] + 0.4 * sum(terms[1:])
        torch.testing.assert_close(MSVTNetLoss()(outputs, labels), expected)
        torch.testing.assert_close(MSVTNetLoss(1.0)(outputs, labels), terms[0])
        torch.testing.assert_close(MSVTNetLoss(0.0)(outputs, labels), sum(terms[1:]))

    def test_optimizer_reaches_all_branches_position_transformer_and_heads(self):
        model = self.make_model()
        eeg = torch.randn(4, 3, 128, requires_grad=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        initial = copy.deepcopy(model.state_dict())
        loss = MSVTNetLoss()(model(eeg), torch.tensor([0, 1, 2, 3]))
        loss.backward()
        self.assertTrue(torch.isfinite(eeg.grad).all())
        for name, value in model.named_parameters():
            self.assertIsNotNone(value.grad, name)
            self.assertTrue(torch.isfinite(value.grad).all(), name)
        for block in (
            model.mstsconv,
            model.branch_head,
            model.transformer,
            model.last_head,
        ):
            self.assertGreater(
                sum(p.grad.abs().sum().item() for p in block.parameters()), 0
            )
        optimizer.step()
        for index in range(4):
            name = f'branch_head.{index}.1.weight'
            self.assertFalse(torch.equal(initial[name], model.state_dict()[name]))

    def test_dropout_optimizer_and_rng_checkpoint_continuation(self):
        torch.manual_seed(14)
        model = self.make_model(dropout=0.3, attn_dropout=0.5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        eeg = torch.randn(3, 3, 128)
        labels = torch.tensor([0, 1, 2])

        def step(net, opt):
            opt.zero_grad()
            loss = MSVTNetLoss()(net(eeg), labels)
            loss.backward()
            opt.step()
            return loss.detach()

        step(model, optimizer)
        buffer = io.BytesIO()
        torch.save(
            dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                rng=torch.get_rng_state(),
            ),
            buffer,
        )
        expected = step(model, optimizer)
        expected_rng = torch.get_rng_state()
        buffer.seek(0)
        checkpoint = torch.load(buffer, weights_only=True)
        restored = self.make_model(dropout=0.3, attn_dropout=0.5)
        restored.load_state_dict(checkpoint['model'], strict=True)
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.001)
        restored_optimizer.load_state_dict(checkpoint['optimizer'])
        torch.set_rng_state(checkpoint['rng'])
        torch.testing.assert_close(
            step(restored, restored_optimizer), expected, rtol=0, atol=0
        )
        for name, value in model.state_dict().items():
            torch.testing.assert_close(
                restored.state_dict()[name], value, rtol=0, atol=0
            )
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))

    def test_eval_does_not_mix_patients(self):
        model = self.make_model(dropout=0.5, attn_dropout=0.5).eval()
        eeg = torch.randn(3, 3, 128)
        with torch.no_grad():
            all_outputs = model(eeg)
            single = model(eeg[:1])
        for full, part in zip(
            (all_outputs[0],) + all_outputs[1], (single[0],) + single[1]
        ):
            torch.testing.assert_close(full[:1], part, rtol=1e-5, atol=1e-6)

    def test_invalid_configuration_and_input_fail(self):
        for config in [
            dict(num_electrodes=True),
            dict(chunk_size=55),
            dict(num_classes=0),
            dict(num_filters=()),
            dict(temporal_kernels=(3,)),
            dict(heads=3),
            dict(depth=0),
            dict(mlp_ratio=0.0),
            dict(mlp_ratio=float('inf')),
            dict(dropout=float('nan')),
            dict(attn_dropout=1.1),
            dict(pool_size1=0),
            dict(return_branch_logits=1),
        ]:
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.make_model(**config)
        model = self.make_model()
        for shape in [(3, 128), (2, 2, 3, 128), (2, 4, 128), (2, 3, 129)]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                model(torch.randn(shape))
        for value in [-1.0, float('nan'), 2.0, True]:
            with self.assertRaises(ValueError):
                MSVTNetLoss(value)

    def test_full_graph_compilation_preserves_training_gradients(self):
        if not hasattr(torch, 'compile'):
            self.skipTest('torch.compile is unavailable.')
        model = self.make_model()
        reference = copy.deepcopy(model)
        eeg = torch.randn(2, 3, 128)
        labels = torch.tensor([0, 1])
        compiled = torch.compile(model, backend='eager', fullgraph=True)
        actual = MSVTNetLoss()(compiled(eeg), labels)
        expected = MSVTNetLoss()(reference(eeg), labels)
        actual.backward()
        expected.backward()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for (_, left), (_, right) in zip(
            model.named_parameters(), reference.named_parameters()
        ):
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
