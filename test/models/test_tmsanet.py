"""TMSA-Net local/global attention, flatten order and complete continuation."""

import copy
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from torcheeg.models import TMSANet
from torcheeg.models.transformer.tmsanet import _TMSAAttention


class TestTMSANet(unittest.TestCase):
    def make_model(self, **overrides):
        config = dict(
            num_electrodes=3,
            chunk_size=149,
            num_classes=4,
            hid_channels=9,
            pool_size=11,
            pool_stride=7,
            heads=4,
            depth=2,
        )
        config.update(overrides)
        return TMSANet(**config)

    def test_default_geometry_features_and_input_layouts(self):
        model = TMSANet().eval()
        self.assertEqual(model.hid_channels, 19)
        self.assertEqual(model.num_tokens, 64)
        self.assertEqual(model.final_layer.in_features, 19 * 64)
        self.assertEqual(model.transformer[1].attention.w_q.out_features, 16)
        self.assertEqual(model.spatial_conv.groups, 1)
        x = torch.randn(2, 22, 1000)
        features, logits = model(x, return_features=True)
        self.assertEqual(features.shape, (2, 1216))
        torch.testing.assert_close(model(x.unsqueeze(1)), logits, rtol=0, atol=0)
        altered = x.clone()
        altered[1] += 3
        torch.testing.assert_close(model(altered)[0], logits[0], rtol=0, atol=0)

    def test_independent_numpy_local_and_global_attention_sum(self):
        module = _TMSAAttention(9, 4, 0, 0).eval()
        x = torch.randn(2, 5, 9, requires_grad=True)
        array = x.detach().numpy()

        def linear(layer, value):
            return value @ layer.weight.detach().numpy().T + layer.bias.detach().numpy()

        local = []
        for conv in module.local_convs:
            kernel = conv.kernel_size[0]
            padded = np.pad(
                array.transpose(0, 2, 1), ((0, 0), (0, 0), (kernel // 2, kernel // 2))
            )
            convolved = (
                np.stack(
                    [
                        np.einsum(
                            'bck,ock->bo',
                            padded[..., i : i + kernel],
                            conv.weight.detach().numpy(),
                        )
                        for i in range(5)
                    ],
                    axis=-1,
                )
                + conv.bias.detach().numpy()[None, :, None]
            )
            local.append(convolved)
        local = np.concatenate(local, axis=1)
        bn = module.local_bn
        local = (
            (local - bn.running_mean.numpy()[None, :, None])
            / np.sqrt(bn.running_var.numpy()[None, :, None] + bn.eps)
            * bn.weight.detach().numpy()[None, :, None]
            + bn.bias.detach().numpy()[None, :, None]
        ).transpose(0, 2, 1)

        def split(value):
            return value.reshape(2, 5, 4, 2).transpose(0, 2, 1, 3)

        queries = split(linear(module.w_q, array))
        values = split(linear(module.w_v, array))
        outputs = []
        for keys in (
            split(linear(module.w_k_local, local)),
            split(linear(module.w_k_global, array)),
        ):
            scores = (queries @ keys.transpose(0, 1, 3, 2)) / np.sqrt(2)
            weights = np.exp(scores - scores.max(-1, keepdims=True))
            weights /= weights.sum(-1, keepdims=True)
            outputs.append(weights @ values)
        merged = (outputs[0] + outputs[1]).transpose(0, 2, 1, 3).reshape(2, 5, 8)
        expected = linear(module.w_o, merged)
        actual = module(x)
        np.testing.assert_allclose(
            actual.detach().numpy(), expected, rtol=1e-5, atol=1e-6
        )
        actual.square().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_embedding_major_flatten_and_pristine_batchnorm(self):
        model = self.make_model().eval()
        for layer in model.modules():
            if isinstance(layer, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                self.assertEqual(layer.num_batches_tracked.item(), 0)
        outputs = []
        hook = model.transformer.register_forward_hook(
            lambda module, args, output: outputs.append(output.detach())
        )
        try:
            features, logits = model(torch.randn(2, 3, 149), return_features=True)
        finally:
            hook.remove()
        expected = outputs[0].numpy().transpose(0, 2, 1).reshape(2, -1)
        np.testing.assert_array_equal(features.detach().numpy(), expected)
        self.assertEqual(features.shape[1], 9 * 20)
        self.assertEqual(logits.shape, (2, 4))

    def test_complete_checkpoint_optimizer_and_rng_next_step_resume(self):
        model = self.make_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        x, target = torch.randn(3, 3, 149), torch.tensor([0, 1, 3])

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

    def test_invalid_configuration_and_input(self):
        for config in (
            dict(num_electrodes=0),
            dict(chunk_size=True),
            dict(num_classes=0),
            dict(hid_channels=3),
            dict(pool_size=150),
            dict(pool_stride=0),
            dict(heads=0),
            dict(depth=0),
            dict(mlp_ratio=0),
            dict(dropout=float('nan')),
            dict(attn_dropout=-1),
            dict(fc_dropout=2),
        ):
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    self.make_model(**config)
        model = self.make_model()
        for bad in (torch.ones(2, 3, 149, dtype=torch.long), 'bad'):
            with self.assertRaises(TypeError):
                model(bad)
        for bad in (
            torch.randn(2, 2, 149),
            torch.randn(2, 3, 148),
            torch.randn(2, 2, 3, 149),
        ):
            with self.assertRaises(ValueError):
                model(bad)

    @unittest.skipUnless(hasattr(torch, 'compile'), 'torch.compile requires PyTorch 2.')
    def test_eager_full_graph_compile(self):
        model = self.make_model().eval()
        x = torch.randn(2, 3, 149)
        torch.testing.assert_close(
            torch.compile(model, backend='eager', fullgraph=True)(x),
            model(x),
            rtol=0,
            atol=0,
        )


if __name__ == '__main__':
    unittest.main()
