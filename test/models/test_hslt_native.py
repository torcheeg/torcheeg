"""Compare the complete pinned LibEER HSLT without substitute NN modules."""

import contextlib
import copy
import hashlib
import importlib.util
import io
import os
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F
from torcheeg.models import HSLT


@unittest.skipUnless(
    os.environ.get('TORCHEEG_HSLT_REFERENCE'),
    'Complete pinned LibEER source supplied separately.',
)
class TestHSLTNative(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(os.environ['TORCHEEG_HSLT_REFERENCE'])
        cls.hashes = {
            'LibEER/models/HSLT.py': '8f5f1e409960d07b420a0a80f1251b8b8c21e1cd56e8aec11a957975829fbade',
            'LibEER/data_utils/constants/deap.py': '410a8385c019cb1d81bb637849ccd85c10058f8afaa6412662daa2eb3b05176d',
            'LibEER/data_utils/constants/seed.py': '43b829bf3be6255616e9d0fc3305e6e06639485796e17073adafc611b4e28bca',
            'LICENSE': 'b9e59357a5f5e338c92348e6efe5482d364284aa3c1f59c023f9bb33c69800ee',
        }
        for name, digest in cls.hashes.items():
            if hashlib.sha256((cls.root / name).read_bytes()).hexdigest() != digest:
                raise AssertionError(
                    'Complete pinned reference source required: ' + name
                )
        sys.path.insert(0, str(cls.root / 'LibEER'))
        path = cls.root / 'LibEER/models/HSLT.py'
        spec = importlib.util.spec_from_file_location('independent_libeer_hslt', path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)
        if Path(cls.module.param_path).exists():
            raise AssertionError(
                'Native constructor must use explicit arguments without a cwd YAML override.'
            )

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(cls.root / 'LibEER'))
        for name, digest in cls.hashes.items():
            if hashlib.sha256((cls.root / name).read_bytes()).hexdigest() != digest:
                raise AssertionError('Reference file changed: ' + name)

    def native(self, **config):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.module.HSLT(**config)

    def test_complete_default_and_custom_initialization_rng_and_states(self):
        for seed, config, reference in [
            (19, {}, {}),
            (
                23,
                dict(
                    num_electrodes=62,
                    num_classes=3,
                    in_channels=7,
                    electrode_embed_dim=9,
                    region_embed_dim=17,
                    electrode_depth=1,
                    region_depth=3,
                ),
                dict(
                    num_electrodes=62,
                    num_classes=3,
                    in_channels=7,
                    De=9,
                    Dr=17,
                    Le=1,
                    Lr=3,
                ),
            ),
        ]:
            with self.subTest(config=config):
                torch.manual_seed(seed)
                native = self.native(**reference)
                rng = torch.get_rng_state().clone()
                torch.manual_seed(seed)
                model = HSLT(**config)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertEqual(set(model.state_dict()), set(native.state_dict()))
                for key, value in native.state_dict().items():
                    torch.testing.assert_close(
                        model.state_dict()[key], value, rtol=0, atol=0
                    )
        default = HSLT()
        self.assertEqual(sum(v.numel() for v in default.parameters()), 373330)
        self.assertEqual(len(default.state_dict()), 368)
        self.assertEqual(len(default.transformers), 27)

    def test_complete_training_eval_loss_all_gradients_and_adamw(self):
        for training in [False, True]:
            for channels, classes in [(32, 2), (62, 3)]:
                with self.subTest(training=training, channels=channels):
                    native = self.native(
                        num_electrodes=channels, num_classes=classes
                    ).train(training)
                    model = HSLT(num_electrodes=channels, num_classes=classes).train(
                        training
                    )
                    model.load_state_dict(native.state_dict(), strict=True)
                    x = torch.randn(3, channels, 5, requires_grad=True)
                    y = x.detach().clone().requires_grad_()
                    torch.manual_seed(43)
                    actual = model(x)
                    rng = torch.get_rng_state().clone()
                    torch.manual_seed(43)
                    expected = native(y)
                    self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    labels = torch.tensor([0, 1, classes - 1])
                    if classes == 2:
                        target = F.one_hot(labels, classes).float()
                        loss = F.binary_cross_entropy(actual, target)
                        other = F.binary_cross_entropy(expected, target)
                    else:
                        loss = F.nll_loss(actual.log(), labels)
                        other = F.nll_loss(expected.log(), labels)
                    torch.testing.assert_close(loss, other, rtol=0, atol=0)
                    loss.backward()
                    other.backward()
                    torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
                    for name, value in native.named_parameters():
                        self.assertIsNotNone(value.grad, name)
                        torch.testing.assert_close(
                            dict(model.named_parameters())[name].grad,
                            value.grad,
                            rtol=0,
                            atol=0,
                        )
                    for net in [model, native]:
                        torch.optim.AdamW(net.parameters(), lr=0.001).step()
                    for name, value in native.state_dict().items():
                        torch.testing.assert_close(
                            model.state_dict()[name], value, rtol=0, atol=0
                        )

    def test_independent_patch_embedding_and_full_attention(self):
        model = HSLT().eval()
        patch = model.transformers[0]
        x = torch.randn(2, 4, 5)
        projected = x @ patch.projection.weight.T + patch.projection.bias
        token = patch.class_token.unsqueeze(0).expand(2, -1, -1)
        expected = torch.cat(
            [token, projected], 1
        ) + patch.position_embedding.weight.unsqueeze(0)
        torch.testing.assert_close(patch(x), expected)
        block = model.transformers[1].blocks[0]
        z = F.relu(block.fc1(block.layernormalization1(expected)))
        attention = block.attention
        q, k, v = F.linear(z, attention.in_proj_weight, attention.in_proj_bias).chunk(
            3, -1
        )

        def heads(t):
            return t.reshape(2, 5, 16, 4).transpose(1, 2)

        scores = (heads(q) @ heads(k).transpose(-1, -2)) / 2
        mixed = (
            (scores.softmax(-1) @ heads(v))
            .transpose(1, 2)
            .contiguous()
            .reshape(2, 5, 64)
        )
        mixed = F.linear(mixed, attention.out_proj.weight, attention.out_proj.bias)
        residual = expected + F.relu(block.fc2(mixed))
        expected_block = residual + block.mlp(block.layernormalization2(residual))
        torch.testing.assert_close(
            block(expected), expected_block, rtol=1e-5, atol=1e-6
        )

    def test_next_step_complete_optimizer_rng_resume(self):
        model = HSLT(return_logits=True).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        x = torch.randn(3, 32, 5)
        target = torch.tensor([0, 1, 0])

        def step(net, opt):
            opt.zero_grad()
            loss = F.cross_entropy(net(x), target)
            loss.backward()
            opt.step()
            return loss.detach()

        step(model, optimizer)
        state = copy.deepcopy(model.state_dict())
        optim_state = copy.deepcopy(optimizer.state_dict())
        rng = torch.get_rng_state().clone()
        restored = HSLT(return_logits=True).train()
        restored.load_state_dict(state, strict=True)
        other = torch.optim.AdamW(restored.parameters(), lr=0.001)
        other.load_state_dict(optim_state)
        torch.set_rng_state(rng)
        expected = step(model, optimizer)
        torch.set_rng_state(rng)
        actual = step(restored, other)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(
                restored.state_dict()[name], value, rtol=0, atol=0
            )

    def test_eager_fullgraph_compile(self):
        model = HSLT(return_logits=True).eval()
        x = torch.randn(2, 32, 5)
        with torch.no_grad():
            expected = model(x)
            actual = torch.compile(model, backend='eager', fullgraph=True)(x)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
