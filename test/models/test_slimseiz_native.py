"""Full unmodified pinned notebook cells, including every native neural layer."""

import copy
import hashlib
import json
import math
import os
from pathlib import Path
from types import ModuleType
import unittest

from einops import rearrange, repeat, einsum
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torcheeg.models import SlimSeiz


@unittest.skipUnless(
    os.environ.get('TORCHEEG_SLIMSEIZ_SOURCE'),
    'Complete pinned author notebook supplied separately.',
)
class TestSlimSeizNative(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = Path(os.environ['TORCHEEG_SLIMSEIZ_SOURCE'])
        source = cls.path.read_bytes()
        if hashlib.sha256(source).hexdigest() != SlimSeiz.SOURCE_SHA256:
            raise AssertionError('Complete pinned notebook required.')
        notebook = json.loads(source)
        cls.original = ModuleType('independent_original_slimseiz')
        cls.original.__dict__.update(
            torch=torch,
            nn=nn,
            F=F,
            math=math,
            rearrange=rearrange,
            repeat=repeat,
            einsum=einsum,
        )
        for index in [1, 7]:
            exec(
                compile(
                    ''.join(notebook['cells'][index]['source']),
                    str(cls.path) + ':cell' + str(index),
                    'exec',
                ),
                cls.original.__dict__,
            )

    @classmethod
    def tearDownClass(cls):
        if hashlib.sha256(cls.path.read_bytes()).hexdigest() != SlimSeiz.SOURCE_SHA256:
            raise AssertionError('Author source changed.')

    def test_complete_default_initialization_rng_and_native_states(self):
        torch.manual_seed(19)
        native = self.original.OneDCNN(input_channels=8)
        rng = torch.get_rng_state().clone()
        torch.manual_seed(19)
        model = SlimSeiz.from_source(self.path)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(set(native.state_dict()), set(model.native_state_dict()))
        self.assertEqual(
            set(model.state_dict()), {'backend.' + key for key in native.state_dict()}
        )
        for key, value in native.state_dict().items():
            torch.testing.assert_close(
                model.native_state_dict()[key], value, rtol=0, atol=0
            )
        self.assertEqual(sum(v.numel() for v in model.parameters()), 22946)
        self.assertEqual(len(model.state_dict()), 28)
        self.assertEqual(model.backend.mixer.d_state, 16)
        self.assertEqual(model.backend.mixer.dt_rank, 2)
        with self.assertRaises(RuntimeError):
            model.load_native_state_dict({}, strict=True)

    def test_complete_train_eval_loss_input_all_gradients_and_updates(self):
        for training in [False, True]:
            for channels, length in [(8, 1024), (3, 33)]:
                with self.subTest(training=training, channels=channels, length=length):
                    native = self.original.OneDCNN(input_channels=channels).train(
                        training
                    )
                    model = SlimSeiz.from_source(
                        self.path,
                        num_electrodes=channels,
                        chunk_size=length,
                        return_features=True,
                    ).train(training)
                    model.load_native_state_dict(native.state_dict(), strict=True)
                    x = torch.randn(3, channels, length, requires_grad=True)
                    y = x.detach().clone().requires_grad_()
                    torch.manual_seed(43)
                    actual, features = model(x)
                    rng = torch.get_rng_state().clone()
                    torch.manual_seed(43)
                    expected, other_features = native(y)
                    self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(features, other_features, rtol=0, atol=0)
                    labels = torch.tensor([0, 1, 0])
                    loss = F.cross_entropy(actual, labels)
                    other = F.cross_entropy(expected, labels)
                    torch.testing.assert_close(loss, other, rtol=0, atol=0)
                    loss.backward()
                    other.backward()
                    torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
                    for name, value in native.named_parameters():
                        self.assertIsNotNone(value.grad, name)
                        self.assertTrue(torch.isfinite(value.grad).all(), name)
                        torch.testing.assert_close(
                            dict(model.backend.named_parameters())[name].grad,
                            value.grad,
                            rtol=0,
                            atol=0,
                        )
                    for net in [model, native]:
                        torch.optim.AdamW(net.parameters(), lr=0.001).step()
                    for name, value in native.state_dict().items():
                        torch.testing.assert_close(
                            model.native_state_dict()[name], value, rtol=0, atol=0
                        )

    def test_independent_numpy_selective_scan_and_gradcheck(self):
        model = SlimSeiz.from_source(self.path)
        rng = np.random.default_rng(19)
        arrays = [
            rng.normal(size=s)
            for s in [(2, 3, 2), (2, 3, 2), (2, 3), (2, 3, 3), (2, 3, 3), (2,)]
        ]
        arrays[1] = np.exp(arrays[1])
        arrays[2] = -np.exp(arrays[2])
        u, delta, a, b, c, d = arrays
        state = np.zeros((2, 2, 3))
        values = []
        for t in range(3):
            state = (
                np.exp(delta[:, t, :, None] * a[None, :, :]) * state
                + delta[:, t, :, None] * b[:, t, None, :] * u[:, t, :, None]
            )
            values.append((state * c[:, t, None, :]).sum(-1) + u[:, t] * d)
        expected = np.stack(values, 1)
        inputs = tuple(
            torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in arrays
        )
        actual = model.backend.mixer.selective_scan(*inputs)
        np.testing.assert_allclose(
            actual.detach().numpy(), expected, rtol=1e-11, atol=1e-11
        )
        self.assertTrue(
            torch.autograd.gradcheck(
                model.backend.mixer.selective_scan,
                inputs,
                eps=1e-6,
                atol=1e-5,
                rtol=1e-4,
            )
        )

    def test_bct_b1ct_minimum_geometry_and_sample_isolation(self):
        model = SlimSeiz.from_source(self.path).eval()
        x = torch.randn(3, 8, 1024)
        with torch.no_grad():
            actual = model(x)
            torch.testing.assert_close(model(x.unsqueeze(1)), actual, rtol=0, atol=0)
            torch.testing.assert_close(model(x[:1]), actual[:1], rtol=1e-5, atol=1e-6)
            self.assertTrue(torch.isfinite(model(torch.zeros_like(x))).all())
        for length in [32, 63, 1025]:
            net = SlimSeiz.from_source(self.path, num_electrodes=1, chunk_size=length)
            self.assertEqual(net(torch.randn(2, 1, length)).shape, (2, 2))
        with self.assertRaises(ValueError):
            model(torch.randn(2, 7, 1024))
        with self.assertRaises(TypeError):
            model(torch.ones(2, 8, 1024, dtype=torch.int64))

    def test_complete_model_optimizer_and_random_next_step_resume(self):
        model = SlimSeiz.from_source(self.path).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        x = torch.randn(3, 8, 1024)
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
        restored = SlimSeiz.from_source(self.path).train()
        restored.load_state_dict(state, strict=True)
        other = torch.optim.AdamW(restored.parameters(), lr=0.001)
        other.load_state_dict(optim_state)
        torch.set_rng_state(rng)
        expected = step(model, optimizer)
        torch.set_rng_state(rng)
        actual = step(restored, other)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(
                restored.state_dict()[key], value, rtol=0, atol=0
            )

    def test_eager_fullgraph_compile(self):
        model = SlimSeiz.from_source(self.path).eval()
        x = torch.randn(2, 8, 1024)
        with torch.no_grad():
            torch.testing.assert_close(
                torch.compile(model, backend='eager', fullgraph=True)(x),
                model(x),
                rtol=1e-5,
                atol=1e-6,
            )
