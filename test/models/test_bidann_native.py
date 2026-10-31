"""Compare all four heads of the complete unmodified pinned LibEER model."""

import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import unittest
import numpy as np
import torch
from torch.nn import functional as F
from torcheeg.models import BiDANN, BiDANNLoss
from torcheeg.models.rnn.bidann import (
    _split_hemispheres,
    _SEED_CHANNELS,
    _LEFT_CHANNELS,
    _RIGHT_CHANNELS,
)


@unittest.skipUnless(
    os.environ.get('TORCHEEG_BIDANN_REFERENCE'),
    'Complete pinned LibEER reference supplied separately.',
)
class TestBiDANNNative(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(os.environ['TORCHEEG_BIDANN_REFERENCE'])
        cls.hashes = {
            'LibEER/models/BiDANN.py': '33c093b3a1df25dbebad7a4fdee0e58448fd0da4714b012d614acdb8c97104d8',
            'LibEER/Trainer/BiDANNTraining.py': 'e36789bb14e80384a8c72e3ee138ffffb8a09c96bee1f980128ee03d55659dd2',
            'LICENSE': 'b9e59357a5f5e338c92348e6efe5482d364284aa3c1f59c023f9bb33c69800ee',
        }
        for name, digest in cls.hashes.items():
            if hashlib.sha256((cls.root / name).read_bytes()).hexdigest() != digest:
                raise AssertionError('Complete pinned reference required: ' + name)
        spec = importlib.util.spec_from_file_location(
            'independent_complete_libeer_bidann', cls.root / 'LibEER/models/BiDANN.py'
        )
        cls.native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.native)

    @classmethod
    def tearDownClass(cls):
        for name, digest in cls.hashes.items():
            if hashlib.sha256((cls.root / name).read_bytes()).hexdigest() != digest:
                raise AssertionError('Reference source changed: ' + name)

    def test_complete_default_custom_states_and_initialization_rng(self):
        for seed, config, reference in [
            (19, {}, {}),
            (
                23,
                dict(in_channels=2, sequence_length=3, adversarial_weight=0.4),
                dict(in_channels=2, sample_length=3, lambda_=0.4),
            ),
        ]:
            torch.manual_seed(seed)
            native = self.native.BiDANN(**reference)
            rng = torch.get_rng_state().clone()
            torch.manual_seed(seed)
            model = BiDANN(hemisphere_layout='libeer', **config)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(set(model.state_dict()), set(native.state_dict()))
            for key, value in native.state_dict().items():
                torch.testing.assert_close(
                    model.state_dict()[key], value, rtol=0, atol=0
                )
        model = BiDANN()
        self.assertEqual(sum(v.numel() for v in model.parameters()), 3348649)
        self.assertEqual(len(model.state_dict()), 94)

    def test_full_train_eval_loss_all_parameter_gradients_bn_and_adamw(self):
        for training in [False, True]:
            native = self.native.BiDANN(lambda_=0.4).train(training)
            model = BiDANN(adversarial_weight=0.4, hemisphere_layout='libeer').train(
                training
            )
            model.load_state_dict(native.state_dict(), strict=True)
            source = torch.randn(3, 9, 62, 5, requires_grad=True)
            target = torch.randn(2, 9, 62, 5, requires_grad=True)
            torch.manual_seed(43)
            actual = model(source, target)
            rng = torch.get_rng_state().clone()
            torch.manual_seed(43)
            expected = native(source.detach(), target.detach())
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            labels = torch.tensor([0, 1, 2])
            domains = torch.tensor([0, 0, 0, 1, 1])
            other = F.cross_entropy(expected[0], labels)
            for head in expected[1:]:
                other = other + F.cross_entropy(head, domains)
            loss = BiDANNLoss()(actual, labels)
            torch.testing.assert_close(loss, other, rtol=0, atol=0)
            loss.backward()
            other.backward()
            self.assertTrue(torch.isfinite(source.grad).all())
            self.assertTrue(torch.isfinite(target.grad).all())
            for name, value in native.named_parameters():
                self.assertIsNotNone(value.grad, name)
                self.assertTrue(torch.isfinite(value.grad).all(), name)
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

    def test_tensor_hemispheres_match_full_native_literal_layout(self):
        for length, features in [(9, 5), (3, 2)]:
            x = torch.randn(2, length, 62, features)
            actual = _split_hemispheres(
                x, _SEED_CHANNELS, _LEFT_CHANNELS, _RIGHT_CHANNELS
            )
            expected = self.native.divide_r_l(
                x,
                self.native.SEED_CHANNEL_NAME,
                self.native.LEFT_CHANNEL_NAME,
                self.native.RIGHT_CHANNEL_NAME,
            )
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_source_only_logits_native_eval_and_sample_isolation(self):
        native = self.native.BiDANN().eval()
        model = BiDANN(hemisphere_layout='libeer').eval()
        model.load_state_dict(native.state_dict(), strict=True)
        x = torch.randn(3, 9, 62, 5)
        with torch.no_grad():
            actual = model(x)
            expected = native(x, x)[0]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(model(x[:1]), actual[:1], rtol=1e-5, atol=1e-6)

    def test_next_step_full_optimizer_rng_and_bn_resume(self):
        model = BiDANN().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        source = torch.randn(3, 9, 62, 5)
        target = torch.randn(2, 9, 62, 5)
        labels = torch.tensor([0, 1, 2])

        def step(net, opt):
            opt.zero_grad()
            loss = BiDANNLoss()(net(source, target), labels)
            loss.backward()
            opt.step()
            return loss.detach()

        step(model, optimizer)
        state = copy.deepcopy(model.state_dict())
        optim_state = copy.deepcopy(optimizer.state_dict())
        rng = torch.get_rng_state().clone()
        restored = BiDANN().train()
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

    def test_eager_compile_with_native_lstm_graph_breaks(self):
        model = BiDANN(hemisphere_layout='libeer').eval()
        x = torch.randn(2, 9, 62, 5)
        with torch.no_grad():
            expected = model(x)
            actual = torch.compile(model, backend='eager')(x)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_aligned_layout_complete_reference_layers_and_all_updates(self):
        native = self.native.BiDANN().train()
        model = BiDANN().train()
        model.load_state_dict(native.state_dict(), strict=True)
        source = torch.randn(3, 9, 62, 5)
        target = torch.randn(2, 9, 62, 5)

        def split(x):
            data = x.numpy()
            return [
                torch.from_numpy(
                    np.ascontiguousarray(
                        data[
                            :,
                            :,
                            [self.native.SEED_CHANNEL_NAME.index(n) for n in names],
                            :,
                        ].reshape(len(x), 9, -1)
                    )
                )
                for names in [
                    self.native.LEFT_CHANNEL_NAME,
                    self.native.RIGHT_CHANNEL_NAME,
                ]
            ]

        ls, rs = split(source)
        lt, rt = split(target)
        torch.manual_seed(43)
        actual = model(source, target)
        rng = torch.get_rng_state().clone()
        torch.manual_seed(43)
        ls = native.l_extractor(ls)
        rs = native.r_extractor(rs)
        lt = native.l_extractor(lt)
        rt = native.r_extractor(rt)
        expected = (
            native.classifer(ls, rs),
            native.l_discriminator(ls, lt),
            native.r_discriminator(rs, rt),
            native.global_discriminator(ls, rs, lt, rt),
        )
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        labels = torch.tensor([0, 1, 2])
        domains = torch.tensor([0, 0, 0, 1, 1])
        loss = BiDANNLoss()(actual, labels)
        other = F.cross_entropy(expected[0], labels)
        for head in expected[1:]:
            other = other + F.cross_entropy(head, domains)
        loss.backward()
        other.backward()
        for name, value in native.named_parameters():
            torch.testing.assert_close(
                dict(model.named_parameters())[name].grad, value.grad, rtol=0, atol=0
            )
        for net in [model, native]:
            torch.optim.AdamW(net.parameters(), lr=0.001).step()
        for name, value in native.state_dict().items():
            torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
