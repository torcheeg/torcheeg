"""Numerical checks against the complete, separately supplied author file."""

import copy
import hashlib
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest

import torch
from torch.nn import functional as F

from torcheeg.models import DBConformer


@unittest.skipUnless(
    os.environ.get('TORCHEEG_DBCONFORMER_SOURCE'),
    'Complete author source is supplied separately.',
)
class TestDBConformerNative(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = Path(os.environ['TORCHEEG_DBCONFORMER_SOURCE'])
        source = cls.path.read_bytes()
        if hashlib.sha256(source).hexdigest() != DBConformer.SOURCE_SHA256:
            raise AssertionError('Unmodified pinned author source is required.')
        cls.original = ModuleType('dbconformer_test_reference')
        benchmark, deterministic = (
            torch.backends.cudnn.benchmark,
            torch.backends.cudnn.deterministic,
        )
        try:
            exec(compile(source, str(cls.path), 'exec'), cls.original.__dict__)
        finally:
            torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = (
                benchmark,
                deterministic,
            )

    def native(self, **config):
        args = SimpleNamespace(
            data_name='BNCI2014001'
            if config.get('drop_last_sample', True)
            else 'MI1-7',
            chn=config.get('num_electrodes', 22),
            patch_size=config.get('patch_size', 125),
            time_sample_num=config.get('chunk_size', 1001),
            class_num=config.get('num_classes', 4),
            spa_dim=config.get('spatial_channels', 16),
            gate_flag=config.get('fusion') == 'gated',
            posemb_flag=config.get('positional_embedding', True),
            branch='all',
            chn_atten_flag=config.get('fusion', 'attention') == 'attention',
        )
        return self.original.DBConformer(
            args,
            emb_size=config.get('hid_channels', 40),
            tem_depth=config.get('temporal_depth', 5),
            chn_depth=config.get('spatial_depth', 5),
            chn=args.chn,
            n_classes=args.class_num,
        )

    def test_complete_default_initialization_rng_and_states(self):
        torch.manual_seed(19)
        native = self.native()
        rng = torch.get_rng_state().clone()
        torch.manual_seed(19)
        model = DBConformer.from_source(self.path)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertEqual(len(model.backend.temporal_transformer), 5)
        self.assertEqual(len(model.backend.spatial_transformer), 5)
        self.assertEqual(model.backend.P, 8)
        self.assertEqual(set(model.native_state_dict()), set(native.state_dict()))
        self.assertEqual(
            set(model.state_dict()), {'backend.' + key for key in native.state_dict()}
        )
        for key, value in native.state_dict().items():
            torch.testing.assert_close(
                model.native_state_dict()[key], value, rtol=0, atol=0
            )
        model.load_native_state_dict(native.state_dict(), strict=True)
        with self.assertRaises(RuntimeError):
            model.load_native_state_dict({}, strict=True)

    def test_complete_train_eval_outputs_gradients_buffers_and_updates(self):
        for training in (False, True):
            with self.subTest(training=training):
                torch.manual_seed(71)
                native = self.native().train(training)
                model = DBConformer(copy.deepcopy(native)).train(training)
                left = torch.randn(3, 22, 1001, requires_grad=True)
                right = left.detach().clone().unsqueeze(1).requires_grad_()
                torch.manual_seed(43)
                actual_features, actual = model(left, return_features=True)
                rng = torch.get_rng_state().clone()
                torch.manual_seed(43)
                features, expected = native(right)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                torch.testing.assert_close(actual_features, features, rtol=0, atol=0)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                labels = torch.tensor([0, 1, 3])
                loss, reference_loss = (
                    F.cross_entropy(actual, labels),
                    F.cross_entropy(expected, labels),
                )
                torch.testing.assert_close(loss, reference_loss, rtol=0, atol=0)
                loss.backward()
                reference_loss.backward()
                torch.testing.assert_close(
                    left.grad, right.grad.squeeze(1), rtol=0, atol=0
                )
                for key, value in native.named_parameters():
                    parameter = dict(model.backend.named_parameters())[key]
                    self.assertIsNotNone(value.grad, key)
                    self.assertTrue(torch.isfinite(value.grad).all(), key)
                    torch.testing.assert_close(
                        parameter.grad, value.grad, rtol=0, atol=0
                    )
                for module in (model, native):
                    torch.optim.AdamW(module.parameters(), lr=0.001).step()
                for key, value in native.state_dict().items():
                    torch.testing.assert_close(
                        model.native_state_dict()[key], value, rtol=0, atol=0
                    )

    def test_native_fusion_positions_and_no_crop_variants(self):
        for fusion in ('attention', 'mean', 'gated'):
            for positions, crop in ((True, False), (False, True)):
                with self.subTest(fusion=fusion, positions=positions, crop=crop):
                    config = dict(
                        num_electrodes=6,
                        chunk_size=250,
                        num_classes=3,
                        hid_channels=20,
                        temporal_depth=2,
                        spatial_depth=1,
                        spatial_channels=8,
                        patch_size=50,
                        fusion=fusion,
                        positional_embedding=positions,
                        drop_last_sample=crop,
                    )
                    torch.manual_seed(23)
                    native = self.native(**config).eval()
                    torch.manual_seed(23)
                    model = DBConformer.from_source(self.path, **config).eval()
                    x = torch.randn(2, 1, 6, 250)
                    a = model(x, return_features=True)
                    b = native(x)
                    for left, right in zip(a, b):
                        torch.testing.assert_close(left, right, rtol=0, atol=0)
                    self.assertEqual(a[0].shape[-1], 20 if fusion == 'gated' else 40)

    def test_cudnn_settings_are_preserved(self):
        previous = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
        try:
            torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = (
                True,
                False,
            )
            DBConformer.from_source(self.path)
            self.assertTrue(torch.backends.cudnn.benchmark)
            self.assertFalse(torch.backends.cudnn.deterministic)
        finally:
            torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = (
                previous
            )

    def test_input_validation_eval_isolation_and_classifier_interface(self):
        model = DBConformer.from_source(self.path).eval()
        x = torch.randn(2, 22, 1001)
        expected = model(x)
        torch.testing.assert_close(model(x.unsqueeze(1)), expected, rtol=0, atol=0)
        changed = x.clone()
        changed[1] += 3
        torch.testing.assert_close(model(changed)[0], expected[0], rtol=0, atol=0)
        for bad in (torch.ones(2, 22, 1001, dtype=torch.long), 'bad'):
            with self.assertRaises(TypeError):
                model(bad)
        for bad in (x[:, :-1], x[..., :-1], x.unsqueeze(1).repeat(1, 2, 1, 1), x[0]):
            with self.assertRaises(ValueError):
                model(bad)
        backend = copy.deepcopy(model.backend)
        backend.branch = 'temporal'
        with self.assertRaises(ValueError):
            DBConformer(backend)
        with self.assertRaises(ValueError):
            DBConformer(model.backend, chunk_size=1000)

    def test_complete_state_optimizer_and_rng_resume(self):
        model = DBConformer.from_source(self.path)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        x, target = torch.randn(2, 22, 1001), torch.tensor([0, 2])

        def step(net, optim):
            optim.zero_grad()
            loss = F.cross_entropy(net(x), target)
            loss.backward()
            optim.step()
            return loss.detach()

        step(model, optimizer)
        state = copy.deepcopy(model.state_dict())
        opt_state = copy.deepcopy(optimizer.state_dict())
        rng = torch.get_rng_state().clone()
        expected = step(model, optimizer)
        restored = DBConformer.from_source(self.path)
        restored.load_state_dict(state, strict=True)
        other = torch.optim.AdamW(restored.parameters(), lr=0.001)
        other.load_state_dict(opt_state)
        torch.set_rng_state(rng)
        torch.testing.assert_close(step(restored, other), expected, rtol=0, atol=0)
        for key, value in restored.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)

    @unittest.skipUnless(hasattr(torch, 'compile'), 'torch.compile requires PyTorch 2.')
    def test_full_graph_eager_compile(self):
        model = DBConformer.from_source(self.path).eval()
        x = torch.randn(2, 22, 1001)
        torch.testing.assert_close(
            torch.compile(model, backend='eager', fullgraph=True)(x),
            model(x),
            rtol=0,
            atol=0,
        )


if __name__ == '__main__':
    unittest.main()
