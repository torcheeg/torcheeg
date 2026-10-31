"""Compare the complete pinned author ADFCNN, including its classifier."""

import copy
import hashlib
import importlib.util
import sys
import subprocess
import os
from pathlib import Path
from types import ModuleType
import unittest

import torch
from torch.nn import functional as F

from torcheeg.models import ADFCNN


@unittest.skipUnless(
    os.environ.get('TORCHEEG_ADFCNN_SOURCE'),
    'Complete author source is supplied separately.',
)
class TestADFCNNNative(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = Path(os.environ['TORCHEEG_ADFCNN_SOURCE'])
        for name, digest in ADFCNN.SOURCE_HASHES.items():
            if hashlib.sha256((cls.path / name).read_bytes()).hexdigest() != digest:
                raise AssertionError('Both complete pinned source files are required.')
        cls.previous = {
            name: sys.modules.get(name)
            for name in ('model', 'model.layers', 'model.ADFCNN')
        }
        package = ModuleType('model')
        package.__path__ = [str(cls.path)]
        sys.modules['model'] = package
        for name in ('layers', 'ADFCNN'):
            spec = importlib.util.spec_from_file_location(
                'model.' + name, cls.path / (name + '.py')
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
        cls.original = sys.modules['model.ADFCNN']

    @classmethod
    def tearDownClass(cls):
        for name, digest in ADFCNN.SOURCE_HASHES.items():
            if hashlib.sha256((cls.path / name).read_bytes()).hexdigest() != digest:
                raise AssertionError('Author source changed during testing.')
        for name, previous in cls.previous.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous

    def test_complete_default_initialization_rng_and_native_states(self):
        torch.manual_seed(19)
        native = self.original.Net(num_channels=22, num_classes=4, sampling_rate=250)
        rng = torch.get_rng_state().clone()
        torch.manual_seed(19)
        model = ADFCNN.from_source(self.path)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertEqual(set(model.native_state_dict()), set(native.state_dict()))
        self.assertEqual(
            set(model.state_dict()), {'backend.' + key for key in native.state_dict()}
        )
        for key, value in native.state_dict().items():
            torch.testing.assert_close(
                model.native_state_dict()[key], value, rtol=0, atol=0
            )
        self.assertEqual(model.backend.backbone.spectral_1[0].kernel_size, (1, 125))
        self.assertEqual(model.backend.classifier.dense[0].kernel_size, (1, 51))
        model.load_native_state_dict(native.state_dict(), strict=True)
        with self.assertRaises(RuntimeError):
            model.load_native_state_dict({}, strict=True)

    def test_native_train_eval_loss_all_gradients_bn_and_updates(self):
        for training in (False, True):
            with self.subTest(training=training):
                native = self.original.Net(
                    num_channels=22, num_classes=4, sampling_rate=250
                ).train(training)
                with torch.no_grad():
                    for layer in native.modules():
                        if isinstance(layer, torch.nn.Conv2d) and hasattr(
                            layer, 'max_norm'
                        ):
                            layer.weight.mul_(30)
                model = ADFCNN.from_source(self.path).train(training)
                model.load_native_state_dict(native.state_dict(), strict=True)
                left = torch.randn(3, 22, 750, requires_grad=True)
                right = left.detach().clone().unsqueeze(1).requires_grad_()
                torch.manual_seed(43)
                actual = model(left)
                rng = torch.get_rng_state().clone()
                torch.manual_seed(43)
                expected = native(right)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(
                    actual.exp().sum(1), torch.ones(3), rtol=1e-6, atol=1e-6
                )
                target = torch.tensor([0, 1, 3])
                loss, other = F.nll_loss(actual, target), F.nll_loss(expected, target)
                torch.testing.assert_close(loss, other, rtol=0, atol=0)
                loss.backward()
                other.backward()
                torch.testing.assert_close(
                    left.grad, right.grad.squeeze(1), rtol=0, atol=0
                )
                for key, parameter in native.named_parameters():
                    self.assertIsNotNone(parameter.grad, key)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), key)
                    torch.testing.assert_close(
                        dict(model.backend.named_parameters())[key].grad,
                        parameter.grad,
                        rtol=0,
                        atol=0,
                    )
                for net in (model, native):
                    torch.optim.AdamW(net.parameters(), lr=0.001).step()
                for key, value in native.state_dict().items():
                    torch.testing.assert_close(
                        model.native_state_dict()[key], value, rtol=0, atol=0
                    )

    def test_native_minimum_and_alternative_geometries(self):
        for channels, length, classes in ((3, 750, 2), (21, 767, 2), (6, 751, 3)):
            with self.subTest(channels=channels, length=length, classes=classes):
                torch.manual_seed(23)
                native = self.original.Net(
                    num_channels=channels, num_classes=classes, sampling_rate=250
                ).eval()
                torch.manual_seed(23)
                model = ADFCNN.from_source(
                    self.path,
                    num_electrodes=channels,
                    chunk_size=length,
                    num_classes=classes,
                ).eval()
                x = torch.randn(2, 1, channels, length)
                torch.testing.assert_close(model(x), native(x), rtol=0, atol=0)
                self.assertEqual(model(x).shape, (2, classes))

    def test_input_validation_and_eval_sample_isolation(self):
        model = ADFCNN.from_source(self.path).eval()
        x = torch.randn(2, 22, 750)
        actual = model(x)
        torch.testing.assert_close(model(x.unsqueeze(1)), actual, rtol=0, atol=0)
        altered = x.clone()
        altered[1] += 3
        torch.testing.assert_close(model(altered)[0], actual[0], rtol=0, atol=0)
        self.assertTrue(torch.isfinite(model(torch.zeros_like(x))).all())
        for bad in (torch.ones(2, 22, 750, dtype=torch.long), 'bad'):
            with self.assertRaises(TypeError):
                model(bad)
        for bad in (x[:, :-1], x[..., :-1], x[0], x.unsqueeze(1).repeat(1, 2, 1, 1)):
            with self.assertRaises(ValueError):
                model(bad)

    def test_complete_model_optimizer_and_rng_next_step_resume(self):
        model = ADFCNN.from_source(self.path)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        x, target = torch.randn(2, 22, 750), torch.tensor([0, 2])

        def step(net, optim):
            optim.zero_grad()
            loss = F.nll_loss(net(x), target)
            loss.backward()
            optim.step()
            return loss.detach()

        step(model, optimizer)
        state, opt_state = (
            copy.deepcopy(model.state_dict()),
            copy.deepcopy(optimizer.state_dict()),
        )
        rng = torch.get_rng_state().clone()
        expected = step(model, optimizer)
        other = ADFCNN.from_source(self.path)
        other.load_state_dict(state, strict=True)
        optim = torch.optim.AdamW(other.parameters(), lr=0.001)
        optim.load_state_dict(opt_state)
        torch.set_rng_state(rng)
        torch.testing.assert_close(step(other, optim), expected, rtol=0, atol=0)
        for key, value in other.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)

    @unittest.skipUnless(hasattr(torch, 'compile'), 'torch.compile requires PyTorch 2.')
    def test_eager_full_graph_compile(self):
        model = ADFCNN.from_source(self.path).eval()
        x = torch.randn(2, 22, 750)
        torch.testing.assert_close(
            torch.compile(model, backend='eager', fullgraph=True)(x),
            model(x),
            rtol=0,
            atol=0,
        )

    def test_cold_loader_keeps_unrelated_modules_and_builtin_import(self):
        root = Path(__file__).resolve().parents[2]
        script = """
import builtins, sys, types, torch
from torcheeg.models import ADFCNN
sentinel = types.ModuleType('model')
layers = types.ModuleType('model.layers')
sys.modules['model'] = sentinel
sys.modules['model.layers'] = layers
original_import = builtins.__import__
model = ADFCNN.from_source(sys.argv[1]).eval()
assert sys.modules['model'] is sentinel
assert sys.modules['model.layers'] is layers
assert builtins.__import__ is original_import
assert torch.isfinite(model(torch.randn(2,22,750))).all()
print('Complete cold loader preserves existing module namespace.')
"""
        result = subprocess.run(
            [sys.executable, '-c', script, str(self.path)],
            env=dict(os.environ, PYTHONPATH=str(root)),
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
