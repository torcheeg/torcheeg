"""Compare the complete pinned author EEGWaveNet, including its classifier."""

import copy
import hashlib
import os
from pathlib import Path
from types import ModuleType
import unittest

import torch
from torch.nn import functional as F

from torcheeg.models import EEGWaveNet


@unittest.skipUnless(
    os.environ.get('TORCHEEG_EEGWAVENET_SOURCE'),
    'Complete author source is supplied separately.',
)
class TestEEGWaveNetNative(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = Path(os.environ['TORCHEEG_EEGWAVENET_SOURCE'])
        source = cls.path.read_bytes()
        if hashlib.sha256(source).hexdigest() != EEGWaveNet.SOURCE_SHA256:
            raise AssertionError(
                'The complete unmodified pinned author file is required.'
            )
        cls.original = ModuleType('eegwavenet_test_reference')
        exec(compile(source, str(cls.path), 'exec'), cls.original.__dict__)

    def test_complete_default_initialization_rng_and_native_states(self):
        torch.manual_seed(19)
        native = self.original.Net(22, 4)
        rng = torch.get_rng_state().clone()
        torch.manual_seed(19)
        model = EEGWaveNet.from_source(self.path)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertEqual(set(model.native_state_dict()), set(native.state_dict()))
        self.assertEqual(
            set(model.state_dict()), {'backend.' + key for key in native.state_dict()}
        )
        for key, value in native.state_dict().items():
            torch.testing.assert_close(
                model.native_state_dict()[key], value, rtol=0, atol=0
            )
        self.assertEqual(model.backend.temp_conv6.groups, 22)
        self.assertEqual(model.backend.chpool5[0].padding, (0,))
        model.load_native_state_dict(native.state_dict(), strict=True)
        with self.assertRaises(RuntimeError):
            model.load_native_state_dict({}, strict=True)

    def test_native_train_eval_loss_all_gradients_bn_and_updates(self):
        for training in (False, True):
            with self.subTest(training=training):
                native = self.original.Net(22, 4).train(training)
                model = EEGWaveNet(copy.deepcopy(native)).train(training)
                left = torch.randn(3, 22, 1000, requires_grad=True)
                right = left.detach().clone().requires_grad_()
                actual, expected = model(left), native(right)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(
                    actual.exp().sum(1), torch.ones(3), rtol=1e-6, atol=1e-6
                )
                target = torch.tensor([0, 1, 3])
                loss, other = F.nll_loss(actual, target), F.nll_loss(expected, target)
                torch.testing.assert_close(loss, other, rtol=0, atol=0)
                loss.backward()
                other.backward()
                torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
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
        for channels, length, classes in ((3, 448, 2), (21, 1024, 2), (6, 1001, 3)):
            with self.subTest(channels=channels, length=length, classes=classes):
                torch.manual_seed(23)
                native = self.original.Net(channels, classes).eval()
                torch.manual_seed(23)
                model = EEGWaveNet.from_source(
                    self.path,
                    num_electrodes=channels,
                    chunk_size=length,
                    num_classes=classes,
                ).eval()
                x = torch.randn(2, 1, channels, length)
                torch.testing.assert_close(
                    model(x), native(x.squeeze(1)), rtol=0, atol=0
                )
                self.assertEqual(model(x).shape, (2, classes))

    def test_input_validation_and_eval_sample_isolation(self):
        model = EEGWaveNet.from_source(self.path).eval()
        x = torch.randn(2, 22, 1000)
        actual = model(x)
        torch.testing.assert_close(model(x.unsqueeze(1)), actual, rtol=0, atol=0)
        altered = x.clone()
        altered[1] += 3
        torch.testing.assert_close(model(altered)[0], actual[0], rtol=0, atol=0)
        for bad in (torch.ones(2, 22, 1000, dtype=torch.long), 'bad'):
            with self.assertRaises(TypeError):
                model(bad)
        for bad in (x[:, :-1], x[..., :-1], x[0], x.unsqueeze(1).repeat(1, 2, 1, 1)):
            with self.assertRaises(ValueError):
                model(bad)

    def test_complete_model_optimizer_and_rng_next_step_resume(self):
        model = EEGWaveNet.from_source(self.path)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        x, target = torch.randn(2, 22, 1000), torch.tensor([0, 2])

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
        other = EEGWaveNet.from_source(self.path)
        other.load_state_dict(state, strict=True)
        optim = torch.optim.AdamW(other.parameters(), lr=0.001)
        optim.load_state_dict(opt_state)
        torch.set_rng_state(rng)
        torch.testing.assert_close(step(other, optim), expected, rtol=0, atol=0)
        for key, value in other.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)

    @unittest.skipUnless(hasattr(torch, 'compile'), 'torch.compile requires PyTorch 2.')
    def test_eager_full_graph_compile(self):
        model = EEGWaveNet.from_source(self.path).eval()
        x = torch.randn(2, 22, 1000)
        torch.testing.assert_close(
            torch.compile(model, backend='eager', fullgraph=True)(x),
            model(x),
            rtol=0,
            atol=0,
        )


if __name__ == '__main__':
    unittest.main()
