import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from torcheeg.models import EEGMamba, EEGMambaClassifier
from torcheeg.models.transformer.eegmamba import _ssd_scan


class TinyEEGMamba(EEGMamba):
    def __init__(self, **kwargs):
        super().__init__(**dict(dict(num_layers=2, state_size=4), **kwargs))


def reference_state(state):
    result = {}
    for name, value in state.items():
        if name.endswith('bias') or name.endswith('mask_encoding'):
            result[name] = torch.zeros_like(value)
        elif name.endswith('dt_bias'):
            result[name] = torch.full_like(value, -3)
        elif name.endswith('.D') or (name.endswith('.weight') and value.ndim == 1):
            result[name] = torch.ones_like(value)
        else:
            offset = int(hashlib.sha256(name.encode()).hexdigest()[:6], 16) % 97
            values = torch.sin(torch.arange(value.numel(), dtype=torch.float64) / 17 + offset) * .02
            result[name] = values.reshape_as(value).to(value.dtype)
    return result


class TestEEGMamba(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eeg = (torch.sin(torch.arange(2400, dtype=torch.float32) / 17) +
                   torch.cos(torch.arange(2400, dtype=torch.float32) / 31)).reshape(2, 2, 3, 200)

    def test_complete_native_numerical_and_gradient_fixture(self):
        fixture = json.loads((Path(__file__).parent/'fixtures/eegmamba_reference.json').read_text())
        model = EEGMamba(block_size=8).eval()
        self.assertEqual(len(model.state_dict()), 117)
        model.load_state_dict(reference_state(model.state_dict()))
        x = self.eeg.clone().requires_grad_()
        mask = torch.zeros(2, 2, 3, dtype=torch.bool)
        mask[:, 0, 0] = True
        mask[0, 1, 2] = True
        features = model.forward_features(x, mask)
        output = model.proj_out(features)
        loss = output.square().mean() + features[..., ::3].mean() * .1
        loss.backward()
        for actual, key in [(features, 'features'), (output, 'reconstruction'), (x.grad, 'input_gradient')]:
            torch.testing.assert_close(actual, torch.tensor(fixture[key]), rtol=3e-4, atol=2e-6)
        self.assertAlmostEqual(loss.item(), fixture['loss'], places=6)
        self.assertEqual(set(dict(model.named_parameters())), set(fixture['gradients']))
        for name, parameter in model.named_parameters():
            expected = fixture['gradients'][name]
            if expected is None:
                self.assertIsNone(parameter.grad, name)
                continue
            torch.testing.assert_close(parameter.grad.flatten()[:8], torch.tensor(expected['first']),
                                       rtol=5e-4, atol=2e-6, msg=name)
            torch.testing.assert_close(parameter.grad.sum(), torch.tensor(expected['sum']), rtol=5e-4, atol=2e-5, msg=name)
            torch.testing.assert_close(parameter.grad.norm(), torch.tensor(expected['norm']), rtol=5e-4, atol=2e-5, msg=name)
        self.assertEqual(x.grad[mask].count_nonzero().item(), 0)

    def test_ssd_matches_independent_recurrence_and_all_gradients(self):
        torch.manual_seed(42)
        x = torch.randn(2, 11, 3, 2, dtype=torch.float64, requires_grad=True)
        dt = torch.rand(2, 11, 3, dtype=torch.float64, requires_grad=True)
        a = (-torch.rand(3, dtype=torch.float64)).requires_grad_()
        b = torch.randn(2, 11, 4, dtype=torch.float64, requires_grad=True)
        c = torch.randn(2, 11, 4, dtype=torch.float64, requires_grad=True)
        inputs = [x, dt, a, b, c]
        state = torch.zeros(2, 3, 2, 4, dtype=torch.float64)
        expected = []
        for index in range(x.shape[1]):
            state = state * torch.exp(dt[:, index]*a)[:, :, None, None]
            state = state + (x[:, index]*dt[:, index, :, None])[..., None] * b[:, index, None, None, :]
            expected.append((state*c[:, index, None, None, :]).sum(-1))
        expected = torch.stack(expected, dim=1)
        expected_grads = torch.autograd.grad(expected.square().sum(), inputs, retain_graph=True)
        for block_size in [1, 4, 8, 16]:
            actual = _ssd_scan(*inputs, block_size=block_size)
            torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
            actual_grads = torch.autograd.grad(actual.square().sum(), inputs, retain_graph=True)
            for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-10, atol=1e-10)
        # A future input must not affect an earlier causal SSM output.
        actual = _ssd_scan(*inputs, block_size=4)
        future_grad = torch.autograd.grad(actual[:, 0].sum(), x)[0][:, 1:]
        self.assertEqual(future_grad.count_nonzero().item(), 0)

    def test_masked_inputs_and_batch_isolation(self):
        model = TinyEEGMamba().eval()
        mask = torch.zeros(2, 2, 3, dtype=torch.bool)
        mask[:, :, 0] = True
        x = self.eeg.clone().requires_grad_()
        output = model(x, mask)
        changed = x.detach().clone()
        changed[mask] += 100
        torch.testing.assert_close(output, model(changed, mask), rtol=0, atol=0)
        output[0].sum().backward()
        self.assertEqual(x.grad[1].count_nonzero().item(), 0)
        self.assertEqual(x.grad[mask].count_nonzero().item(), 0)
        with torch.no_grad():
            # Keep the same batch/GEMM layout when testing sample isolation.
            # Different batch sizes may select different float32 CPU kernels.
            other_sample = x.detach().clone()
            other_sample[1] = other_sample[1]*13+7
            torch.testing.assert_close(model(other_sample)[:1], model(x)[:1], rtol=0, atol=0)
        self.assertEqual(model(x[:1, :1, :1]).shape, (1, 1, 1, 200))

    def test_training_dropout_optimizer_and_round_trip(self):
        model = TinyEEGMamba().train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        before = model.patch_embedding.spectral_proj[0].weight.detach().clone()
        output = model(self.eeg)
        self.assertFalse(torch.equal(output, model(self.eeg)))
        output.square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        optimizer.step()
        self.assertFalse(torch.equal(before, model.patch_embedding.spectral_proj[0].weight))
        model.eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.pt'
            torch.save(model.state_dict(), path)
            restored = TinyEEGMamba.from_pretrained(path).eval()
            torch.testing.assert_close(restored(self.eeg), model(self.eeg), rtol=0, atol=0)

    def test_classifier_freeze_full_backbone_and_gradients(self):
        backbone = TinyEEGMamba()
        model = EEGMambaClassifier(backbone, num_electrodes=2, num_patches=3, freeze_backbone=True)
        before = {name: value.clone() for name, value in backbone.state_dict().items()}
        model.train()
        self.assertFalse(backbone.training)
        self.assertTrue(model.classifier.training)
        loss = torch.nn.functional.cross_entropy(model(self.eeg), torch.tensor([0, 1]))
        loss.backward()
        self.assertTrue(all(p.grad is None for p in backbone.parameters()))
        self.assertTrue(all(p.grad is not None for p in model.classifier.parameters()))
        for name, value in backbone.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        trainable = EEGMambaClassifier(TinyEEGMamba(), num_electrodes=2, num_patches=3).eval()
        trainable(self.eeg).square().sum().backward()
        self.assertIsNotNone(trainable.backbone.encoder.layers[0].mixer.A_log.grad)
        self.assertIsNone(trainable.backbone.proj_out[0].weight.grad)

    def test_strict_loading_and_corruptions(self):
        state = TinyEEGMamba().state_dict()
        key = 'encoder.layers.0.mixer.A_log'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.pt'
            torch.save({'module.'+name: value for name, value in state.items()}, path)
            TinyEEGMamba.from_pretrained(path, prefix='module.')
            with self.assertRaises(ValueError): TinyEEGMamba.from_pretrained(path)
            for value in [None, state[key][:1], state[key].double(), torch.full_like(state[key], float('nan'))]:
                bad = dict(state)
                if value is None: del bad[key]
                else: bad[key] = value
                torch.save(bad, path)
                with self.assertRaises(ValueError): TinyEEGMamba.from_pretrained(path)
            for bad in [dict(state, extra=torch.zeros(1)), dict(model=state), dict(state, bad='not a tensor')]:
                torch.save(bad, path)
                with self.assertRaises(ValueError): TinyEEGMamba.from_pretrained(path)
            safe = Path(directory)/'model.safetensors'
            save_file(state, safe)
            loaded = TinyEEGMamba.from_pretrained(safe)
            for name, value in loaded.state_dict().items():
                torch.testing.assert_close(value, state[name], rtol=0, atol=0)
            mixed = {'module.'+name: value for name, value in state.items()}
            mixed['extra'] = torch.zeros(1)
            torch.save(mixed, path)
            with self.assertRaises(ValueError): TinyEEGMamba.from_pretrained(path, prefix='module.')

    def test_invalid_configuration_and_inputs(self):
        for kwargs in [dict(num_layers=False), dict(state_size=0), dict(head_dim=3), dict(block_size=0)]:
            with self.assertRaises(ValueError): EEGMamba(**kwargs)
        model = TinyEEGMamba()
        for x in [self.eeg.long(), self.eeg[..., :199], self.eeg[:0], self.eeg[:, :0], torch.zeros(2, 200)]:
            with self.assertRaises(ValueError): model(x)
        for mask in [torch.zeros(2, 2, 3), torch.zeros(2, 2, dtype=torch.bool), []]:
            with self.assertRaises(ValueError): model(self.eeg, mask)
        with self.assertRaises(ValueError): TinyEEGMamba.from_pretrained('missing', prefix=None)
        with self.assertRaises(ValueError): EEGMambaClassifier(model, dropout=1)
        with self.assertRaises(ValueError): EEGMambaClassifier(model, freeze_backbone=1)
        with self.assertRaises(ValueError): EEGMambaClassifier(model)(self.eeg)


if __name__ == '__main__':
    unittest.main()
